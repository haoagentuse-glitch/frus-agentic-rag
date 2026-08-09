"""External judge for answer correctness.

Deliberately kept off the evidence path. The judge never supplies facts to the
system under test and never sees the corpus beyond the gold documents; it only
scores an already-produced answer. The closed-corpus rule ("FRUS or abstain")
is about where evidence comes from, not about how a report is scored.

Falls back to a deterministic verdict when no key is configured, so the eval
still runs offline — with `judge: "unavailable"` recorded rather than a
silently invented score.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from functools import lru_cache

import httpx
from pydantic import BaseModel

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9.]+)')


class Verdict(BaseModel):
    # None when the judge could not reach a verdict. A failed call used to score
    # 0.0, which averages in as "answered wrongly": one ablation variant
    # exhausted the Gemini daily quota partway through, recorded 195 errors as
    # zeros, and reported a correctness of 0.000 that looked like a catastrophic
    # regression rather than a missing measurement.
    score: float | None  # 0.0 - 1.0, or None if unjudged
    correct: bool | None
    reason: str = ""
    judge: str = "gemini"


SYSTEM = """You score an answer about US diplomatic history against a gold reference.

The REFERENCE section carries the text of the documents the answer was supposed
to be based on. Where a document was too long to include whole it is marked
[TRUNCATED]; treat material missing for that reason as unverified, not as false.

Score 1.0 if the answer states what the reference supports, with no invented
facts. Score 0.0 if it CONTRADICTS the reference, or invents specifics of a kind
the reference shows are wrong. An answer that is accurate but partial — it
reports the document faithfully and stops short — scores 0.6-0.9, not 0.0.

Reply with JSON only: {"score": <0-1>, "correct": <bool>, "reason": "<one sentence>"}"""


def _env(name: str, default: str = "") -> str:
    """.env files written on Windows carry a trailing \\r that poisons URLs and headers."""
    return (os.getenv(name) or default).strip()


def _endpoint() -> str:
    base = _env("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")
    # The key in .env points at an /interactions path used by another project;
    # the models endpoint is what generateContent needs.
    base = base.replace("/interactions", "").rstrip("/")
    model = _env("GEMINI_MODEL", "gemini-3.5-flash-lite")
    return f"{base}/models/{model}:generateContent"


def provider() -> str:
    """Which judge to call. DeepSeek wins when both keys are present.

    Not a preference for the model so much as for the billing: the Gemini free
    tier is 500 requests a day, and running out of it mid-sweep is what turned
    one ablation variant into 195 errors recorded as zeros. A paid endpoint
    removes the failure mode for less than a dollar over the whole D-series.
    """
    if _env("DEEPSEEK_API_KEY"):
        return "deepseek"
    if _env("GEMINI_API_KEY"):
        return "gemini"
    return "none"


def available() -> bool:
    return provider() != "none"


@lru_cache(maxsize=512)
def gold_excerpt(document_id: str, max_chars: int = 60000) -> str:
    """A gold document's text, as complete as the budget allows.

    The judge was previously handed bare FRUS ids such as `frus1949v07p1:d89`,
    which say nothing about what the document contains. It could only score
    fluency, and it marked answers correct that cited nothing and hit no gold
    document. Scoring against the reference requires the reference.

    It then required only 2,500 characters of the first four chunks, while the
    prompt told the judge it was reading the full text and to score 0.0 for any
    specific the reference did not carry. Measured across the gold set the judge
    saw 29.2% of what the model saw, and 3.4% on the worst case — so an answer
    drawn correctly from the second half of a despatch was instructed to be
    marked wrong. That is not strictness, it is a rubric applied to a document
    the judge was not shown. 60,000 characters covers every gold document here
    but the largest, and costs about a cent a sweep.
    """
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect

    try:
        volume_id, doc_id = document_id.split(":", 1)
        tbl = connect().open_table(CHUNKS_TABLE)
        rows = (
            tbl.search()
            .where(f"volume_id = '{volume_id}' AND document_id = '{doc_id}'")
            .select(["chunk_id", "head", "text", "ordinal"])
            .limit(200)
            .to_list()
        )
    except Exception:
        return ""
    if not rows:
        return ""
    rows.sort(key=lambda r: r.get("ordinal", 0))
    head = rows[0].get("head", "")
    joined = " ".join(r["text"] for r in rows)
    body = joined[:max_chars]
    if len(joined) > max_chars:
        # Say so, so the rubric can treat the gap as unverified rather than as
        # a fact the document contradicts.
        body += "\n[TRUNCATED]"
    return f"[{document_id}] {head}\n{body}"


def _reference_block(gold_documents: list[str]) -> str:
    parts = [t for d in gold_documents if (t := gold_excerpt(d))]
    return "\n\n---\n\n".join(parts) if parts else "(gold documents unavailable)"


async def judge_answer(
    question: str,
    answer_text: str,
    gold_documents: list[str],
    answerable: bool,
    outcome: str,
    timeout: float = 120.0,
) -> Verdict:
    # Abstention correctness is decidable without a model, so never spend a
    # judge call on it — the free tier is 500 requests a day.
    if not answerable:
        ok = outcome == "abstain"
        return Verdict(
            score=1.0 if ok else 0.0,
            correct=ok,
            reason="unanswerable: abstention is the correct outcome",
            judge="deterministic",
        )
    if outcome == "abstain":
        return Verdict(
            score=0.0,
            correct=False,
            reason="abstained on an answerable question",
            judge="deterministic",
        )
    if not available():
        # Also None, for the same reason: a run without a key has not measured
        # correctness, and reporting it as zero understates every system equally
        # while looking like a result.
        return Verdict(
            score=None,
            correct=None,
            reason="no GEMINI_API_KEY configured",
            judge="unavailable",
        )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE — the gold FRUS documents, in full text:\n"
        f"{_reference_block(gold_documents)}\n\n"
        f"ANSWER UNDER TEST:\n{answer_text[:6000]}\n\nScore it."
    )
    who = provider()
    if who == "deepseek":
        # OpenAI-compatible. The legacy deepseek-chat / deepseek-reasoner
        # aliases were retired on 2026-07-24; v4-flash and v4-pro replace them.
        base = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        base = base.rstrip("/")
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {_env('DEEPSEEK_API_KEY')}"}
        payload = {
            "model": _env("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        def _extract(data: dict) -> str:
            return data["choices"][0]["message"]["content"]
    else:
        url = _endpoint()
        headers = {"x-goog-api-key": _env("GEMINI_API_KEY")}
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }

        def _extract(data: dict) -> str:
            return data["candidates"][0]["content"]["parts"][0]["text"]

    text = ""
    last_exc: Exception | None = None
    # Timeouts on a home connection are transient; a judge that gives up on the
    # first one silently turns into "everything scored 0".
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(url, headers=headers, json=payload)
                r.raise_for_status()
                text = _extract(r.json())
            break
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                await asyncio.sleep(2 * (attempt + 1))
    if not text:
        # str(ReadTimeout) is empty, so the class name has to carry the diagnosis.
        detail = f"{type(last_exc).__name__}: {last_exc}".rstrip(": ")
        return Verdict(score=None, correct=None, reason=f"judge error: {detail}", judge="error")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _SCORE_RE.search(text)
        if not m:
            return Verdict(
                score=None, correct=None, reason="unparseable judge reply", judge="error"
            )
        data = {"score": float(m.group(1)), "correct": float(m.group(1)) >= 0.5}
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    return Verdict(
        score=score,
        correct=bool(data.get("correct", score >= 0.5)),
        reason=str(data.get("reason", ""))[:300],
        judge=who,
    )


async def health() -> dict:
    if not available():
        return {"available": False, "reason": "no DEEPSEEK_API_KEY or GEMINI_API_KEY"}
    v = await judge_answer("test", "The sky is blue.", ["frusTEST:d1"], True, "answer")
    return {"available": v.judge in ("gemini", "deepseek"), "judge": v.judge, "reason": v.reason}


def health_sync() -> dict:
    return asyncio.run(health())
