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

import httpx
from pydantic import BaseModel

_SCORE_RE = re.compile(r'"score"\s*:\s*([0-9.]+)')


class Verdict(BaseModel):
    score: float  # 0.0 - 1.0
    correct: bool
    reason: str = ""
    judge: str = "gemini"


SYSTEM = """You score an answer about US diplomatic history against a gold reference.

Score 1.0 only if the answer states what the gold documents support, with no
invented facts. Score 0.0 if it contradicts them, or asserts specifics the gold
does not carry. A correct refusal to answer an unanswerable question scores 1.0.
An answer that hedges but gets the substance right scores 0.6-0.9.

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


def available() -> bool:
    return bool(_env("GEMINI_API_KEY"))


async def judge_answer(
    question: str,
    answer_text: str,
    gold_documents: list[str],
    answerable: bool,
    outcome: str,
    timeout: float = 60.0,
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
        return Verdict(
            score=0.0,
            correct=False,
            reason="no GEMINI_API_KEY configured",
            judge="unavailable",
        )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"GOLD DOCUMENTS (FRUS ids):\n{', '.join(gold_documents) or '(none)'}\n\n"
        f"ANSWER UNDER TEST:\n{answer_text[:6000]}\n\nScore it."
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                _endpoint(),
                headers={"x-goog-api-key": _env("GEMINI_API_KEY")},
                json=payload,
            )
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as exc:
        return Verdict(score=0.0, correct=False, reason=f"judge error: {exc}", judge="error")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = _SCORE_RE.search(text)
        if not m:
            return Verdict(
                score=0.0, correct=False, reason="unparseable judge reply", judge="error"
            )
        data = {"score": float(m.group(1)), "correct": float(m.group(1)) >= 0.5}
    score = max(0.0, min(1.0, float(data.get("score", 0.0))))
    return Verdict(
        score=score,
        correct=bool(data.get("correct", score >= 0.5)),
        reason=str(data.get("reason", ""))[:300],
    )


async def health() -> dict:
    if not available():
        return {"available": False, "reason": "GEMINI_API_KEY unset"}
    v = await judge_answer("test", "The sky is blue.", ["frusTEST:d1"], True, "answer")
    return {"available": v.judge == "gemini", "judge": v.judge, "reason": v.reason}


def health_sync() -> dict:
    return asyncio.run(health())
