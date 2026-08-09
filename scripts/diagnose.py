"""D-series diagnostics: find the cause before building the next version.

Each experiment changes one thing. Everything else is frozen — same questions,
same generation prompt, same model, same context ceiling, same retrieval and
call budgets, same reranker, same judge, same abstention rule. Where an
experiment has to break that, it says so in its own record rather than being
compared against the others.

    python scripts/diagnose.py d0    # rebuild the per-question table, no model
    python scripts/diagnose.py d2    # gold evidence handed straight to the model
    python scripts/diagnose.py d3 --models qwen3:4b-instruct,qwen3:8b
    python scripts/diagnose.py d6    # classify the retrieval misses

D1 (citation mechanism) is not here: it is a code change that already landed,
and its experiment is the paired ablation with FRUS_CITATION_MODE=model.
D4 is a config flag (FRUS_CONTEXT_EXPAND_NEIGHBOURS) measured through the normal
evaluator. D5 needs a graph node that does not exist yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

CASES = Path("eval/gold_cases.jsonl")
REPORTS = Path("reports")


def load_cases() -> dict:
    return {
        json.loads(x)["case_id"]: json.loads(x) for x in CASES.read_text().splitlines() if x.strip()
    }


# --- D0 -------------------------------------------------------------------


def d0(args) -> None:
    """One row per question, from records that already exist. No model calls.

    Deliberately reports which fields are derivable from the stored runs and
    which are not: `n_gold_in_context` needs the synthesis window, which the
    run records never captured. Leaving it null is the honest answer; filling it
    from document recall would repeat the mistake this whole exercise exposed —
    a cited document is not evidence delivered.
    """
    cases = load_cases()
    rows: list[dict] = []
    for path in sorted(REPORTS.glob("runs_*.jsonl")):
        variant = path.stem.replace("runs_", "")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            case = cases.get(r["case_id"], {})
            rows.append(
                {
                    "variant": variant,
                    "system": r["system"],
                    "case_id": r["case_id"],
                    "kind": r["kind"],
                    "language": r["language"],
                    "answerable": r["answerable"],
                    "n_gold": r.get("n_gold"),
                    "gold_in_top20": r.get("candidate_recall"),
                    "gold_in_union": r.get("agent_union_recall"),
                    "gold_accepted": r.get("accepted_recall"),
                    "gold_cited": r.get("citation_recall"),
                    "n_gold_hit_union": r.get("n_gold_hit_union"),
                    # Not derivable from stored runs: see docstring.
                    "evidence_chunk_in_context": None,
                    "abstained": r["outcome"] != "answer",
                    "abstain_reason": _abstain_reason(r),
                    "citation_valid": (r.get("n_citations") or 0) > 0,
                    "correct": r.get("judge_correct"),
                    "judge": r.get("judge"),
                    "retrieval_calls": r.get("retrieval_calls"),
                    "llm_calls": r.get("llm_calls"),
                    "n_context_docs": r.get("n_accepted_docs"),
                    "latency_s": r.get("latency_s"),
                    "notes": (case.get("notes") or "")[:120],
                }
            )
    out = REPORTS / "d0_diagnostic_table.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"{len(rows)} rows -> {out}")
    ans = [r for r in rows if r["answerable"]]
    print("abstain reasons:", Counter(r["abstain_reason"] for r in ans if r["abstained"]))
    print("by kind, correct rate:")
    for kind in sorted({r["kind"] for r in ans}):
        sub = [r for r in ans if r["kind"] == kind and r["correct"] is not None]
        if sub:
            print(
                f"  {kind:20s} n={len(sub):4d} "
                f"{sum(bool(r['correct']) for r in sub) / len(sub):.3f}"
            )


def _abstain_reason(r: dict) -> str:
    if r["outcome"] == "answer":
        return ""
    if not r["answerable"]:
        return "correctly abstained"
    if (r.get("agent_union_recall") or 0) == 0:
        return "no gold retrieved"
    if (r.get("n_citations") or 0) == 0:
        return "citation gate"
    return "other"


# --- D2 -------------------------------------------------------------------


async def d2(args) -> None:
    """Hand the model the gold documents in full. No retrieval at all.

    The single most informative experiment available, and nearly free: it
    separates "the pipeline did not deliver the evidence" from "the model cannot
    use the evidence". If accuracy here is far above the pipeline's, retrieval
    and context assembly are the bottleneck; if it is barely above, further
    retrieval work has little left to win.

    Retrieval budget is 0 by construction — recorded, not compared.
    """
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.prompts import SYNTH_SYSTEM_EN, SYNTH_SYSTEM_ZH, synth_user
    from frus_agentic_rag.agent.schemas import AgentAnswer
    from frus_agentic_rag.config import get_settings
    from frus_agentic_rag.evaluation.judge import judge_answer

    cases = [c for c in load_cases().values() if c.get("answerable", True)]
    if args.limit:
        cases = cases[: args.limit]
    settings = get_settings()
    client = get_client()
    rows: list[dict] = []

    for n, case in enumerate(cases, 1):
        gold = _gold_evidence(case.get("gold_documents") or [])
        if not gold:
            continue
        for lang in ("zh-TW", "en"):
            q = case["question_zh"] if lang == "zh-TW" else case["question_en"]
            try:
                draft = await client.structured(
                    SYNTH_SYSTEM_ZH if lang == "zh-TW" else SYNTH_SYSTEM_EN,
                    synth_user(q, gold),
                    AgentAnswer,
                    num_predict=settings.ollama_num_predict_synthesis,
                )
                text = draft.answer_text
            except Exception as exc:
                text = ""
                print(f"  synthesis failed: {type(exc).__name__}: {exc}", flush=True)
            v = await judge_answer(
                q, text, case.get("gold_documents") or [], True, "answer" if text else "abstain"
            )
            rows.append(
                {
                    "case_id": case["case_id"],
                    "kind": case["kind"],
                    "language": lang,
                    "n_gold_docs": len(case.get("gold_documents") or []),
                    "n_gold_chunks": len(gold),
                    "answered": bool(text),
                    "judge_score": v.score,
                    "judge": v.judge,
                    "answer_preview": text[:200],
                }
            )
        print(f"[{n}/{len(cases)}] {case['case_id']}", flush=True)

    (REPORTS / "d2_gold_evidence_direct.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    )
    _report_d2(rows)


def _gold_evidence(document_ids: list[str]) -> list:
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect
    from frus_agentic_rag.models import Evidence

    tbl = connect().open_table(CHUNKS_TABLE)
    out = []
    for doc in document_ids:
        vol, _, did = doc.partition(":")
        if not re.fullmatch(r"[\w\-.]+", vol) or not re.fullmatch(r"[\w\-.]+", did):
            continue
        rows = (
            tbl.search()
            .where(f"volume_id = '{vol}' AND document_id = '{did}'")
            .select(
                [
                    "chunk_id",
                    "volume_id",
                    "document_id",
                    "doc_number",
                    "subtype",
                    "head",
                    "date_from",
                    "date_to",
                    "text",
                ]
            )
            .limit(200)
            .to_list()
        )
        for r in sorted(rows, key=lambda x: x["chunk_id"]):
            out.append(
                Evidence(
                    evidence_id=r["chunk_id"],
                    volume_id=r["volume_id"],
                    document_id=r["document_id"],
                    doc_number=r.get("doc_number", ""),
                    subtype=r["subtype"],
                    head=r.get("head", ""),
                    date_from=r.get("date_from", ""),
                    date_to=r.get("date_to", ""),
                    text=r["text"],
                    score=1.0,
                )
            )
    return out


def _report_d2(rows: list[dict]) -> None:
    scored = [r for r in rows if r["judge_score"] is not None]
    if not scored:
        print("no judged rows (judge unavailable?)")
        return
    m = sum(r["judge_score"] for r in scored) / len(scored)
    print(f"\nD2 gold-evidence-direct correctness: {m:.3f}  (n={len(scored)})")
    for key in ("kind", "language"):
        print(f"  by {key}:")
        for k in sorted({r[key] for r in scored}):
            sub = [r["judge_score"] for r in scored if r[key] == k]
            print(f"    {k:20s} n={len(sub):3d} {sum(sub) / len(sub):.3f}")
    print(
        "\nCompare against the pipeline's own answerable correctness. A large gap "
        "means retrieval and context assembly; a small one means the generator."
    )


# --- D3 -------------------------------------------------------------------


async def d3(args) -> None:
    """The same gold evidence through several generators.

    Runs only where D2 already failed, and never re-retrieves: comparing models
    through a full RAG pipeline would let retrieval noise decide which model
    looks better.
    """
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.prompts import SYNTH_SYSTEM_EN, SYNTH_SYSTEM_ZH, synth_user
    from frus_agentic_rag.agent.schemas import AgentAnswer
    from frus_agentic_rag.evaluation.judge import judge_answer

    src = REPORTS / "d2_gold_evidence_direct.jsonl"
    if not src.exists():
        raise SystemExit("run d2 first")
    failures = [
        json.loads(x)
        for x in src.read_text().splitlines()
        if x.strip() and (json.loads(x)["judge_score"] or 0) < 1.0
    ]
    cases = load_cases()
    models = args.models.split(",")
    rows: list[dict] = []
    for model in models:
        client = get_client()
        client.model = model  # type: ignore[attr-defined]
        for n, f in enumerate(failures, 1):
            case = cases[f["case_id"]]
            q = case["question_zh"] if f["language"] == "zh-TW" else case["question_en"]
            gold = _gold_evidence(case.get("gold_documents") or [])
            try:
                draft = await client.structured(
                    SYNTH_SYSTEM_ZH if f["language"] == "zh-TW" else SYNTH_SYSTEM_EN,
                    synth_user(q, gold),
                    AgentAnswer,
                    num_predict=3072,
                )
                text = draft.answer_text
            except Exception:
                text = ""
            v = await judge_answer(
                q, text, case.get("gold_documents") or [], True, "answer" if text else "abstain"
            )
            rows.append(
                {
                    "model": model,
                    "case_id": f["case_id"],
                    "language": f["language"],
                    "kind": f["kind"],
                    "judge_score": v.score,
                    "was": f["judge_score"],
                }
            )
            print(f"[{model} {n}/{len(failures)}] {f['case_id']}", flush=True)
    (REPORTS / "d3_model_comparison.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    )
    print()
    for model in models:
        sub = [
            r["judge_score"] for r in rows if r["model"] == model and r["judge_score"] is not None
        ]
        if sub:
            print(f"  {model:28s} n={len(sub):3d} recovered={sum(sub) / len(sub):.3f}")


# --- D6 -------------------------------------------------------------------

_TITLE = re.compile(r"(titled|標題為|Doc\.|document\s+\d+|第\s*\d+\s*號)", re.I)
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{4}\s*年")


def d6(args) -> None:
    """Classify only the questions with no gold in the top 20.

    Classifying before choosing a fix is the point: the cheapest remedy differs
    per cause, and the retrieval ceiling is worth spending on only where the
    largest class actually is.
    """
    src = REPORTS / "score_distribution.jsonl"
    if not src.exists():
        raise SystemExit("run scripts/score_distribution.py first")
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    by_q: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_q.setdefault((r["case_id"], r["language"]), []).append(r)

    cases = load_cases()
    misses: list[dict] = []
    for (cid, lang), rs in by_q.items():
        if any(r["is_gold"] for r in rs):
            continue
        case = cases[cid]
        q = case["question_zh"] if lang == "zh-TW" else case["question_en"]
        kind = rs[0]["kind"]
        # Order matters: the first matching cause is the one to fix first.
        if _TITLE.search(q) or _DATE.search(q):
            cause = "A precise title/date/person"
        elif lang == "zh-TW":
            cause = "B cross-language"
        elif kind == "multihop":
            cause = "D multi-hop bridging"
        elif kind == "correction":
            cause = "C modern phrasing vs archival wording"
        else:
            cause = "E general semantic miss"
        misses.append(
            {"case_id": cid, "language": lang, "kind": kind, "cause": cause, "question": q[:120]}
        )

    (REPORTS / "d6_retrieval_misses.jsonl").write_text(
        "\n".join(json.dumps(m, ensure_ascii=False) for m in misses)
    )
    counts = Counter(m["cause"] for m in misses)
    total = sum(counts.values()) or 1
    print(f"{len(misses)} of {len(by_q)} question-language pairs have no gold in the top 20\n")
    remedy = {
        "A precise title/date/person": "BM25 field weighting, metadata filters",
        "B cross-language": "multilingual embedding or query translation",
        "C modern phrasing vs archival wording": "document-side expansion (doc2query)",
        "D multi-hop bridging": "query decomposition / iterative retrieval",
        "E general semantic miss": "embedding model or late interaction",
    }
    for cause, n in counts.most_common():
        print(f"  {cause:42s} {n:3d} ({n / total:.0%})  -> {remedy[cause]}")
    print("\nOnly the largest class is worth paying for.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("d0")
    p2 = sub.add_parser("d2")
    p2.add_argument("--limit", type=int, default=None)
    p3 = sub.add_parser("d3")
    p3.add_argument("--models", default="qwen3:4b-instruct")
    sub.add_parser("d6")
    args = ap.parse_args()

    if args.cmd == "d0":
        d0(args)
    elif args.cmd == "d6":
        d6(args)
    else:
        asyncio.run({"d2": d2, "d3": d3}[args.cmd](args))


main()
