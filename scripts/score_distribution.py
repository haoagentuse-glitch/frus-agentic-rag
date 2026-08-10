"""Cross-encoder score distribution against gold labels.

Retrieval only — no LLM, no graph — so a full pass over the validation set costs
seconds and can be repeated freely. For every (case, language) it records one row
per retrieved passage: the raw cross-encoder logit, its rank, and whether the
document is gold. That is the input to choosing `score_keep_absolute` and
`score_keep_margin`; the recommendations printed at the end are a starting point
from the measured separation, not a substitute for reading the distribution.

    FRUS_GPU=1 ./scripts/dev.sh python scripts/score_distribution.py [--limit N]

Writes reports/score_distribution.jsonl (one row per passage) and
reports/score_distribution.json (the summary).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

from frus_agentic_rag.models import SearchFilters
from frus_agentic_rag.retrieval.hybrid import hybrid_search

CASES = Path("eval/gold_cases.jsonl")
ROWS = Path("reports/score_distribution.jsonl")
SUMMARY = Path("reports/score_distribution.json")


def quantiles(xs: list[float], qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{int(q * 100)}": round(s[min(len(s) - 1, int(q * len(s)))], 4) for q in qs}


def separation(gold: list[float], other: list[float]) -> dict:
    """How far apart the two populations sit, and where a cut would land.

    `recall_at_floor` is the fraction of gold passages a floor would keep;
    `kept_fraction` is the fraction of everything it would keep. A useful
    threshold has the first near 1.0 and the second well below it.
    """
    if not gold or not other:
        return {}
    out: dict = {
        "gold_n": len(gold),
        "other_n": len(other),
        "gold": quantiles(gold),
        "other": quantiles(other),
        "gold_mean": round(st.mean(gold), 4),
        "other_mean": round(st.mean(other), 4),
    }
    # Floors that keep 100%, 95% and 90% of gold, and what each costs.
    everything = sorted(gold + other)
    cuts = {}
    for keep in (1.0, 0.95, 0.90):
        idx = int((1 - keep) * len(gold))
        floor = sorted(gold)[min(idx, len(gold) - 1)]
        kept = sum(1 for x in everything if x >= floor)
        cuts[f"gold_recall_{keep:.2f}"] = {
            "floor": round(floor, 4),
            "kept_fraction": round(kept / len(everything), 4),
        }
    out["candidate_floors"] = cuts
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    cases = [json.loads(x) for x in CASES.read_text().splitlines() if x.strip()]
    if args.limit:
        # Even coverage across kinds rather than the first N, which are all lookup.
        by_kind: dict[str, list] = defaultdict(list)
        for c in cases:
            by_kind[c["kind"]].append(c)
        per = max(1, args.limit // max(1, len(by_kind)))
        cases = [c for group in by_kind.values() for c in group[:per]]

    rows: list[dict] = []
    for n, case in enumerate(cases, 1):
        gold = set(case.get("gold_documents") or [])
        for lang in ("zh-TW", "en"):
            q = case["question_zh"] if lang == "zh-TW" else case["question_en"]
            hits = await hybrid_search(
                q, SearchFilters(subtypes=["historical-document"]), args.top_k, "main"
            )
            for e in hits:
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "kind": case["kind"],
                        "language": lang,
                        "answerable": case.get("answerable", True),
                        "evidence_id": e.evidence_id,
                        "document": f"{e.volume_id}:{e.document_id}",
                        "rank_rerank": e.rank_rerank,
                        "rerank_score": e.rerank_score,
                        "rrf_score": e.score,
                        "is_gold": f"{e.volume_id}:{e.document_id}" in gold,
                    }
                )
        print(f"[{n}/{len(cases)}] {case['case_id']} rows={len(rows)}", flush=True)

    ROWS.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    scored = [r for r in rows if r["rerank_score"] is not None]
    summary: dict = {
        "rows": len(rows),
        "scored": len(scored),
        "cases": len({r["case_id"] for r in rows}),
    }
    if not scored:
        summary["error"] = "no cross-encoder scores; run with FRUS_GPU=1 and a reranker path"
        SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(summary, indent=2))
        return

    def slice_rows(pred) -> dict:
        sub = [r for r in scored if pred(r)]
        return separation(
            [r["rerank_score"] for r in sub if r["is_gold"]],
            [r["rerank_score"] for r in sub if not r["is_gold"]],
        )

    summary["overall"] = slice_rows(lambda r: True)
    summary["by_language"] = {
        lang: slice_rows(lambda r, la=lang: r["language"] == la) for lang in ("zh-TW", "en")
    }
    summary["by_kind"] = {
        kind: slice_rows(lambda r, k=kind: r["kind"] == k)
        for kind in sorted({r["kind"] for r in scored})
    }
    summary["by_language_kind"] = {
        f"{lang}/{kind}": slice_rows(
            lambda r, la=lang, k=kind: r["language"] == la and r["kind"] == k
        )
        for lang in ("zh-TW", "en")
        for kind in sorted({r["kind"] for r in scored})
    }

    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("rows", "scored", "cases", "overall")}, indent=2))
    print("\nper-language / per-kind detail in", SUMMARY)
    print(
        "\nSet FRUS_SCORE_KEEP_ABSOLUTE / FRUS_SCORE_KEEP_MARGIN from the floor that "
        "holds gold recall at 1.00 in the WORST language/kind slice, not the overall one."
    )


Path("reports").mkdir(exist_ok=True)
asyncio.run(main())
