"""Did the passage that actually answers the question reach the model?

Every recall figure in this project is computed over documents: a run counts as
having found the evidence if any chunk of the gold document was retrieved. On a
corpus chunked at 512 tokens that is a weak claim. A FRUS despatch runs to
several chunks, and citing the right document says nothing about whether the
model saw the sentence carrying the fact.

The 15 runs that cited a gold document and were still judged wrong are the
symptom. This measures the thing document recall cannot see:

  document_hit   - any chunk of the gold document reached the synthesis window
  evidence_hit   - the chunk that best answers the question reached it
  gap            - document_hit and not evidence_hit

The gold set labels documents, not chunks, so the answering chunk has to be
identified rather than looked up: every chunk of each gold document is scored
against the question with the same cross-encoder the pipeline ranks with, and
the top one is taken as the evidence chunk. That is a proxy, and it is the
pipeline's own notion of relevance — which is the right one here, because the
question is whether the pipeline's own ranking carried the right chunk through,
not whether a human would have chosen the same one.

    FRUS_GPU=1 ./scripts/dev.sh python scripts/evidence_coverage.py \\
        --runs reports/runs_B_min_margin.jsonl --only-cited-wrong

Writes reports/evidence_coverage.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from frus_agentic_rag.agent.prompts import select_synthesis_evidence
from frus_agentic_rag.agent.run import answer

CASES = Path("eval/gold_cases.jsonl")
OUT = Path("reports/evidence_coverage.json")


def gold_chunks(document_ids: list[str]) -> list[dict]:
    """Every chunk of every gold document, in document order."""
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect

    tbl = connect().open_table(CHUNKS_TABLE)
    out: list[dict] = []
    for doc in document_ids:
        vol, _, did = doc.partition(":")
        rows = (
            tbl.search()
            .where(f"volume_id = '{vol}' AND document_id = '{did}'")
            .select(["chunk_id", "volume_id", "document_id", "text"])
            .limit(200)
            .to_list()
        )
        out.extend(sorted(rows, key=lambda r: r["chunk_id"]))
    return out


def best_chunk(question: str, chunks: list[dict]) -> tuple[str | None, float | None, int]:
    """The chunk of the gold document that best answers the question."""
    from frus_agentic_rag.config import get_settings
    from frus_agentic_rag.retrieval import rerank as rr

    if not chunks:
        return None, None, 0
    if len(chunks) == 1:
        return chunks[0]["chunk_id"], None, 1
    limit = get_settings().reranker_max_chars
    scores = rr.score_pairs([(question, c["text"][:limit]) for c in chunks])
    if scores is None:
        return None, None, len(chunks)
    i = max(range(len(chunks)), key=lambda j: scores[j])
    return chunks[i]["chunk_id"], round(float(scores[i]), 4), len(chunks)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="reports/runs_B_min_margin.jsonl")
    ap.add_argument("--system", default="B3")
    ap.add_argument(
        "--only-cited-wrong",
        action="store_true",
        help="only runs that cited a gold document and were still judged wrong",
    )
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = {
        json.loads(x)["case_id"]: json.loads(x) for x in CASES.read_text().splitlines() if x.strip()
    }
    runs = [json.loads(x) for x in Path(args.runs).read_text().splitlines() if x.strip()]
    targets = [r for r in runs if r["system"] == args.system and r["answerable"]]
    if args.only_cited_wrong:
        targets = [
            r
            for r in targets
            if r["outcome"] == "answer"
            and (r.get("citation_recall") or 0) > 0
            and r.get("judge_score") == 0
        ]
    targets = targets[: args.limit] if args.limit else targets

    rows: list[dict] = []
    for n, r in enumerate(targets, 1):
        case = cases[r["case_id"]]
        q = case["question_zh"] if r["language"] == "zh-TW" else case["question_en"]
        gold_docs = case.get("gold_documents") or []
        a = await answer(q, language=r["language"], system=args.system)

        # What the model actually saw: the synthesis window, not the retrieval set.
        accepted = {e.evidence_id for e in a.evidence}
        window = {e.evidence_id for e in select_synthesis_evidence(a.evidence)}

        per_doc = []
        for doc in gold_docs:
            chunks = gold_chunks([doc])
            ids = {c["chunk_id"] for c in chunks}
            bid, bscore, ntotal = best_chunk(q, chunks)
            per_doc.append(
                {
                    "document": doc,
                    "chunks_in_document": ntotal,
                    "evidence_chunk": bid,
                    "evidence_chunk_score": bscore,
                    "document_hit_retrieved": bool(ids & accepted),
                    "document_hit_window": bool(ids & window),
                    "evidence_hit_retrieved": bid in accepted if bid else False,
                    "evidence_hit_window": bid in window if bid else False,
                }
            )

        rows.append(
            {
                "case_id": r["case_id"],
                "language": r["language"],
                "kind": r["kind"],
                "outcome": a.outcome,
                "window_size": len(window),
                "gold_documents": len(gold_docs),
                "per_document": per_doc,
            }
        )
        hit = sum(1 for d in per_doc if d["evidence_hit_window"])
        print(
            f"[{n}/{len(targets)}] {r['case_id']}/{r['language']} "
            f"evidence chunk in window {hit}/{len(per_doc)}",
            flush=True,
        )

    docs = [d for row in rows for d in row["per_document"]]
    n = len(docs) or 1
    summary = {
        "runs": len(rows),
        "gold_documents": len(docs),
        "document_recall_window": round(sum(d["document_hit_window"] for d in docs) / n, 4),
        "evidence_chunk_recall_window": round(sum(d["evidence_hit_window"] for d in docs) / n, 4),
        "document_recall_retrieved": round(sum(d["document_hit_retrieved"] for d in docs) / n, 4),
        "evidence_chunk_recall_retrieved": round(
            sum(d["evidence_hit_retrieved"] for d in docs) / n, 4
        ),
        # The number this script exists to produce: right document, wrong chunk.
        "document_hit_but_evidence_missed": sum(
            1 for d in docs if d["document_hit_window"] and not d["evidence_hit_window"]
        ),
        "mean_chunks_per_gold_document": round(sum(d["chunks_in_document"] for d in docs) / n, 2),
        "rows": rows,
    }
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


Path("reports").mkdir(exist_ok=True)
asyncio.run(main())
