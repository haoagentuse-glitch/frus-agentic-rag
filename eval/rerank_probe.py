"""A/B the cross-encoder against RRF order on the same candidates."""

import json
import statistics
import time
from collections import defaultdict

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import SearchFilters
from frus_agentic_rag.retrieval import rerank as rr
from frus_agentic_rag.retrieval.hybrid import bm25_search, dense_search, head_search, rrf_fuse

s = get_settings()
f = SearchFilters(subtypes=["historical-document"])

print("reranker:", rr.warm(), flush=True)
with open("eval/gold_cases.jsonl", encoding="utf-8") as fh:
    cases = [c for c in (json.loads(x) for x in fh) if c["answerable"]]


def docs(ev):
    return {f"{e.volume_id}:{e.document_id}" for e in ev}


agg = defaultdict(lambda: defaultdict(list))
t_rrf = t_rr = 0.0
for c in cases:
    gold = set(c["gold_documents"])
    lex = bm25_search(c["question_en"], f, s.candidate_k)
    hds = head_search(c["question_en"], f, s.candidate_k)
    dns = dense_search(c["question_en"], f, s.candidate_k)

    t = time.perf_counter()
    base = rrf_fuse([lex, hds, dns], s.rrf_k, 10)
    t_rrf += time.perf_counter() - t

    wide = rrf_fuse([lex, hds, dns], s.rrf_k, s.rerank_candidates)
    t = time.perf_counter()
    scored = rr.rerank(c["question_en"], [e.model_dump() for e in wide], 10)
    t_rr += time.perf_counter() - t
    by = {e.evidence_id: e for e in wide}
    reran = [by[r["evidence_id"]] for r in scored]

    for name, ev in (("RRF", base), ("rerank", reran)):
        r = len(gold & docs(ev)) / len(gold)
        agg[name]["all"].append(r)
        agg[name][c["kind"]].append(r)

kinds = ["all", "lookup", "multihop", "correction", "correction_matched"]
print(f"\n{'':10}" + "".join(f"{k:>20}" for k in kinds))
for name in ("RRF", "rerank"):
    print(
        f"{name:10}"
        + "".join(
            f"{statistics.fmean(agg[name][k]):>20.3f}" if agg[name][k] else f"{'-':>20}"
            for k in kinds
        )
    )
print(
    f"\n每題耗時  RRF {t_rrf / len(cases) * 1000:.0f} ms   rerank {t_rr / len(cases) * 1000:.0f} ms"
)
