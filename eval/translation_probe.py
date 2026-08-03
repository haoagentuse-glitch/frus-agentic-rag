"""Does translating the query before retrieval matter, and by how much?"""

import asyncio
import json

from frus_agentic_rag.agent.llm import get_client
from frus_agentic_rag.agent.prompts import PLANNER_SYSTEM, planner_user
from frus_agentic_rag.agent.schemas import QueryPlan
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import SearchFilters
from frus_agentic_rag.retrieval.hybrid import hybrid_search_sync

s = get_settings()
f = SearchFilters(subtypes=["historical-document"])
with open("eval/gold_cases.jsonl", encoding="utf-8") as fh:
    cases = [c for c in (json.loads(line) for line in fh) if c["answerable"]]


async def main():
    c = get_client()
    raw_h = tr_h = en_h = T = 0
    for case in cases:
        gold = set(case["gold_documents"])
        zh = case["question_zh"]
        # 1. raw Chinese, as B0 does it
        raw = {f"{h.volume_id}:{h.document_id}" for h in hybrid_search_sync(zh, f, 10)}
        # 2. planner-translated English, as B1-B3 do it
        plan = await c.structured(PLANNER_SYSTEM, planner_user(zh), QueryPlan)
        q = plan.subqueries[0].query if plan.subqueries else zh
        tr = {f"{h.volume_id}:{h.document_id}" for h in hybrid_search_sync(q, f, 10)}
        # 3. the human-written English question, as an upper bound
        en = {
            f"{h.volume_id}:{h.document_id}" for h in hybrid_search_sync(case["question_en"], f, 10)
        }
        raw_h += len(gold & raw)
        tr_h += len(gold & tr)
        en_h += len(gold & en)
        T += len(gold)
    print(f"gold docs = {T}")
    print(f"  raw zh query      (B0 path)     {raw_h}/{T} = {raw_h / T:.3f}")
    print(f"  planner-translated (B1-B3 path) {tr_h}/{T} = {tr_h / T:.3f}")
    print(f"  human en question  (ceiling)    {en_h}/{T} = {en_h / T:.3f}")


asyncio.run(main())
