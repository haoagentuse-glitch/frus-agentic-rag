"""One real question per system: did the inverted grader and sentence focus fire?"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from frus_agentic_rag.agent.run import answer

CASES = Path("eval/gold_cases.jsonl")


async def main() -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    multihop = next(c for c in cases if c["kind"] == "multihop")
    lookup = next(c for c in cases if c["kind"] == "lookup")

    for case in (multihop, lookup):
        for lang in ("zh-TW", "en"):
            q = case["question_zh"] if lang == "zh-TW" else case["question_en"]
            a = await answer(q, language=lang, system="B3")
            disp = next((t for t in a.trace if t.get("node") == "dispatch_retrieval"), {})
            grade = next((t for t in a.trace if t.get("node") == "grade_evidence"), {})
            gate = next((t for t in a.trace if t.get("node") == "validate_citations"), {})
            print(
                json.dumps(
                    {
                        "case": case["case_id"],
                        "lang": lang,
                        "outcome": a.outcome,
                        "focus": (disp.get("detail") or {}).get("sentence_focus"),
                        "grade": {
                            k: v for k, v in (grade.get("detail") or {}).items() if k != "hops"
                        },
                        "gate": gate.get("detail"),
                        "latency_s": a.latency_s,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


asyncio.run(main())
