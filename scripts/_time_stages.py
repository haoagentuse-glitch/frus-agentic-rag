"""Where the wall clock goes, per node, with and without sentence focus."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from frus_agentic_rag.agent.run import answer

CASES = Path("eval/gold_cases.jsonl")


async def one(case: dict, lang: str, label: str) -> None:
    q = case["question_zh"] if lang == "zh-TW" else case["question_en"]
    a = await answer(q, language=lang, system="B3")
    nodes = [(t.get("node"), t.get("duration_s")) for t in a.trace]
    focus = next(
        (
            (t.get("detail") or {}).get("sentence_focus")
            for t in a.trace
            if t.get("node") == "dispatch_retrieval"
        ),
        None,
    )
    print(
        json.dumps(
            {
                "label": label,
                "case": case["case_id"],
                "lang": lang,
                "outcome": a.outcome,
                "total_s": a.latency_s,
                "nodes": nodes,
                "focus_kept": (focus or {}).get("kept_ratio"),
                "focus_skipped": (focus or {}).get("skipped"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


async def main() -> None:
    cases = [json.loads(x) for x in CASES.read_text().splitlines() if x.strip()]
    picks = [
        (next(c for c in cases if c["kind"] == "lookup"), "en"),
        (next(c for c in cases if c["kind"] == "multihop"), "zh-TW"),
    ]
    label = os.environ.get("FRUS_LABEL", "run")
    for case, lang in picks:
        await one(case, lang, label)


asyncio.run(main())
