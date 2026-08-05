"""Rerun the answerable-but-abstained runs and record why the gate blocked."""

from __future__ import annotations

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from frus_agentic_rag.agent.run import answer, answer_2step

REPORT = Path("reports/agent_ablation.json")
CASES = Path("eval/gold_cases.jsonl")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "reports/abstain_probe.json")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def load_cases() -> dict:
    return {
        json.loads(line)["case_id"]: json.loads(line)
        for line in CASES.read_text().splitlines()
        if line.strip()
    }


async def main() -> None:
    runs = json.load(REPORT.open())["runs"]
    cases = load_cases()
    targets = [
        r
        for r in runs
        if r["answerable"] and r["outcome"] != "answer" and (r.get("agent_union_recall") or 0) > 0
    ][:LIMIT]

    out = []
    for i, r in enumerate(targets, 1):
        case = cases[r["case_id"]]
        q = case["question_zh"] if r["language"] == "zh-TW" else case["question_en"]
        try:
            if r["system"] == "B0-2step":
                a = await answer_2step(q, language=r["language"])
            else:
                a = await answer(q, language=r["language"], system=r["system"])
        except Exception as exc:
            out.append(
                {
                    **{k: r[k] for k in ("case_id", "system", "language", "kind")},
                    "crash": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        gate = next((t for t in a.trace if t.get("node") == "validate_citations"), {})
        synth = next((t for t in a.trace if t.get("node") == "synthesize"), {})
        grade = next((t for t in a.trace if t.get("node") == "grade_evidence"), {})
        retrieved = {e.evidence_id for e in a.evidence}
        claim_ids = [i for c in a.claims for i in c.evidence_ids]
        rec = {
            "case_id": r["case_id"],
            "system": r["system"],
            "language": r["language"],
            "kind": r["kind"],
            "outcome_then": r["outcome"],
            "outcome_now": a.outcome,
            "n_claims": len(a.claims),
            "n_claim_ids": len(claim_ids),
            "ids_not_retrieved": [i for i in claim_ids if i not in retrieved][:5],
            "gate_errors": (gate.get("detail") or {}).get("errors", [])[:6],
            "gate_error_detail": gate.get("detail"),
            "synth_detail": synth.get("detail"),
            "synth_error": synth.get("error"),
            "grade_detail": {k: v for k, v in (grade.get("detail") or {}).items() if k != "hops"},
            "answer_head": (a.answer_text or "")[:160],
        }
        out.append(rec)
        print(
            f"[{i}/{len(targets)}] {r['case_id']} {r['system']} {r['language']} "
            f"-> {a.outcome} claims={len(a.claims)} err={rec['gate_errors'][:1]}",
            flush=True,
        )

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    kinds = Counter()
    for rec in out:
        errs = rec.get("gate_errors") or []
        if rec.get("crash"):
            kinds["crash"] += 1
        elif rec.get("outcome_now") == "answer":
            kinds["now answers (nondeterministic)"] += 1
        elif not errs:
            kinds["no gate errors recorded"] += 1
        else:
            for e in errs:
                for tag in (
                    "no claims",
                    "never retrieved",
                    "not accepted by the grader",
                    "malformed evidence id",
                    "model-written URL",
                    "no draft answer",
                    "has no evidence id",
                    "not a citable",
                    "not a published",
                ):
                    if tag in e:
                        kinds[tag] += 1
                        break
                else:
                    kinds[e[:40]] += 1
    print()
    print(json.dumps(kinds, ensure_ascii=False, indent=2))


asyncio.run(main())
