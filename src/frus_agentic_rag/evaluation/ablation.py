"""Bilingual B0-B3 ablation.

Primary metrics are deterministic and reproducible offline: evidence recall
against gold documents, route accuracy, citation precision, abstention
precision/recall, calls and latency. Answer correctness is the one metric that
needs a model, and it is reported separately with the judge named, so a reader
can discount it independently.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

from frus_agentic_rag.agent.nodes import rule_route
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.evaluation.gold import load_cases

SYSTEM_ROUTES = {"lookup", "simple", "timeline", "complex", "status"}


# The evidence pipeline has three lossy stages. A single recall number cannot
# say which one dropped a gold document, so each stage is measured separately:
#
#   candidate_recall      the retriever alone, one search, @candidate_k
#   agent_union_recall    every document the agent saw across all its rounds
#   accepted_recall       what the grader kept
#   citation_recall       what survived the citation gate
#
# candidate -> union isolates the value of multi-round planning; union ->
# accepted isolates the grader; accepted -> citation isolates the gate.


def _recall(gold: set[str], got: set[str]) -> float | None:
    return len(gold & got) / len(gold) if gold else None


def _precision(gold: set[str], got: set[str]) -> float | None:
    return len(gold & got) / len(got) if got and gold else None


async def candidate_recall(question: str, gold: set[str], limit: int) -> dict:
    """The retriever on its own, independent of any system under test."""
    from frus_agentic_rag.models import SearchFilters
    from frus_agentic_rag.retrieval.hybrid import hybrid_search

    hits = await hybrid_search(question, SearchFilters(subtypes=["historical-document"]), limit)
    docs = {f"{h.volume_id}:{h.document_id}" for h in hits}
    return {"recall": _recall(gold, docs), "n_documents": len(docs), "n_chunks": len(hits)}


async def _run_case(case: dict, system: str, language: str, use_judge: bool) -> dict:
    from frus_agentic_rag.agent.run import answer as run_answer
    from frus_agentic_rag.agent.run import answer_2step
    from frus_agentic_rag.evaluation import judge as judge_mod

    question = case["question_zh"] if language == "zh-TW" else case["question_en"]
    gold = set(case.get("gold_documents", []))

    t0 = time.perf_counter()
    try:
        if system == "B0-2step":
            result = await answer_2step(question, language=language)
        else:
            result = await run_answer(question, language=language, system=system)
        error = ""
    except Exception as exc:
        return {
            "case_id": case["case_id"],
            "kind": case["kind"],
            "system": system,
            "language": language,
            "error": f"{type(exc).__name__}: {exc}",
            "outcome": "error",
            "latency_s": round(time.perf_counter() - t0, 2),
        }

    union = set(result.retrieved_document_ids)
    accepted = set(result.accepted_document_ids)
    cited = set(result.cited_document_ids)
    claims = result.claims
    claim_coverage = sum(1 for c in claims if c.evidence_ids) / len(claims) if claims else None

    verdict: dict[str, Any] = {"judge": "skipped", "score": None, "correct": None}
    if use_judge:
        v = await judge_mod.judge_answer(
            question, result.answer_text, sorted(gold), case["answerable"], result.outcome
        )
        verdict = v.model_dump()

    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "system": system,
        "language": language,
        "expected_route": case.get("route"),
        "rule_route": rule_route(question),
        "actual_route": next(
            (
                t.get("detail", {}).get("route")
                for t in result.trace
                if t.get("node") == "plan_query"
            ),
            None,
        ),
        "outcome": result.outcome,
        "answerable": case["answerable"],
        "n_gold": len(gold),
        # --- the three lossy stages, measured separately ---
        "agent_union_recall": _recall(gold, union),
        "accepted_recall": _recall(gold, accepted),
        "citation_recall": _recall(gold, cited),
        "citation_precision": _precision(gold, cited),
        "n_union_docs": len(union),
        "n_accepted_docs": len(accepted),
        "n_cited_docs": len(cited),
        "n_gold_hit_union": len(gold & union),
        "grader_drop": len(gold & union) - len(gold & accepted),
        "citation_drop": len(gold & accepted) - len(gold & cited),
        "claim_citation_coverage": claim_coverage,
        "n_citations": len(result.citations),
        "llm_calls": result.llm_calls,
        "retrieval_calls": result.retrieval_calls,
        "latency_s": result.latency_s,
        "correction_used": any(t.get("node") == "rewrite_missing" for t in result.trace),
        "judge_score": verdict.get("score"),
        "judge_correct": verdict.get("correct"),
        "judge": verdict.get("judge"),
        "error": error,
        "answer_preview": result.answer_text[:200],
    }


def _mean(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _pct(xs: list, p: float) -> float | None:
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    k = min(len(vals) - 1, round(p * (len(vals) - 1)))
    return round(vals[k], 3)


def _macro_f1(rows: list[dict], key: str = "actual_route") -> float | None:
    pairs = [(r["expected_route"], r.get(key)) for r in rows if r.get("expected_route")]
    if not pairs:
        return None
    f1s = []
    for label in SYSTEM_ROUTES:
        tp = sum(1 for e, a in pairs if e == label and a == label)
        fp = sum(1 for e, a in pairs if e != label and a == label)
        fn = sum(1 for e, a in pairs if e == label and a != label)
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return round(statistics.fmean(f1s), 4) if f1s else None


def summarise(rows: list[dict]) -> dict:
    ok = [r for r in rows if r["outcome"] != "error"]
    answerable = [r for r in ok if r["answerable"]]
    unanswerable = [r for r in ok if not r["answerable"]]

    abstained = [r for r in ok if r["outcome"] == "abstain"]
    correct_abstentions = [r for r in abstained if not r["answerable"]]

    def by_kind(kind: str) -> list[dict]:
        return [r for r in ok if r["kind"] == kind]

    # Capability breakdown is pre-declared, not chosen after seeing results:
    # correction cases are reported on their own rather than dropped from the
    # aggregate, which would be discarding a question type for failing.
    kinds = {
        k: {
            "n": len(by_kind(k)),
            "candidate_recall": _mean([r.get("candidate_recall") for r in by_kind(k)]),
            "agent_union_recall": _mean([r["agent_union_recall"] for r in by_kind(k)]),
            "accepted_recall": _mean([r["accepted_recall"] for r in by_kind(k)]),
            "citation_recall": _mean([r["citation_recall"] for r in by_kind(k)]),
        }
        for k in ("lookup", "multihop", "correction")
    }

    return {
        "n_runs": len(rows),
        "n_errors": len(rows) - len(ok),
        # --- stage-by-stage recall over answerable cases (macro) ---
        "candidate_recall": _mean([r.get("candidate_recall") for r in answerable]),
        "agent_union_recall": _mean([r["agent_union_recall"] for r in answerable]),
        "accepted_recall": _mean([r["accepted_recall"] for r in answerable]),
        "citation_recall": _mean([r["citation_recall"] for r in answerable]),
        "planning_gain": _delta(
            _mean([r["agent_union_recall"] for r in answerable]),
            _mean([r.get("candidate_recall") for r in answerable]),
        ),
        "grader_loss": _delta(
            _mean([r["accepted_recall"] for r in answerable]),
            _mean([r["agent_union_recall"] for r in answerable]),
        ),
        "citation_loss": _delta(
            _mean([r["citation_recall"] for r in answerable]),
            _mean([r["accepted_recall"] for r in answerable]),
        ),
        "by_kind": kinds,
        "correction_success_rate": _mean(
            [1.0 if (r["agent_union_recall"] or 0) > 0 else 0.0 for r in by_kind("correction")]
        ),
        # The matched variant asks about the same gold document using the
        # archival term. The gap between the two is what a corrective retrieval
        # has to close; closing it is the only evidence that correction works.
        "correction_pair_gap_pp": _delta(
            _mean([r["agent_union_recall"] for r in by_kind("correction_matched")]),
            _mean([r["agent_union_recall"] for r in by_kind("correction")]),
        ),
        # --- everything downstream of retrieval ---
        "answer_correctness": _mean(
            [1.0 if r["judge_correct"] else 0.0 for r in ok if r["judge_correct"] is not None]
        ),
        "judge_score_mean": _mean([r["judge_score"] for r in ok]),
        "route_macro_f1": _macro_f1(ok),
        "rule_route_macro_f1": _macro_f1(ok, key="rule_route"),
        "simple_over_decomposed_pct": round(
            100
            * sum(
                1
                for r in ok
                if r["expected_route"] == "simple" and r.get("actual_route") == "complex"
            )
            / max(1, sum(1 for r in ok if r["expected_route"] == "simple")),
            2,
        ),
        "correction_rate": round(sum(1 for r in ok if r["correction_used"]) / max(1, len(ok)), 4),
        "abstention_precision": round(len(correct_abstentions) / len(abstained), 4)
        if abstained
        else None,
        "abstention_recall": round(len(correct_abstentions) / len(unanswerable), 4)
        if unanswerable
        else None,
        "false_answer_on_unanswerable": sum(1 for r in unanswerable if r["outcome"] == "answer"),
        "citation_precision": _mean([r["citation_precision"] for r in answerable]),
        "claim_citation_coverage": _mean([r["claim_citation_coverage"] for r in ok]),
        # Retrieval budget. B1 issuing more searches than B0 is a confound in any
        # recall comparison between them, so it is reported alongside, not buried.
        "retrieval_calls_mean": _mean([r["retrieval_calls"] for r in ok]),
        "llm_calls_mean": _mean([r["llm_calls"] for r in ok]),
        "latency_p50": _pct([r["latency_s"] for r in ok], 0.50),
        "latency_p95": _pct([r["latency_s"] for r in ok], 0.95),
    }


def _delta(a: float | None, b: float | None) -> float | None:
    """Percentage-point change between two stages."""
    return round((a - b) * 100, 2) if a is not None and b is not None else None


def paired_by_case(rows_a: list[dict], rows_b: list[dict]) -> dict:
    """Paired comparison over semantic cases.

    The zh and en wordings of one case are two observations of the same
    question, not two independent samples; pairing on case_id keeps that
    structure instead of inflating n by treating them as independent.
    """

    def index(rows: list[dict]) -> dict[tuple[str, str], dict]:
        return {(r["case_id"], r["language"]): r for r in rows if r["outcome"] != "error"}

    a, b = index(rows_a), index(rows_b)
    shared = sorted(set(a) & set(b))
    diffs = [
        (a[k]["agent_union_recall"] - b[k]["agent_union_recall"])
        for k in shared
        if a[k]["agent_union_recall"] is not None and b[k]["agent_union_recall"] is not None
    ]
    if not diffs:
        return {"n_pairs": 0}
    wins = sum(1 for d in diffs if d > 0)
    losses = sum(1 for d in diffs if d < 0)
    return {
        "n_pairs": len(diffs),
        "mean_difference_pp": round(statistics.fmean(diffs) * 100, 2),
        "wins": wins,
        "losses": losses,
        "ties": len(diffs) - wins - losses,
        "stdev_pp": round(statistics.stdev(diffs) * 100, 2) if len(diffs) > 1 else None,
    }


def retriever_fingerprint() -> dict:
    """What the retriever actually was for a run.

    Before the dense index exists, `dense_search` filters on `embedded = true`
    and returns nothing, so RRF fuses one arm and every number describes BM25
    alone. Resuming a sweep across that boundary would silently average
    BM25-only runs with hybrid runs, which is worse than having no number.
    """
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect

    try:
        tbl = connect().open_table(CHUNKS_TABLE)
        rows = tbl.count_rows()
        embedded = tbl.count_rows("embedded = true")
    except Exception:
        return {"rows": 0, "embedded": 0, "mode": "unavailable"}
    return {
        "rows": rows,
        "embedded": embedded,
        "mode": "bm25-only" if embedded == 0 else ("hybrid" if embedded == rows else "partial"),
    }


def _load_completed(path: Path, fingerprint: dict) -> tuple[dict[tuple[str, str, str], dict], int]:
    if not path.exists():
        return {}, 0
    done: dict[tuple[str, str, str], dict] = {}
    discarded = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("retriever", {}).get("mode") != fingerprint["mode"]:
                discarded += 1
                continue
            done[(r["system"], r["language"], r["case_id"])] = r
    return done, discarded


async def run_ablation(
    systems: list[str],
    cases_path: Path,
    languages: list[str],
    limit: int | None = None,
    use_judge: bool = True,
    out: Path | None = None,
    resume: bool = True,
) -> dict:
    from frus_agentic_rag.evaluation import judge as judge_mod

    settings = get_settings()
    cases = load_cases(cases_path)
    if limit:
        cases = cases[:limit]

    # A run recorded while Ollama is unreachable looks exactly like a decision to
    # abstain: every node catches the connection error, falls back, and the case
    # is written as `abstain` in about a second. Forty-two such rows were logged
    # after a WSL restart and would have been read as the system refusing to
    # answer. Fail loudly instead of recording fiction.
    from frus_agentic_rag.agent.llm import get_client

    health = await get_client().health()
    if not health.get("target_present"):
        raise RuntimeError(
            f"model {health.get('target')} is not loaded in Ollama at {health.get('host')}; "
            f"available: {health.get('models')}. Start it before evaluating — a run against "
            "a dead Ollama records infrastructure failure as abstention."
        )
    print(f"ollama: {health['target']} ready at {health['host']}", flush=True)

    judge_state = "enabled" if (use_judge and judge_mod.available()) else "unavailable"

    # A full bilingual 5-system sweep is 300 sequential Ollama runs — hours on
    # one 4060. Each result is appended as it lands so an interrupted sweep can
    # be resumed, and so partial results are readable while it is still running.
    runs_path = (out or settings.reports_dir / "agent_ablation.json").with_name(
        "ablation_runs.jsonl"
    )
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = retriever_fingerprint()
    completed, discarded = _load_completed(runs_path, fingerprint) if resume else ({}, 0)
    if not resume and runs_path.exists():
        runs_path.unlink()
    print(
        f"retriever: {fingerprint['mode']} "
        f"({fingerprint['embedded']}/{fingerprint['rows']} chunks embedded)",
        flush=True,
    )
    if discarded:
        print(
            f"discarding {discarded} checkpointed runs from a different retriever mode; "
            "they will be re-run",
            flush=True,
        )

    # The retriever floor depends on the question, not on the system under test,
    # so it is measured once per (case, language) and attached to every row.
    # Without it, "the agent found 0.35" has no denominator.
    floors: dict[tuple[str, str], float | None] = {}
    for language in languages:
        for case in cases:
            q = case["question_zh"] if language == "zh-TW" else case["question_en"]
            gold = set(case.get("gold_documents", []))
            floors[(case["case_id"], language)] = (
                (await candidate_recall(q, gold, settings.candidate_k))["recall"] if gold else None
            )
    print(f"candidate recall measured for {len(floors)} (case, language) pairs", flush=True)

    rows: list[dict] = []
    per_system: dict[str, dict] = {}
    total = len(systems) * len(languages) * len(cases)
    n = 0

    for system in systems:
        sys_rows: list[dict] = []
        for language in languages:
            for case in cases:
                n += 1
                key = (system, language, case["case_id"])
                if key in completed:
                    completed[key].setdefault(
                        "candidate_recall", floors.get((case["case_id"], language))
                    )
                    sys_rows.append(completed[key])
                    rows.append(completed[key])
                    continue
                r = await _run_case(case, system, language, use_judge)
                # Same failure mode mid-sweep: Ollama dying part-way through.
                # Claimed LLM calls that took no time did not happen.
                if r.get("llm_calls", 0) >= 2 and r.get("latency_s", 99) < 3.0:
                    raise RuntimeError(
                        f"{system}/{language} {case['case_id']} reported "
                        f"{r['llm_calls']} LLM calls in {r['latency_s']}s — Ollama is not "
                        "responding. Stopping so the sweep is not filled with fictitious "
                        "abstentions; fix Ollama and re-run with --resume."
                    )
                r["retriever"] = fingerprint
                r["candidate_recall"] = floors.get((case["case_id"], language))
                sys_rows.append(r)
                rows.append(r)
                with runs_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                print(
                    f"[{n}/{total}] [{system}/{language}] {r['case_id']}: {r['outcome']} "
                    f"cand={r.get('candidate_recall')} union={r.get('agent_union_recall')} "
                    f"cited={r.get('citation_recall')} {r['latency_s']}s",
                    flush=True,
                )
        per_system[system] = summarise(sys_rows)

    rows_by_system: dict[str, list[dict]] = {}
    for r in rows:
        rows_by_system.setdefault(r["system"], []).append(r)
    base_rows = rows_by_system.get("B0") or rows_by_system.get("B0-2step") or []
    paired = {
        s_: paired_by_case(rows_by_system[s_], base_rows)
        for s_ in rows_by_system
        if base_rows and s_ not in ("B0", "B0-2step")
    }

    baseline = per_system.get("B0") or per_system.get("B0-2step")
    b3 = per_system.get("B3")
    gates = _gates(baseline, b3) if baseline and b3 else {}

    report = {
        "cases": str(cases_path),
        "n_cases": len(cases),
        "languages": languages,
        "systems": systems,
        "gold_case_provenance": {
            "authored_by": "claude-generated from the indexed corpus",
            "review_status": "unreviewed",
            "caveat": (
                "Questions were drafted from the same corpus the retriever indexes. "
                "Treat these numbers as a regression signal, not as an external "
                "evaluation of historical accuracy."
            ),
        },
        "retriever": fingerprint,
        "judge": {
            "state": judge_state,
            "model": judge_mod.os.getenv("GEMINI_MODEL", "") if judge_state == "enabled" else "",
            "caveat": (
                "answer_correctness comes from an external LLM judge and is the only "
                "non-deterministic metric here. Every other metric is reproducible offline."
            ),
        },
        "per_system": per_system,
        "paired_vs_baseline": paired,
        "gates": gates,
        "runs": rows,
    }

    out = out or settings.reports_dir / "agent_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {k: v for k, v in report.items() if k != "runs"}


def _gates(b0: dict, b3: dict) -> dict:
    """The pre-registered thresholds from the build spec, evaluated literally."""

    def gain(key: str) -> float | None:
        a, b = b0.get(key), b3.get(key)
        return round((b - a) * 100, 2) if a is not None and b is not None else None

    def kind_gain(kind: str, key: str) -> float | None:
        a = (b0.get("by_kind") or {}).get(kind, {}).get(key)
        b = (b3.get("by_kind") or {}).get(kind, {}).get(key)
        return round((b - a) * 100, 2) if a is not None and b is not None else None

    recall_gain = kind_gain("multihop", "agent_union_recall")
    correctness_gain = gain("answer_correctness")
    p95_b0, p95_b3 = b0.get("latency_p95"), b3.get("latency_p95")

    # Retrieval budget is not equal across systems, so a recall gain is not
    # attributable to planning until the budget is matched. Reported next to
    # the gate rather than folded into it.
    budget_b0, budget_b3 = b0.get("retrieval_calls_mean"), b3.get("retrieval_calls_mean")
    budget_confound = (
        budget_b3 is not None and budget_b0 is not None and budget_b3 > budget_b0 * 1.2
    )

    passed_quality = any(g is not None and g >= 10.0 for g in (recall_gain, correctness_gain))
    return {
        "1_multihop_gain_pp": {
            "recall": recall_gain,
            "correctness": correctness_gain,
            "threshold_pp": 10.0,
            "pass": passed_quality,
            "retrieval_calls_b0": budget_b0,
            "retrieval_calls_b3": budget_b3,
            "budget_confounded": budget_confound,
            "attribution_note": (
                "B3 issues more retrievals than B0; a recall gain cannot be "
                "attributed to planning until the budget is matched"
                if budget_confound
                else "retrieval budgets are comparable"
            ),
            "consequence": (
                "README may claim an agentic improvement"
                if passed_quality
                else "README must NOT claim an agentic improvement; default to the "
                "smallest system with evidence"
            ),
        },
        "2_route_macro_f1": {
            "value": b3.get("route_macro_f1"),
            "rule_first_value": b3.get("rule_route_macro_f1"),
            "threshold": 0.80,
            "over_decomposition_pct": b3.get("simple_over_decomposed_pct"),
            "pass": (b3.get("route_macro_f1") or 0) >= 0.80
            and (b3.get("simple_over_decomposed_pct") or 0) <= 20,
            "consequence": "switch to a rule-first router; LLM only produces subqueries",
        },
        "3_unanswerable_leakage": {
            "false_answers": b3.get("false_answer_on_unanswerable"),
            "pass": (b3.get("false_answer_on_unanswerable") or 0) == 0,
        },
        "4_citation_validity": {
            "citation_precision": b3.get("citation_precision"),
            "claim_coverage": b3.get("claim_citation_coverage"),
        },
        "5_latency": {
            "b0_p95": p95_b0,
            "b3_p95": p95_b3,
            "ratio": round(p95_b3 / p95_b0, 2) if p95_b0 and p95_b3 else None,
            "pass": bool(p95_b3 and p95_b3 <= 30 and (not p95_b0 or p95_b3 <= 2.5 * p95_b0)),
            "consequence": "cut subqueries 3->2, keep one correction; else default the UI to B1",
        },
    }
