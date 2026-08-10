"""Run the probes from docs/EVAL_DESIGN.md, one mechanism at a time.

    python scripts/run_probes.py p8 p4 p5 p9      # no generation, seconds
    FRUS_GPU=1 ... python scripts/run_probes.py p1 p2   # generation, no judge
    FRUS_GPU=1 ... python scripts/run_probes.py all

Each probe prints its own pass criterion and writes reports/probe_<name>.json.
Seven of the nine never call the judge, which is deliberate: the judge has
returned 0.0 and 0.6 for the same input and moved 28 points across machines, so
a diagnosis that depends on it has worse resolution than the effects it looks
for.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

PROBES = Path("eval/probes")
REPORTS = Path("reports")


def load(name: str) -> list[dict]:
    path = PROBES / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing — run scripts/build_probes.py first")
    cases = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    if not cases:
        raise SystemExit(f"{path} is empty — a probe with no cases measures nothing")
    return cases


def save(name: str, payload: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"probe_{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _docs(evidence) -> set[str]:
    return {f"{e.volume_id}:{e.document_id}" for e in evidence}


# --- P8 routing: planner only ----------------------------------------------


async def p8() -> dict:
    """Route classification. No retrieval, no generation, no judge."""
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.prompts import PLANNER_SYSTEM, planner_user
    from frus_agentic_rag.agent.schemas import QueryPlan

    cases = load("p8_route")
    client = get_client()
    rows = []
    for c in cases:
        try:
            plan = await client.structured(
                PLANNER_SYSTEM, planner_user(c["question_en"]), QueryPlan
            )
            got = plan.route
        except Exception as exc:
            got = f"error: {type(exc).__name__}"
        rows.append({**{k: c[k] for k in ("case_id", "expected_route")}, "actual": got})
        print(f"  {c['case_id']:18s} expected={c['expected_route']:9s} got={got}", flush=True)

    acc = sum(1 for r in rows if r["actual"] == r["expected_route"]) / len(rows)
    by_intent: dict[str, list[str]] = {}
    for r in rows:
        by_intent.setdefault(r["expected_route"], []).append(r["actual"])
    consistency = sum(1 for v in by_intent.values() if len(set(v)) == 1) / len(by_intent)
    return {
        "criterion": "accuracy >= 0.80 and every intent routed consistently across its 3 phrasings",
        "accuracy": round(acc, 3),
        "phrasing_consistency": round(consistency, 3),
        "pass": acc >= 0.80 and consistency == 1.0,
        "rows": rows,
    }


# --- P4 / P5 / P9: retrieval only -------------------------------------------


async def _retrieval(name: str, languages=("en", "zh-TW"), top_k: int = 20) -> list[dict]:
    from frus_agentic_rag.models import SearchFilters
    from frus_agentic_rag.retrieval.hybrid import hybrid_search

    rows = []
    for c in load(name):
        for lang in languages:
            q = c["question_zh"] if lang == "zh-TW" else c["question_en"]
            hits = await hybrid_search(q, SearchFilters(subtypes=["historical-document"]), top_k)
            got = [f"{h.volume_id}:{h.document_id}" for h in hits]
            gold = set(c["gold_documents"])
            rank = next((i + 1 for i, d in enumerate(got) if d in gold), None)
            rows.append(
                {
                    "case_id": c["case_id"],
                    "language": lang,
                    "rank_of_first_gold": rank,
                    "recall_at_20": len(gold & set(got)) / len(gold) if gold else None,
                    "n_gold": len(gold),
                }
            )
            print(f"  {c['case_id']:10s} {lang:5s} rank={rank}", flush=True)
    return rows


async def p4() -> dict:
    rows = await _retrieval("p4_precise")
    at1 = sum(1 for r in rows if r["rank_of_first_gold"] == 1) / len(rows)
    at10 = sum(1 for r in rows if (r["rank_of_first_gold"] or 99) <= 10) / len(rows)
    return {
        "criterion": "a question quoting a unique title and its date should rank that document 1st",
        "hit_at_1": round(at1, 3),
        "hit_at_10": round(at10, 3),
        "pass": at1 >= 0.80,
        "rows": rows,
    }


async def p5() -> dict:
    """Cross-language retrieval with the date supplied as a filter.

    The first version put the date in the question text and scored 0.0 in both
    languages — dates live in the index, not in the prose, so neither English
    nor Chinese could match on them and the probe measured nothing. The date is
    known by construction, so it belongs in SearchFilters; what is left for
    retrieval to do is match the country, which is the cross-language question.
    """
    from frus_agentic_rag.models import SearchFilters
    from frus_agentic_rag.retrieval.hybrid import hybrid_search

    rows = []
    for c in load("p5_crosslang"):
        date = c["anchor"]["date"]
        for lang in ("en", "zh-TW"):
            q = c["question_zh"] if lang == "zh-TW" else c["question_en"]
            hits = await hybrid_search(
                q,
                SearchFilters(subtypes=["historical-document"], date_from=date, date_to=date),
                20,
            )
            got = [f"{h.volume_id}:{h.document_id}" for h in hits]
            gold = set(c["gold_documents"])
            rank = next((i + 1 for i, d in enumerate(got) if d in gold), None)
            rows.append(
                {
                    "case_id": c["case_id"],
                    "language": lang,
                    "country": c["anchor"]["country"],
                    "rank_of_first_gold": rank,
                    "candidates_in_window": len(got),
                }
            )
            print(f"  {c['case_id']:10s} {lang:5s} rank={rank} window={len(got)}", flush=True)
    by = {lang: [r for r in rows if r["language"] == lang] for lang in ("en", "zh-TW")}
    res = {
        lang: round(sum(1 for r in v if (r["rank_of_first_gold"] or 99) <= 10) / len(v), 3)
        for lang, v in by.items()
        if v
    }
    return {
        "criterion": "the same document, asked in Chinese with no Latin anchors, within top 10",
        "hit_at_10": res,
        "gap_pp": round((res.get("en", 0) - res.get("zh-TW", 0)) * 100, 1),
        "pass": res.get("zh-TW", 0) >= 0.5,
        "rows": rows,
    }


async def p9() -> dict:
    """Timeline: does the date filter return the right window, in order?"""
    from frus_agentic_rag.models import SearchFilters
    from frus_agentic_rag.retrieval.tools import get_toolbox

    rows = []
    for c in load("p9_timeline"):
        hits = await get_toolbox().timeline_search(
            c["question_en"],
            c["date_from"],
            c["date_to"],
            SearchFilters(
                volume_ids=[c["volume_id"]],
                date_from=c["date_from"],
                date_to=c["date_to"],
                subtypes=["historical-document"],
            ),
            20,
            "main",
        )
        got = _docs(hits)
        gold = set(c["gold_documents"])
        dates = [h.date_from for h in hits if h.date_from]
        rows.append(
            {
                "case_id": c["case_id"],
                "recall": len(gold & got) / len(gold) if gold else None,
                "n_returned": len(hits),
                # build_where implements OVERLAP (date_to >= from AND
                # date_from <= to), so a document spanning the window is
                # returned even when its own start date is outside it. Both are
                # defensible for a timeline; the semantics had simply never been
                # written down, so report each rather than assert one.
                "all_starts_in_range": all(c["date_from"] <= d <= c["date_to"] for d in dates),
                "chronological": dates == sorted(dates),
                "starts_before_window": [d for d in dates if d < c["date_from"]][:3],
            }
        )
        print(
            f"  {c['case_id']:8s} recall={rows[-1]['recall']} "
            f"starts_in_range={rows[-1]['all_starts_in_range']}",
            flush=True,
        )
    n = len(rows)
    return {
        "criterion": "gold recall >= 0.8 and results in date order; range is overlap-based",
        "mean_recall": round(sum(r["recall"] or 0 for r in rows) / n, 3),
        "all_starts_in_range": sum(r["all_starts_in_range"] for r in rows) / n,
        "chronological": sum(r["chronological"] for r in rows) / n,
        "filter_semantics": "overlap, not containment — see the note in the rows",
        "pass": sum(r["chronological"] for r in rows) == n
        and sum(r["recall"] or 0 for r in rows) / n >= 0.8,
        "rows": rows,
    }


# --- P1 grounding, P2 order, P7 abstention: generation, no judge -------------


async def p1() -> dict:
    """Ask with the only answering document removed. Answering is fabricating."""
    from frus_agentic_rag.agent.run import answer

    rows = []
    for c in load("p1_grounding"):
        for lang in ("en", "zh-TW"):
            q = c["question_zh"] if lang == "zh-TW" else c["question_en"]
            a = await answer(q, language=lang, system="B3", exclude_documents=c["gold_documents"])
            leaked = bool(set(a.retrieved_document_ids) & set(c["gold_documents"]))
            rows.append(
                {
                    "case_id": c["case_id"],
                    "language": lang,
                    "outcome": a.outcome,
                    "abstained": a.outcome != "answer",
                    "gold_leaked_into_retrieval": leaked,
                    "answer_head": a.answer_text[:160],
                }
            )
            print(
                f"  {c['case_id']:8s} {lang:5s} {a.outcome}{'  LEAK' if leaked else ''}", flush=True
            )
    valid = [r for r in rows if not r["gold_leaked_into_retrieval"]]
    rate = sum(r["abstained"] for r in valid) / len(valid) if valid else 0.0
    return {
        "criterion": "with the answer withheld the system must abstain; answering is fabrication",
        "abstention_rate": round(rate, 3),
        "fabrications": sum(1 for r in valid if not r["abstained"]),
        "excluded_for_leak": len(rows) - len(valid),
        "pass": rate >= 0.90,
        "rows": rows,
    }


_ATTRIB = re.compile(r"\b((?:Mr|Dr|Sir|President|Secretary|Ambassador)\.? [A-Z][a-z]{3,})")


async def p2() -> dict:
    """Same evidence, two orderings. Who-said-what must not depend on position."""
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.prompts import synth_system, synth_user
    from frus_agentic_rag.agent.schemas import AgentAnswer
    from frus_agentic_rag.config import get_settings
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect
    from frus_agentic_rag.models import Evidence

    tbl = connect().open_table(CHUNKS_TABLE)
    client = get_client()
    rows = []
    for c in load("p2_order"):
        vol, _, did = c["gold_documents"][0].partition(":")
        raw = (
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
            .limit(50)
            .to_list()
        )
        ev = [
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
            for r in sorted(raw, key=lambda x: x["chunk_id"])
        ]
        if len(ev) < 2:
            continue
        pairs = []
        for order in ("forward", "reverse"):
            seq = ev if order == "forward" else list(reversed(ev))
            try:
                draft = await client.structured(
                    synth_system("en"),
                    synth_user(c["question_en"], seq),
                    AgentAnswer,
                    num_predict=get_settings().ollama_num_predict_synthesis,
                )
                pairs.append(
                    {(m, cl.text[:60]) for cl in draft.claims for m in _ATTRIB.findall(cl.text)}
                )
            except Exception:
                pairs.append(set())
        actors = [{a for a, _ in p} for p in pairs]
        same_actors = actors[0] == actors[1]
        overlap = (
            len(actors[0] & actors[1]) / len(actors[0] | actors[1])
            if (actors[0] | actors[1])
            else 1.0
        )
        rows.append(
            {
                "case_id": c["case_id"],
                "chunks": len(ev),
                "actors_forward": sorted(actors[0]),
                "actors_reverse": sorted(actors[1]),
                "same_actor_set": same_actors,
                "actor_overlap": round(overlap, 3),
            }
        )
        print(
            f"  {c['case_id']:8s} chunks={len(ev):2d} "
            f"same_actors={same_actors} overlap={overlap:.2f}",
            flush=True,
        )
    n = len(rows) or 1
    stable = sum(r["same_actor_set"] for r in rows) / n
    return {
        "criterion": "reordering the same evidence must not change which people are named",
        "actor_set_stability": round(stable, 3),
        "mean_overlap": round(sum(r["actor_overlap"] for r in rows) / n, 3),
        "pass": stable >= 0.80,
        "rows": rows,
    }


async def p7() -> dict:
    """Unpublished volumes must abstain; false premises must be corrected."""
    from frus_agentic_rag.agent.run import answer

    rows = []
    for c in load("p7_abstain"):
        a = await answer(c["question_zh"], language="zh-TW", system="B3")
        ok: bool
        if c["kind"] == "unpublished":
            ok = a.outcome != "answer"
        else:
            # A correction must mention the true date, not the claimed one.
            true_date = c["must_correct"]["true_date"]
            ok = true_date in a.answer_text or true_date[:4] in a.answer_text
        rows.append(
            {
                "case_id": c["case_id"],
                "kind": c["kind"],
                "outcome": a.outcome,
                "ok": ok,
                "answer_head": a.answer_text[:160],
            }
        )
        print(f"  {c['case_id']:10s} {c['kind']:14s} {a.outcome:8s} ok={ok}", flush=True)
    by = {}
    for k in ("unpublished", "false_premise"):
        s = [r for r in rows if r["kind"] == k]
        if s:
            by[k] = round(sum(r["ok"] for r in s) / len(s), 3)
    return {
        "criterion": "planned volumes -> abstain; false premise -> state the true date",
        "by_kind": by,
        "pass": by.get("unpublished", 0) >= 0.9 and by.get("false_premise", 0) >= 0.5,
        "rows": rows,
    }


# --- P3 volume, P6 multihop: need the judge ---------------------------------


async def p3() -> dict:
    """Same question, increasing evidence. Isolates dilution from overflow."""
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.prompts import synth_system, synth_user
    from frus_agentic_rag.agent.schemas import AgentAnswer
    from frus_agentic_rag.config import get_settings
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect
    from frus_agentic_rag.evaluation.judge import judge_answer
    from frus_agentic_rag.models import Evidence

    tbl = connect().open_table(CHUNKS_TABLE)
    cols = [
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

    def ev_of(rows) -> list[Evidence]:
        return [
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
            for r in rows
        ]

    filler = tbl.search().where("subtype = 'historical-document'").select(cols).limit(200).to_list()
    client = get_client()
    rows = []
    for c in load("p3_volume"):
        vol, _, did = c["gold_documents"][0].partition(":")
        gold_rows = (
            tbl.search()
            .where(f"volume_id = '{vol}' AND document_id = '{did}'")
            .select(cols)
            .limit(20)
            .to_list()
        )
        if not gold_rows:
            continue
        gold = ev_of(gold_rows[:1])
        noise = ev_of([r for r in filler if r["document_id"] != did])
        for level in c["levels"]:
            seq = gold + noise[: max(0, level - 1)]
            try:
                draft = await client.structured(
                    synth_system("en"),
                    synth_user(c["question_en"], seq),
                    AgentAnswer,
                    num_predict=get_settings().ollama_num_predict_synthesis,
                )
                text = draft.answer_text
            except Exception:
                text = ""
            v = await judge_answer(
                c["question_en"], text, c["gold_documents"], True, "answer" if text else "abstain"
            )
            rows.append(
                {
                    "case_id": c["case_id"],
                    "level": level,
                    "score": v.score,
                    "judge": v.judge,
                    "reason": (v.reason or "")[:140],
                }
            )
            print(f"  {c['case_id']:8s} level={level:2d} score={v.score}", flush=True)
    curve = {}
    for lv in sorted({r["level"] for r in rows}):
        s = [r["score"] for r in rows if r["level"] == lv and r["score"] is not None]
        if s:
            curve[lv] = round(sum(s) / len(s), 3)
    return {
        "criterion": "score must not fall as irrelevant evidence is added around the answer",
        "curve_by_level": curve,
        "pass": bool(curve) and min(curve.values()) >= 0.7 * max(curve.values()),
        "rows": rows,
    }


async def p6() -> dict:
    """Multi-hop with historian-authored gold: both documents must be used."""
    from frus_agentic_rag.agent.run import answer
    from frus_agentic_rag.evaluation.judge import judge_answer

    rows = []
    for c in load("p6_multihop"):
        for lang in ("en", "zh-TW"):
            q = c["question_zh"] if lang == "zh-TW" else c["question_en"]
            a = await answer(q, language=lang, system="B3")
            gold = set(c["gold_documents"])
            v = await judge_answer(q, a.answer_text, sorted(gold), True, a.outcome)
            rows.append(
                {
                    "case_id": c["case_id"],
                    "language": lang,
                    "outcome": a.outcome,
                    # "both" alone cannot tell a total retrieval miss from the
                    # multi-hop failure. The first run reported 0.0 and it took
                    # a separate check to establish that one of the two was
                    # always found; the count is recorded now so it cannot.
                    "n_gold_retrieved": len(gold & set(a.retrieved_document_ids)),
                    "both_retrieved": gold <= set(a.retrieved_document_ids),
                    "both_cited": gold <= set(a.cited_document_ids),
                    "score": v.score,
                    "reason": (v.reason or "")[:140],
                }
            )
            print(
                f"  {c['case_id']:8s} {lang:5s} "
                f"both_retrieved={rows[-1]['both_retrieved']} score={v.score}",
                flush=True,
            )
    n = len(rows) or 1
    sc = [r["score"] for r in rows if r["score"] is not None]
    return {
        "criterion": "both cross-referenced documents retrieved, and the answer uses both",
        "mean_gold_retrieved_of_2": round(sum(r["n_gold_retrieved"] for r in rows) / n, 3),
        "at_least_one": round(sum(r["n_gold_retrieved"] > 0 for r in rows) / n, 3),
        "both_retrieved": round(sum(r["both_retrieved"] for r in rows) / n, 3),
        "both_cited": round(sum(r["both_cited"] for r in rows) / n, 3),
        "correctness": round(sum(sc) / len(sc), 3) if sc else None,
        "pass": sum(r["both_retrieved"] for r in rows) / n >= 0.7,
        "rows": rows,
    }


ALL = {"p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6, "p7": p7, "p8": p8, "p9": p9}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", nargs="+", help="p1..p9 or 'all'")
    args = ap.parse_args()
    names = list(ALL) if "all" in args.probes else args.probes

    summary = {}
    for name in names:
        if name not in ALL:
            raise SystemExit(f"unknown probe {name}; choose from {sorted(ALL)}")
        print(f"\n=== {name} ===", flush=True)
        res = await ALL[name]()
        save(name, res)
        summary[name] = {k: v for k, v in res.items() if k != "rows"}
        print(json.dumps(summary[name], ensure_ascii=False, indent=2), flush=True)

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}  pass={v.get('pass')}  {v.get('criterion', '')[:70]}")


asyncio.run(main())
