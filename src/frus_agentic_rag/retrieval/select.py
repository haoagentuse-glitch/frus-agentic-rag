"""Deterministic evidence selection from cross-encoder scores.

This replaces the LLM grader. Measured over the ablation, that stage removed 40%
of the retrieved documents and 15 gold documents with them, on a window of ten
passages capped at eight accepts by its own schema; inverted so that omission
meant "keep", it then removed nothing at all across every run tried. Either way
the decision was not being made on evidence quality.

The cross-encoder has already read each passage against the query and produced a
relevance logit. Thresholding that number is the same judgement without a
generation step: no id to copy wrong, no window to fall outside of, no schema
cap, and it costs nothing beyond scoring that already happened.

Three controls, because one is not enough:

- absolute: a floor. Passages the cross-encoder scored below it are off-topic in
  their own right.
- margin: a relative floor, `best - margin`. A question whose whole candidate
  pool scores low still has a best passage, and the absolute floor alone would
  empty it.
- per-hop minimum: however the two floors fall, each hop keeps at least this
  many passages, so a multi-hop answer cannot lose a hop entirely.

All three are unset by default. Cross-encoder logits are not calibrated across
models or query languages, so a threshold picked by intuition is a guess about a
distribution nobody has looked at; `scripts/score_distribution.py` produces that
distribution against gold labels, and the numbers belong in config only once it
has been run. Until then this keeps everything and records what it would have
done, which is the honest no-op.
"""

from __future__ import annotations

from typing import Any

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import Evidence


def score_rows(evidence: list[Evidence], gold_documents: set[str] | None = None) -> list[dict]:
    """One row per passage: what it scored, where it ranked, whether it is gold.

    Emitted whether or not filtering is active. The rows are the input to
    threshold selection, and a threshold chosen without them is a guess.
    """
    gold = gold_documents or set()
    rows = []
    for i, e in enumerate(evidence):
        rows.append(
            {
                "evidence_id": e.evidence_id,
                "document": f"{e.volume_id}:{e.document_id}",
                "hop": e.hop or "main",
                "position": i,
                "rank_rerank": e.rank_rerank,
                "rerank_score": e.rerank_score,
                "rrf_score": e.score,
                "is_gold": f"{e.volume_id}:{e.document_id}" in gold,
            }
        )
    return rows


def select_evidence(evidence: list[Evidence]) -> tuple[list[str], dict[str, Any]]:
    """Return (accepted ids, decision record).

    Keeps everything when any control is unset or when the passages carry no
    cross-encoder score — an unscored passage is unjudged, not rejected, which
    is the distinction the LLM grader kept getting wrong in both directions.
    """
    settings = get_settings()
    ids = [e.evidence_id for e in evidence]
    scored = [e for e in evidence if e.rerank_score is not None]

    record: dict[str, Any] = {
        "mode": "score",
        "passages": len(evidence),
        "scored": len(scored),
        "absolute": settings.score_keep_absolute,
        "margin": settings.score_keep_margin,
        "min_per_hop": settings.score_min_per_hop,
    }

    if not scored:
        record["skipped"] = "no cross-encoder scores"
        return ids, record
    if settings.score_keep_absolute is None and settings.score_keep_margin is None:
        record["skipped"] = "no threshold configured; run scripts/score_distribution.py"
        record["score_range"] = [
            round(min(e.rerank_score or 0.0 for e in scored), 4),
            round(max(e.rerank_score or 0.0 for e in scored), 4),
        ]
        return ids, record

    best = max(e.rerank_score or 0.0 for e in scored)
    floor = float("-inf")
    if settings.score_keep_absolute is not None:
        floor = max(floor, settings.score_keep_absolute)
    if settings.score_keep_margin is not None:
        floor = max(floor, best - settings.score_keep_margin)

    keep = {e.evidence_id for e in scored if (e.rerank_score or 0.0) >= floor}
    # Unscored passages are kept: nothing judged them.
    keep.update(e.evidence_id for e in evidence if e.rerank_score is None)

    # Then restore each hop's floor, best-scoring first.
    by_hop: dict[str, list[Evidence]] = {}
    for e in scored:
        by_hop.setdefault(e.hop or "main", []).append(e)
    restored: list[str] = []
    for hop, group in by_hop.items():
        group.sort(key=lambda e: e.rerank_score or 0.0, reverse=True)
        held = [e for e in group if e.evidence_id in keep]
        for e in group[: settings.score_min_per_hop]:
            if e.evidence_id not in keep:
                keep.add(e.evidence_id)
                restored.append(e.evidence_id)
        record.setdefault("per_hop", {})[hop] = {
            "candidates": len(group),
            "above_floor": len(held),
        }

    record.update(
        {
            "best": round(best, 4),
            "floor": round(floor, 4),
            "accepted": len(keep),
            "dropped": len(ids) - len(keep),
            "restored_by_min_per_hop": restored,
        }
    )
    return [i for i in ids if i in keep], record


def missing_hops(evidence: list[Evidence], subqueries: list) -> list[str]:
    """Hops whose best passage never cleared the floor, for the B3 retry.

    The corrective loop used to fire on the grader's verdict. Without a grader
    the same signal is arithmetic: a hop whose best candidate is below the
    absolute floor did not find what it was sent for.
    """
    settings = get_settings()
    if settings.score_keep_absolute is None:
        return []
    best: dict[str, float] = {}
    for e in evidence:
        if e.rerank_score is None:
            continue
        hop = e.hop or "main"
        best[hop] = max(best.get(hop, float("-inf")), e.rerank_score)
    out = []
    for sq in subqueries:
        hop = getattr(sq, "hop_id", "main")
        if best.get(hop, float("-inf")) < settings.score_keep_absolute:
            out.append(getattr(sq, "query", hop))
    return out
