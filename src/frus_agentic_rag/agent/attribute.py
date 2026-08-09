"""Post-hoc citation attribution: the model writes prose, code writes citations.

Asking the generator to copy chunk ids verbatim made citation a transcription
task, and transcription is where it failed. Measured over 350 runs, 57 of 79
abstentions on answerable questions were answers that carried claims, every
claim carrying an id, with zero citations surviving — the ids were not the ones
retrieved. In Chinese the abstention rate was 40.7% against 12.0% in English,
which is what a copying failure looks like: the same model writing the same
evidence, worse at reproducing an ASCII identifier while generating Chinese.

So the identifier is removed from the generation task. The model writes claims;
the cross-encoder that already ranked the passages then scores each claim
against each accepted passage and code attaches the ids. A claim can no longer
cite something that was not retrieved, because the model never names anything.

What this does NOT do is check that the passage supports the claim. It attaches
the passage a claim most resembles, which is the passage it was written from —
so a claim that over-reads its source scores against that source highly. A
score floor was tried and removed: measured over 29 claims the best-passage
score never fell below 0.285, and no threshold separates "written from this
passage" from "entailed by this passage", because the score does not encode the
difference. Every claim therefore gets a citation, and the citation means
provenance, not verification. An entailment stage would be the honest fix.
"""

from __future__ import annotations

from typing import Any

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import Claim, Evidence


def attribute(claims: list[Claim], evidence: list[Evidence]) -> tuple[list[Claim], dict]:
    """Replace each claim's evidence ids with cross-encoder-chosen ones.

    Relative, not absolute. Cross-encoder logits are uncalibrated and shift with
    question type — measured on this corpus, gold passages sit at +1.86 on lookup
    questions and -1.32 on multi-hop ones — so a fixed floor would attribute
    everything on one kind and nothing on another. Each claim keeps its best
    passage plus whatever sits within `attribution_margin` of it.
    """
    settings = get_settings()
    record: dict[str, Any] = {"mode": "post_hoc", "claims": len(claims), "evidence": len(evidence)}
    if not claims or not evidence:
        record["skipped"] = "nothing to attribute"
        return claims, record

    from frus_agentic_rag.retrieval import rerank as rr

    if not rr.available():
        # Falling back to the model's own ids is worse than saying so: it would
        # silently reinstate the failure this stage exists to remove.
        record["skipped"] = "cross-encoder unavailable; kept model-written ids"
        return claims, record

    pairs: list[tuple[str, str]] = []
    for c in claims:
        for e in evidence:
            pairs.append((c.text, e.text[: settings.reranker_max_chars]))
    scores = rr.score_pairs(pairs)
    if scores is None:
        record["skipped"] = "scoring unavailable; kept model-written ids"
        return claims, record

    out: list[Claim] = []
    best_scores: list[float] = []
    n = len(evidence)
    for i, c in enumerate(claims):
        row = scores[i * n : (i + 1) * n]
        ranked = sorted(zip(evidence, row, strict=True), key=lambda x: x[1], reverse=True)
        best = ranked[0][1]
        best_scores.append(round(float(best), 4))
        keep = [
            e.evidence_id
            for e, s in ranked[: settings.attribution_top_k]
            if best - s <= settings.attribution_margin
        ]
        out.append(Claim(text=c.text, evidence_ids=keep))

    record.update(
        {
            "attributed": len(out),
            "best_scores": best_scores,
            "margin": settings.attribution_margin,
            # Kept so the two schemes can be compared on the same run rather
            # than across two sweeps.
            "model_written_ids_valid": sum(
                1
                for c in claims
                if c.evidence_ids
                and all(i in {e.evidence_id for e in evidence} for i in c.evidence_ids)
            ),
        }
    )
    return out, record
