"""Deterministic citation gate.

No LLM call happens here. Either every claim rests on an evidence id that was
actually retrieved and actually exists in the corpus, or the run abstains. The
citation block itself is rebuilt from the validated evidence, so a model that
mangles a URL costs nothing — we never use its URL.
"""

from __future__ import annotations

import re
from functools import lru_cache

from frus_agentic_rag.retrieval.models import Claim, Evidence, canonical_url

CHUNK_ID_RE = re.compile(r"^[A-Za-z0-9\-.]+:[A-Za-z0-9_\-.]+:\d+$")
URL_RE = re.compile(r"https?://\S+")
CITABLE_SUBTYPES = {"historical-document"}


@lru_cache(maxsize=1)
def _published_volumes() -> frozenset[str]:
    from frus_agentic_rag.ingest.manifest import load_manifest

    try:
        rows = load_manifest().to_pylist()
    except FileNotFoundError:
        return frozenset()
    return frozenset(r["volume_id"] for r in rows if r["status"] == "published")


def validate(
    claims: list[Claim],
    answer_text: str,
    retrieved: list[Evidence],
    accepted_ids: list[str] | None = None,
) -> tuple[list[str], list[Evidence], list[str]]:
    """Returns (errors, cited evidence, citation lines).

    A non-empty error list is a blocking failure: the caller must abstain.
    """
    errors: list[str] = []
    by_id = {e.evidence_id: e for e in retrieved}
    allowed = set(accepted_ids) if accepted_ids else set(by_id)
    published = _published_volumes()

    cited: dict[str, Evidence] = {}

    if not claims:
        errors.append("no claims: answer carries no citable statements")

    for i, claim in enumerate(claims):
        if not claim.evidence_ids:
            errors.append(f"claim {i} has no evidence id")
            continue
        for eid in claim.evidence_ids:
            if not CHUNK_ID_RE.match(eid):
                errors.append(f"claim {i}: malformed evidence id {eid!r}")
                continue
            if eid not in by_id:
                errors.append(f"claim {i}: evidence id {eid!r} was never retrieved")
                continue
            if eid not in allowed:
                errors.append(f"claim {i}: evidence id {eid!r} was not accepted by the grader")
                continue
            ev = by_id[eid]
            if ev.volume_id not in published and published:
                errors.append(f"claim {i}: volume {ev.volume_id} is not a published volume")
                continue
            if ev.subtype not in CITABLE_SUBTYPES:
                errors.append(
                    f"claim {i}: {eid} is a {ev.subtype}, not a citable historical document"
                )
                continue
            cited[eid] = ev

    # The model must not smuggle in its own links; ours are built below.
    for url in URL_RE.findall(answer_text):
        errors.append(f"answer text contains a model-written URL: {url}")

    ordered = sorted(cited.values(), key=lambda e: (e.volume_id, e.document_id, e.evidence_id))
    lines = _citation_lines(ordered)
    return errors, ordered, lines


def _citation_lines(evidence: list[Evidence]) -> list[str]:
    """One line per document, not per chunk — the document is the citable unit."""
    seen: dict[tuple[str, str], Evidence] = {}
    for e in evidence:
        seen.setdefault((e.volume_id, e.document_id), e)
    out = []
    for (vol, doc), e in seen.items():
        num = f"Doc. {e.doc_number}" if e.doc_number else doc
        date = f", {e.date_from}" if e.date_from else ""
        head = f" — {e.head}" if e.head else ""
        out.append(f"[{vol} {num}{date}]{head}\n  {canonical_url(vol, doc)}")
    return out


def repair_claims(claims: list[Claim], retrieved: list[Evidence]) -> list[Claim]:
    """Drop unusable ids without an LLM call.

    Cheap salvage: a claim that still has one good id survives; one that has
    none does not, and the caller then abstains on an empty claim list.
    """
    valid = {e.evidence_id for e in retrieved if e.subtype in CITABLE_SUBTYPES}
    out = []
    for c in claims:
        keep = [e for e in c.evidence_ids if e in valid]
        if keep:
            out.append(Claim(text=c.text, evidence_ids=keep))
    return out


def strip_urls(text: str) -> str:
    return URL_RE.sub("", text).strip()
