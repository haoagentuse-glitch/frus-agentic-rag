"""Entailment check: does the cited passage actually say what the claim says?

Post-hoc attribution closed one gap and left another open. It guarantees that a
citation points at a passage that was retrieved, and that the passage is the one
the claim most resembles — which is the passage the claim was written from. It
does not check that the passage supports the claim, and no threshold on the
cross-encoder score can, because the score cannot separate "written from this"
from "entailed by this": measured over 29 claims the best-passage score never
fell below 0.285.

The grounding probe measured what that costs. With the only document that could
answer a question withheld, the system answered 7 of 12 times — a 58%
fabrication rate, established without knowing any correct answer.

So this asks the question directly, per claim, against the passage that was
attached. It is a far weaker demand than judging historical correctness: the
model reads one short claim and one passage and says whether the second states
the first. Omission is not contradiction — a claim the passage neither supports
nor denies is unsupported, and unsupported claims lose their citation, which the
gate then drops.

One LLM call for the whole answer, not one per claim: the budget is three calls
and this is the fourth, so it has to be cheap.
"""

from __future__ import annotations

from typing import Any

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import Claim, Evidence

VERIFY_SYSTEM = """You check whether a passage states a claim. Nothing else.

For each numbered claim you are given every passage it was attributed to,
separated by ---. Answer supported=true if those passages together state the
claim, or state something the claim follows from directly; the claim may rest on
two of them rather than one. Answer supported=false if they are about a
different subject, a different person, or simply do not mention it.

Not mentioning something is NOT support. Being plausible is NOT support. Do not
use anything you know about history; judge only what the passage says.

Return one entry per claim, in the same order, with no commentary."""


def verify_user(pairs: list[tuple[str, str]], max_chars: int) -> str:
    """`passage` here is every attributed passage, joined."""
    blocks = [
        f"CLAIM {i}: {claim}\nPASSAGE(S) {i}: {passage[:max_chars]}"
        for i, (claim, passage) in enumerate(pairs)
    ]
    return (
        "\n\n---\n\n".join(blocks)
        + f"\n\nReturn {len(pairs)} entries as JSON, one per claim, in order."
    )


async def verify(
    claims: list[Claim], evidence: list[Evidence], calls_used: int = 0
) -> tuple[list[Claim], dict]:
    """Strip the citation from any claim its passage does not support.

    Yields to the call budget rather than exceeding it. On the default
    configuration this is the third call (plan, synthesise, verify) and there is
    room; only `grader_mode=llm` together with a correction reaches four before
    getting here, and that mode exists for comparison. The budget is a
    pre-registered bound, so it is not raised to fit a stage added later — but
    the skip is recorded, because an unverified answer must not look like a
    verified one.
    """
    settings = get_settings()
    record: dict[str, Any] = {"enabled": settings.entailment_check, "claims": len(claims)}
    if not settings.entailment_check or not claims:
        record["skipped"] = "disabled" if not settings.entailment_check else "no claims"
        return claims, record
    if calls_used >= settings.budgets.max_llm_calls:
        record["skipped"] = f"llm budget exhausted ({calls_used}/{settings.budgets.max_llm_calls})"
        record["unverified"] = len(claims)
        return claims, record

    by_id = {e.evidence_id: e for e in evidence}
    pairs: list[tuple[str, str]] = []
    checkable: list[int] = []
    budget = settings.entailment_max_chars
    for i, c in enumerate(claims):
        # Every passage attached to the claim, not just the first. Attribution
        # keeps up to three, and a claim drawn from two of them is not stated by
        # either alone: checking only the first rejected 14 of 19 claims on
        # questions whose gold document had been retrieved, and turned one
        # answer into an abstention.
        texts = [by_id[j].text for j in c.evidence_ids if j in by_id]
        if not texts:
            continue
        share = max(300, budget // len(texts))
        pairs.append((c.text, "\n---\n".join(t[:share] for t in texts)))
        checkable.append(i)
    if not pairs:
        record["skipped"] = "no claim carries a retrievable passage"
        return claims, record

    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.agent.schemas import ClaimSupport

    try:
        result = await get_client().structured(
            VERIFY_SYSTEM,
            verify_user(pairs, settings.entailment_max_chars),
            ClaimSupport,
        )
    except Exception as exc:
        # A verifier that did not run has not refuted anything. Stripping
        # citations on failure would turn an outage into mass abstention.
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["skipped"] = "verifier unavailable; claims left as attributed"
        return claims, record

    verdicts = {v.index: v.supported for v in result.claims}
    out: list[Claim] = []
    stripped: list[int] = []
    for i, c in enumerate(claims):
        if i in checkable and verdicts.get(checkable.index(i), True) is False:
            stripped.append(i)
            out.append(Claim(text=c.text, evidence_ids=[]))
        else:
            out.append(c)

    record.update(
        {
            "checked": len(pairs),
            "unsupported": len(stripped),
            "unverified": len(claims) - len(pairs),
        }
    )
    return out, record
