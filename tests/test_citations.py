"""The citation gate is the last thing between the model and a false claim."""

from __future__ import annotations

import pytest

from frus_agentic_rag.agent import citations as cite
from frus_agentic_rag.models import Claim, Evidence


def ev(eid: str = "frus1969-76v17:d4:0", subtype: str = "historical-document") -> Evidence:
    vol, doc, _ = eid.rsplit(":", 2)
    return Evidence(
        evidence_id=eid,
        volume_id=vol,
        document_id=doc,
        doc_number="4",
        subtype=subtype,  # type: ignore[arg-type]
        head="NSSM 14",
        date_from="1969-02-05",
        text="body",
    )


@pytest.fixture(autouse=True)
def _no_manifest(monkeypatch):
    """Run without a manifest so the published-volume check is a no-op."""
    monkeypatch.setattr(cite, "_published_volumes", lambda: frozenset())


def test_valid_claim_passes():
    e = ev()
    errors, cited, lines = cite.validate(
        [Claim(text="x", evidence_ids=[e.evidence_id])], "an answer", [e]
    )
    assert errors == []
    assert cited == [e]
    assert lines[0].startswith("[frus1969-76v17 Doc. 4, 1969-02-05]")
    assert "https://history.state.gov/historicaldocuments/frus1969-76v17/d4" in lines[0]


def test_invented_id_is_blocking():
    e = ev()
    errors, cited, _ = cite.validate(
        [Claim(text="x", evidence_ids=["frus1958-60v01:d999:0"])], "a", [e]
    )
    assert any("never retrieved" in x for x in errors)
    assert cited == []


def test_malformed_id_is_blocking():
    e = ev()
    errors, _, _ = cite.validate([Claim(text="x", evidence_ids=["not-an-id"])], "a", [e])
    assert any("malformed" in x for x in errors)


def test_retrieved_but_not_accepted_is_blocking():
    e = ev()
    errors, _, _ = cite.validate(
        [Claim(text="x", evidence_ids=[e.evidence_id])], "a", [e], accepted_ids=[]
    )
    assert any("not accepted" in x for x in errors)


def test_editorial_note_is_not_citable():
    e = ev(subtype="editorial-note")
    errors, _, _ = cite.validate([Claim(text="x", evidence_ids=[e.evidence_id])], "a", [e])
    assert any("not a citable historical document" in x for x in errors)


def test_no_claims_is_blocking():
    assert any("no claims" in x for x in cite.validate([], "prose", [ev()])[0])


def test_model_written_url_is_blocking():
    e = ev()
    errors, _, _ = cite.validate(
        [Claim(text="x", evidence_ids=[e.evidence_id])],
        "See https://example.com/made-up",
        [e],
    )
    assert any("model-written URL" in x for x in errors)


def test_repair_drops_bad_ids_but_keeps_the_claim():
    e = ev()
    claims = [Claim(text="x", evidence_ids=[e.evidence_id, "frus1958-60v01:d999:0"])]
    repaired = cite.repair_claims(claims, [e])
    assert repaired == [Claim(text="x", evidence_ids=[e.evidence_id])]


def test_repair_drops_a_claim_with_nothing_left():
    assert cite.repair_claims([Claim(text="x", evidence_ids=["frus1:d9:0"])], [ev()]) == []


def test_citation_lines_are_one_per_document_not_per_chunk():
    a, b = ev("frus1969-76v17:d4:0"), ev("frus1969-76v17:d4:1")
    _, _, lines = cite.validate(
        [Claim(text="x", evidence_ids=[a.evidence_id, b.evidence_id])], "a", [a, b]
    )
    assert len(lines) == 1
