"""Sentence filtering must never change what a citation means."""

from __future__ import annotations

from frus_agentic_rag.retrieval.focus import _assemble, split_sentences


def test_split_keeps_abbreviations_and_document_numbers_intact() -> None:
    text = (
        "Mr. Nolting called on the Secretary at 3 p.m. "
        "He referred to despatch No. 421 of October 20. "
        "The Secretary said he would consider it."
    )
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Mr. Nolting called on the Secretary at 3 p.m."
    assert sents[1].startswith("He referred")


def test_split_keeps_initials_with_the_name() -> None:
    sents = split_sentences("Frederick E. Nolting reported. The Secretary concurred.")
    assert sents == ["Frederick E. Nolting reported.", "The Secretary concurred."]


def test_split_errs_towards_merging_when_an_abbreviation_is_ambiguous() -> None:
    """`Jr. The` is a real boundary; `Jr. Nolting` is not, and nothing in the
    string distinguishes them. The splitter merges, because the two failures are
    not symmetric: an over-merged unit costs a few extra sentences of context,
    while an over-split one leaves a bare "Mr." to be scored as its own passage.
    """
    assert len(split_sentences("A memo by Frederick E. Nolting, Jr. The reply came.")) == 1


def test_split_handles_paragraphs() -> None:
    sents = split_sentences("First point.\n\nSecond point. Third point.")
    assert sents == ["First point.", "Second point.", "Third point."]


def test_assemble_marks_gaps_and_preserves_reading_order() -> None:
    sents = ["A.", "B.", "C.", "D.", "E."]
    assert _assemble(sents, {0, 1}) == "A. B."
    assert _assemble(sents, {0, 3}) == "A. […] D."
    assert _assemble(sents, {4, 0}) == "A. […] E."


def test_focus_is_a_no_op_without_a_reranker(monkeypatch) -> None:
    from frus_agentic_rag.retrieval import focus, rerank

    monkeypatch.setattr(rerank, "available", lambda: False)
    from frus_agentic_rag.models import Evidence

    ev = [
        Evidence(
            evidence_id="frus1949v07p1:d376:0",
            volume_id="frus1949v07p1",
            document_id="d376",
            subtype="historical-document",
            text="Long passage. " * 100,
        )
    ]
    out, stats = focus.focus_evidence("anything", ev)
    assert out is ev
    assert stats["skipped"] == "reranker unavailable"
    assert out[0].text_focus == ""
