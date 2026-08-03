"""TEI parsing and chunking against a fixture that carries the awkward cases."""

from __future__ import annotations

import pytest
from lxml import etree

from frus_agentic_rag.ingest import tei
from frus_agentic_rag.ingest.manifest import read_volume_meta

FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"
     xmlns:frus="http://history.state.gov/frus/ns/1.0"
     xml:id="frusTEST">
  <teiHeader><fileDesc>
    <titleStmt>
      <title type="complete">Foreign Relations, Test Volume</title>
      <title type="volume">Testland, 1970</title>
      <editor role="primary">A. Editor</editor>
    </titleStmt>
    <publicationStmt>
      <idno type="frus">frusTEST</idno>
      <date type="publication-date">2001</date>
      <date type="content-date" notBefore="1970-01-01T00:00:00-05:00"
            notAfter="1970-12-31T23:59:59-05:00">1970</date>
    </publicationStmt>
  </fileDesc></teiHeader>
  <text><body>
    <div type="compilation">
      <div type="document" subtype="historical-document" n="1" xml:id="d1"
           frus:doc-dateTime-min="1970-03-01T00:00:00-05:00"
           frus:doc-dateTime-max="1970-03-01T23:59:59-05:00">
        <head>1. Memorandum of Conversation<note n="1" type="source"
              xml:id="d1fn1">Source: Test Archive. Secret.</note></head>
        <opener><dateline>Washington, <date when="1970-03-01">March 1, 1970</date>.</dateline></opener>
        <!-- an editorial comment that used to crash the walk -->
        <p>The <persName>Secretary</persName> spoke about policy.<note n="2">A footnote
           that must not land mid-sentence.</note> He then left.</p>
        <pb n="12"/>
        <p>A second paragraph.</p>
      </div>
      <div type="document" subtype="editorial-note" n="2" xml:id="d2">
        <head>2. Editorial Note</head>
        <p>Context written by the editors, not a primary document.</p>
      </div>
    </div>
  </body></text>
</TEI>
"""


@pytest.fixture
def volume(tmp_path):
    p = tmp_path / "frusTEST.xml"
    p.write_text(FIXTURE, encoding="utf-8")
    return p


def test_manifest_counts_documents_and_marks_published(volume):
    meta = read_volume_meta(volume)
    assert meta.volume_id == "frusTEST"
    assert meta.status == "published"
    assert meta.n_document_divs == 2
    assert meta.n_historical_documents == 1
    assert meta.content_date_from == "1970-01-01"
    assert meta.editors == ["A. Editor"]


def test_planned_stub_has_no_document_divs(tmp_path):
    stub = FIXTURE.split("<text>")[0] + "<text><body><div type='compilation'/></body></text></TEI>"
    p = tmp_path / "frusPLANNED.xml"
    p.write_text(stub, encoding="utf-8")
    assert read_volume_meta(p).status == "planned"


def test_only_historical_documents_are_yielded(volume):
    docs = list(tei.iter_documents(volume, "frusTEST"))
    assert [d.document_id for d in docs] == ["d1"]
    assert docs[0].subtype == "historical-document"


def test_editorial_notes_are_available_when_asked_for(volume):
    docs = list(tei.iter_documents(volume, "frusTEST", include_editorial_notes=True))
    assert {d.document_id for d in docs} == {"d1", "d2"}


def test_footnotes_and_page_breaks_are_stripped_from_the_body(volume):
    doc = next(iter(tei.iter_documents(volume, "frusTEST")))
    assert "must not land mid-sentence" not in doc.text
    assert "Source: Test Archive" not in doc.text
    # The tail after </note> still belongs to the sentence.
    assert "He then left." in doc.text
    assert "A second paragraph." in doc.text


def test_head_is_not_duplicated_into_the_body(volume):
    doc = next(iter(tei.iter_documents(volume, "frusTEST")))
    assert doc.head == "1. Memorandum of Conversation"
    assert not doc.text.startswith("1. Memorandum of Conversation")


def test_comment_nodes_do_not_crash_the_walk(volume):
    # Regression: lxml raises on itertext over a _Comment, which killed the
    # full-corpus parse at volume 478 of 552.
    assert "<!--" in volume.read_text()
    assert next(iter(tei.iter_documents(volume, "frusTEST"))).text


def test_metadata_is_carried_onto_the_document(volume):
    doc = next(iter(tei.iter_documents(volume, "frusTEST")))
    assert doc.doc_number == "1"
    assert doc.date_from == "1970-03-01"
    assert doc.source_note.startswith("Source: Test Archive")
    assert "Secretary" in doc.persons
    assert doc.url == "https://history.state.gov/historicaldocuments/frusTEST/d1"


def test_processing_instructions_are_skipped():
    div = etree.fromstring(
        '<div xmlns="http://www.tei-c.org/ns/1.0" type="document" '
        'subtype="historical-document"><p>text<?php ?> here</p></div>'
    )
    assert "text" in tei._body_text(div)
