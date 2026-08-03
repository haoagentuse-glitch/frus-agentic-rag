"""Streaming TEI parser.

Only `div[@type='document'][@subtype='historical-document']` reaches the main
evidence corpus. Editorial notes, front/back matter, indexes and planned stubs
are kept but routed to the aux store so they can never be cited as a document.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from lxml import etree

from frus_agentic_rag.config import TEI_NS, get_settings
from frus_agentic_rag.retrieval.models import FrusDocument

FRUS_NS = "http://history.state.gov/frus/ns/1.0"
T = f"{{{TEI_NS}}}"
X = "{http://www.w3.org/XML/1998/namespace}"
F = f"{{{FRUS_NS}}}"

# Elements whose text is apparatus, not the document body.
_APPARATUS = {f"{T}note", f"{T}pb", f"{T}fw"}
# The head is captured separately and prepended to every chunk, so including it
# in the body would duplicate the title in chunk 0.
_BODY_SKIP = _APPARATUS | {f"{T}head"}


def _clean(s: str) -> str:
    return " ".join(s.split())


def _body_text(div: etree._Element) -> str:
    """Text of the document with footnotes and page breaks stripped out.

    Footnotes are editorial apparatus; inlining them mid-sentence corrupts the
    passage a retriever has to match against.
    """
    parts: list[str] = []

    def walk(el: etree._Element) -> None:
        if el.tag in _BODY_SKIP and el is not div:
            # Keep the tail: text after </note> belongs to the parent sentence.
            if el.tail:
                parts.append(el.tail)
            return
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
        if el.tail:
            parts.append(el.tail)

    walk(div)
    return _clean("".join(parts))


def _head_text(div: etree._Element) -> str:
    head = div.find(f"{T}head")
    if head is None:
        return ""
    parts = [head.text or ""]
    for child in head:
        if child.tag not in _APPARATUS:
            parts.append("".join(child.itertext()))
        if child.tail:
            parts.append(child.tail)
    return _clean("".join(parts))


def _source_note(div: etree._Element) -> str:
    note = div.find(f"{T}head/{T}note[@type='source']")
    if note is None:
        return ""
    return _clean("".join(note.itertext()))


def _persons(div: etree._Element) -> list[str]:
    seen: dict[str, None] = {}
    for p in div.iter(f"{T}persName"):
        name = _clean("".join(p.itertext()))
        if name:
            seen.setdefault(name, None)
    return list(seen)[:40]


def _date_display(div: etree._Element) -> str:
    dl = div.find(f"{T}opener/{T}dateline")
    if dl is not None:
        return _clean("".join(dl.itertext()))
    d = div.find(f".//{T}date[@when]")
    return _clean("".join(d.itertext())) if d is not None else ""


def iter_documents(
    path: Path, volume_id: str, include_editorial_notes: bool = False
) -> Iterator[FrusDocument]:
    """Yield citable document divs from one volume, streaming."""
    wanted = {"historical-document"}
    if include_editorial_notes:
        wanted.add("editorial-note")

    context = etree.iterparse(str(path), events=("end",), tag=f"{T}div")
    for _, div in context:
        if div.get("type") != "document":
            continue
        subtype = div.get("subtype") or ""
        if subtype not in wanted:
            _drop(div)
            continue

        doc_id = div.get(f"{X}id") or ""
        if not doc_id:
            _drop(div)
            continue

        text = _body_text(div)
        if not text:
            _drop(div)
            continue

        yield FrusDocument(
            volume_id=volume_id,
            document_id=doc_id,
            doc_number=div.get("n") or "",
            subtype=subtype,  # type: ignore[arg-type]
            head=_head_text(div),
            date_from=(div.get(f"{F}doc-dateTime-min") or "")[:10],
            date_to=(div.get(f"{F}doc-dateTime-max") or "")[:10],
            date_display=_date_display(div),
            persons=_persons(div),
            text=text,
            source_note=_source_note(div),
        )
        _drop(div)


def _drop(el: etree._Element) -> None:
    """Free the subtree and its now-dead preceding siblings."""
    el.clear()
    parent = el.getparent()
    if parent is not None:
        while el.getprevious() is not None:
            del parent[0]


def volume_path(volume_id: str) -> Path:
    return get_settings().raw_dir / f"{volume_id}.xml"
