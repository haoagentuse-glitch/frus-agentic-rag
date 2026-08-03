"""Scan the pinned FRUS snapshot into a manifest before anything is parsed.

The manifest is the inventory of record: it decides which volumes are published
and therefore eligible for the evidence corpus, and it pins every file to a
SHA-256 so a later rebuild can prove it read the same bytes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

from frus_agentic_rag.config import FRUS_SOURCE_COMMIT, TEI_NS, XML_NS, get_settings
from frus_agentic_rag.models import VolumeMeta

T = f"{{{TEI_NS}}}"
X = f"{{{XML_NS}}}"


def sha256_file(path: Path, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def _text(el: etree._Element | None) -> str:
    if el is None:
        return ""
    return " ".join("".join(el.itertext()).split())


def read_volume_meta(path: Path) -> VolumeMeta:
    """Parse only the header plus div counts. Cheap enough to run over all 694."""
    tree = etree.parse(str(path))
    root = tree.getroot()

    vol_id = root.get(f"{X}id") or path.stem
    idno = root.find(f".//{T}publicationStmt/{T}idno[@type='frus']")
    if idno is not None and idno.text:
        vol_id = idno.text.strip()

    titles = {t.get("type"): _text(t) for t in root.iterfind(f".//{T}titleStmt/{T}title")}
    editors = [_text(e) for e in root.iterfind(f".//{T}titleStmt/{T}editor") if _text(e)]

    pub_year = ""
    date_from = date_to = ""
    for d in root.iterfind(f".//{T}publicationStmt/{T}date"):
        if d.get("type") == "publication-date" and not pub_year:
            pub_year = _text(d)
        elif d.get("type") == "content-date":
            date_from = (d.get("notBefore") or "")[:10]
            date_to = (d.get("notAfter") or "")[:10]

    doc_divs = root.xpath("//tei:div[@type='document']", namespaces={"tei": TEI_NS})
    hist = [d for d in doc_divs if d.get("subtype") == "historical-document"]

    return VolumeMeta(
        volume_id=vol_id,
        title_complete=titles.get("complete", ""),
        title_volume=titles.get("volume", ""),
        sub_series=titles.get("sub-series", ""),
        volume_number=titles.get("volume-number", ""),
        publication_year=pub_year,
        content_date_from=date_from,
        content_date_to=date_to,
        editors=editors,
        # A planned volume is a stub: it has a header but carries no document divs.
        status="published" if doc_divs else "planned",
        n_document_divs=len(doc_divs),
        n_historical_documents=len(hist),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


MANIFEST_SCHEMA = pa.schema(
    [
        ("volume_id", pa.string()),
        ("path", pa.string()),
        ("size_bytes", pa.int64()),
        ("sha256", pa.string()),
        ("source_commit", pa.string()),
        ("status", pa.string()),
        ("title_complete", pa.string()),
        ("title_volume", pa.string()),
        ("sub_series", pa.string()),
        ("volume_number", pa.string()),
        ("publication_year", pa.string()),
        ("content_date_from", pa.string()),
        ("content_date_to", pa.string()),
        ("editors", pa.list_(pa.string())),
        ("n_document_divs", pa.int32()),
        ("n_historical_documents", pa.int32()),
    ]
)


def build_manifest(raw_dir: Path | None = None, out: Path | None = None) -> pa.Table:
    settings = get_settings()
    raw_dir = raw_dir or settings.raw_dir
    out = out or settings.manifest_path
    out.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for path in sorted(raw_dir.glob("*.xml")):
        meta = read_volume_meta(path)
        rows.append(
            {
                **meta.model_dump(exclude={"sha256", "size_bytes"}),
                "path": str(path),
                "sha256": meta.sha256,
                "size_bytes": meta.size_bytes,
                "source_commit": FRUS_SOURCE_COMMIT,
            }
        )

    table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)
    return table


def load_manifest(path: Path | None = None) -> pa.Table:
    path = path or get_settings().manifest_path
    return pq.read_table(path)


def published_volume_ids(path: Path | None = None) -> list[str]:
    table = load_manifest(path)
    mask = pa.compute.equal(table["status"], "published")
    return table.filter(mask)["volume_id"].to_pylist()
