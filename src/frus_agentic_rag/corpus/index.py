"""Parse volumes to Parquet, then load them into LanceDB with a BM25 index.

Order matters: the full lexical index is built first and is usable on its own,
while dense vectors land per volume afterwards. That way a half-finished
embedding run still leaves a working retriever.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.corpus.chunk import chunk_document
from frus_agentic_rag.corpus.manifest import load_manifest
from frus_agentic_rag.corpus.tei import iter_documents
from frus_agentic_rag.models import Chunk

CHUNKS_TABLE = "chunks"


def chunk_schema(dim: int) -> pa.Schema:
    return pa.schema(
        [
            ("chunk_id", pa.string()),
            ("volume_id", pa.string()),
            ("document_id", pa.string()),
            ("ordinal", pa.int32()),
            ("n_chunks_in_doc", pa.int32()),
            ("subtype", pa.string()),
            ("head", pa.string()),
            ("doc_number", pa.string()),
            ("date_from", pa.string()),
            ("date_to", pa.string()),
            ("persons", pa.list_(pa.string())),
            ("text", pa.string()),
            ("n_tokens", pa.int32()),
            ("vector", pa.list_(pa.float32(), dim)),
            ("embedded", pa.bool_()),
        ]
    )


def chunks_to_table(chunks: Iterable[Chunk], dim: int) -> pa.Table:
    rows = [{**c.model_dump(), "vector": [0.0] * dim, "embedded": False} for c in chunks]
    return pa.Table.from_pylist(rows, schema=chunk_schema(dim))


def _atomic_write(table: pa.Table, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(out)


def parse_volume(volume_id: str, path: Path, dim: int) -> int:
    """Parse one volume to `chunks/{volume_id}.parquet`. Returns chunk count."""
    settings = get_settings()
    out = settings.chunks_dir / f"{volume_id}.parquet"

    chunks: list[Chunk] = []
    for doc in iter_documents(path, volume_id):
        chunks.extend(chunk_document(doc))

    _atomic_write(chunks_to_table(chunks, dim), out)
    return len(chunks)


def parse_all(
    limit_volumes: int | None = None,
    volume_ids: list[str] | None = None,
    resume: bool = True,
) -> dict:
    settings = get_settings()
    manifest = load_manifest()
    rows = manifest.to_pylist()
    published = [r for r in rows if r["status"] == "published"]
    if volume_ids:
        wanted = set(volume_ids)
        published = [r for r in published if r["volume_id"] in wanted]
    if limit_volumes:
        published = published[:limit_volumes]

    per_volume: dict[str, int] = {}
    stats: dict[str, Any] = {
        "volumes": 0,
        "chunks": 0,
        "skipped": 0,
        "per_volume": per_volume,
    }
    for row in published:
        vid = row["volume_id"]
        out = settings.chunks_dir / f"{vid}.parquet"
        if resume and out.exists():
            n = pq.read_metadata(out).num_rows
            stats["skipped"] += 1
        else:
            n = parse_volume(vid, Path(row["path"]), settings.embed_dim)
        stats["volumes"] += 1
        stats["chunks"] += n
        per_volume[vid] = n
    return stats


# --- LanceDB -------------------------------------------------------------


def connect():
    import lancedb

    settings = get_settings()
    settings.lancedb_uri.parent.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(str(settings.lancedb_uri))


def load_into_lancedb(rebuild: bool = False, batch_volumes: int = 20) -> dict:
    settings = get_settings()
    db = connect()
    files = sorted(settings.chunks_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no chunk parquet under {settings.chunks_dir}")

    if rebuild and CHUNKS_TABLE in db.table_names():
        db.drop_table(CHUNKS_TABLE)

    total = 0
    table = None
    buf: list[pa.Table] = []

    def flush() -> None:
        nonlocal table, buf
        if not buf:
            return
        batch = pa.concat_tables(buf)
        if table is None and CHUNKS_TABLE not in db.table_names():
            table = db.create_table(CHUNKS_TABLE, data=batch)
        else:
            table = table or db.open_table(CHUNKS_TABLE)
            table.add(batch)
        buf = []

    for f in files:
        buf.append(pq.read_table(f))
        total += buf[-1].num_rows
        if len(buf) >= batch_volumes:
            flush()
    flush()

    tbl = db.open_table(CHUNKS_TABLE)
    return {"rows": tbl.count_rows(), "parquet_rows": total, "volumes": len(files)}


def build_bm25_index() -> dict:
    """Native tantivy FTS over the chunk text. Usable without any vectors."""
    db = connect()
    tbl = db.open_table(CHUNKS_TABLE)
    tbl.create_fts_index("text", replace=True, use_tantivy=False)
    return {"rows": tbl.count_rows(), "index": "fts:text"}


def build_ann_index(num_partitions: int | None = None) -> dict:
    db = connect()
    tbl = db.open_table(CHUNKS_TABLE)
    n = tbl.count_rows()
    parts = num_partitions or max(1, min(4096, int(n**0.5)))
    tbl.create_index(
        metric="cosine",
        vector_column_name="vector",
        num_partitions=parts,
        num_sub_vectors=64,
        replace=True,
    )
    return {"rows": n, "num_partitions": parts}


# --- corpus stats --------------------------------------------------------


def corpus_stats(out: Path | None = None) -> dict:
    settings = get_settings()
    manifest = load_manifest().to_pylist()
    published = [r for r in manifest if r["status"] == "published"]

    files = sorted(settings.chunks_dir.glob("*.parquet"))
    chunk_rows = sum(pq.read_metadata(f).num_rows for f in files)

    stats = {
        "source_commit": manifest[0]["source_commit"] if manifest else "",
        "xml_files": len(manifest),
        "published_volumes": len(published),
        "planned_volumes": len(manifest) - len(published),
        "type_document_divs": sum(r["n_document_divs"] for r in manifest),
        "historical_document_divs": sum(r["n_historical_documents"] for r in manifest),
        "editorial_note_divs": sum(
            r["n_document_divs"] - r["n_historical_documents"] for r in manifest
        ),
        "parsed_volumes": len(files),
        "chunks": chunk_rows,
    }
    try:
        db = connect()
        if CHUNKS_TABLE in db.table_names():
            tbl = db.open_table(CHUNKS_TABLE)
            stats["lancedb_rows"] = tbl.count_rows()
            stats["lancedb_embedded"] = tbl.count_rows("embedded = true")
    except Exception as exc:  # index not built yet is a normal state
        stats["lancedb_error"] = str(exc)

    out = out or settings.reports_dir / "corpus_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats
