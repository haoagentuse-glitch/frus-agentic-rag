"""BGE-M3 embedding with per-volume checkpointing.

The unit of resume is the volume: a run that dies at volume 300 restarts at 300,
not at zero. Vectors are written back into the Parquet shard first and merged
into LanceDB second, so a crash between the two is recoverable by re-merging.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.index.build import CHUNKS_TABLE, connect

_MODEL: Any = None


def get_model(device: str | None = None) -> Any:
    global _MODEL
    settings = get_settings()
    device = device or settings.embed_device
    if _MODEL is None or getattr(_MODEL, "_frus_device", None) != device:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(settings.bge_model_path, device=device)
        model.max_seq_length = settings.max_chunk_tokens
        model._frus_device = device  # type: ignore[attr-defined]
        _MODEL = model
    return _MODEL


def encode(texts: list[str], device: str | None = None, batch_size: int | None = None) -> np.ndarray:
    settings = get_settings()
    model = get_model(device)
    return model.encode(
        texts,
        batch_size=batch_size or settings.embed_batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)


def encode_with_oom_backoff(texts: list[str], device: str, batch_size: int) -> tuple[np.ndarray, int]:
    """Halve the batch on CUDA OOM rather than losing the whole run: 16 -> 8 -> 4."""
    import torch

    bs = batch_size
    while True:
        try:
            return encode(texts, device=device, batch_size=bs), bs
        except torch.cuda.OutOfMemoryError:
            if bs <= 2:
                raise
            torch.cuda.empty_cache()
            bs //= 2


# --- state ---------------------------------------------------------------


def _state_path() -> Path:
    return get_settings().data_dir / "index" / "embed_state.json"


def load_state() -> dict:
    p = _state_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"done": [], "batch_size": get_settings().embed_batch_size, "stats": {}}


def save_state(state: dict) -> None:
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(p)


# --- main loop -----------------------------------------------------------


def embed_volume(
    parquet_path: Path, device: str, batch_size: int, merge: bool = True
) -> dict:
    table = pq.read_table(parquet_path)
    n = table.num_rows
    if n == 0:
        return {"chunks": 0, "seconds": 0.0, "batch_size": batch_size}

    texts = table["text"].to_pylist()
    t0 = time.perf_counter()
    vectors, used_bs = encode_with_oom_backoff(texts, device, batch_size)
    elapsed = time.perf_counter() - t0

    dim = vectors.shape[1]
    vec_col = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dim
    )
    table = table.set_column(table.schema.get_field_index("vector"), "vector", vec_col)
    table = table.set_column(
        table.schema.get_field_index("embedded"), "embedded", pa.array([True] * n)
    )

    tmp = parquet_path.with_suffix(".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(parquet_path)

    if merge:
        merge_into_lancedb(table)

    return {
        "chunks": n,
        "seconds": round(elapsed, 2),
        "chunks_per_s": round(n / elapsed, 2) if elapsed else 0.0,
        "batch_size": used_bs,
        "peak_vram_gib": _peak_vram(device),
    }


def _peak_vram(device: str) -> float:
    if device != "cuda":
        return 0.0
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**30, 3)


def merge_into_lancedb(table: pa.Table) -> None:
    db = connect()
    if CHUNKS_TABLE not in db.table_names():
        db.create_table(CHUNKS_TABLE, data=table)
        return
    tbl = db.open_table(CHUNKS_TABLE)
    (
        tbl.merge_insert("chunk_id")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute(table)
    )


def embed_all(
    resume: bool = True,
    device: str | None = None,
    batch_size: int | None = None,
    limit_volumes: int | None = None,
    merge: bool = True,
) -> dict:
    settings = get_settings()
    device = device or settings.embed_device
    state = load_state()
    batch_size = batch_size or state.get("batch_size", settings.embed_batch_size)

    files = sorted(settings.chunks_dir.glob("*.parquet"))
    done = set(state["done"]) if resume else set()
    todo = [f for f in files if f.stem not in done]
    if limit_volumes:
        todo = todo[:limit_volumes]

    t0 = time.perf_counter()
    total_chunks = 0
    for i, f in enumerate(todo, 1):
        r = embed_volume(f, device, batch_size, merge=merge)
        batch_size = r["batch_size"]
        total_chunks += r["chunks"]
        state["done"] = sorted({*state["done"], f.stem})
        state["batch_size"] = batch_size
        state["stats"][f.stem] = r
        save_state(state)
        rate = total_chunks / max(1e-9, time.perf_counter() - t0)
        print(
            f"[{i}/{len(todo)}] {f.stem}: {r['chunks']} chunks "
            f"@ {r.get('chunks_per_s', 0)}/s (bs={batch_size}) "
            f"| cumulative {rate:.1f}/s",
            flush=True,
        )

    elapsed = time.perf_counter() - t0
    return {
        "volumes_embedded": len(todo),
        "volumes_total": len(files),
        "volumes_done": len(state["done"]),
        "chunks": total_chunks,
        "seconds": round(elapsed, 1),
        "chunks_per_s": round(total_chunks / elapsed, 2) if elapsed else 0.0,
        "device": device,
        "batch_size": batch_size,
    }


def benchmark(n_chunks: int = 10_000, device: str | None = None, batch_size: int | None = None) -> dict:
    """Measure throughput on a real sample before committing to the full run."""
    settings = get_settings()
    device = device or settings.embed_device
    batch_size = batch_size or settings.embed_batch_size

    texts: list[str] = []
    for f in sorted(settings.chunks_dir.glob("*.parquet")):
        texts.extend(pq.read_table(f, columns=["text"])["text"].to_pylist())
        if len(texts) >= n_chunks:
            break
    texts = texts[:n_chunks]
    if not texts:
        raise FileNotFoundError("no parsed chunks to benchmark; run `frus ingest` first")

    tok_total = sum(len(t) for t in texts) / 4  # rough chars->tokens

    get_model(device)  # warm the model out of the timed section
    encode(texts[: min(64, len(texts))], device=device, batch_size=batch_size)
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    vectors, used_bs = encode_with_oom_backoff(texts, device, batch_size)
    elapsed = time.perf_counter() - t0

    total_chunks = _estimate_total_chunks()
    rate = len(texts) / elapsed
    result = {
        "n_chunks": len(texts),
        "device": device,
        "batch_size": used_bs,
        "seconds": round(elapsed, 2),
        "chunks_per_s": round(rate, 2),
        "approx_tokens_per_s": round(tok_total / elapsed, 1),
        "peak_vram_gib": _peak_vram(device),
        "dim": int(vectors.shape[1]),
        "estimated_total_chunks": total_chunks,
        "estimated_full_hours": round(total_chunks / rate / 3600, 2) if rate else None,
        "estimated_full_hours_with_30pct_buffer": (
            round(total_chunks / rate / 3600 * 1.3, 2) if rate else None
        ),
    }
    out = settings.reports_dir / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result


def _estimate_total_chunks() -> int:
    """Extrapolate full-corpus chunk count from parsed volumes and the manifest."""
    from frus_agentic_rag.ingest.manifest import load_manifest

    settings = get_settings()
    files = sorted(settings.chunks_dir.glob("*.parquet"))
    if not files:
        return 0
    parsed_chunks = sum(pq.read_metadata(f).num_rows for f in files)

    manifest = load_manifest().to_pylist()
    by_id = {r["volume_id"]: r for r in manifest}
    parsed_docs = sum(by_id.get(f.stem, {}).get("n_historical_documents", 0) for f in files)
    all_docs = sum(r["n_historical_documents"] for r in manifest)
    if parsed_docs == 0:
        return parsed_chunks
    return int(round(parsed_chunks / parsed_docs * all_docs))
