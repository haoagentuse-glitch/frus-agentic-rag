"""Cross-encoder reranking.

BM25 and BGE-M3 both score similarity: the query and the passage are turned into
representations independently and compared. Nothing in that pipeline ever reads
the two together, so a passage on the right subject that does not answer the
question scores as highly as one that does.

Measured on this corpus, that is the largest single loss in the system. Taking
50 candidates per hop puts 91.7% of gold documents in the candidate pool, but
the 14 that reach the synthesis window carry only 37.5% — 54 percentage points
lost to ordering alone. Retrieving more made it worse, not better, because a
wider pool means the ranking decides more.

A cross-encoder reads query and passage in one forward pass, so the two attend
to each other and the score is a relevance judgement rather than a distance.
It is far too slow to run over 723,557 chunks, which is why it reranks the
candidates the cheap arms produce rather than replacing them.
"""

from __future__ import annotations

import threading
from typing import Any

from frus_agentic_rag.config import get_settings

_MODEL: Any = None
_DISABLED = False
# Model loading is guarded by a lock. `hybrid_search` runs one thread per hop,
# and three threads entering an unguarded loader all saw `_MODEL is None` and
# called `from_pretrained` on the same checkpoint at once. The collision left
# weights on the meta device, so the subsequent `.to(device)` raised "Cannot
# copy out of meta tensor" — reproducibly on multi-hop questions, never on
# single-hop ones, which is what made it look like a model or library problem.
_LOAD_LOCK = threading.Lock()


def available() -> bool:
    return bool(get_settings().reranker_model_path) and not _DISABLED


def resolve_device(configured: str) -> str:
    """Prefer CUDA when it is there.

    Measured on this machine, the same 50-pair rerank takes 1.0s on GPU and
    62.6s on CPU. Defaulting to CPU and relying on the caller to pass a flag
    made a 60x penalty the consequence of forgetting an environment variable,
    which is not a default worth keeping.
    """
    if configured != "cpu":
        return configured
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def get_model(device: str | None = None) -> Any:
    """Load the cross-encoder, or disable reranking for the rest of the process.

    Built directly on transformers rather than sentence-transformers'
    CrossEncoder. That wrapper raised "Cannot copy out of meta tensor" from its
    own `.to(device)` inside the agent while loading fine in a fresh process,
    and the same checkpoint loads reliably through AutoModelForSequenceClassification
    in every configuration tested. Reranking is an optional stage over
    candidates the cheap arms already produced, so a load failure disables it
    and falls back to RRF rather than taking retrieval down.
    """
    global _MODEL, _DISABLED
    settings = get_settings()
    device = device or resolve_device(settings.reranker_device)
    if _MODEL is not None and _MODEL.get("device") == device:
        return _MODEL

    with _LOAD_LOCK:
        # Re-check: another thread may have finished while this one waited.
        if _DISABLED:
            return None
        if _MODEL is not None and _MODEL.get("device") == device:
            return _MODEL
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        try:
            tok = AutoTokenizer.from_pretrained(settings.reranker_model_path, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                settings.reranker_model_path, local_files_only=True
            )
            model.eval().to(device)
            if device == "cuda" and settings.embed_fp16:
                model.half()
        except Exception as exc:
            _DISABLED = True
            print(
                f"[rerank] disabled, falling back to RRF order: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        _MODEL = {"tok": tok, "model": model, "device": device, "torch": torch}
    return _MODEL


def _score(bundle: dict, pairs: list[tuple[str, str]]) -> list[float]:
    torch = bundle["torch"]
    tok, model, device = bundle["tok"], bundle["model"], bundle["device"]
    settings = get_settings()
    out: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(pairs), settings.reranker_batch_size):
            batch = pairs[i : i + settings.reranker_batch_size]
            enc = tok(
                [a for a, _ in batch],
                [b for _, b in batch],
                padding=True,
                truncation=True,
                max_length=settings.reranker_max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.view(-1).float()
            out.extend(logits.tolist())
    return out


def rerank(query: str, rows: list[dict], top_k: int, text_key: str = "text") -> list[dict]:
    """Reorder candidates by cross-encoder relevance, truncating to top_k.

    Returns the input untouched when the model is unavailable or the candidate
    list is already short enough to be worth no reordering — the cost is a GPU
    forward pass per candidate, so it only pays where the ordering matters.
    """
    settings = get_settings()
    if not available() or len(rows) <= 1 or top_k >= len(rows):
        return rows[:top_k]

    bundle = get_model()
    if bundle is None:
        return rows[:top_k]
    pairs = [(query, r.get(text_key, "")[: settings.reranker_max_chars]) for r in rows]
    scores = _score(bundle, pairs)
    ordered = sorted(zip(rows, scores, strict=True), key=lambda x: float(x[1]), reverse=True)
    out = []
    for rank, (row, score) in enumerate(ordered[:top_k], start=1):
        row = {**row, "_rerank_score": round(float(score), 6), "_rerank": rank}
        out.append(row)
    return out


def warm() -> dict:
    """Load the model and report where it ended up, for startup checks."""
    if not available():
        return {"available": False, "reason": "FRUS_RERANKER_MODEL_PATH unset"}
    m = get_model()
    if m is None:
        return {"available": False, "reason": "model failed to load; using RRF order"}
    return {
        "available": True,
        "device": m["device"],
        "max_length": get_settings().reranker_max_length,
    }
