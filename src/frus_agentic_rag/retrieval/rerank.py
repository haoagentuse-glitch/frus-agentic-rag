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

from typing import Any

from frus_agentic_rag.config import get_settings

_MODEL: Any = None
_DISABLED = False


def available() -> bool:
    return bool(get_settings().reranker_model_path) and not _DISABLED


def get_model(device: str | None = None) -> Any:
    """Load the cross-encoder, or disable reranking for the rest of the process.

    Reranking is an optional stage over candidates the cheap arms already
    produced. A missing or half-downloaded model must degrade to RRF order, not
    take retrieval down with it.
    """
    global _MODEL, _DISABLED
    settings = get_settings()
    device = device or settings.reranker_device
    if _MODEL is None or getattr(_MODEL, "_frus_device", None) != device:
        import torch
        from sentence_transformers import CrossEncoder

        # sentence-transformers 3.x names this `automodel_args` on CrossEncoder,
        # not `model_kwargs` as on SentenceTransformer.
        kwargs: dict[str, Any] = {"local_files_only": True}
        if device == "cuda" and settings.embed_fp16:
            kwargs["automodel_args"] = {"torch_dtype": torch.float16}
        try:
            model = CrossEncoder(
                settings.reranker_model_path,
                device=device,
                max_length=settings.reranker_max_length,
                **kwargs,
            )
        except Exception as exc:
            _DISABLED = True
            print(
                f"[rerank] disabled, falling back to RRF order: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
        model._frus_device = device  # type: ignore[attr-defined]
        _MODEL = model
    return _MODEL


def rerank(query: str, rows: list[dict], top_k: int, text_key: str = "text") -> list[dict]:
    """Reorder candidates by cross-encoder relevance, truncating to top_k.

    Returns the input untouched when the model is unavailable or the candidate
    list is already short enough to be worth no reordering — the cost is a GPU
    forward pass per candidate, so it only pays where the ordering matters.
    """
    settings = get_settings()
    if not available() or len(rows) <= 1 or top_k >= len(rows):
        return rows[:top_k]

    model = get_model()
    if model is None:
        return rows[:top_k]
    pairs = [(query, r.get(text_key, "")[: settings.reranker_max_chars]) for r in rows]
    scores = model.predict(
        pairs,
        batch_size=settings.reranker_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
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
        "device": getattr(m, "_frus_device", "?"),
        "max_length": get_settings().reranker_max_length,
    }
