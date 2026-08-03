"""Token-window chunking against the real BGE-M3 tokenizer.

Windows are sliced by character offset rather than decoded from token ids, so
the stored text is byte-identical to the source passage. Chunks never span two
documents: the document is the citable unit, and a chunk that straddled two of
them could not be attributed.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.retrieval.models import Chunk, FrusDocument

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerFast


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerFast:
    from transformers import AutoTokenizer

    settings = get_settings()
    tok = AutoTokenizer.from_pretrained(settings.bge_model_path, local_files_only=True)
    if not tok.is_fast:  # offset mapping is what makes the slicing exact
        raise RuntimeError("BGE-M3 tokenizer must be a fast tokenizer")
    return tok


def chunk_document(doc: FrusDocument) -> list[Chunk]:
    settings = get_settings()
    max_tokens = settings.max_chunk_tokens
    overlap = settings.chunk_overlap_tokens
    stride = max_tokens - overlap
    if stride <= 0:
        raise ValueError("chunk_overlap_tokens must be smaller than max_chunk_tokens")

    tok = get_tokenizer()
    # The head is prepended to every window so an isolated chunk still carries
    # its document's title; it is not part of the offset arithmetic.
    prefix = f"{doc.head}\n\n" if doc.head else ""

    enc = tok(doc.text, add_special_tokens=False, return_offsets_mapping=True)
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    n = len(offsets)

    if n == 0:
        return []

    windows: list[tuple[int, int, int]] = []  # (start_char, end_char, n_tokens)
    start = 0
    while start < n:
        end = min(start + max_tokens, n)
        c0 = offsets[start][0]
        c1 = offsets[end - 1][1]
        windows.append((c0, c1, end - start))
        if end >= n:
            break
        start += stride

    total = len(windows)
    chunks: list[Chunk] = []
    for i, (c0, c1, ntok) in enumerate(windows):
        body = doc.text[c0:c1].strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.volume_id}:{doc.document_id}:{i}",
                volume_id=doc.volume_id,
                document_id=doc.document_id,
                ordinal=i,
                n_chunks_in_doc=total,
                subtype=doc.subtype,
                head=doc.head,
                doc_number=doc.doc_number,
                date_from=doc.date_from,
                date_to=doc.date_to,
                persons=doc.persons,
                text=prefix + body,
                n_tokens=ntok,
            )
        )
    return chunks
