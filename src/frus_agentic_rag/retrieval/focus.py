"""Sentence-level relevance filtering with the cross-encoder.

The grader decides which chunks survive. Inside a surviving chunk the passage
that answers the question is usually one or two sentences of a 512-token block
of diplomatic prose, and the rest is protocol, salutations and unrelated agenda
items the model has to read past. Selecting chunks cannot fix that; it is a
different granularity.

This runs the same cross-encoder that already ranks the chunks, at sentence
granularity, and rewrites `text_focus` to the sentences that scored. Nothing is
deleted: `text` keeps the full passage, the chunk id is untouched, and the
citation gate still validates against whole chunks. Only what the prompts render
changes.

Two things it deliberately does not do. It never drops a chunk — a chunk whose
every sentence scores badly still reaches the grader, because removing it here
would duplicate a decision that is made, and measured, elsewhere. And it carries
the neighbours of every sentence it keeps: FRUS prose is dense with anaphora
("he replied that the proposal was acceptable"), and a sentence lifted out of
its neighbourhood loses the referent that made it evidence.
"""

from __future__ import annotations

import re

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import Evidence

# Sentence boundary: terminal punctuation, optional closing quote/bracket, then
# whitespace. Diplomatic cables are full of "Mr." and "No. 421", so a bare
# `[.!?]\s` split shreds them; requiring the next character to open a new
# sentence costs a few missed boundaries and avoids that.
_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+(?=[A-Z"\'(\[])')
_PARAGRAPH = re.compile(r"\n{2,}")

# Titles are what actually breaks here, because the word after one is a name and
# so is capitalised: "Mr. Nolting" satisfies the boundary pattern exactly. The
# same goes for the initials FRUS uses constantly ("Frederick E. Nolting").
_ABBREV = frozenset(
    {
        # titles
        "mr", "mrs", "ms", "dr", "prof", "rev", "hon",
        "col", "gen", "lt", "capt", "adm", "amb",
        # name particles
        "st", "jr", "sr",
        # citation furniture, everywhere in FRUS editorial notes
        "no", "vol", "pp", "ch", "art", "sec", "asst", "dept", "govt",
    }
)  # fmt: skip
_TRAILING = re.compile(r"([A-Za-z]+)\.$")


def _ends_in_abbreviation(fragment: str) -> bool:
    m = _TRAILING.search(fragment.rstrip("\"')]"))
    if not m:
        return False
    word = m.group(1)
    return word.lower() in _ABBREV or (len(word) == 1 and word.isupper())


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for para in _PARAGRAPH.split(text):
        para = para.strip()
        if not para:
            continue
        pieces = [s.strip() for s in _BOUNDARY.split(para) if s.strip()]
        for piece in pieces:
            if parts and _ends_in_abbreviation(parts[-1]):
                parts[-1] = f"{parts[-1]} {piece}"
            else:
                parts.append(piece)
    return parts


def _assemble(sentences: list[str], keep: set[int]) -> str:
    """Rebuild in reading order, marking where sentences were skipped."""
    out: list[str] = []
    prev = -1
    for i in sorted(keep):
        if prev >= 0 and i > prev + 1:
            out.append("[…]")
        out.append(sentences[i])
        prev = i
    return " ".join(out)


def focus_active() -> tuple[bool, str]:
    """Whether sentence filtering will actually run, and why not when it won't.

    The CPU check is not caution, it is arithmetic. Reranking scores one pair
    per candidate — 50, about a second on GPU and a minute on CPU. Sentence
    filtering scores one pair per SENTENCE, which on this corpus is six to
    fifteen times more, so the stage that costs six seconds with a GPU costs six
    minutes without one, per retrieval round. Running it anyway would turn a
    missing `--gpus` flag into a sweep that never finishes, which is how this
    would be discovered. One caller, so the fingerprint and the node agree on
    what happened.
    """
    from frus_agentic_rag.retrieval import rerank as rr

    settings = get_settings()
    if not settings.focus_sentences:
        return False, "disabled"
    if not rr.available():
        return False, "reranker unavailable"
    if rr.resolve_device(settings.reranker_device) == "cpu":
        return False, "cross-encoder on cpu; too slow at sentence granularity"
    return True, ""


def focus_evidence(question: str, evidence: list[Evidence]) -> tuple[list[Evidence], dict]:
    """Set `text_focus` on each passage to its question-relevant sentences.

    Every sentence of every passage is scored in one batch rather than one call
    per passage: the cost is dominated by the number of pairs, and batching them
    keeps the GPU busy instead of paying per-passage launch overhead.
    """
    settings = get_settings()
    stats: dict = {"passages": len(evidence)}
    active, reason = focus_active()
    if not active or not evidence:
        stats["skipped"] = reason or "no evidence"
        return evidence, stats

    from frus_agentic_rag.retrieval import rerank as rr

    # Which passages are worth splitting, and their sentences.
    plans: list[tuple[int, list[str]]] = []
    pairs: list[tuple[str, str]] = []
    for idx, e in enumerate(evidence):
        if len(e.text) < settings.focus_min_chars:
            continue
        sents = split_sentences(e.text)
        if len(sents) <= settings.focus_max_sentences:
            continue
        if len(pairs) + len(sents) > settings.focus_max_pairs:
            break
        plans.append((idx, sents))
        pairs.extend((question, s[: settings.reranker_max_chars]) for s in sents)

    if not pairs:
        stats["skipped"] = "nothing long enough to trim"
        return evidence, stats

    scores = rr.score_pairs(pairs)
    if scores is None:
        stats["skipped"] = "scoring unavailable"
        return evidence, stats

    out = list(evidence)
    cursor = 0
    before = after = 0
    for idx, sents in plans:
        chunk_scores = scores[cursor : cursor + len(sents)]
        cursor += len(sents)
        ranked = sorted(range(len(sents)), key=lambda i: chunk_scores[i], reverse=True)
        keep: set[int] = set()
        for i in ranked[: settings.focus_max_sentences]:
            lo = max(0, i - settings.focus_context_sentences)
            hi = min(len(sents), i + settings.focus_context_sentences + 1)
            keep.update(range(lo, hi))
        text_focus = _assemble(sents, keep)
        before += len(evidence[idx].text)
        after += len(text_focus)
        out[idx] = evidence[idx].model_copy(update={"text_focus": text_focus})

    stats.update(
        {
            "trimmed": len(plans),
            "sentences_scored": len(pairs),
            "chars_before": before,
            "chars_after": after,
            "kept_ratio": round(after / before, 3) if before else None,
        }
    )
    return out, stats
