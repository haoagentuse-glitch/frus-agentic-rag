"""Prompts. Short and imperative — a 4B model degrades fast on long instructions."""

from __future__ import annotations

from frus_agentic_rag.models import Evidence

PLANNER_SYSTEM = """You plan retrieval over Foreign Relations of the United States (FRUS), \
a closed corpus of declassified US diplomatic documents. You never answer from memory.

simple is the default. Choose another route only if its test is clearly met;
when two seem to fit, prefer the one listed later.

- lookup: the question POINTS AT a document it can already name — it quotes a
  title, gives a document number, or gives a volume id. Asking about a person,
  a place or an event is NOT lookup, however specific: "Who was the ambassador
  to Japan in 1954?" names no document and is simple.
- status: asks whether a volume or period is published, planned, or exists at
  all. Anything about what FRUS covers rather than what a document says.
- timeline: asks for a sequence or an order across a span of dates. Two years
  mentioned in a comparison do not make it a timeline.
- complex: needs a comparison, a cause, or two or more distinct facts. A
  question joining two asks with "and" is complex, however short it is.
- simple: one fact, one retrieval. This is the answer unless another test above
  is clearly met.

Rules:
- Any factual or historical question needs retrieval. Only greetings and questions \
about how to use this system may set needs_retrieval=false.
- Write subqueries in ENGLISH; the corpus is English. Write at most 3.
- simple/lookup/status: exactly 1 subquery. timeline: 1-2. complex: 2-3.
- Put date constraints in date_from/date_to as yyyy-mm-dd, not in the query text.
- answer_language is the language of the user's question: zh-TW or en.
- required_evidence lists what must be shown for an answer to stand."""


def planner_user(question: str) -> str:
    return f"User question:\n{question}\n\nReturn the plan as JSON."


GRADER_SYSTEM = """You judge whether retrieved FRUS passages cover what a question needs.

You are NOT a historian here. Do not add facts. Do not guess. Judge coverage only.

For each hop:
- supported: the passages state the needed fact directly.
- partial: related but the specific fact is missing.
- unsupported: nothing relevant.

rejected_evidence_ids: take the EVIDENCE passages one at a time and list the ids \
of the ones that carry nothing towards this hop — a different country, a \
different period, or a different subject that merely shares a name or a phrase \
with the question. The test for each passage is whether deleting it would change \
the answer; if it would not, reject it. Keep a passage that carries even part of \
what the hop needs, and keep it when you cannot tell. Copy ids verbatim from the \
EVIDENCE shown; never invent one. An empty list is permitted, but it asserts \
that you checked every passage and that all of them contribute.

If a hop is partial or unsupported, give corrective_query: ONE short English \
retrieval query, under 20 words, that would find the missing fact. Do not explain \
your reasoning anywhere in the JSON — only the fields, and keep them terse."""


# Qwen3 4B degrades on long prompts, so the grader judges a window rather than
# everything. That is only safe because omission no longer means rejection —
# see grade_evidence. Widened from 6 once the context budget was measured: the
# grader prompt sits near 3k of the 8192 window at 10.
GRADER_MAX_EVIDENCE = 10
GRADER_EVIDENCE_CHARS = 700
# The synthesis window is budgeted in DOCUMENTS, not chunks. Budgeting in chunks
# let adjacent chunks of one document take several slots each: measured over ten
# multi-hop questions, 15.8 accepted documents collapsed to 6.4 distinct ones in
# an 8-chunk window, and 8 of 24 gold documents that retrieval had already found
# never reached the model. Context is not the constraint — 8 passages fill 28%
# of the 8192 window and 16 fill 50%.
SYNTH_MAX_DOCUMENTS = 14
SYNTH_MAX_EVIDENCE = 18
SYNTH_EVIDENCE_CHARS = 950


def _fmt_evidence(evidence: list[Evidence], max_chars: int = 1200) -> str:
    lines = []
    for e in evidence:
        # text_focus is set only when sentence filtering ran; text is the fallback.
        body = (e.text_focus or e.text)[:max_chars]
        lines.append(
            f"[id: {e.evidence_id}] ({e.volume_id} Doc.{e.doc_number} {e.date_from})\n"
            f"{e.head}\n{body}"
        )
    return "\n\n---\n\n".join(lines) if lines else "(no evidence retrieved)"


def grader_user(question: str, hops: list[str], shown: list[Evidence]) -> str:
    """`shown` is exactly what the grader is judging; the caller does the slicing.

    Slicing here hid from the caller which evidence had actually been presented,
    and evidence the grader never saw was then dropped for not being named.
    """
    hop_list = "\n".join(f"- {h}" for h in hops) or "- (single hop) answer the question"
    return (
        f"QUESTION:\n{question}\n\nHOPS TO COVER:\n{hop_list}\n\n"
        f"EVIDENCE:\n{_fmt_evidence(shown, GRADER_EVIDENCE_CHARS)}\n\n"
        "Return the grade as JSON."
    )


# The id instruction has to track citation_mode. Setting the flag to "model"
# while leaving a prompt that says "do not write identifiers" is not a test of
# the old scheme — it is a gate demanding ids from a model told not to produce
# them, and it abstained on 27 of 28 questions when run that way.
_ZH_IDS = {
    "post_hoc": (
        "- 把答案拆成若干條主張，每條是一個可被單一段落佐證的完整句子。"
        "引用由程式產生，不要自己寫編號。"
    ),
    "model": "- 每個主張都要附上使用到的 evidence id，id 必須逐字複製，不可自創。",
}
_EN_IDS = {
    "post_hoc": (
        "- Break the answer into claims, each a complete sentence one passage could support.\n"
        "  Citations are attached by code; do not write identifiers yourself."
    ),
    "model": (
        "- Every claim carries the evidence ids it rests on, copied verbatim. Never invent an id."
    ),
}

_SYNTH_SYSTEM_ZH_TMPL = """你依據 FRUS 原始文件回答，並且只能使用下面提供的 EVIDENCE。

規則：
{ids}
- EVIDENCE 沒說的就不要說。不要補充你自己知道的歷史。
- 文件互相衝突時並列陳述，不要替史家下定論。
- 用繁體中文（台灣）作答。人名、機構、文件標題後面附英文原文。
- answer_text 不要放網址；引用區塊由程式產生。
- limitations 說明證據沒有涵蓋到什麼。"""

_SYNTH_SYSTEM_EN_TMPL = """You answer from FRUS primary documents using ONLY the EVIDENCE below.

Rules:
{ids}
- If the EVIDENCE does not say it, do not say it. Do not add outside history.
- Where documents conflict, present both; do not adjudicate.
- Do not put URLs in answer_text; the citation block is generated by code.
- limitations states what the evidence does not cover."""


def synth_system(language: str) -> str:
    """The synthesis prompt for the configured citation mode.

    The templates are private because a caller that imported them directly and
    skipped the format sent a system prompt containing the literal text
    "{ids}" — which is what scripts/diagnose.py did for the whole of D2 and D3,
    dropping the first rule from every prompt in both experiments. Nothing
    failed; the runs completed and produced numbers.
    """
    from frus_agentic_rag.config import get_settings

    mode = get_settings().citation_mode
    if language == "zh-TW":
        return _SYNTH_SYSTEM_ZH_TMPL.format(ids=_ZH_IDS[mode])
    return _SYNTH_SYSTEM_EN_TMPL.format(ids=_EN_IDS[mode])


def select_synthesis_evidence(evidence: list[Evidence]) -> list[Evidence]:
    """Fill the window with distinct documents first, then depth.

    One pass takes the best chunk of each document in score order, so a
    multi-document answer can see every document it needs. A second pass spends
    whatever slots remain on further chunks of documents already included, which
    is what a single-document question wants.
    """
    by_doc: dict[str, list[Evidence]] = {}
    for e in evidence:
        by_doc.setdefault(f"{e.volume_id}:{e.document_id}", []).append(e)

    first: list[Evidence] = []
    rest: list[Evidence] = []
    for chunks in by_doc.values():
        chunks.sort(key=lambda e: e.score, reverse=True)
        first.append(chunks[0])
        rest.extend(chunks[1:])

    first.sort(key=lambda e: e.score, reverse=True)
    rest.sort(key=lambda e: e.score, reverse=True)
    selected = first[:SYNTH_MAX_DOCUMENTS]
    selected.extend(rest[: max(0, SYNTH_MAX_EVIDENCE - len(selected))])
    return selected


def _ids_hint() -> str:
    from frus_agentic_rag.config import get_settings

    if get_settings().citation_mode == "post_hoc":
        return " Leave evidence_ids empty — the cross-encoder attaches them afterwards."
    return " Every claim must carry evidence ids copied verbatim from the ids above."


def synth_user(question: str, evidence: list[Evidence]) -> str:
    body = _fmt_evidence(select_synthesis_evidence(evidence), SYNTH_EVIDENCE_CHARS)
    return (
        f"QUESTION:\n{question}\n\nEVIDENCE:\n{body}\n\n"
        "Return the answer as JSON. claims must not be empty: every sentence of "
        "answer_text has to appear as a claim." + _ids_hint()
    )


ABSTAIN_ZH = (
    "根據目前檢索到的 FRUS 文件，無法支持一個有憑據的回答。"
    "FRUS 只收錄已出版卷次的解密外交文件；"
    "若該主題屬於尚未出版的卷次，或用詞與檔案不符，就會檢索不到。"
)
ABSTAIN_EN = (
    "The retrieved FRUS documents do not support a grounded answer. "
    "FRUS covers only published volumes of declassified US diplomatic records; "
    "a topic in an unpublished volume, or phrased unlike the archival wording, will not be found."
)
