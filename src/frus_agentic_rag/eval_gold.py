"""Draft gold cases from the corpus.

These are machine-drafted and NOT historian-reviewed. Every record carries
`review_status: "unreviewed"` and `authored_by: "claude-generated"`, and the
ablation report repeats that caveat, because questions drafted from the same
corpus the retriever indexes are a self-referential benchmark: they measure
whether the pipeline can find text it was shown, not whether a historian would
call the answer correct. Treat the numbers as a regression signal, not as an
external evaluation.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.ingest.manifest import load_manifest

SEED = 20260803

# Volumes with well-known, distinctive subject matter make better multi-hop
# anchors than an arbitrary sample of 552.
_STOP = re.compile(r"^(the|a|an|of|to|and|in|on|for|by|from|memorandum|telegram|letter)$", re.I)

# Diplomatic boilerplate. A "correction" case keyed on "Secretary" or "State"
# matches half the corpus and measures nothing.
_BOILERPLATE = {
    "Secretary",
    "State",
    "Department",
    "Government",
    "Excellency",
    "Ambassador",
    "Minister",
    "Legation",
    "Embassy",
    "Consul",
    "Consulate",
    "President",
    "Washington",
    "American",
    "United",
    "States",
    "Charge",
    "Affaires",
    "Sir",
    "Note",
    "Telegram",
    "Memorandum",
    "Despatch",
    "Enclosure",
    "Reference",
    "Source",
    "Confidential",
    "Secret",
    "January",
    "February",
    "March",
    "April",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
}


def _load_docs(volume_id: str, limit: int = 4000) -> list[dict]:
    settings = get_settings()
    path = settings.chunks_dir / f"{volume_id}.parquet"
    if not path.exists():
        return []
    tbl = pq.read_table(
        path,
        columns=[
            "chunk_id",
            "volume_id",
            "document_id",
            "doc_number",
            "head",
            "date_from",
            "text",
            "ordinal",
        ],
    )
    rows = [r for r in tbl.to_pylist() if r["ordinal"] == 0 and len(r["text"]) > 400]
    return rows[:limit]


def _topic(head: str) -> str:
    """Strip the leading document number off a head to get a subject phrase."""
    return re.sub(r"^\d+\.\s*", "", head).strip()


def _norm_head(head: str) -> str:
    return _topic(head).lower()


def _rare_entity_docs(docs: list[dict], lo: int, hi: int) -> list[tuple[str, list[dict]]]:
    """Terms occurring in between `lo` and `hi` documents, with those documents.

    Rarity is what makes the term usable as a gold anchor: a term in 40 documents
    cannot define a 3-document answer set.
    """
    by_term: dict[str, list[dict]] = {}
    for d in docs:
        for term in set(_keywords(d["text"], n=25)):
            by_term.setdefault(term, []).append(d)
    return [(term, ds) for term, ds in by_term.items() if lo <= len(ds) <= hi and len(term) >= 5]


def _head_frequencies() -> dict[str, int]:
    """How many documents corpus-wide share each head, for uniqueness filtering."""
    settings = get_settings()
    counts: dict[str, int] = {}
    for path in sorted(settings.chunks_dir.glob("*.parquet")):
        tbl = pq.read_table(path, columns=["head", "ordinal"])
        for head, ordinal in zip(tbl["head"].to_pylist(), tbl["ordinal"].to_pylist(), strict=True):
            if ordinal != 0 or not head:
                continue
            key = _norm_head(head)
            counts[key] = counts.get(key, 0) + 1
    return counts


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n+")
_CAPWORD = re.compile(r"\b[A-Z][a-z]{3,}\b")


def _keywords(text: str, n: int = 6) -> list[str]:
    """Proper nouns, taken only from mid-sentence positions.

    A plain `[A-Z][a-z]{3,}` sweep also collects sentence-initial common words —
    measured anchors included "Section", "Draft", "Presently" and "Suppose",
    which retrieve nothing useful, while genuine proper nouns (Daland, Istrian)
    hit their gold documents exactly. Skipping the first token of each sentence
    is a cheap way to keep only the latter.
    """
    seen: dict[str, None] = {}
    for sentence in _SENTENCE_SPLIT.split(text):
        tokens = sentence.split()
        for tok in tokens[1:]:  # the first token is capitalised by position
            for w in _CAPWORD.findall(tok):
                if not _STOP.match(w) and w not in _BOILERPLATE:
                    seen.setdefault(w, None)
    return list(seen)[:n]


def build_gold_cases(
    n_lookup: int = 10,
    n_multihop: int = 10,
    n_correction: int = 5,
    n_unanswerable: int = 5,
    out: Path = Path("eval/gold_cases.jsonl"),
) -> dict:
    rng = random.Random(SEED)
    settings = get_settings()
    manifest = [r for r in load_manifest().to_pylist() if r["status"] == "published"]
    parsed = {p.stem for p in settings.chunks_dir.glob("*.parquet")}
    usable = [r for r in manifest if r["volume_id"] in parsed and r["n_historical_documents"] > 40]
    if not usable:
        raise FileNotFoundError("no parsed volumes; run `frus ingest` first")

    rng.shuffle(usable)
    cases: list[dict] = []

    # 1. Single-document lookup.
    #
    # FRUS heads are standard diplomatic correspondence forms: "The Secretary of
    # State to the Consulate General at Batavia" occurs on 216 chunks across the
    # corpus. A lookup case keyed on the head alone is undecidable — measured, not
    # assumed — so candidates must have a head that is rare corpus-wide, and the
    # question carries the date as a disambiguator.
    head_counts = _head_frequencies()
    for row in usable:
        if len([c for c in cases if c["kind"] == "lookup"]) >= n_lookup:
            break
        docs = _load_docs(row["volume_id"])
        candidates = [
            d
            for d in docs
            if len(_topic(d["head"])) >= 12
            and d["date_from"]
            and head_counts.get(_norm_head(d["head"]), 99) <= 2
        ]
        if not candidates:
            continue
        d = rng.choice(candidates)
        topic = _topic(d["head"])
        cases.append(
            {
                "case_id": f"lookup-{len(cases):02d}",
                "kind": "lookup",
                "route": "lookup",
                "question_en": (
                    f"What does the FRUS document titled '{topic}', dated {d['date_from']}, say?"
                ),
                "question_zh": (
                    f"FRUS 中標題為「{topic}」、日期為 {d['date_from']} 的文件說了什麼？"
                ),
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "gold_volume_ids": [d["volume_id"]],
                "answerable": True,
                "notes": (
                    f"head occurs on {head_counts.get(_norm_head(d['head']), 0)} "
                    f"documents corpus-wide; date {d['date_from']}"
                ),
            }
        )

    # 2. Multi-document questions anchored on a rare entity.
    #
    # The gold set has to BE the answer, not a random sample. An earlier version
    # picked 2-4 arbitrary documents from a volume and asked a subject question
    # about them; retrieval scored 0/30, correctly, because nothing tied those
    # documents to that question. Here the anchor term occurs in exactly 2-4
    # documents of the volume, so those documents are what a correct answer must
    # cite, and the question is genuinely multi-hop over them.
    for row in usable:
        if len([c for c in cases if c["kind"] == "multihop"]) >= n_multihop:
            break
        docs = _load_docs(row["volume_id"])
        if len(docs) < 8:
            continue
        anchored = _rare_entity_docs(docs, lo=2, hi=4)
        if not anchored:
            continue
        term, picked = rng.choice(anchored)
        picked.sort(key=lambda d: d["date_from"] or "")
        subject = row["title_volume"] or row["title_complete"]
        d0, d1 = picked[0], picked[-1]
        cases.append(
            {
                "case_id": f"multihop-{len(cases):02d}",
                "kind": "multihop",
                "route": "complex",
                "question_en": (
                    f"In the FRUS volume on {subject}, what was discussed regarding "
                    f"{term}, and how did the position develop between "
                    f"{d0['date_from']} and {d1['date_from']}? Cite every document."
                ),
                "question_zh": (
                    f"在關於{subject}的 FRUS 卷次中，針對 {term} 討論了什麼？"
                    f"立場在 {d0['date_from']} 到 {d1['date_from']} 之間如何演變？"
                    "請引用所有相關文件。"
                ),
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}" for d in picked],
                "gold_volume_ids": [row["volume_id"]],
                "answerable": True,
                "notes": (
                    f"anchor term '{term}' appears in exactly {len(picked)} documents "
                    "of this volume; those are the gold set"
                ),
            }
        )

    # 3. Vocabulary-mismatch cases: ask with a paraphrase the archive does not use,
    #    so a first retrieval is expected to miss and a correction to recover.
    for row in usable:
        if len([c for c in cases if c["kind"] == "correction"]) >= n_correction:
            break
        docs = _load_docs(row["volume_id"])
        if len(docs) < 8:
            continue
        # Anchor on a term unique to one document, then ask about it in everyday
        # wording rather than the archival phrasing, so a first retrieval that
        # keys on the paraphrase can miss and the correction has work to do.
        unique = _rare_entity_docs(docs, lo=1, hi=1)
        if not unique:
            continue
        term, (d,) = rng.choice(unique)
        cases.append(
            {
                "case_id": f"correction-{len(cases):02d}",
                "kind": "correction",
                "route": "simple",
                "question_en": (
                    f"I remember something about {term} being talked over in these "
                    "papers — what did they end up settling on?"
                ),
                "question_zh": f"我記得這批文件裡談到 {term} 的事，最後是怎麼定案的？",
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "gold_volume_ids": [row["volume_id"]],
                "answerable": True,
                "notes": (
                    f"anchor term '{term}' is unique to this document in the volume; "
                    "question is deliberately paraphrased, expects a corrective retrieval"
                ),
            }
        )

    # 4. Unanswerable: planned volumes and out-of-scope topics.
    planned = [r for r in load_manifest().to_pylist() if r["status"] == "planned"]
    unanswerable = [
        {
            "question_en": "What does FRUS say about the 2019 US-China trade negotiations?",
            "question_zh": "FRUS 對 2019 年美中貿易談判有什麼記載？",
            "notes": "outside the published date range entirely",
        },
        {
            "question_en": "Summarise the FRUS volume on the 2011 Arab Spring cables.",
            "question_zh": "請摘要 FRUS 中關於 2011 年阿拉伯之春電報的那一卷。",
            "notes": "no such volume exists",
        },
        {
            "question_en": "Which FRUS document records the minutes of the 2024 NATO summit?",
            "question_zh": "哪一份 FRUS 文件記錄了 2024 年北約峰會的會議紀錄？",
            "notes": "post-dates the corpus",
        },
        {
            "question_en": (
                f"What are the contents of the planned volume {planned[0]['volume_id']}?"
                if planned
                else "What are the contents of a volume that is only planned?"
            ),
            "question_zh": (
                f"尚未出版的 {planned[0]['volume_id']} 這一卷內容是什麼？"
                if planned
                else "尚未出版的那一卷內容是什麼？"
            ),
            "notes": "planned stub: manifest knows it, corpus has no documents",
        },
        {
            "question_en": "What is the current US ambassador to Japan's phone number?",
            "question_zh": "現任美國駐日大使的電話號碼是什麼？",
            "notes": "not a historical-document question at all",
        },
    ]
    for i, u in enumerate(unanswerable[:n_unanswerable]):
        cases.append(
            {
                "case_id": f"unanswerable-{i:02d}",
                "kind": "unanswerable",
                "route": "simple",
                "question_en": u["question_en"],
                "question_zh": u["question_zh"],
                "gold_documents": [],
                "gold_volume_ids": [],
                "answerable": False,
                "notes": u["notes"],
            }
        )

    for c in cases:
        c["authored_by"] = "claude-generated"
        c["review_status"] = "unreviewed"

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for c in cases:
        counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    return {
        "path": str(out),
        "total": len(cases),
        "by_kind": counts,
        "review_status": "unreviewed — machine-drafted from the corpus, needs a human pass",
        "seed": SEED,
    }


def load_cases(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
