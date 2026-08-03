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


def _keywords(text: str, n: int = 6) -> list[str]:
    """Distinctive capitalised terms: rare enough that a paraphrase can miss them."""
    words = [
        w
        for w in re.findall(r"[A-Z][a-z]{3,}", text)
        if not _STOP.match(w) and w not in _BOILERPLATE
    ]
    seen: dict[str, None] = {}
    for w in words:
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
    for row in usable:
        if len([c for c in cases if c["kind"] == "lookup"]) >= n_lookup:
            break
        docs = _load_docs(row["volume_id"])
        if not docs:
            continue
        d = rng.choice(docs)
        topic = _topic(d["head"])
        if len(topic) < 12:
            continue
        cases.append(
            {
                "case_id": f"lookup-{len(cases):02d}",
                "kind": "lookup",
                "route": "lookup",
                "question_en": f"What does the FRUS document titled '{topic}' say?",
                "question_zh": f"FRUS 中標題為「{topic}」的文件內容說了什麼？",
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "gold_volume_ids": [d["volume_id"]],
                "answerable": True,
                "notes": f"date {d['date_from']}",
            }
        )

    # 2. Multi-document comparison / timeline within one volume, 2-4 golds.
    for row in usable:
        if len([c for c in cases if c["kind"] == "multihop"]) >= n_multihop:
            break
        docs = _load_docs(row["volume_id"])
        if len(docs) < 8:
            continue
        picked = rng.sample(docs, k=rng.choice([2, 3, 4]))
        picked.sort(key=lambda d: d["date_from"] or "")
        subject = row["title_volume"] or row["title_complete"]
        d0, d1 = picked[0], picked[-1]
        cases.append(
            {
                "case_id": f"multihop-{len(cases):02d}",
                "kind": "multihop",
                "route": "complex",
                "question_en": (
                    f"On the subject of {subject}, how did the US position develop between "
                    f"{d0['date_from']} and {d1['date_from']}? Cite the documents."
                ),
                "question_zh": (
                    f"關於{subject}，美方立場在 {d0['date_from']} 到 {d1['date_from']} "
                    "之間如何演變？請引用文件。"
                ),
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}" for d in picked],
                "gold_volume_ids": [row["volume_id"]],
                "answerable": True,
                "notes": f"{len(picked)} gold documents",
            }
        )

    # 3. Vocabulary-mismatch cases: ask with a paraphrase the archive does not use,
    #    so a first retrieval is expected to miss and a correction to recover.
    for row in usable:
        if len([c for c in cases if c["kind"] == "correction"]) >= n_correction:
            break
        docs = _load_docs(row["volume_id"])
        if not docs:
            continue
        d = rng.choice(docs)
        kws = _keywords(d["text"])
        if len(kws) < 3:
            continue
        cases.append(
            {
                "case_id": f"correction-{len(cases):02d}",
                "kind": "correction",
                "route": "simple",
                "question_en": (
                    f"What was decided in the discussions involving {kws[0]} and {kws[1]}? "
                    "Use everyday wording, not the archival phrasing."
                ),
                "question_zh": f"關於 {kws[0]} 與 {kws[1]} 的討論，最後決定了什麼？",
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "gold_volume_ids": [row["volume_id"]],
                "answerable": True,
                "notes": "paraphrased wording; expects a corrective retrieval",
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
