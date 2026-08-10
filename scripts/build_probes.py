"""Build the probe case files described in docs/EVAL_DESIGN.md.

Ground truth here is constructed, not judged. A question built from a document's
own title and date has an answer that is a lookup; a question built from a
footnote's cross-reference has an answer a State Department historian wrote.
Neither is a machine's guess about which document is relevant, which is what the
current gold set is and why its error rate is unknown.

    python scripts/build_probes.py

Writes eval/probes/*.jsonl. Idempotent: the same corpus produces the same cases,
so a probe result is comparable across runs.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

OUT = Path("eval/probes")
SEED = 20260810


def _table():
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect

    return connect().open_table(CHUNKS_TABLE)


def _docs(limit: int = 40000) -> list[dict]:
    """One row per document: its first chunk carries head, date and number."""
    rows = (
        _table()
        .search()
        .where("subtype = 'historical-document' AND ordinal = 0")
        .select(["chunk_id", "volume_id", "document_id", "doc_number", "head", "date_from", "text"])
        .limit(limit)
        .to_list()
    )
    return [r for r in rows if r.get("head") and r.get("date_from")]


def _distinctive(docs: list[dict]) -> list[dict]:
    """Documents whose title appears once in the corpus.

    FRUS reuses titles heavily — "The Secretary of State to the Consulate
    General at Batavia" covers 216 chunks — so a title-only question has no
    single right answer. Requiring uniqueness is what makes the ground truth
    constructive rather than a guess.
    """
    by_head: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        by_head[d["head"].strip()].append(d)
    return [v[0] for v in by_head.values() if len(v) == 1 and 40 < len(v[0]["head"]) < 160]


# --- P4 precise lookup ------------------------------------------------------


def build_p4(docs: list[dict], rng: random.Random, n: int = 10) -> list[dict]:
    """Title + date questions. The gold document is a lookup, not a judgement."""
    pool = [d for d in _distinctive(docs) if len(d["text"]) > 800]
    rng.shuffle(pool)
    out = []
    for d in pool[:n]:
        out.append(
            {
                "probe": "P4",
                "case_id": f"p4-{len(out):02d}",
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "question_en": (
                    f"Which FRUS document is titled '{d['head']}' and dated {d['date_from']}?"
                ),
                "question_zh": (
                    f"FRUS 中標題為「{d['head']}」、日期為 {d['date_from']} 的是哪一份文件？"
                ),
                "ground_truth": "constructive: unique title + date from TEI metadata",
            }
        )
    return out


# --- P5 cross-language ------------------------------------------------------

# Countries with a settled Chinese rendering that appear in FRUS heads. A
# question needs an anchor to identify one document, and in Chinese it cannot be
# the English title — so it is a country plus an exact date, both of which
# survive translation without introducing a guess.
_COUNTRIES = [
    ("Mexico", "墨西哥"),
    ("China", "中國"),
    ("Japan", "日本"),
    ("France", "法國"),
    ("Germany", "德國"),
    ("Russia", "俄國"),
    ("Cuba", "古巴"),
    ("Spain", "西班牙"),
    ("Italy", "義大利"),
    ("Turkey", "土耳其"),
    ("Korea", "韓國"),
    ("Brazil", "巴西"),
]


def build_p5(docs: list[dict], rng: random.Random, n: int = 6) -> list[dict]:
    """Chinese questions with no Latin characters that still identify one document.

    Every question in the current gold set embeds the English title, so it
    measures lexical matching and never the cross-language path. Being merely
    anchorless is not enough either: "a 1901 ambassador document" matches
    hundreds, so a failure would say nothing about translation. The anchor is a
    country plus an exact date, unique in the corpus, and carried in Chinese.

    Both properties are asserted, not assumed — a case with a Latin character in
    its Chinese question, or with more than one document at that country-date,
    is dropped.
    """
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in docs:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.get("date_from") or ""):
            continue
        for en, _zh in _COUNTRIES:
            if re.search(rf"\b{en}\b", d["head"]):
                by_key[(en, d["date_from"])].append(d)
    unique = {k: v[0] for k, v in by_key.items() if len(v) == 1}

    out: list[dict] = []
    used: set[str] = set()
    for (en, date), d in sorted(unique.items(), key=lambda kv: kv[0]):
        if en in used or len(d["text"]) < 800:
            continue
        zh_country = dict(_COUNTRIES)[en]
        y, mth, day = date.split("-")
        zh = f"{y} 年 {int(mth)} 月 {int(day)} 日關於{zh_country}的那份外交文件說了什麼？"
        if re.search(r"[A-Za-z]", zh):
            continue
        used.add(en)
        out.append(
            {
                "probe": "P5",
                "case_id": f"p5-{len(out):02d}",
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "anchor": {"country": en, "date": date},
                "question_en": f"What does the {date} diplomatic document on {en} say?",
                "question_zh": zh,
                "ground_truth": (
                    "constructive: unique country+date; zh question has no Latin anchors"
                ),
            }
        )
        if len(out) >= n:
            break
    return out


# --- P9 timeline ------------------------------------------------------------


def build_p9(docs: list[dict], rng: random.Random, n: int = 5) -> list[dict]:
    """Date-bounded questions whose answer set comes from the date index."""
    by_vol: dict[str, list[dict]] = defaultdict(list)
    for d in docs:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["date_from"] or ""):
            by_vol[d["volume_id"]].append(d)
    vols = [v for v, ds in by_vol.items() if len(ds) >= 12]
    rng.shuffle(vols)
    out = []
    for vol in vols[:n]:
        ds = sorted(by_vol[vol], key=lambda x: x["date_from"])
        lo, hi = ds[2]["date_from"], ds[7]["date_from"]
        inrange = [f"{d['volume_id']}:{d['document_id']}" for d in ds if lo <= d["date_from"] <= hi]
        out.append(
            {
                "probe": "P9",
                "case_id": f"p9-{len(out):02d}",
                "volume_id": vol,
                "date_from": lo,
                "date_to": hi,
                "gold_documents": inrange,
                "question_en": (
                    f"List the FRUS documents in volume {vol} between {lo} and {hi}, in order."
                ),
                "question_zh": f"依時間順序列出 {vol} 卷中 {lo} 到 {hi} 之間的 FRUS 文件。",
                "ground_truth": "constructive: date range filter over the index",
            }
        )
    return out


# --- P7 abstention ----------------------------------------------------------


def build_p7(docs: list[dict], rng: random.Random) -> list[dict]:
    """Two constructive kinds: unpublished volumes, and altered premises."""
    from frus_agentic_rag.corpus.manifest import load_manifest

    out: list[dict] = []
    planned = [
        r
        for r in load_manifest().to_pylist()
        if r["status"] != "published" and r.get("title_volume")
    ]
    rng.shuffle(planned)
    for r in planned[:6]:
        out.append(
            {
                "probe": "P7",
                "kind": "unpublished",
                "case_id": f"p7u-{len(out):02d}",
                "gold_documents": [],
                "must_abstain": True,
                "question_en": f"What does FRUS record about {r['title_volume']}?",
                "question_zh": f"FRUS 對於「{r['title_volume']}」記載了什麼？",
                "ground_truth": f"constructive: volume {r['volume_id']} is status=planned",
            }
        )

    # Altered premise: the document exists, the date in the question does not.
    pool = [d for d in _distinctive(docs) if len(d["text"]) > 800]
    rng.shuffle(pool)
    for d in pool[:4]:
        real = d["date_from"]
        wrong = f"{int(real[:4]) + 7}{real[4:]}"
        out.append(
            {
                "probe": "P7",
                "kind": "false_premise",
                "case_id": f"p7f-{len([x for x in out if x['kind'] == 'false_premise']):02d}",
                "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
                "must_abstain": False,
                "must_correct": {"claimed_date": wrong, "true_date": real},
                "question_en": (
                    f"The FRUS document '{d['head']}' is dated {wrong}. What does it say?"
                ),
                "question_zh": f"FRUS 文件「{d['head']}」的日期是 {wrong}，它說了什麼？",
                "ground_truth": "constructive: the stated date is wrong by seven years",
            }
        )
    return out


# --- P1 grounding, P2 order, P3 volume, P6 multihop --------------------------


def build_p1(docs: list[dict], rng: random.Random, n: int = 6) -> list[dict]:
    """Reuses P4's constructive lookups: ask, with the answer withheld."""
    return [
        {
            **c,
            "probe": "P1",
            "case_id": f"p1-{i:02d}",
            "must_abstain": True,
            "exclude_documents": c["gold_documents"],
            "ground_truth": "constructive: the only document that answers this is removed",
        }
        for i, c in enumerate(build_p4(docs, rng, n))
    ]


def build_p2(docs: list[dict], rng: random.Random, n: int = 8) -> list[dict]:
    """Documents with several named people, so who-said-what can be cross-wired."""
    pool = []
    for d in _distinctive(docs):
        names = set(
            re.findall(
                r"\b(?:Mr|Dr|Sir|President|Secretary|Ambassador)\.? [A-Z][a-z]{3,}", d["text"]
            )
        )
        if len(names) >= 3 and len(d["text"]) > 1200:
            pool.append((d, sorted(names)))
    rng.shuffle(pool)
    return [
        {
            "probe": "P2",
            "case_id": f"p2-{i:02d}",
            "gold_documents": [f"{d['volume_id']}:{d['document_id']}"],
            "actors": names[:6],
            "question_en": "Who said what in this exchange? Attribute each statement.",
            "question_zh": "這次往來中誰說了什麼？請逐項指出發言者。",
            "ground_truth": "none needed: the two orderings are compared against each other",
        }
        for i, (d, names) in enumerate(pool[:n])
    ]


def build_p3(docs: list[dict], rng: random.Random, n: int = 5) -> list[dict]:
    """One question per case; the runner varies how much evidence it is given."""
    return [
        {**c, "probe": "P3", "case_id": f"p3-{i:02d}", "levels": [2, 4, 8, 16]}
        for i, c in enumerate(build_p4(docs, rng, n))
    ]


def _volumes_with_document_refs(raw: Path, want: int) -> list[Path]:
    """Volumes whose footnotes cross-reference other documents by id.

    Only later volumes do. Measured: frus1969-76v39 carries 5,995 such refs and
    frus1949v07p1 carries none, because the 19th-century volumes reference page
    anchors instead. Scanning a random sample of the 694 files therefore found
    nothing, which is what the empty-family guard caught.
    """
    out = []
    for f in sorted(raw.glob("*.xml")):
        if "Index" in f.stem:
            continue
        try:
            # Only the head of the file: enough to tell whether this volume uses
            # document refs at all, and 694 whole volumes is 2.8GB of reading.
            with f.open("rb") as fh:
                head = fh.read(400_000).decode("utf-8", errors="ignore")
        except OSError:
            continue
        if 'target="#d' in head:
            out.append(f)
    # Every era that has them, not the alphabetical tail: taking the last twelve
    # filenames put all eight multi-hop cases in the 1981-88 volumes.
    return out[:want] if len(out) <= want else out


def build_p6(rng: random.Random, n: int = 8) -> list[dict]:
    """Multi-hop pairs taken from footnote cross-references.

    A footnote pointing at another document is a relevance link between two
    documents, written by the volume's editors. The current gold set instead
    pairs documents sharing a proper noun, which is co-occurrence rather than a
    relation anyone asserted.

    Parsed with lxml, not a regex: TEI attributes here span lines and vary in
    order, and a pattern assuming `xml:id` precedes `type` matched nothing.
    """
    from lxml import etree

    from frus_agentic_rag.config import TEI_NS, XML_NS, get_settings

    files = _volumes_with_document_refs(get_settings().raw_dir, want=12)
    rng.shuffle(files)
    out: list[dict] = []
    for f in files:
        try:
            tree = etree.parse(str(f))
        except etree.XMLSyntaxError:
            continue
        vol = f.stem
        ids = {
            d.get(f"{{{XML_NS}}}id")
            for d in tree.iter(f"{{{TEI_NS}}}div")
            if d.get("type") == "document"
        }
        for div in tree.iter(f"{{{TEI_NS}}}div"):
            if div.get("type") != "document" or div.get("subtype") != "historical-document":
                continue
            src = div.get(f"{{{XML_NS}}}id")
            head_el = div.find(f"{{{TEI_NS}}}head")
            head = " ".join(head_el.itertext() if head_el is not None else []).strip()
            if not src or not head or len(head) < 30:
                continue
            targets = [
                (r.get("target") or "")[1:]
                for r in div.iter(f"{{{TEI_NS}}}ref")
                if (r.get("target") or "").startswith("#d")
            ]
            targets = [t for t in targets if t != src and t in ids]
            if not targets:
                continue
            out.append(
                {
                    "probe": "P6",
                    "case_id": f"p6-{len(out):02d}",
                    "gold_documents": [f"{vol}:{src}", f"{vol}:{targets[0]}"],
                    "question_en": (
                        f"In FRUS volume {vol}, what does the document titled "
                        f"'{head[:120]}' say, and what does the document it "
                        f"cross-references add?"
                    ),
                    "question_zh": (
                        f"在 FRUS {vol} 卷中，標題為「{head[:120]}」的文件說了什麼？"
                        f"它交叉引用的那份文件又補充了什麼？"
                    ),
                    "ground_truth": "expert: a footnote cross-reference by the volume editors",
                }
            )
            break
        if len(out) >= n:
            break
    return out


# --- P8 routing -------------------------------------------------------------

# Hand-written, because the label is the question's intent and no corpus lookup
# defines it. Three phrasings per route so consistency is measurable separately
# from accuracy.
P8 = [
    (
        "lookup",
        [
            "What does FRUS document frus1949v07p1 d376 say?",
            "Show me document 376 in frus1949v07p1.",
            "Quote FRUS frus1949v07p1, document 376.",
        ],
    ),
    (
        "status",
        [
            "Has the FRUS volume for 1977-1980 Iran been published?",
            "Is there a published FRUS volume covering Iran in 1977?",
            "Which FRUS volumes on Iran are still planned rather than published?",
        ],
    ),
    (
        "timeline",
        [
            "How did US policy on Berlin develop between 1948 and 1949?",
            "Give the sequence of events on Berlin from 1948 to 1949.",
            "In what order did the Berlin decisions of 1948-1949 happen?",
        ],
    ),
    (
        "complex",
        [
            "Compare US policy towards Cuba before and after 1961.",
            "What is the difference between the 1960 and 1962 US positions on Cuba?",
            "Why did US policy on Cuba change after 1961, and what caused it?",
        ],
    ),
    (
        "simple",
        [
            "Who was the US ambassador to Japan in 1954?",
            "Name the US ambassador in Tokyo in 1954.",
            "In 1954, which American held the ambassadorship to Japan?",
        ],
    ),
]


def build_p8() -> list[dict]:
    return [
        {"probe": "P8", "case_id": f"p8-{r}-{i}", "expected_route": r, "question_en": q}
        for r, qs in P8
        for i, q in enumerate(qs)
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs = _docs()
    print(f"documents with head+date: {len(docs)}, distinctive titles: {len(_distinctive(docs))}")

    families = {
        "p1_grounding": build_p1(docs, random.Random(SEED)),
        "p2_order": build_p2(docs, random.Random(SEED + 1)),
        "p3_volume": build_p3(docs, random.Random(SEED + 2)),
        "p4_precise": build_p4(docs, random.Random(SEED)),
        "p5_crosslang": build_p5(docs, random.Random(SEED + 3)),
        "p6_multihop": build_p6(random.Random(SEED + 4)),
        "p7_abstain": build_p7(docs, random.Random(SEED + 5)),
        "p8_route": build_p8(),
        "p9_timeline": build_p9(docs, random.Random(SEED + 6)),
    }
    for name, cases in families.items():
        path = OUT / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases))
        print(f"  {name:16s} {len(cases):3d} cases -> {path}")
    total = sum(len(c) for c in families.values())
    print(f"total {total} cases")
    missing = [n for n, c in families.items() if not c]
    if missing:
        raise SystemExit(
            f"EMPTY FAMILIES: {missing} — a probe with no cases silently measures nothing"
        )


main()
