"""Against the real LanceDB index. No Ollama, no GPU — retrieval only.

Skipped automatically when the index is absent, so a fresh checkout still runs
`pytest` green before `frus ingest` has been run.
"""

from __future__ import annotations

import pytest

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect
from frus_agentic_rag.models import SearchFilters
from frus_agentic_rag.retrieval.hybrid import build_where, hybrid_search_sync

pytestmark = pytest.mark.integration


def _has_index() -> bool:
    try:
        return CHUNKS_TABLE in connect().table_names()
    except Exception:
        return False


requires_index = pytest.mark.skipif(not _has_index(), reason="no LanceDB index built")


@pytest.fixture(scope="module")
def table():
    return connect().open_table(CHUNKS_TABLE)


@requires_index
def test_index_matches_the_parsed_corpus(table):
    settings = get_settings()
    import pyarrow.parquet as pq

    parquet_rows = sum(pq.read_metadata(f).num_rows for f in settings.chunks_dir.glob("*.parquet"))
    assert table.count_rows() == parquet_rows


@requires_index
def test_only_published_volumes_are_indexed(table):
    from frus_agentic_rag.corpus.manifest import load_manifest

    published = {r["volume_id"] for r in load_manifest().to_pylist() if r["status"] == "published"}
    indexed = {r["volume_id"] for r in table.search().select(["volume_id"]).limit(20000).to_list()}
    assert indexed <= published


@requires_index
def test_no_editorial_notes_in_the_evidence_corpus(table):
    assert table.count_rows("subtype != 'historical-document'") == 0


@requires_index
def test_bm25_finds_a_known_document():
    hits = hybrid_search_sync(
        "National Security Study Memorandum 14 China policy",
        SearchFilters(volume_ids=["frus1969-76v17"], subtypes=["historical-document"]),
        10,
    )
    assert hits
    assert all(h.volume_id == "frus1969-76v17" for h in hits)
    assert any(h.rank_bm25 is not None for h in hits)


@requires_index
def test_volume_filter_is_honoured():
    hits = hybrid_search_sync(
        "treaty negotiations",
        SearchFilters(volume_ids=["frus1862"], subtypes=["historical-document"]),
        10,
    )
    assert hits
    assert {h.volume_id for h in hits} == {"frus1862"}


@requires_index
def test_date_filter_excludes_out_of_range_documents():
    hits = hybrid_search_sync(
        "correspondence",
        SearchFilters(date_from="1900-01-01", date_to="1910-12-31"),
        10,
    )
    for h in hits:
        # A document overlaps the window if it starts before the end of it.
        assert not h.date_from or h.date_from <= "1910-12-31"


@requires_index
def test_evidence_ids_round_trip_to_canonical_urls():
    hits = hybrid_search_sync("diplomatic recognition", SearchFilters(), 5)
    for h in hits:
        assert h.evidence_id.startswith(f"{h.volume_id}:{h.document_id}:")
        assert h.url == (
            f"https://history.state.gov/historicaldocuments/{h.volume_id}/{h.document_id}"
        )


def test_filter_literals_reject_injection():
    with pytest.raises(ValueError, match="unsafe filter literal"):
        build_where(SearchFilters(volume_ids=["frus1862' OR '1'='1"]))


def test_empty_filters_produce_no_where_clause():
    assert build_where(SearchFilters()) is None


@requires_index
async def test_series_status_answers_from_the_manifest():
    from frus_agentic_rag.retrieval.tools import LiveToolbox

    status = await LiveToolbox().get_series_status("frus1969-76v17")
    assert status["n_published_total"] == 552
    assert status["n_planned_total"] == 142
    assert status["matches"]
    assert status["matches"][0]["status"] == "published"


@requires_index
async def test_lookup_document_returns_one_document_in_order():
    from frus_agentic_rag.retrieval.tools import LiveToolbox

    hits = await LiveToolbox().lookup_document("frus1969-76v17", "d4")
    assert hits
    assert {h.document_id for h in hits} == {"d4"}
    assert [h.evidence_id for h in hits] == sorted(h.evidence_id for h in hits)


@requires_index
def test_dense_index_retrieves_the_chunk_it_encoded():
    """Guards the whole dense arm: stored vectors must match freshly encoded ones."""
    import numpy as np

    from frus_agentic_rag.corpus.embed import encode

    table = connect().open_table(CHUNKS_TABLE)
    row = (
        table.search()
        .where("volume_id = 'frus1969-76v17'")
        .select(["chunk_id", "text", "vector"])
        .limit(3)
        .to_list()
    )[-1]

    fresh = encode([row["text"]])[0]
    stored = np.array(row["vector"], dtype=np.float32)
    assert float(stored @ fresh) > 0.99

    settings = get_settings()
    hits = (
        table.search(fresh, vector_column_name="vector")
        .nprobes(settings.ann_nprobes)
        .refine_factor(settings.ann_refine_factor)
        .select(["chunk_id"])
        .limit(5)
        .to_list()
    )
    assert row["chunk_id"] in [h["chunk_id"] for h in hits]


@requires_index
@pytest.mark.parametrize(
    "question",
    [
        "美國承認共產中國的討論",
        "古巴飛彈危機期間的外交電報",
        "第二次世界大戰後對日本的佔領政策",
    ],
)
def test_chinese_queries_are_served_by_the_dense_arm_alone(question):
    """The reason the dense arm exists.

    These carry no English anchor, so BM25 over an English corpus returns
    nothing at all. If dense ever stops answering them, the system has silently
    lost its bilingual capability while every lexical metric stays green.
    """
    from frus_agentic_rag.retrieval.hybrid import bm25_search, dense_search

    filters = SearchFilters(subtypes=["historical-document"])
    assert bm25_search(question, filters, 10) == []
    assert len(dense_search(question, filters, 10)) == 10


@requires_index
def test_fusion_is_never_worse_than_its_lexical_arm_on_an_english_query():
    """Equal-weight RRF once scored below BM25 alone; this is that regression."""
    from frus_agentic_rag.retrieval.hybrid import bm25_search

    question = "National Security Study Memorandum 14 China policy"
    filters = SearchFilters(volume_ids=["frus1969-76v17"], subtypes=["historical-document"])
    lexical = bm25_search(question, filters, get_settings().candidate_k)
    fused = hybrid_search_sync(question, filters, 10)

    top_lexical = {r["chunk_id"] for r in lexical[:5]}
    assert top_lexical & {e.evidence_id for e in fused}
