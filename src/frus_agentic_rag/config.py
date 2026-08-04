"""Central settings. Everything path- or budget-like lives here, not inline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# FRUS canonical identifiers. The LLM never builds these; code does.
FRUS_CANONICAL_BASE = "https://history.state.gov/historicaldocuments"

# The snapshot the whole corpus is pinned to.
FRUS_SOURCE_COMMIT = "4b4c402f0cce25144ded2198ba9566b6c37c7c49"
FRUS_SOURCE_REPO = "/mnt/d/Project/sideProject/interview/frus-measurement"

TEI_NS = "http://www.tei-c.org/ns/1.0"
XML_NS = "http://www.w3.org/XML/1998/namespace"


class Budgets(BaseSettings):
    """Hard bounds on the graph. Exceeding any of these ends the run."""

    max_subqueries: int = 3
    max_retrieval_rounds: int = 2
    max_llm_calls: int = 4
    max_corrections: int = 1
    recursion_limit: int = 25


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRUS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -------------------------------------------------------------
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw/frus/volumes")
    lancedb_uri: Path = Path("data/index/lancedb")
    reports_dir: Path = Path("reports")
    checkpoint_db: Path = Path("data/index/checkpoints.sqlite")

    # --- embedding ---------------------------------------------------------
    bge_model_path: str = ""
    embed_device: Literal["cpu", "cuda"] = "cpu"
    embed_batch_size: int = 16
    embed_fp16: bool = True
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64
    embed_dim: int = 1024

    # --- generation --------------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b-instruct"
    ollama_num_ctx: int = 8192
    ollama_temperature: float = 0.0
    ollama_timeout_s: float = 90.0
    ollama_num_predict: int = 1024
    # Synthesis needs its own ceiling. At 1024 the answer JSON was truncated
    # mid-sentence on evidence-rich questions and failed validation 25% of the
    # time; the same call completes in ~1500 tokens with room to spare. The
    # tight default stays for planner and grader, whose outputs are small and
    # whose arrays are what the cap is guarding against.
    ollama_num_predict_synthesis: int = 3072

    # --- retrieval ---------------------------------------------------------
    top_k: int = 10
    candidate_k: int = 50
    rrf_k: int = 60
    # Fusion weights for [lexical, dense]. NOT tuned: on the gold set every
    # weight from 0.0 to 1.0 lands within one document of the others, which is
    # noise on 43 gold documents. Chosen on the principle that lexical should
    # lead when both arms fire, while dense stays strong enough to carry a query
    # the lexical arm cannot answer at all — a Chinese question over this English
    # corpus returns zero BM25 hits. See README "Why both arms are kept".
    rrf_weight_lexical: float = 1.0
    rrf_weight_head: float = 1.0
    rrf_weight_dense: float = 0.5
    # ANN search depth. The IVF_PQ default scans too few partitions on 723k
    # vectors; measured dense recall@10 rose from 0.070 to 0.116 at 400/20.
    ann_nprobes: int = 400
    ann_refine_factor: int = 20
    # Cross-encoder reranking over the fused candidates. Empty path disables it
    # and the pipeline falls back to RRF order. rerank_candidates is how many of
    # the fused list the cross-encoder scores; measured, 91.7% of gold documents
    # sit inside 50 per hop while RRF's top 14 carried 37.5%.
    reranker_model_path: str = ""
    reranker_device: Literal["cpu", "cuda"] = "cpu"
    reranker_batch_size: int = 16
    reranker_max_length: int = 512
    reranker_max_chars: int = 1800
    rerank_candidates: int = 50

    # --- eval judge (optional, external) -----------------------------------
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_MODEL")
    gemini_base_url: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta",
        alias="GEMINI_BASE_URL",
    )

    budgets: Budgets = Budgets()

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "processed" / "chunks"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "processed" / "manifest.parquet"

    @property
    def aux_dir(self) -> Path:
        """Non-evidence material: planned stubs, indexes, front/back matter."""
        return self.data_dir / "processed" / "aux"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
