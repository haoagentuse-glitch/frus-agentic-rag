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
    # How much survives the rerank. The point of a wide candidate pool is to give
    # the cross-encoder something to choose from, not to hand 50 passages to the
    # grader and the synthesiser: carrying the full pool downstream pushed the
    # synthesis prompt to the edge of the 8192 window and took a query from ~20s
    # to 60-100s. The synthesis window needs 14 documents, the grader sees 10.
    evidence_after_rerank: int = 20
    # Sentence-level filtering with the same cross-encoder. It rewrites what the
    # prompts render, not what is retrieved or citable. Off via
    # FRUS_FOCUS_SENTENCES=false, which is how the paired run measures it.
    focus_sentences: bool = True
    focus_max_sentences: int = 4
    focus_context_sentences: int = 1
    # Below this a passage is already about as short as trimming would make it.
    focus_min_chars: int = 400
    # Ceiling on cross-encoder pairs per call. 20 passages of ~15 sentences is
    # 300; the cap stops an unusually sentence-dense pool from turning one
    # rerank-sized cost into several.
    focus_max_pairs: int = 400
    # Evidence selection for B2/B3. "score" filters deterministically on the
    # cross-encoder logit; "llm" restores the grader call, kept only so the two
    # can be compared on the same gold set.
    grader_mode: Literal["score", "llm"] = "score"
    # Deliberately unset. Cross-encoder logits are uncalibrated — they differ by
    # model, by query language and by question type — so a threshold chosen
    # before looking at the distribution is a guess. Run
    # `scripts/score_distribution.py`, read the report, then set these. Until
    # then selection keeps everything and records what it would have done.
    score_keep_absolute: float | None = None
    score_keep_margin: float | None = None
    score_min_per_hop: int = 3
    # Upper bound on what survives, per hop. With min and max set equal and no
    # margin, selection degenerates to fixed top-k — the baseline any adaptive
    # rule has to beat before its complexity is worth carrying.
    score_max_per_hop: int | None = None

    # D4: widen each surviving passage with its neighbours inside the same
    # document before synthesis. Cheap because the chunks are already indexed —
    # no re-embedding, no new index — which is why it is tried before late
    # chunking or contextual retrieval, both of which need a full rebuild.
    context_expand_neighbours: int = 0

    # --- citation attribution ------------------------------------------------
    # "post_hoc": the model writes claims, the cross-encoder attaches the ids.
    # "model": the old scheme, where the model copies ids verbatim — kept only
    # so the two can be compared on the same gold set.
    citation_mode: Literal["post_hoc", "model"] = "post_hoc"
    attribution_top_k: int = 3
    # OFF, because the verifier was measured and it does not work. Given
    # sentences taken verbatim from the passage they came from — the easiest
    # possible positive, since a passage entails its own sentences — it answered
    # "not supported" 70% of the time. It never accepted an unrelated sentence,
    # so it is not lax; it rejects almost everything. The grounding probe's
    # apparent improvement (7 fabrications to 3) came from indiscriminate
    # rejection, not from verification, which is why shipping it on by default
    # would have been shipping a number rather than a fix.
    #
    # The stage stays: the gap it addresses is real and the machinery is right.
    # What it needs is a verifier that works, and the measurement points at one.
    # On the same probe qwen3:8b scores 0.80 true-positive against 4B's 0.30,
    # with false-positive still near zero — the same shape as D3, where 8B gained
    # 28.5pp on lookup by reading a passage more carefully. 0.80 is still short
    # of the 0.90 bar on the easiest possible positives, so 8B is a lead rather
    # than a fix, and `entailment_verifier_model` exists to test it without
    # changing the generator.
    entailment_check: bool = False
    # Empty means "use the generation model". Set to qwen3:8b or larger to run
    # the check on a model that can do it.
    entailment_verifier_model: str = ""
    entailment_max_chars: int = 900
    # Relative, like the evidence filter and for the same measured reason: the
    # logits shift by three units between question kinds, so a fixed floor
    # attributes everything on one kind and nothing on another.
    attribution_margin: float = 2.0
    # There is deliberately no support threshold here, and that is a gap worth
    # stating rather than papering over. A floor was tried at -6.0; measured
    # over 29 claims the best-passage score ran 0.285 to 8.953, so it never
    # rejected anything and no value would have helped. The reason is not
    # calibration: the cross-encoder finds the passage a claim was written FROM,
    # so a claim invented by over-reading a passage scores against that passage
    # highly. It measures provenance, not support.
    #
    # So this pipeline currently checks that every claim has a source, and does
    # not check that the source says it. Closing that needs an entailment check
    # (claim vs passage, NLI-style), which is a separate stage, not a number.

    # --- eval judge (optional, external) -----------------------------------
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-3.5-flash-lite", alias="GEMINI_MODEL")
    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    deepseek_model: str = Field(default="deepseek-v4-flash", alias="DEEPSEEK_MODEL")
    deepseek_base_url: str = Field(default="https://api.deepseek.com", alias="DEEPSEEK_BASE_URL")
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
