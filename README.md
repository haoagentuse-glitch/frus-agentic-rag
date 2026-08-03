# FRUS Bounded Agentic RAG

Question answering over the **full published Foreign Relations of the United States**
corpus — 552 published volumes, 306,016 historical documents, 723,557 chunks —
with a bounded LangGraph agent that either cites real FRUS documents or refuses
to answer.

The design claim is narrow and testable: *adaptive planning, evidence grading and
a deterministic citation gate earn their latency on multi-hop historical
questions*. A fixed 2-step RAG is kept as `B0` precisely so that claim can be
falsified rather than asserted.

```
question
  → plan / route ─┬─ lookup ─ simple ─ timeline ─ status
                  └─ complex → 2-3 English subqueries
  → BM25 + BGE-M3 → RRF
  → grade evidence ─┬─ sufficient → synthesize
                    ├─ missing    → rewrite the missing hop, retry once
                    └─ exhausted  → abstain
  → deterministic citation validation ─┬─ valid   → answer
                                       └─ invalid → abstain
```

**Hard budgets, enforced on the graph edges, not by a recursion limit:**
at most 3 subqueries, 2 retrieval rounds, 1 correction, 4 LLM calls.

## What is actually verified

Everything in this section was produced by running the system on this machine
(RTX 4060 Laptop 8 GB, 32 GB RAM, WSL2). Numbers not listed here are not claimed.

| | |
|---|---|
| Corpus snapshot | `frus-measurement` @ `4b4c402f0cce25144ded2198ba9566b6c37c7c49` |
| XML files | 694 |
| Published volumes / planned stubs | 552 / 142 |
| `div[@type='document']` | 314,483 |
| `historical-document` (indexed as evidence) | 306,016 |
| `editorial-note` (kept, never citable) | 8,467 |
| Chunks (512 BGE-M3 tokens, 64 overlap) | 723,557 |
| **BM25 coverage** | **552 / 552 volumes** |
| **Dense coverage** | **723,557 / 723,557 chunks** (552 / 552 volumes) |
| BGE-M3 throughput (fp16, batch 16, 10k sample) | **32.4 chunks/s**, peak 1.26 GiB VRAM |
| Full dense index, measured | ETA 6.2 h · **completed** |
| Route consistency under paraphrase (10 intents × 3) | **0.80** LLM vs 0.40 rule-first |
| Route accuracy | **0.90** LLM vs 0.80 rule-first |
| Structured-output valid rate | **1.00** (30/30) |
| Graph termination (`frus graph-smoke`) | 5/5 paths, all within budget |
| Tests | 44 passed · ruff + mypy clean |

See `reports/corpus_stats.json`, `benchmark.json`, `graph_smoke.json`,
`route_stability.json`.

The route numbers settle two pre-registered gates. Gate 6 (consistency ≥ 0.80)
passes at exactly 0.80, and the rule-first router that gate would have fallen
back to is *worse* on this corpus — 0.40 consistency, 0.80 accuracy — so the
LLM router stands on evidence rather than on preference. Gate 11 wanted a 0.95
structured-output valid rate; it is 1.00, with a median planner latency of
3.2 s.

### Known UNKNOWN

- **Agentic quality gain.** Whether B3 beats B0 by the pre-registered 10
  percentage points on multi-hop is what `frus eval` exists to answer. Until
  `reports/agent_ablation.json` says so, this README claims no improvement.

## Base image

`Dockerfile` starts `FROM jobshift:latest`, a locally-built image that already
carries Python 3.12, uv, Torch and Sentence Transformers. **This is layer reuse
for a one-day build, not a dependency.** FRUS uses no Jobshift code and no
Jobshift data; the BGE-M3 cache is mounted read-only. Any Python 3.12 image with
CUDA-capable Torch would substitute by editing one line.

Torch resolves to `2.13.0+cu130` from PyPI rather than the cu124 index the spec
suggested: torch is a *transitive* dependency of sentence-transformers, and
`[tool.uv.sources]` only binds direct dependencies. Verified working on the 4060
(the host driver reports CUDA 13.3, so cu130 is the closer match).

## Quick start

```bash
docker compose build
docker compose up -d ollama
docker compose exec ollama ollama pull qwen3:4b-instruct
```

Then the corpus. `scripts/dev.sh` runs the same code against the host venv
without a rebuild round-trip; `docker compose run --rm dev` runs it in the image.

```bash
./scripts/dev.sh frus manifest
./scripts/dev.sh frus ingest
./scripts/dev.sh frus search "Nixon visit to China 1972"
./scripts/dev.sh frus ask "尼克森政府對中國政策如何演變？" --system B3
```

### Building the dense index

Stop Ollama first — 8 GB of VRAM does not hold both comfortably, and the
throughput figure above assumes an otherwise idle GPU.

```bash
docker compose stop ollama
FRUS_GPU=1 FRUS_EMBED_DEVICE=cuda ./scripts/dev.sh frus embed --all --resume
docker compose up -d ollama
./scripts/dev.sh frus index --ann
```

The run checkpoints per volume in `data/index/embed_state.json`. Killing it and
re-running with `--resume` picks up at the next unfinished volume; it does not
start over. Batch size halves on CUDA OOM (16 → 8 → 4) and the reduced size is
remembered.

## Evaluation

```bash
./scripts/dev.sh frus gold-build
./scripts/dev.sh frus eval --systems B0-2step,B0,B1,B2,B3
```

**The gold cases are machine-drafted from the indexed corpus and are not
historian-reviewed.** Every record carries `review_status: "unreviewed"` and the
report repeats the caveat. Questions drawn from the corpus under test measure
whether the pipeline can re-find text it was shown — a retrieval regression
signal, not an external judgement of historical accuracy. A human pass over
`eval/gold_cases.jsonl` is the single highest-value thing anyone can add here.

Primary metrics are deterministic and reproducible offline: All-Evidence
Recall@10, route macro-F1, citation precision, claim citation coverage,
abstention precision/recall, LLM and retrieval calls, p50/p95 latency. Answer
correctness needs a model and is reported separately with the judge named
(Gemini via `GEMINI_API_KEY`, falling back to `judge: "unavailable"` rather than
inventing a score). The judge scores finished answers; it never supplies
evidence, so the closed-corpus rule is intact.

`reports/route_stability.json` settles gates 6 and 11.
`reports/agent_ablation.json` evaluates the pre-registered gates literally,
including the one that says this README may not claim an agentic improvement
without 10 percentage points of multi-hop gain.

## Observability

Traces go to [Arize Phoenix](https://github.com/Arize-ai/phoenix) over OTLP.

```bash
docker compose -f Phoenix/compose.yaml up -d   # http://localhost:6006
```

Client settings live in the **repo-root** `.env` (`PHOENIX_COLLECTOR_ENDPOINT`,
`PHOENIX_PROJECT`); `Phoenix/.env` configures only the server. Leave the
endpoint unset and tracing compiles away to no-op context managers.

Because the Ollama calls are raw httpx rather than a LangChain chat model, no
auto-instrumentor would see them; spans are emitted by hand following
OpenInference conventions, so Phoenix renders proper LLM spans with prompts,
token counts and schema validity, and RETRIEVER spans carrying the retrieved
chunks.

## Serving

```bash
docker compose up -d api ui     # API :8010, Streamlit :8511
```

The UI is a trace viewer, not a chat window: route, subqueries, tool calls,
accepted evidence, whether a correction fired, and the citation gate's verdict
are all on screen. A system whose selling point is bounded decisions has to make
those decisions inspectable.

## Design decisions worth defending

**The citation gate runs no LLM call.** An evidence id that was invented, never
retrieved, not accepted by the grader, or that points at an editorial note is a
blocking failure. The citation block is rebuilt from validated evidence, so a
model that mangles a URL costs nothing — its URL is never used. `repair_claims`
salvages a claim that still has one good id; a claim with none does not survive.

**No web fallback.** FRUS is a closed canonical corpus. A question it cannot
answer gets a refusal that explains the corpus boundary, not a plausible
paragraph from somewhere else.

**Every schema list carries `maxItems`.** This is load-bearing. Without it,
Ollama's grammar-constrained decoder emitted 14,000 tokens on a single grade
call before the read timeout fired. Conversely `maxLength`, `minLength` and
`minItems` make Ollama return *"failed to parse grammar"* (400) — bisected
against a live server. String length is bounded by `num_predict` instead.

**Multi-hop evidence is interleaved round-robin, not sorted by score.** RRF
scores from different subqueries are not on a comparable scale, so sorting the
pooled list lets one strong hop take every slot the grader sees, and a genuine
multi-hop question then grades as unsupported on a hop that was actually
retrieved.

**BM25 is built before any vector exists.** A half-finished embedding run still
leaves a working retriever, and `dense_search` filters on `embedded = true` so
un-embedded chunks never rank as noise.

### Why both arms are kept

With the dense index built, hybrid retrieval first scored *below* lexical alone
on the gold set — 0.186 against 0.209. Two findings came out of that.

The IVF_PQ default scans too few partitions on 723k vectors. Dense recall@10
rose from 0.070 to 0.116 at `nprobes=400, refine_factor=20`, and that alone
brought fusion back to parity with its lexical arm. RRF is weighted so the
weaker arm cannot crowd out the stronger one's hits at ranks 5–10.

The weight is **not tuned**. Sweeping it from 0.0 to 1.0 moves the gold-set
score by at most one document out of 43 — noise on a set that size. Reporting
the best of those as an improvement would be fitting to it.

So the dense arm is justified on evidence the gold set cannot produce. Asked a
Chinese question carrying no English anchor, BM25 over this English corpus
returns nothing at all:

| Query (zh-TW, no English anchor) | BM25 | Dense top hit |
|---|---|---|
| 美國承認共產中國的討論 | **0 hits** | Memorandum, Office of Chinese Affairs |
| 古巴飛彈危機期間的外交電報 | **0 hits** | Telegram From the Embassy in Cuba |
| 關於巴拿馬運河主權的談判 | **0 hits** | Memorandum From Secretary Rusk to President Johnson |
| 第二次世界大戰後對日本的佔領政策 | **0 hits** | Report by the State-War-Navy Coordinating Subcommittee for the Far East |
| 美蘇限制戰略武器談判 | **0 hits** | Telegram From the Delegation to the Strategic Arms Limitation Talks |

That is the entire bilingual capability of this system, and the benchmark is
blind to it: every generated question embeds English titles and proper nouns,
so the gold set measures lexical retrieval and understates dense by
construction. Three regression tests pin the behaviour, because otherwise the
capability could be lost while every lexical metric stayed green.

The honest summary is that this corpus rewards lexical search for English
questions and *requires* dense search for Chinese ones, and the gold set can
only see the first half of that.

## Repository layout

Four packages, one per stage of the pipeline. `models.py` sits at the top
because every stage shares those types.

```
src/frus_agentic_rag/
  config.py  models.py  cli.py  api.py  ui.py  observability.py
  corpus/      manifest tei chunk index embed        # build it
  retrieval/   hybrid tools                          # search it
  agent/       state schemas prompts nodes edges graph run llm citations fakes smoke
  evaluation/  gold ablation route judge             # measure it

tests/          four flat test modules; integration ones skip without an index
docs/           SPEC.md (the build spec), DECISIONS.md, STATE.md
eval/           gold_cases.jsonl
reports/        corpus_stats, benchmark, graph_smoke, agent_ablation, route_stability
Dockerfile  compose.yaml  Phoenix/compose.yaml  scripts/dev.sh
```

`Phoenix/` stays a directory on purpose: `docker compose -f Phoenix/compose.yaml`
treats it as the project directory and reads `Phoenix/.env` for the server's
variable substitution. Client settings belong in the repo-root `.env`.

## Quality gates

```bash
docker compose run --rm dev uv run ruff format --check .
docker compose run --rm dev uv run ruff check .
docker compose run --rm dev uv run mypy src
docker compose run --rm dev uv run pytest --cov=frus_agentic_rag
docker compose run --rm dev uv run frus graph-smoke
```
