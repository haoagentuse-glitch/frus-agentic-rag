# FRUS RAG 搭建操作指令（交給執行 Agent）

目標：在 WSL2 Ubuntu／RTX 4060 8GB 上，搭建全已發布 FRUS 的雙語 hybrid RAG。先交付全量 BM25，再以可續跑方式完成 dense index。詳細決策見 `rag-portfolio-one-day-plan.md`。

## 0. 執行契約

- 程式碼放在 `D:\Project\sideProject\interview\frus-rag`；所有 Python／Docker 指令從 WSL 執行。
- 若目錄已是 Git repo，先確認分支；不得直接修改 `main/master`，建立 `feat/frus-rag-mvp`。
- 不使用 LangChain；檢索流程必須可觀察、可單獨評估。
- 長任務以 volume checkpoint；中斷後不得從頭重跑。
- 不修改 `jobshift:latest`、Jobshift 原始碼或資料；只衍生新 image，BGE-M3 cache 唯讀掛載。

## 1. 先確認並沿用 Jobshift GPU 環境

已查到的 WSL 資產：

- image：`jobshift:latest`，Python 3.12.13，uv-managed `/opt/venv`。
- Torch `2.6.0+cu124`、Sentence Transformers `3.4.1`、Transformers `4.57.6`。
- image 已可透過 `--gpus all` 看見 RTX 4060。
- BGE-M3 cache：`/home/haoche_nitro_v15/jobshift/data/hf_cache`，約 4.3GB；snapshot `5617a9f61b028005a4858fdac845db406aefb181`。
- 可參考：`/home/haoche_nitro_v15/jobshift/Dockerfile`、`docker-compose.yml`、`jobshift/embed.py`。

先重跑 smoke test：

```bash
docker image inspect jobshift:latest
docker run --rm --gpus all --entrypoint nvidia-smi jobshift:latest
docker run --rm --entrypoint uv jobshift:latest \
  pip list --python /opt/venv/bin/python
test -d /home/haoche_nitro_v15/jobshift/data/hf_cache/hub/models--BAAI--bge-m3
```

建立 `frus-rag:latest` 時可先 `FROM jobshift:latest`，但必須由本專案自己的 `pyproject.toml` 與 `uv.lock` 決定依賴。模型 cache 掛載：

```yaml
volumes:
  - /home/haoche_nitro_v15/jobshift/data/hf_cache:/data/hf_cache:ro
```

不得把整個 Jobshift `data/` 掛成可寫。

## 2. 建立現代 Python 專案

使用 Python 3.12、src layout、uv、Ruff、mypy、pytest：

```bash
mkdir -p /mnt/d/Project/sideProject/interview/frus-rag
cd /mnt/d/Project/sideProject/interview/frus-rag
uv init --package --python 3.12
uv add lxml pyarrow lancedb sentence-transformers \
  fastapi "uvicorn[standard]" streamlit pydantic-settings orjson typer rich httpx
uv add --dev ruff mypy pytest pytest-cov pre-commit
```

在加入 Torch 前，沿用 Jobshift 已驗證的 CUDA 12.4 source：

```toml
[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]
```

然後執行 `uv add "torch>=2.5,<3" && uv lock`。不得讓 uv 靜默換成 CPU-only Torch。

在 `pyproject.toml` 設定：

- `requires-python = ">=3.12,<3.13"`
- Ruff：target `py312`、line length 100，啟用 `E,W,F,I,UP,B,SIM,RUF`。
- pytest：`testpaths = ["tests"]`、strict markers/config。
- mypy：Python 3.12、`check_untyped_defs = true`。
- 提供 console script：`frus = "frus_rag.cli:app"`。

建立目錄：

```text
src/frus_rag/{config,manifest,tei,chunk,embed,index,retrieval,generation,api,ui,cli}.py
tests/fixtures/  eval/  reports/  data/
Dockerfile  compose.yaml  .env.example  README.md
```

## 3. 固定來源並建立 manifest

不要重下載。先使用已固定的來源：

```text
Windows: D:\Project\sideProject\interview\frus-measurement
WSL:     /mnt/d/Project/sideProject/interview/frus-measurement
commit:  4b4c402f0cce25144ded2198ba9566b6c37c7c49
```

實作 `frus manifest`：

1. 掃描全部 694 個 XML，記錄 path、size、SHA-256、volume ID、source commit、parse status。
2. `type="document"` 基準為 314,483 個 div；差異必須可解釋。
3. 只有 `tei:div[@type='document'][@subtype='historical-document']` 進主正文 corpus。
4. planned placeholder、總索引、front/back matter 另存 catalog，不進主正文向量索引。

## 4. TEI 解析與切塊

使用 `lxml.etree.iterparse` streaming 解析，不用 regex 當 production parser：

- 不跨 `document_id` 切塊。
- 最多 512 BGE-M3 tokenizer tokens、overlap 64，段落優先 packing。
- 保存 volume/document/chapter、日期 interval、頁碼、人物、terms、footnote parent、canonical URL。
- 每卷輸出 `data/chunks/<volume_id>.parquet`，先寫 `.tmp`，校驗後 atomic rename。
- stable `chunk_id`；重跑不得新增重複資料；truncation count 必須為 0。

先做六年代 fixtures，再做 3 卷 small/medium/large dry run。通過後才跑全量 parse。

## 5. 技術選型與索引順序

固定第一版：

| 層 | 選型 |
|---|---|
| 原始／中間資料 | TEI/XML → per-volume Parquet |
| Lexical | LanceDB FTS／BM25 |
| Dense | `BAAI/bge-m3`，1024d、FP16 encode、float32 normalized output |
| 融合 | Reciprocal Rank Fusion（RRF） |
| Reranker | 第一日不做；只有 eval 證明 nDCG@10 +0.02 才加入 |
| Generator | Ollama `qwen2.5:7b`、4k context；OOM 才降 1.5B |
| API／UI | FastAPI + Streamlit |

第一版只用 BGE-M3 的 dense embedding；lexical 仍由 LanceDB BM25 負責。不要為了使用 BGE-M3 的 sparse／ColBERT 能力而先加入 `FlagEmbedding`，除非基準測試另立實驗證明值得。

順序必須是：**全量 parse → 全量 BM25 → 10k dense benchmark → checkpointed dense → RRF → generation/UI**。

## 6. BGE-M3 benchmark gate

衍生 image 後，用既有唯讀 cache、GPU 與最終 chunk 設定測 10,000 chunks：

- batch 從 16 起測，OOM 時 16 → 8 → 4。
- 記錄 chunks/s、tokens/s、peak VRAM、寫 Parquet 吞吐、向量維度與 norm。
- 初估全量 65–100 萬 chunks；若希望 dense 階段 8 小時內完成，需持續約 23–35 chunks/s。
- 若 BGE-M3 過慢，不可直接宣稱全量 dense 完成：先保留全量 BM25，改測 `intfloat/multilingual-e5-small`，並以相同 10k sample 比較速度與 Recall@10。

每卷 embedding 寫入 `data/embeddings/<model_revision>/<volume_id>.parquet`；模型 revision、batch、max length、耗時與 checksum 寫入 `build_state.jsonl`。

## 7. Retrieval、回答與引用

- 英文：BM25 + BGE-M3 dense → RRF。
- 中文：原中文 query 走 multilingual dense；Qwen 產生短英文 lexical query 再走 BM25 → RRF。
- top 4–5 evidence 送 Qwen；回答每個主要主張附 `[volume_id/document_id]`。
- URL 固定為 `https://history.state.gov/historicaldocuments/{volume_id}/{document_id}`。
- planned、找不到證據或 citation resolver 失敗時必須拒答。
- debug panel 顯示 BM25、dense、RRF 排名與各階段 latency。

## 8. 評估與完成條件

先做 20 題 mini-qrels，至少含英文、中文跨語言、exact fact、跨文件與 unanswerable。分別跑 BM25、dense、RRF，報 Recall@5/10、MRR@10、nDCG@10。

交付前必跑：

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest --cov=frus_rag
uv run frus manifest
uv run frus ingest --limit-volumes 3
uv run frus benchmark --chunks 10000
```

完成回報只交：

1. 實際測試輸出與失敗項目。
2. `reports/corpus_stats.json`、`benchmark.json`、`retrieval_metrics.json`。
3. BM25 coverage `552/552`；dense coverage `X/552` 與 ETA。
4. 啟動指令、介面 URL、已知限制。

不可用「pipeline 已啟動」冒充全量完成；只有 manifest、checksums 與 coverage 證明後才能寫「全量 hybrid index」。
