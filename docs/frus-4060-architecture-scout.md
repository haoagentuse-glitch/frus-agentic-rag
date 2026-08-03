# FRUS 全量雙語 RAG：RTX 4060 8GB 的研究架構與執行計畫

研究日期：2026-08-03（Asia/Taipei）  
定位：AI Engineer 面試作品集；Windows 11；單張 RTX 4060 8GB；CPU、RAM、SSD 容量與速度尚未提供。

## 0. 先給結論、失敗條件與改判證據

### 結論

**VERIFIED — 推薦架構：官方 FRUS TEI/XML → 可恢復的 Parquet 中間層 → `multilingual-e5-small` 全量向量化 → LanceDB OSS 的 BM25/FTS＋向量混合檢索 → RRF → 選配多語 cross-encoder → Ollama `qwen2.5:7b` 生成附文件級引用的中英回答。** LanceDB 是內嵌式資料庫，不需要在 Windows 上另外維運 JVM、Docker 服務或常駐伺服器；官方文件同時涵蓋 FTS、混合檢索、RRF、向量索引與 upsert，對只有一天完成作品集、且主機 RAM 未知的情境，失敗面最小。[LanceDB Quickstart](https://docs.lancedb.com/quickstart)（文件未標日期，查閱 2026-08-03）、[Full-text Search](https://docs.lancedb.com/search/full-text-search)（文件未標日期，查閱 2026-08-03）、[Hybrid Search](https://docs.lancedb.com/search/hybrid-search)（文件未標日期，查閱 2026-08-03）。

**VERIFIED — 4060 8GB 可以做這個系統，但必須分時使用 GPU。** 離線建索引時卸載 Ollama，只讓 0.1B 的 E5 embedding 模型使用 CUDA；線上問答時則讓 Qwen 7B 佔用 GPU，query embedding、LanceDB 與 reranker 先放 CPU。Ollama 官方說明指出 context 與 parallel request 都會增加記憶體需求，且可用 `keep_alive`、最大載入模型數與平行數控制常駐模型。[Ollama Context length](https://docs.ollama.com/context-length)（文件未標日期，查閱 2026-08-03）、[Ollama FAQ](https://docs.ollama.com/faq)（文件未標日期，查閱 2026-08-03）。

**UNVERIFIED — 「完整 FRUS 全量索引能在一天內建完」目前不能成立。** 2023 年 KG-FRUS 論文描述的衍生資料已超過 30 萬份外交文件，但目前全庫切塊數、CPU、RAM、SSD 與實際 embedding throughput 都未知。一天的可交付承諾應是：完整 pipeline、可展示 MVP、實測資源模型、斷點續跑全量工作；全量完成時間只能由 10k/50k chunk pilot 外推。[KG-FRUS](https://arxiv.org/abs/2311.01606)（發表 2023-11-03，查閱 2026-08-03）。

### 前三個失敗條件與預登記改判證據

| # | 失敗條件（先登記，避免事後硬拗） | 廉價驗證 | 觸發後如何改判 |
|---|---|---|---|
| 1 | **LanceDB 在這台 Windows 的安裝、全量尺度或查詢表現不穩** | 先做 Python 安裝 smoke test；再以 50k chunks 測冷／熱 p95、程序 RSS、ANN Recall@10 | 若安裝失敗無可重現修法，或調參後 ANN Recall@10 < 0.90，或檢索 p95 > 1.5 秒，改用 Qdrant on-disk；若已確認 Docker 與充足 RAM、且需要企業級 lexical 功能，再改 OpenSearch。LanceDB 官方文件**未查到**對此使用者確切 Windows/Python 組合的 wheel 保證，因此 smoke test 是必要 gate。 |
| 2 | **Qwen 7B 在 8GB 下無法全 GPU、延遲過高或 OOM** | `ollama ps` 記錄 VRAM/CPU 配置；固定 4k context、同一 prompt 跑 20 次冷／熱測試 | 若非全 GPU 或 warm p95 > 20 秒：context 4096→3072、證據 5→4；仍失敗就改 `qwen2.5:1.5b`。這是生成器降級，不重建索引。門檻是本專案工程目標，不是官方保證。 |
| 3 | **混合檢索或 reranker 沒有帶來可量化收益** | 60 題 qrels 比較 BM25、dense、RRF、RRF＋reranker | 若 hybrid 的 nDCG@10 未比最佳單路高至少 0.02，中文 query 改走 dense-only、英文保留 hybrid；若 reranker 未再提高 0.02 或增加 p95 超過 1.5 秒，就移除 reranker。這些也是預登記的作品集驗收門檻。 |

## 1. 為什麼 FRUS 值得做 RAG

**VERIFIED — 專業性與資料價值足夠。** Foreign Relations of the United States（FRUS）自 1861 年起是美國政府對重大外交政策決策與活動的官方歷史紀錄，資料具有長時間跨度、跨國事件、人物與政策脈絡；這類問題通常需要跨文件檢索與可追溯引文，不只是把單篇文章交給 LLM。[Office of the Historian — About FRUS](https://history.state.gov/historicaldocuments/about-frus)（頁面未標日期，查閱 2026-08-03）。

**VERIFIED — 取得容易且授權清楚。** Office of the Historian 說明已出版卷冊可線上閱讀與下載，內容屬 public domain；官方 GitHub repository 提供 TEI P5 XML 主檔，每卷一檔，`volume id` 與 `document id` 發布後視為 canonical/stable。[FRUS FAQ](https://history.state.gov/about/faq/what-is-frus)（頁面未標日期，查閱 2026-08-03）、[historyatstate/frus](https://github.com/historyatstate/frus)（release v1.0.18，2026-03-13；查閱 2026-08-03）、[Developer Resources](https://history.state.gov/developer)（頁面未標日期，查閱 2026-08-03）。

**VERIFIED — 範圍足以稱為「全量」。** 2018 年官方歷史諮詢委員會紀錄指出，當時 512 卷 back catalog 已可用 TEI/XML 形式搜尋、閱讀與下載；2024 年會議紀錄則說 repository 涵蓋全部印刷卷、legacy electronic-only 卷及部分 microfiche。這兩個數字／描述不是 2026 當下的精確 volume count，所以實作仍須由 repository manifest 現場統計。[December 2018 HAC](https://history.state.gov/about/hac/December-2018)（發布 2018-12，查閱 2026-08-03）、[September 2024 HAC](https://history.state.gov/about/hac/September-2024)（發布 2024-09，查閱 2026-08-03）。

### 可在面試中說的產品故事

「讓研究者用中文提問、從英文外交原始文件找證據，回答時附可回到官方文件的引用」比一般 PDF Chat 更有辨識度。展示題可包括：

- 史學：古巴飛彈危機決策脈絡，Kennedy 政府內部意見如何變化？
- 哲學／政治思想：不同時期的自由、人權、現實主義語彙如何出現在政策文件？
- 藝術外交：冷戰期間文化、展覽、宣傳或 UNESCO 如何被外交政策運用？
- 跨語言：中文問題檢索英文證據，再以中文回答；英文原文保留在 citation panel。

## 2. 資料契約與切塊

### 2.1 來源固定與可重現性

1. 從官方 `historyatstate/frus` repository 下載，不爬網頁。
2. 將 release tag 或 commit SHA 寫入 `source_manifest.json`；不要直接追蹤會變動的 `main`。
3. 每個 TEI 檔記錄路徑、byte size、SHA-256、`volume_id`、文件數與 parse 狀態。
4. `document_id` 作引用主鍵；page-break ID 只作顯示資訊。官方 repository 說 page-break IDs 不屬 canonical、日後可能改變，因此不可拿它作唯一鍵。[historyatstate/frus](https://github.com/historyatstate/frus)（release v1.0.18，2026-03-13；查閱 2026-08-03）。

### 2.2 TEI 解析規則

- parent document：以 TEI 中的文件級 `<div>`／官方 document identity 為邊界，不跨 document 混切。
- paragraph-first：先保留 `<p>`、標題、日期、發文／收文地、人物、備註與 footnote 關係，再做 token-aware packing。
- chunk body：目標 320–384 E5 tokens，overlap 40–60；硬上限依 embedding tokenizer 計算，不以字元估算。
- footnote：短註跟隨所在段；長註另成 `kind=note` chunk 並保存 `parent_chunk_id`。
- front matter、目錄、編者說明保留但標記 `kind`；預設搜尋可針對 `kind=document|note`，避免目錄污染排序。
- 每個 chunk 至少保存：`chunk_id`、`volume_id`、`document_id`、`section_path`、`text`、`token_count`、`date_raw`、`persons`、`places`、`kind`、`source_url`、`source_sha`、`parser_version`。
- `chunk_id = SHA256(volume_id | document_id | section_path | start_offset | end_offset | normalized_text_sha)`；相同設定重跑不得產生新 ID。
- **驗收硬規則：embedding 前的 truncation count 必須為 0。** 超長資料回到 parser 重切，不接受 tokenizer 靜默截斷。

### 2.3 雙語的真實邊界

FRUS corpus 主要是英文；「雙語」指中文／英文 query 與答案，不是把全部原文機器翻譯後再索引。`multilingual-e5-small` 的多語對齊負責中文 query→英文 passage 的 dense retrieval；英文 query 再加 BM25。中文 BM25 對英文原文通常沒有 lexical overlap，不能假裝它會有效，因此是否讓中文 query 進 hybrid 必須由 qrels 決定。

## 3. 選定架構

```text
一次性／增量離線（Ollama 卸載，GPU 只給 E5）
官方 TEI/XML
  → manifest + XML validation
  → document-aware chunking
  → chunks/<volume>.parquet（原子寫入）
  → multilingual-e5-small, CUDA FP16
  → embeddings/<volume>.parquet（原子寫入）
  → versioned LanceDB table
  → FTS(BM25, first pass 無 positions)
  → IVF_HNSW_SQ
  → integrity + retrieval + ANN recall gates
  → 切換 active_index.json

線上（GPU 只給 Qwen；其餘先走 CPU）
中／英文問題
  → E5 query embedding（CPU）
  → [英文：BM25 + vector]／[中文：先測 dense-only vs hybrid]
  → RRF
  → multilingual cross-encoder（CPU，選配）
  → top 4–5 chunks
  → Qwen2.5 7B（Ollama, GPU, 4k context）
  → 中／英文答案 + FRUS volume/document URL + evidence snippets
```

這個拆分的目的不是追求最多元件，而是確保任何時刻只有一個大工作負載佔 GPU：離線 E5 或線上 Qwen，不同時常駐。

## 4. 模型、精度與實際起始參數

### 4.1 Embedding：`intfloat/multilingual-e5-small`

**VERIFIED。** 模型卡標示 MIT、約 0.1B 參數、384 維、最大 512 tokens，並要求即使非英文也使用 `query:`／`passage:` prefix；長內容會被截斷。mE5 論文發布於 2024-02-08。[Model card](https://huggingface.co/intfloat/multilingual-e5-small)（頁面未標發布日，查閱 2026-08-03）、[mE5 paper](https://arxiv.org/abs/2402.05672)（發布 2024-02-08，查閱 2026-08-03）。

| 場景 | 裝置／精度 | 起始 batch | `max_length` | 其他 |
|---|---|---:|---:|---|
| 全量 passage embedding | CUDA FP16 | 32 | 384 | `passage:` prefix、normalize embeddings、length bucket；輸出以 float32 保存 |
| 線上 query embedding | CPU FP32 | 1 | 128 | `query:` prefix、normalize；若 CPU p95 不達標再測 ONNX |

Sentence Transformers 官方文件說 FP16 可在 GPU 上加速、通常只有極小準確率差異；其 API 預設 batch 32，但最適 batch 取決於硬體、模型、精度與長度，因此 32 只是 pilot 起點，不是官方對 4060 的 benchmark。[Inference efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)（文件未標日期，查閱 2026-08-03）、[SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)（文件未標日期，查閱 2026-08-03）。

OOM fallback：32→16→8；每次記錄 chunks/s 與 peak VRAM。batch 8 仍 OOM 時，先確認 Ollama與其他 CUDA 程序已卸載，再改 CPU/ONNX；不要立刻更換 embedding 模型，否則全部向量與 qrels baseline 都失效。

### 4.2 Reranker：`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`（選配）

**VERIFIED。** 模型卡標示 Apache-2.0、約 0.1B、多語 MiniLMv2，使用翻譯版 mMARCO 訓練，以 query-passage pair 聯合評分；tokenizer 設定的上限是 512。[Model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1)（頁面未標日期，查閱 2026-08-03）、[tokenizer_config.json](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/main/tokenizer_config.json)（檔案日期未查到，查閱 2026-08-03）。

- 線上起點：CPU FP32、batch 8、max_length 512，RRF top 30 → reranked top 8。
- 不把 reranker 納入離線 indexing。
- 第一日不要同時把 reranker 與 Qwen 放 GPU；只有在 CPU gate 失敗且 reranker 的 nDCG 收益達標時，才另測 ONNX INT8，並以 score/ranking parity 做回歸。
- LanceDB 內建 cross-encoder 範例的預設模型是英文模型；本案應手動接上述多語模型，不使用英文預設值。[LanceDB Cross-encoder reranker](https://docs.lancedb.com/reranking/cross-encoder)（文件未標日期，查閱 2026-08-03）。

### 4.3 Generator：Ollama `qwen2.5:7b`

**VERIFIED。** Ollama model library 標示 Qwen2.5 支援包含中文在內的多語，`7b` 套件約 4.7GB，廣告 context 上限 32K；但 Ollama 對低於 24GiB VRAM 的預設 context 是 4K，而且 context 增大會增加記憶體，所以本案不追 32K。[Qwen2.5 library](https://registry.ollama.com/library/qwen2.5)（頁面標示約 1 年前更新，查閱 2026-08-03）、[Context length](https://docs.ollama.com/context-length)（文件未標日期，查閱 2026-08-03）。

- 起點：`qwen2.5:7b`、記錄實際 model digest／manifest、`num_ctx=4096`、parallel=1、最大載入模型數=1。
- prompt budget：system/instruction 約 300 tokens；4–5 個 evidence chunks 共約 2,000–2,500；問題與 metadata 約 200；保留約 700–1,000 給回答與 tokenizer 誤差。
- 回答規則：只用 evidence；每段 claim 綁定 `[volume_id / document_id]`；證據不足就回答「資料中找不到足夠依據」。
- `ollama ps` 記錄 `size_vram`；離線 indexing 前用 `keep_alive=0` 或停止模型，確認 GPU 已釋放。[Running models API](https://docs.ollama.com/api/ps)（文件未標日期，查閱 2026-08-03）、[Windows](https://docs.ollama.com/windows)（文件未標日期，查閱 2026-08-03）。

## 5. 檢索資料庫比較與選型

| 選項 | 能力 | 對本機的主要代價 | 判定 |
|---|---|---|---|
| **LanceDB OSS** | 內嵌、向量、BM25 FTS、hybrid/RRF、disk-first ANN、Python 官方 SDK | Windows exact wheel 組合需先 smoke test；OSS index/compaction 要手動管理 | **首選。** 最少服務依賴，適合一天作品集與未知 RAM。 |
| **Qdrant** | dense＋sparse named vectors、RRF hybrid；vectors/HNSW 可 on-disk | 完整服務通常要多一個 server/container；中文 query 對英文 BM25 仍無詞面重疊 | **第一備援。** LanceDB 實測 scale gate 失敗時採用。 |
| **OpenSearch** | 成熟 BM25、k-NN 與 hybrid；可用 on-disk vector mode | JVM、Docker/WSL 與系統 RAM 調校；Windows Docker 安裝文件要求至少 4GB host memory，WSL 還需調 `vm.max_map_count` | **不是此機第一選擇。** 已有 OpenSearch 維運環境或要展示企業搜尋才值得。 |
| **Haystack InMemoryDocumentStore** | 無外部服務，pipeline 拼裝快 | 官方明確定位為實驗用途、非 production；資料留在 RAM | **只做最小 smoke test，不做 FRUS 全量。** Haystack 是 orchestration framework，不取代持久化資料庫。 |

依據：[Qdrant Hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/)（文件未標日期，查閱 2026-08-03）、[Qdrant Vertical scaling](https://qdrant.tech/documentation/guides/capacity-planning/)（文件未標日期，查閱 2026-08-03）、[Qdrant Overview](https://qdrant.tech/documentation/overview/)（文件未標日期，查閱 2026-08-03）、[OpenSearch Docker](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/)（頁面約於 2026-07 更新，查閱 2026-08-03）、[OpenSearch k-NN engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/)（文件未標日期，查閱 2026-08-03）、[OpenSearch memory-optimized search](https://docs.opensearch.org/latest/vector-search/optimizing-storage/memory-optimized-search/)（文件未標日期，查閱 2026-08-03）、[Haystack InMemoryDocumentStore](https://docs.haystack.deepset.ai/docs/inmemorydocumentstore)（頁面約於 2026-07 更新，查閱 2026-08-03）。

### LanceDB 索引設定

- FTS：先建立 BM25，不開 positions；官方文件說 positions 可支援 phrase query，但會顯著增加 index 大小與建置時間。等 phrase-query evaluation 證明需要再開。[Full-text Search](https://docs.lancedb.com/search/full-text-search)（文件未標日期，查閱 2026-08-03）。
- 融合：先用 RRF，因 dense 與 keyword 分數不可直接比較；再用 qrels 決定是否換 reranker。[Hybrid evaluation](https://docs.lancedb.com/reranking/eval)（文件未標日期，查閱 2026-08-03）。
- ANN：384 維先用 `IVF_HNSW_SQ`，它是官方建議的 recall/latency 折衷；`ef_construction=150`，partitions 從 `max(1, N // 1,048,576)` 起測。這是官方 starting guidance 加上防止小資料得到 0 partition 的實作護欄。[Vector index](https://docs.lancedb.com/indexing/vector-index)（文件未標日期，查閱 2026-08-03）。
- 索引品質：在固定樣本用 `bypass_vector_index()` 跑 flat exact，計算 ANN Recall@10，不可只看主觀答案。
- 寫入：以 stable `chunk_id` 做 `merge_insert(...).when_not_matched_insert_all()`；官方文件支援依 key 比對 incoming rows，並建議 join key 建 scalar index。[Updating and Modifying Table Data](https://docs.lancedb.com/tables/update)（文件未標日期，查閱 2026-08-03）。

## 6. 資源與建置時間：先量測，不假裝知道

### 6.1 必須先產出的 `corpus_stats.json`

```json
{
  "source_commit": "...",
  "tei_files": 0,
  "documents": 0,
  "chunks": 0,
  "utf8_text_bytes": 0,
  "tokens_p50": 0,
  "tokens_p95": 0,
  "tokens_max": 0,
  "truncation_count": 0
}
```

截至研究日，**未查到**官方對目前 repository 的精確 document/chunk/token/FTS bytes 統計；任何固定 GB 或小時數都應視為猜測。

### 6.2 磁碟估算公式

令 `N` 為 chunks、`d=384`、`B_text` 為 chunk table 文字與 metadata 的實測 bytes：

- raw float32 vector：`N × 384 × 4 bytes`。
- 每 100k chunks 約 153,600,000 bytes，即 146.5 MiB；1M chunks 約 1.43 GiB。這只是 raw vectors，不是整個資料庫。
- LanceDB 文件稱 `IVF_HNSW_SQ` 壓縮 index 通常略大於 raw vector 的四分之一；原始 vector 仍在 table，所以不能把「1/4」誤寫成總容量。[Vector index](https://docs.lancedb.com/indexing/vector-index)（文件未標日期，查閱 2026-08-03）。
- 在 50k sample 測：`r_table = table_bytes / B_text`、`r_fts = fts_bytes / B_text`、`r_ann = ann_bytes / raw_vector_bytes`。
- 全量預估：`source_TEI + r_table×B_text + raw_vectors + r_fts×B_text + r_ann×raw_vectors + checkpoints + 30% headroom`。
- 若預估後會吃掉可用磁碟超過 80%，停止全量 build；這是工程安全門檻，不是產品規格。

Qdrant 若作備援，官方 capacity guidance 的 full-precision vector RAM 估法是 `N × dimensions × 4 × 1.5`，另有 HNSW、payload、WAL 與 headroom；它也支援將 vectors/HNSW 放磁碟。這再次說明不能用「有 8GB VRAM」推導資料庫 RAM 足夠。[Qdrant Capacity planning](https://qdrant.tech/documentation/guides/capacity-planning/)（文件未標日期，查閱 2026-08-03）。

### 6.3 時間估算與 benchmark 程序

先跑 10k chunks 暖機，再跑 3 組 50k sample；各 stage 單獨計時：

1. `R_parse`：documents/s。
2. `R_embed`：chunks/s，另記 tokens/s、batch、max_length、peak VRAM。
3. `R_ingest`：rows/s。
4. `T_fts_sample`、`T_ann_sample`：sample index 秒數與 bytes。
5. 用實際全量 `D` documents、`N` chunks 外推：  
   `T_est = 1.30 × (D/R_parse + N/R_embed + N/R_ingest + scaled_FTS + scaled_ANN)`。

1.30 是本案為重試、cache 差異與尾端長 chunk 保留的 buffer，不是文獻常數。報告要同時呈現 best/median/worst 三組 sample，不只報最快一次。GPU 用 `nvidia-smi` 與 framework peak allocation 觀測；RAM 用 process RSS；SSD 用 stage bytes/s。CPU/RAM/SSD 未知時，**不提供虛假的「全量 6 小時」承諾**。

## 7. 可恢復、可重跑、可切換的 build 設計

### 7.1 目錄與 checkpoint

```text
data/
  raw/frus/                         # pinned source checkout
  manifests/source_manifest.json
  chunks/<volume_id>.parquet
  embeddings/<model_revision>/<volume_id>.parquet
  build_state.jsonl
  indexes/frus_<sourceSHA>_<modelSHA>_<chunkConfig>/
  active_index.json
```

- 以 volume 為恢復單位；極大 volume 可再按 5,000 chunks 分片。
- 所有 Parquet 先寫 `.tmp`，關閉並驗證 row count/checksum 後再 atomic rename。
- `build_state.jsonl` 每筆包含 stage、partition、input checksum、output checksum、rows、開始／結束時間、model revision、狀態與錯誤。
- embeddings Parquet 是 source of truth；資料庫損壞或索引參數改變時，不必再跑 GPU。
- ingestion 用 stable `chunk_id` insert-only merge，重跑不重複；完成 ingestion 後才建 scalar、FTS 與 ANN index。
- 新版一律建在新的 versioned directory；完整驗證通過後只替換 `active_index.json` 指標，舊版至少保留到新版本展示完成。
- 任一 model revision、tokenizer、normalize、prefix 或 chunk config 改變，都產生新的 index version，不在舊向量上局部混寫。

### 7.2 中斷與 OOM 演練

- 人為在第 2 個 volume embedding 中止程序；重啟後應跳過 checksum 相同的完成分片，只重做不完整 `.tmp`。
- 人為把 batch 調到 OOM；捕捉錯誤、清理該 batch、batch 對半、從最後完成 partition 恢復。
- full build 不與 query service 同跑；若必須展示，query service 固定使用上一版 active index。
- 建置完成後比對：source documents coverage 100%、chunk ID duplicate=0、null vector=0、vector norm 異常=0、truncation=0。

## 8. 一天 MVP 與全量排程

### Day 1：可面試展示的 MVP（約 10 個有效工時）

| 時段 | 產物 | Gate |
|---|---|---|
| 0:00–1:00 | 固定 source/model revision；LanceDB、Ollama、CUDA smoke test | E5 能 encode；Qwen 能回答；Lance FTS/vector query 可跑 |
| 1:00–3:00 | TEI parser、schema、3 卷或至少 10k chunks | deterministic chunk IDs；truncation=0 |
| 3:00–4:30 | pilot embedding、三組 throughput/VRAM 測試 | 自動 OOM batch fallback；可重啟 |
| 4:30–5:30 | LanceDB FTS、vector、RRF、citation URL | BM25/dense/hybrid 都能獨立跑 |
| 5:30–7:00 | 先做 20 題 mini-qrels；決定是否保留 reranker | 產出 Recall/MRR/nDCG 表，不只 demo cherry-pick |
| 7:00–8:30 | Ollama evidence-grounded answer；中英文 UI/API | 每個主張有 document citation；無證據會拒答 |
| 8:30–9:30 | 中止恢復、OOM、空結果、壞 XML 演練 | 重啟不重複；錯誤可定位到 volume |
| 9:30–10:00 | 啟動全量 parse/embed job；保存 ETA 與資源報告 | ETA 來自 sample，不寫拍腦數字 |

### 全量工作（Day 1 後，時間由實測決定）

1. 全 repository manifest＋parse；產出確切 documents/chunks/tokens。
2. 按 volume 逐批 E5 embedding；每批完成即校驗與 checkpoint。
3. 從全部 Parquet 建全量 versioned LanceDB table。
4. 建 FTS，再建 `IVF_HNSW_SQ`；兩步分開記時間與容量。
5. 做 integrity suite、ANN exact-vs-index recall、60 題 retrieval evaluation。
6. 通過 gate 後切 `active_index.json`；保留 build report、失敗分片與環境 lockfile。

若 Day 1 結束時全量仍在跑，作品集要誠實顯示「已完成 X/Y volumes、實測 ETA、可恢復 checkpoint」，這比宣稱已全量但無 coverage proof 更有工程說服力。

## 9. 評估設計與驗收

### 9.1 60 題 retrieval qrels

- 20 題 exact fact：人物、日期、地點、文件編號。
- 15 題 thematic：政策理由、事件脈絡、概念變化。
- 10 題中文問→英文證據的 cross-lingual retrieval。
- 10 題 multi-document comparison。
- 5 題 corpus 不足以回答的 unanswerable。

每題至少有 gold `volume_id/document_id`；由官方 URL 人工核對。依序跑：BM25-only、dense-only、RRF hybrid、RRF＋reranker，報 Recall@5/10、MRR@10、nDCG@10。LanceDB 官方指出 dense 與 keyword scores 不宜直接比較、也沒有單一 reranker 永遠最佳，因此模型選擇須回到這組 qrels。[Hybrid evaluation](https://docs.lancedb.com/reranking/eval)（文件未標日期，查閱 2026-08-03）。

### 9.2 系統驗收矩陣

| 層 | 指標 | MVP 目標／規則 |
|---|---|---|
| 資料 | source coverage、duplicate、null、truncation | sample/full 對應階段 coverage 100%；其餘皆 0 |
| ANN | ANN Recall@10 vs flat exact | ≥0.95 為目標；<0.90 觸發 DB/索引改判 |
| Retrieval | Recall@10、MRR@10、nDCG@10 | 報四個 baseline；不預先保證絕對值 |
| Hybrid | 對最佳單路 nDCG@10 增益 | 至少 +0.02 才保留在該語言 route |
| Reranker | nDCG 增益與新增 p95 | 至少 +0.02 且新增 ≤1.5 秒才保留 |
| Citation | 引文 precision、claim coverage、URL 可開啟率 | precision ≥0.90；URL sample 100% |
| Refusal | 5 題 unanswerable | 不得捏造 FRUS 引文；目標 5/5 正確拒答 |
| Latency | query embed/retrieve/rerank/generate 分段 p50/p95 | 冷、熱分開；retrieval p95 目標 ≤1.5 秒、E2E warm ≤20 秒 |

所有數值門檻都是本專案預先登記的工程驗收值，不是外部 benchmark；硬體資料補齊後可在跑測試前修訂一次，之後不可為了讓結果好看而移動門柱。

## 10. 面試展示順序

1. 中文輸入一道跨語言問題，顯示 dense 與 BM25 各自候選、RRF 排名與 reranker 前後差異。
2. 展開一筆 evidence：原文、volume/document metadata、官方 URL、chunk 前後鄰文。
3. 顯示回答中的 citation 對應證據；再問一題資料外問題，展示拒答。
4. 打開 evaluation dashboard：四條 retrieval baseline、ANN recall、中文／英文拆分、p95。
5. 展示 build manifest 與中斷恢復紀錄，說明 GPU 為何離線／線上分時。

這樣面試重點會落在資料工程、IR baseline、可量化改進、資源約束與可靠性，而不是「把 LangChain 範例接上一個模型」。

## 11. 主張狀態與尚未知道的事

- **VERIFIED：** FRUS 官方性、公開下載／public domain、TEI repository、canonical volume/document IDs、E5 模型規格、LanceDB hybrid/FTS/vector/upsert 能力、Ollama context/併發控制，均有上列官方文件或原始論文。
- **UNVERIFIED：** 此特定 Windows 主機的 LanceDB wheel 相容性、CPU query latency、Qwen 7B 是否全 GPU、全量 chunks、全量磁碟與 wall time；只能由 smoke test 與 pilot 回答。
- **REFUTED：** 「8GB VRAM 可推出整套系統資源足夠」不成立；資料庫主要受 RAM/SSD、生成受 VRAM/context、建置受 CPU/GPU/IO 共同限制。
- **未查到：** 2026-08-03 當下官方 FRUS repository 的精確 document/token/chunk 數；官方對本機特定 Windows/Python/LanceDB 組合的相容性保證；此硬體的官方 E5/Qwen 吞吐 benchmark。

## 12. 來源帳本

本報告**沒有直接引文**；以上皆為來源內容的中文轉述或本案工程推論，兩者已用 VERIFIED／UNVERIFIED 與「工程門檻」區分。

| 來源 | 發布／更新日期 | 查閱日 | 用途 |
|---|---:|---:|---|
| [Office of the Historian — About FRUS](https://history.state.gov/historicaldocuments/about-frus) | 未查到 | 2026-08-03 | 官方定位、歷史範圍 |
| [FRUS FAQ](https://history.state.gov/about/faq/what-is-frus) | 未查到 | 2026-08-03 | 線上取得、public domain |
| [historyatstate/frus](https://github.com/historyatstate/frus) | release v1.0.18，2026-03-13 | 2026-08-03 | TEI 主檔、ID 穩定性、授權 |
| [Developer Resources](https://history.state.gov/developer) | 未查到 | 2026-08-03 | 每卷單一 TEI、repo/API |
| [December 2018 HAC](https://history.state.gov/about/hac/December-2018) | 2018-12 | 2026-08-03 | 512 卷歷史快照 |
| [September 2024 HAC](https://history.state.gov/about/hac/September-2024) | 2024-09 | 2026-08-03 | repository 涵蓋範圍 |
| [KG-FRUS paper](https://arxiv.org/abs/2311.01606) | 2023-11-03 | 2026-08-03 | 30 萬以上文件的量級下界 |
| [mE5 paper](https://arxiv.org/abs/2402.05672) | 2024-02-08 | 2026-08-03 | 多語 embedding 研究來源 |
| [multilingual-e5-small model card](https://huggingface.co/intfloat/multilingual-e5-small) | 未查到 | 2026-08-03 | 參數、維度、長度、prefix |
| [mMARCO MiniLM reranker card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) | 未查到 | 2026-08-03 | reranker 規格／授權 |
| [Sentence Transformers efficiency](https://sbert.net/docs/sentence_transformer/usage/efficiency.html) | 未查到 | 2026-08-03 | FP16 與推論調校 |
| [LanceDB Quickstart](https://docs.lancedb.com/quickstart) | 未查到 | 2026-08-03 | embedded OSS |
| [LanceDB Full-text Search](https://docs.lancedb.com/search/full-text-search) | 未查到 | 2026-08-03 | BM25/positions |
| [LanceDB Hybrid Search](https://docs.lancedb.com/search/hybrid-search) | 未查到 | 2026-08-03 | hybrid/RRF |
| [LanceDB Vector index](https://docs.lancedb.com/indexing/vector-index) | 未查到 | 2026-08-03 | IVF/HNSW/SQ、容量與調參 |
| [LanceDB Updating data](https://docs.lancedb.com/tables/update) | 未查到 | 2026-08-03 | merge_insert/upsert |
| [Qdrant Hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/) | 未查到 | 2026-08-03 | dense/sparse/RRF |
| [Qdrant Capacity planning](https://qdrant.tech/documentation/guides/capacity-planning/) | 未查到 | 2026-08-03 | RAM 估算與 headroom |
| [OpenSearch Docker](https://docs.opensearch.org/latest/install-and-configure/install-opensearch/docker/) | 約 2026-07 更新 | 2026-08-03 | Windows/Docker 維運成本 |
| [OpenSearch k-NN engines](https://docs.opensearch.org/latest/mappings/supported-field-types/knn-methods-engines/) | 未查到 | 2026-08-03 | HNSW 記憶體模型 |
| [Haystack InMemoryDocumentStore](https://docs.haystack.deepset.ai/docs/inmemorydocumentstore) | 約 2026-07 更新 | 2026-08-03 | 實驗用途限制 |
| [Ollama Qwen2.5 library](https://registry.ollama.com/library/qwen2.5) | 頁面標示約 1 年前更新 | 2026-08-03 | 模型尺寸、多語、context |
| [Ollama Context length](https://docs.ollama.com/context-length) | 未查到 | 2026-08-03 | VRAM/context 關係 |
| [Ollama FAQ](https://docs.ollama.com/faq) | 未查到 | 2026-08-03 | keep-alive、模型數與平行數 |

