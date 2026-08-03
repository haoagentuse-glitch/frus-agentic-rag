# FRUS 全量雙語 RAG：AI Engineer 面試作品集執行計畫

版本：2026-08-03  
目標：以一台 RTX 4060 Laptop 8GB，在本機完成可重建、可評估、能中英文問答的全量 FRUS RAG。  
產品暫名：**FRUS Evidence Explorer｜美國外交史證據檢索器**

---

## 1. 結論先行

### 難度判定

**VERIFIED｜可以做，整體難度中等偏難（約 6.5/10）。**

困難不在於 FRUS 是否能下載，也不在於 8GB VRAM 能否存下整個資料庫。向量資料放在 SSD／資料庫，GPU 只負責分批計算 embedding 與生成答案。真正的難點是：

1. 正確解析跨年代、結構略有差異的 TEI/XML，並保留可靠引用。
2. 對 30 萬份以上文件做可中斷、可續跑的全量切塊與 embedding。
3. 證明 BM25、dense retrieval、hybrid 和 reranker 哪一個真的有效，而不是只展示一次好看的問答。

**UNVERIFIED｜無法在尚未實測前承諾「一天內完成全量索引＋評估＋介面」。** 官方與研究資料足以確認資料是 30 萬份文件以上的量級，但目前已發布正文的精確文件／切塊數與這台機器的 embeddings/s 尚未量完。一天應承諾的是：完成可展示 MVP、建立全量可恢復工作流、跑出實測 ETA，並啟動全量索引；全量何時結束由 10k／50k chunk benchmark 決定。

### 三個預登記失敗條件

| 失敗條件 | 廉價驗證 | 觸發後改判 |
|---|---|---|
| TEI provenance 不可靠 | 六年代分層抽 100 chunks 回查官方 URL、parent document 與 footnote | canonical resolution < 99%、任何 planned stub 成為證據，立即停止全量 embedding，先修 parser |
| 全庫 dense 成本超過作品集價值 | 10% 分層 build 外推時間／磁碟，並比較 cross-era retrieval | 推估 >24 小時或超出磁碟安全線，仍保留全已發布文件的 BM25＋metadata；dense 改分年代路由，不再聲稱「全庫 dense」 |
| Hybrid／reranker 沒有客觀收益 | 60 題 qrels 比 BM25、dense、RRF、RRF＋reranker | nDCG@10 未比最佳單路高 0.02 就移除該層；Qwen 7B 若 warm p95 >20 秒則縮 context，仍失敗改 1.5B |

---

## 2. 為什麼 FRUS 適合這個作品集

**VERIFIED｜題材夠專業。** Foreign Relations of the United States 是美國政府自 1861 年起對重大外交政策決策與活動的官方文件紀錄，適合處理跨文件、跨年代、人物關係、政策理由與證據追溯問題。[Office of the Historian — About FRUS](https://history.state.gov/historicaldocuments/about-frus)

**VERIFIED｜取得與授權簡單。** Office of the Historian 的官方 GitHub repository 提供每卷一份的 TEI P5 XML；內容屬 public domain，不需要爬網站，也沒有反爬蟲依賴。[historyatstate/frus](https://github.com/historyatstate/frus)、[Developer Resources](https://history.state.gov/developer)

**VERIFIED｜規模足以需要 RAG。** KG-FRUS 論文描述的衍生資料已超過 30 萬份外交文件；官方 2021 年資料則曾描述約 550 卷、31 萬份文件。這是量級證據，不當成 2026 repository 的精確計數。[KG-FRUS](https://arxiv.org/abs/2311.01606)、[August 2021 HAC](https://history.state.gov/about/hac/August-2021)

### 2026-08-03 現場盤點

固定官方 commit `4b4c402f0cce25144ded2198ba9566b6c37c7c49` 後，實際 sparse checkout 的 `volumes/` 有：

- **694 個 XML entries**。
- **3,379,455,993 bytes，約 3.15 GiB**。
- 官方 ebooks 頁面目前列出 **552 卷**；快照也正好有 **552 個 XML 含 `<div type="document">`**。
- **142 個 XML 不含 document div**；其中 141 個小於 10 KB，部分是 planned placeholder。例如 `frus1993-00v48.xml` 只有狀態資訊，沒有歷史文件正文。
- 快照共有 **314,483 個 `type="document"` div**；這個基準包含 historical document、editorial note 等 document subtype，不能直接當成最終主正文數。

因此「694 XML = 694 卷可向量化正文」已被**反證**。694 是需要 inventory 的來源檔數，不是 published volume 或 historical-document 數。

**範圍誠實說明。** FRUS 是外交史一手／編纂史料，不是哲學經典資料庫。它的「史哲」價值在於研究自由、人權、現實主義、主權、正義等概念如何進入實際政策語言；不能把它包裝成純哲學問答系統。

### 面試故事

> 我不是做另一個 PDF Chat。我把 160 多年的官方外交檔案做成可重建的混合檢索系統，讓使用者用中文問英文史料，並把每個答案追溯到 FRUS 的卷、文件與原文證據；同時用 qrels 證明每一層檢索設計是否有用。

可展示的問題：

- 「古巴飛彈危機期間，美方對封鎖與空襲各自有哪些理由？」
- 「不同時期的美國外交文件如何使用 freedom、human rights 與 sovereignty？」
- 「冷戰時期文化、展覽、廣播與 UNESCO 如何被當作外交工具？」
- 「比較兩個政府對中國承認問題的政策理由，列出原始文件。」
- 資料外問題：要求系統承認證據不足，而不是捏造引用。

---

## 3. 「全做」的資料範圍

### 原始層全部納入

- pinned release／commit 內 `volumes/` 的 694 個 TEI/XML 全部進 manifest 並嘗試解析；552 個已發布卷是正文 ingest 範圍。
- 已發布的 `tei:div[@type='document'][@subtype='historical-document']` 全部進主文件 corpus，包括電報、會議紀錄、備忘錄與信函。
- volume、chapter、document、日期、地點、人物、terms、來源註、分類等可解析 metadata。
- front matter、目錄、人物索引與編者說明也解析保存，但以 `kind` 隔離。

### 不混入回答證據

- planned placeholder、errata、目錄項、索引詞與純排版文字不和正文放在同一 evidence pool。
- 掃描圖像不是本版 RAG 的主要來源；官方 TEI 文字才是 source of truth。
- 不把全庫翻譯成中文再索引。雙語指中英文 query／answer，英文原文保持不變。

### 完整性的定義

「全量完成」必須同時滿足：

- manifest 中 694 個 TEI entries 都有 parse 結果或具名錯誤。
- 已發布 historical-document coverage = 100%；planned stub 得到 0 documents 是正確結果。
- duplicate chunk ID = 0、null vector = 0、靜默 truncation = 0。
- 每筆證據能組回官方 `volume_id/document_id` URL。
- `source_commit`、模型 revision、parser version、chunk config 都可追溯。

---

## 4. 硬體評估與資源策略

已在本機確認：

| 資源 | 實際規格 | 判定 |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB | 足以分批跑 0.1B embedding 模型，也能跑 7B Q4 生成；不應同時常駐 |
| RAM | 31.7 GiB | 足以做 streaming parse 與內嵌式 disk-first DB；仍需量測全量索引峰值 |
| CPU | i5-13420H，12 logical processors | XML、Parquet、FTS 與 CPU query 可行；不是主要生成裝置 |
| SSD | workspace 所在磁碟有充足可用空間 | 完整作品先預留 20–25 GiB；sample 外推超過可用空間 80% 時停止 |

### GPU 分時

1. **離線建索引：**卸載 Ollama，只把 `multilingual-e5-small` 放 GPU。
2. **建立資料庫索引：**GPU 釋放，工作由 CPU／SSD 完成。
3. **線上展示：**Qwen 7B Q4 使用 GPU；query embedding 與 reranker 先走 CPU。

這避免 E5、reranker、Qwen 同時爭搶 8GB VRAM。

### 向量容量

使用 384 維 float32 時：

`raw vector bytes = chunks × 384 × 4`

- 100k chunks 約 146.5 MiB raw vectors。
- 1M chunks 約 1.43 GiB raw vectors。

**UNVERIFIED｜以 512-token／64-overlap 的初始假設，全量約 65–100 萬 chunks。** 對應 384 維 raw vectors 約 0.93–1.43 GiB。實際磁碟還要加 3.15 GiB XML、原文、metadata、FTS、ANN index、checkpoints 與版本備份，因此完整作品先預留 **20–25 GiB**，再用 50k sample 實測比例覆寫估算。

---

## 5. 選定架構

```text
官方 FRUS TEI/XML
  ↓ pinned commit + manifest + checksum
document-aware parser
  ↓ 每卷 Parquet，可恢復 checkpoint
multilingual-e5-small（CUDA FP16，離線分批）
  ↓ embedding Parquet：向量的可重建來源
LanceDB OSS
  ├─ BM25 / FTS
  ├─ 384d dense vector index
  └─ metadata / scalar indexes
  ↓ Reciprocal Rank Fusion（RRF）
選配 multilingual cross-encoder reranker
  ↓ top 4–5 evidence chunks
Qwen 2.5 7B Q4（Ollama，4k context）
  ↓
中／英文答案 + FRUS 卷／文件引用 + 原文 evidence panel
```

### 為什麼選 LanceDB

- 內嵌式，不需要 Docker、JVM 或另一個常駐服務。
- 同時有 FTS/BM25、向量搜尋、hybrid/RRF、index 與 upsert。
- disk-first 架構較適合 32GB RAM 的單機作品集。

**備援不是裝飾：** LanceDB scale gate 失敗就換 Qdrant on-disk；只有要展示企業級搜尋維運時才換 OpenSearch。Haystack 可做 orchestration，但 InMemoryDocumentStore 不作全量 FRUS 儲存層。

### 為什麼不是只做向量搜尋

- BM25 擅長人名、地名、文件編號、罕見專有詞。
- Dense retrieval 擅長中文問英文，以及語意相近但字面不同的問題。
- RRF 能融合兩路名次，不需要硬比較不可直接對齊的分數。
- Reranker 只在 qrels 證明有收益後保留。

補充：BM25 本身就是 lexical retrieval 的核心演算法，不是「BM25 與 lexical」兩個平行技術。

---

## 6. 雙語檢索設計

FRUS 原文以英文為主，因此中文 query 沒有足夠 lexical overlap，不能直接期待英文 BM25 有效。

### 英文問題

`original query → BM25 + multilingual E5 dense → RRF`

### 中文問題

並行兩路：

1. 原中文 query → multilingual E5 → 英文 passages。
2. Qwen 產生短英文檢索式／關鍵詞 → BM25。

再以 RRF 合併。UI 同時顯示原問題、英文 lexical query、兩路候選與融合後排名，讓跨語言設計可觀察。若中文 hybrid 在 qrels 沒有比 dense-only 好 0.02 nDCG@10，就改為中文 dense-only。

---

## 7. 解析與切塊規格

### 原則

- 絕不跨 `document_id` 切塊。
- 先保留段落、標題、日期、發文／收文地、人物與 footnote 關係，再 token-aware packing。
- 起始值為最多 512 E5 tokens、overlap 64；以段落優先 packing，不用固定字元數。prefix 與特殊 token 必須包含在 token budget 內。
- 短腳註跟隨正文；長腳註另成 `kind=note`，保存 `parent_chunk_id`。
- front matter、目錄、編者說明另標 `kind`，預設不當正文證據；人物與術語 ID 只在卷內有效，不冒充全系列 canonical entity ID。
- 日期保留 `min/max + precision`；出版年與文件日期分開，range query 使用 interval overlap。
- embedding 前 truncation count 必須為 0；超長內容回 parser 重切。

### 最小 schema

```text
chunk_id
volume_id / document_id / chapter_id
section_path / kind / parent_chunk_id
date_raw / persons / places / terms
text / token_count
source_url / source_commit / source_sha
parser_version / chunk_config
embedding_model_revision
```

`chunk_id` 由來源 ID、位置與 normalized text hash 決定；相同設定重跑必須產生相同 ID。官方 repository 指出 volume/document ID 是穩定引用，page-break ID 可能改變，因此 page ID 不作唯一主鍵。

---

## 8. 模型與起始參數

### Embedding

`intfloat/multilingual-e5-small`

- 約 0.1B、384 維、最大 512 tokens、MIT。
- passage 必加 `passage:`；query 必加 `query:`；向量 normalize。
- 全量：CUDA FP16，batch 32 起測；OOM 時 32 → 16 → 8。
- 線上 query：CPU FP32，batch 1；若 p95 不達標再測 ONNX。
- 全量每卷先寫 embeddings Parquet，再匯入 DB；更換索引參數不用重跑 GPU。

來源：[multilingual-e5-small model card](https://huggingface.co/intfloat/multilingual-e5-small)、[mE5 paper](https://arxiv.org/abs/2402.05672)

### Reranker（選配）

`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

- CPU、batch 8，RRF top 30 → rerank top 8。
- 不參與離線 indexing。
- nDCG@10 增益未達 0.02，或新增 p95 超過 1.5 秒就刪除。

### Generator

`qwen2.5:7b`，Ollama Q4，`num_ctx=4096`，parallel=1，常駐模型數=1。

- 每次只送 top 4–5 chunks。
- 每個主要 claim 綁定 `[volume_id / document_id]`。
- 無足夠證據時明確拒答。
- 若 8GB 實測失敗，縮短 context；仍失敗改 `qwen2.5:1.5b`。

---

## 9. 可恢復的全量建置

```text
data/
  raw/frus/                              # pinned source
  manifests/source_manifest.json
  chunks/<volume_id>.parquet
  embeddings/<model_revision>/<volume_id>.parquet
  build_state.jsonl
  indexes/frus_<source>_<model>_<chunk_config>/
  active_index.json
```

### 每個 stage 的規則

1. 以 volume 為 checkpoint；過大 volume 再切每 5,000 chunks。
2. 先寫 `.tmp`，校驗 rows/checksum 後 atomic rename。
3. `build_state.jsonl` 記錄輸入 checksum、輸出 checksum、rows、耗時、模型版本與錯誤。
4. embeddings Parquet 是向量 source of truth；DB 或 ANN 損壞不重跑 embedding。
5. 新索引建在 versioned directory，驗證通過才切換 `active_index.json`。
6. 模型、tokenizer、prefix、normalize 或 chunk config 改變，一律建立新版本，不混寫舊向量。

### 必做故障演練

- 在第二卷 embedding 時中止；重啟應跳過完成分片。
- 故意把 batch 調到 OOM；系統應自動減半並由上一 checkpoint 恢復。
- 放入壞 XML；錯誤必須指到 volume，不能讓整批靜默缺資料。
- build 與 demo 分離；展示服務固定使用上一版 active index。

---

## 10. 先實測，再承諾全量時間

第一個 benchmark 產物為：

```json
{
  "source_commit": "...",
  "tei_files": 0,
  "documents": 0,
  "chunks": 0,
  "tokens_p50": 0,
  "tokens_p95": 0,
  "truncation_count": 0,
  "parse_docs_per_sec": 0,
  "embed_chunks_per_sec": 0,
  "embed_tokens_per_sec": 0,
  "peak_vram_mib": 0,
  "estimated_full_hours": 0
}
```

程序：

1. 10k chunks 暖機。
2. 跑三組 50k sample，分別記錄 parse、embed、ingest、FTS、ANN。
3. 保存 best／median／worst，不只拿最快一次。
4. 用全量 documents/chunks 與 median throughput 外推，加 30% 重試與長尾 buffer。

若全量落在 65–100 萬 chunks，要在 8 小時內只完成 dense 階段，持續端到端吞吐至少約 **23–35 chunks/s**。這是算術門檻，不是 4060 的預測效能。

`T_est = 1.30 × (parse + embed + ingest + FTS + ANN)`

**只有這個輸出能回答「全量要跑多久」。**

---

## 11. 一天執行表

一天指約 10 個有效工時；目標是可面試展示並正式啟動全量工作，不把未完成的全量假裝完成。

| 時段 | 工作與產物 | Gate |
|---|---|---|
| 0:00–1:00 | 固定 source/model revision；LanceDB、E5、Ollama smoke test | 三項都能在本機啟動 |
| 1:00–2:30 | manifest、TEI parser、六年代 fixtures、10k chunks | deterministic IDs、truncation=0、planned stub 不進正文 |
| 2:30–4:00 | 目標完成 552 卷 inventory／parse 與全量 BM25 | 552/552 有狀態；314,483 document-div 基準差異可解釋 |
| 4:00–5:15 | 三組 dense benchmark、OOM fallback、全量 ETA | 有 chunks/s、tokens/s、peak VRAM；啟動 checkpointed dense |
| 5:15–6:15 | 對目前 dense coverage 建 BM25、dense、RRF、citation URL | 三種檢索可獨立執行 |
| 6:15–7:15 | 20 題 mini-qrels，先做 retrieval baseline | 有 Recall/MRR/nDCG，不靠主觀 demo |
| 7:15–8:30 | Qwen grounded answer、中英文 API／UI | 引用可點回官方文件、無證據拒答 |
| 8:30–9:20 | 中止恢復、OOM、壞 XML、空結果演練 | 重啟不重複、錯誤可定位 |
| 9:20–10:00 | README、架構圖、benchmark report；全量 dense 背景續跑 | 顯示 X/552 volumes、ETA、checkpoint |

### Day 1 後的全量完成流程

1. 若 Day 1 未完成，補齊全 repository parse 與全量 BM25。
2. 對 552 個已發布卷逐卷 E5 embedding 並 checkpoint。
3. 從 Parquet 建全量 LanceDB table、FTS 與 ANN。
4. 跑 integrity suite 與 ANN exact-vs-index recall。
5. 擴充為 60 題 qrels，完成四條 retrieval baseline。
6. 全部 gate 通過後切換 active index、錄製完整 demo。

若 Day 1 結束時仍在跑，README 必須寫「完成 X/Y volumes、實測 ETA、最後 checkpoint」；這比沒有 coverage 證據的「全量完成」更像成熟 AI Engineer。

---

## 12. 評估設計

### 60 題 qrels

| 類型 | 題數 | 目的 |
|---|---:|---|
| 人物、日期、地點、文件編號等 exact fact | 15 | 測 BM25 與 metadata |
| 政策理由、事件脈絡、概念變化 | 15 | 測 dense 與語意檢索 |
| 中文問題 → 英文證據 | 10 | 測跨語言 retrieval |
| 跨文件／跨政府比較 | 10 | 測 multi-document retrieval |
| 文化外交、藝術、宣傳、UNESCO | 5 | 維持史藝特色 |
| unanswerable | 5 | 測拒答與引用誠實性 |

每題至少人工核對一個 gold `volume_id/document_id`，並依年代分層抽樣，避免評估只集中在冷戰。

### 必報 baseline

1. BM25-only。
2. dense-only。
3. BM25 + dense + RRF。
4. RRF + reranker。

### 指標與門檻

| 層 | 指標 | 規則 |
|---|---|---|
| 資料 | coverage、duplicate、null、truncation | coverage 100%；其餘 0 |
| ANN | Recall@10 vs flat exact | 目標 ≥0.95；<0.90 觸發 DB 改判 |
| Retrieval | Recall@5/10、MRR@10、nDCG@10 | 中／英文與題型分開報 |
| Hybrid | 對最佳單路 nDCG@10 | 至少 +0.02 才保留 |
| Reranker | nDCG 增益／新增 p95 | +0.02 且 ≤1.5 秒才保留 |
| Citation | precision、claim coverage、URL 開啟率 | precision ≥0.90；抽樣 URL 100% |
| Refusal | 5 題 unanswerable | 目標 5/5 不捏造引用 |
| Latency | retrieve/rerank/generate p50/p95 | 冷、熱分開；不只報 E2E 平均 |

---

## 13. 介面與面試展示順序

介面只需要四區：

1. 中／英文問題輸入。
2. 回答與 inline citations。
3. evidence panel：英文原文、volume/document、日期、官方 URL、相鄰段落。
4. debug/evaluation panel：BM25、dense、RRF、rerank 排名與 latency。

展示順序：

1. 用中文問跨語言問題，顯示中文 dense 與英文 lexical query 的候選差異。
2. 展開一筆 evidence，點回官方 FRUS 文件。
3. 切換 BM25／dense／hybrid，展示量化結果而非口頭宣稱。
4. 問一題 corpus 外問題，展示拒答。
5. 打開 build report，說明 8GB GPU 分時、checkpoint 與全量 coverage。

---

## 14. Repository 交付結構

```text
frus-rag/
  README.md
  pyproject.toml
  configs/
    source.yaml
    chunking.yaml
    models.yaml
  src/
    ingest/manifest.py
    ingest/parse_tei.py
    ingest/chunk.py
    index/embed.py
    index/build_lancedb.py
    retrieval/bm25.py
    retrieval/dense.py
    retrieval/hybrid.py
    retrieval/rerank.py
    generation/answer.py
    api/main.py
    ui/app.py
  eval/
    qrels.jsonl
    run_retrieval_eval.py
    run_answer_eval.py
  tests/
    fixtures/
    test_parser.py
    test_chunking.py
    test_resume.py
    test_citations.py
  reports/
    corpus_stats.json
    benchmark.json
    retrieval_metrics.json
  data/                     # gitignored
```

README 首頁只呈現：問題、為何需要 hybrid RAG、架構圖、實測硬體、資料規模、四條 baseline、範例問答、限制與重建指令。

---

## 15. 最終驗收

作品集只有在以下項目具備證據時才算完成：

- [ ] 官方來源 commit／release 已固定，manifest 與 checksums 已保存。
- [ ] 全部 TEI 有 parse 狀態，source document coverage = 100%。
- [ ] 全量 embedding 能 checkpoint、resume，OOM 演練通過。
- [ ] BM25、dense、RRF 可獨立查詢，citation URL 可回到官方文件。
- [ ] 60 題 qrels 與四條 baseline 已保存，不只展示 cherry-picked 問題。
- [ ] 中英文問題可用；中文 lexical expansion 在 UI 可觀察。
- [ ] unanswerable 不捏造 FRUS 引文。
- [ ] README 報告實際 documents、chunks、build time、disk、VRAM、p50/p95。
- [ ] 未達門檻的 reranker／hybrid route 已誠實移除或標成負結果。

---

## 16. 主張狀態

- **VERIFIED：** FRUS 的官方性、public-domain／直接下載、TEI repository；固定快照的 694 XML、552 個含 document div 的卷、314,483 個 document div 與約 3.15 GiB checkout；本機 RTX 4060 8,188 MiB、31.7 GiB RAM、i5-13420H；E5、LanceDB、Ollama 的上列官方能力。
- **UNVERIFIED：** historical-document subtype 的最終正文／chunk 數（初估 65–100 萬 chunks）、這台機器的全量 build 時數、Qwen 7B 實際 p95、LanceDB 全量 ANN recall；都必須由本計畫的正式 parser／benchmark 回答。
- **REFUTED：** 「8GB VRAM 放不下全量 FRUS，所以不能做」不成立；全量資料與索引主要在 SSD，embedding 可分批。反過來，「有 8GB VRAM，所以一天一定做得完」也不成立。

## 主要來源

- [Office of the Historian — About FRUS](https://history.state.gov/historicaldocuments/about-frus)
- [Office of the Historian — FRUS Ebooks](https://history.state.gov/historicaldocuments/ebooks)
- [Office of the Historian — Citing FRUS](https://history.state.gov/historicaldocuments/citing-frus)
- [Official FRUS TEI repository](https://github.com/historyatstate/frus)
- [Office of the Historian — Developer Resources](https://history.state.gov/developer)
- [KG-FRUS paper](https://arxiv.org/abs/2311.01606)
- [multilingual-e5-small model card](https://huggingface.co/intfloat/multilingual-e5-small)
- [LanceDB Quickstart](https://docs.lancedb.com/quickstart)
- [LanceDB Full-text Search](https://docs.lancedb.com/search/full-text-search)
- [LanceDB Hybrid Search](https://docs.lancedb.com/search/hybrid-search)
- [Ollama Qwen2.5 library](https://registry.ollama.com/library/qwen2.5)
