# RAG 搭建方法與 FRUS 是否使用 LangChain：Scout 報告

檢索日期：2026-08-03  
問題：RAG 專案通常有哪些搭建方式？對一日完成、全量 FRUS、BM25＋BGE-M3／dense＋RRF 的面試作品，為何可能先不用 LangChain？  
動機：區分可查證的框架能力與本案工程取捨，不把個人偏好說成產業事實。

## 結論三句

1. **已確立：**RAG 並不等於 LangChain；共同骨架仍是 ingestion／chunking／embedding／store／retrieval／rerank／generation，而實作可選直接 SDK、自寫 pipeline，或 LangChain、LlamaIndex、Haystack 等編排框架。
2. **已確立＋工程判斷：**FRUS 第一版是固定、可預測的 2-step hybrid RAG；LanceDB 已直接提供 BM25、向量搜尋與預設 RRF，Sentence Transformers 直接提供 query／document embedding，所以一日版本沒有必須由 LangChain 補上的核心能力。
3. **發展中：**先不用 LangChain 是縮小一天內的依賴與除錯面，不是說 LangChain 不適合 RAG；若後續加入多工具、動態路由、迴圈驗證、持久狀態或跨供應商切換，再導入 LangChain／LangGraph會更有價值。

## 三態地圖

### 已確立

- LangChain 官方把 RAG 分成 2-Step、Agentic 與 Hybrid；其中 2-Step 的檢索固定發生在生成之前，控制高、延遲較可預測。來源：LangChain Retrieval 官方文件，頁面未標發表日，2026-08-03 擷取。  
  https://docs.langchain.com/oss/python/langchain/retrieval
- LangChain 的價值之一是標準化 document loader、splitter、embedding、vector store、retriever 等介面，便於替換元件。來源同上。
- LangChain／LangGraph 1.0 已採 semantic versioning 且被指定為 LTS；不能再把「一定頻繁 breaking」當作拒用理由。官方同時指出第三方／部分 partner packages 的穩定政策可能不同。來源：LangChain Release Policy，頁面未標發表日，2026-08-03 擷取。  
  https://docs.langchain.com/oss/python/release-policy
- LlamaIndex 以資料索引、nodes、retrievers、query engine、response synthesis 為中心，也同時提供高階 API 與低階組合 API。來源：LlamaIndex 官方文件，頁面未標發表日，2026-08-03 擷取。  
  https://developers.llamaindex.ai/python/framework/getting_started/concepts/  
  https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/
- Haystack 3.0 以可重用 components 與 pipelines 組裝 production RAG；pipeline 原生支援 branches、loops、async／parallel execution，且列出 BM25、embedding、hybrid retrievers 與各類 rankers。來源：Haystack 3.0 官方文件，頁面未標發表日，2026-08-03 擷取。  
  https://docs.haystack.deepset.ai/docs/intro  
  https://docs.haystack.deepset.ai/docs/pipelines  
  https://docs.haystack.deepset.ai/docs/retrievers  
  https://docs.haystack.deepset.ai/docs/rankers
- LanceDB Python API 原生支援 `vector`、`fts`、`hybrid` 查詢；FTS 以 BM25 排序，hybrid 查詢預設以 RRF 合併結果。來源：LanceDB Python 官方 API，頁面未標發表日，2026-08-03 擷取。  
  https://lancedb.github.io/lancedb/python/python/
- Sentence Transformers 官方建議資訊檢索任務分別使用 `encode_query()` 與 `encode_document()`；這表示 embedding 階段可直接使用模型 SDK，不需經過 RAG 框架。來源：Sentence Transformers 官方文件，頁面未標發表日，2026-08-03 擷取。  
  https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html

### 發展中／工程推論

- 「直接 SDK 會比 LangChain 更容易除錯」不是普遍定律；對本案成立的理由是步驟少、資料格式特殊，而且需要直接觀察 BM25、dense、RRF 各路輸出。若流程擴大，框架提供的 tracing、state、routing 反而可能降低維護成本。
- 「框架有額外 overhead」在依賴數與抽象層數上成立，但本次未找到可跨框架、可重現且適用 FRUS 的 latency／memory benchmark；不應宣稱 LangChain 明顯拖慢查詢。
- 「業界最常用 LangChain」或各框架的可靠市占排序，本次未查到具方法透明、可比較的權威資料；只能說它們都是有正式文件與持續版本的主流選項，不能排名。

### 未查到

- 未查到同一份 FRUS corpus、同一硬體、同一模型下，直接 SDK、LangChain、LlamaIndex、Haystack 的公平效能比較。
- 未查到能證明面試官普遍偏好某一框架的可靠調查。

## 常見搭建方法

| 方法 | 實際做法 | 優點 | 代價 | 適合情境 |
|---|---|---|---|---|
| 直接 SDK／自寫 deterministic pipeline | 自己寫資料 schema 與函式；直接呼叫 lxml、Sentence Transformers、LanceDB、模型 server | 控制透明、依賴少、容易針對領域資料客製、各階段方便做 ablation | 介面、observability、retry、state 等需自己補 | 固定 2-step RAG、一天 MVP、特殊 TEI 資料 |
| LangChain＋必要時 LangGraph | 用標準 Document／Retriever／Tool 介面；複雜流程以 graph nodes、edges、checkpoint 編排 | 整合面廣、容易換供應商，適合 tools、agents、routing、stateful workflow | 多一層抽象與套件；客製資料語意仍需自行處理 | 多來源、多工具、agentic RAG、需要 tracing／state |
| LlamaIndex | 用 Document／Node、index、retriever、query engine 與 response synthesizer 組裝 | 資料索引與 retrieval 抽象完整，高／低階 API 都有 | 須理解其資料模型；領域 parser 仍可能客製 | 文件／知識庫導向、複合 index 與 retriever |
| Haystack | 用明確 components 建 indexing/query pipeline DAG，再接 retriever、joiner、ranker、generator | Pipeline 結構清楚；branches、loops、async 與 hybrid 元件完整 | 初始元件與 pipeline 設定較多 | 重視顯式 pipeline、部署與元件測試的 RAG |
| 託管式 RAG／雲端 knowledge base | 將解析、索引、向量庫或生成委由雲服務 | 上線快、少維運 | 成本、資料外傳、供應商綁定、可控性較低 | 企業交付，不適合本案「開源地端模型」主敘事 |

> 上表是能力與工程情境比較，不是市占排名。

## 為什麼 FRUS 第一版先不用 LangChain

### 1. 問題是固定管線，不是 agent 決策問題

本案的 query path 可明確寫成：

```text
中／英 query
  -> 正規化與英文 lexical query
  -> BM25 top-k + dense top-k
  -> RRF
  -> optional reranker
  -> context packing
  -> local LLM answer
  -> FRUS citation validation
```

LangChain 官方也把固定 retrieve-then-generate 列為 2-Step RAG，並描述為 straightforward、predictable。本案暫時不需要 LLM 判斷「要不要檢索」或「該呼叫哪個工具」，因此 agent framework 不會提高核心檢索品質。

### 2. 資料的難點在 FRUS TEI，而不是通用 loader

FRUS 需要保留 volume、document、chapter、date、persons、footnotes、canonical URL 與 source commit，並排除 planned／front matter 對正文檢索的污染。這些是領域規則，即使使用框架仍要自行寫 parser、schema、checkpoint 與驗證。第一天先把這些面試價值最高的部分做對，比把資料轉成某框架的通用 `Document` 更重要。

### 3. 底層元件已原生涵蓋核心能力

- BGE-M3／E5：Sentence Transformers 直接批次產生 document 與 query vectors。
- BM25、vector、hybrid、RRF：LanceDB 直接提供。
- reranker：直接使用 CrossEncoder／FlagEmbedding 或容器內現有模型。
- generation：直接呼叫 Ollama、llama.cpp 或 OpenAI-compatible local endpoint。

因此 LangChain 在第一版主要提供「統一包裝與編排」，不是缺少它就無法完成的演算法。

### 4. 面試展示需要露出檢索工程，而非只展示 chain

直接 pipeline 可以清楚輸出：

- BM25 與 dense 各自 top-k；
- RRF 前後排名；
- metadata filters；
- 每階段 latency；
- citation 到 FRUS document ID 的對應；
- BM25-only、dense-only、hybrid 的離線評估。

這是本案的工程敘事取捨；不是宣稱使用 LangChain 就不能觀察這些資訊。

## 何時應改用 LangChain／LangGraph

符合任一條，就重新評估導入：

1. 新增 FRUS、網路搜尋、SQL、人物知識圖譜等多種工具，讓 LLM 動態決定路徑。
2. 需要 query rewrite -> retrieve -> grade -> retry -> answer validation 的條件分支或迴圈。
3. 需要可持久化的對話狀態、人工審核、暫停／恢復工作流。
4. 同時支援多家 LLM／embedding／vector store，替換供應商成為常態需求。
5. 團隊已統一採用 LangSmith／LangGraph observability 與 deployment stack。

LangChain 官方 custom workflow 文件明示 LangGraph 可混合 deterministic nodes 與 agent nodes，適合上述擴充；頁面未標發表日，2026-08-03 擷取：  
https://docs.langchain.com/oss/python/langchain/multi-agent/custom-workflow

## 對 FRUS repo 的具體建議

維持「無 LangChain 依賴，但保留未來可包裝性」：

```python
class Embedder(Protocol):
    def encode_documents(self, texts: list[str]) -> ndarray: ...
    def encode_query(self, text: str) -> ndarray: ...

class Retriever(Protocol):
    def search(self, query: str, *, top_k: int, filters: dict | None = None) -> list[Hit]: ...

class Generator(Protocol):
    def answer(self, query: str, contexts: list[Hit]) -> Answer: ...
```

這些介面後續能被包成 LangChain Retriever／Tool、LlamaIndex Retriever 或 Haystack Component，不需要重寫 FRUS parser 和評估資料。

## 引述與概括分離

### 短引述

- LangChain 對 2-Step RAG：`Retrieval always happens before generation. Simple and predictable.`（官方文件；2026-08-03 擷取）
- LlamaIndex 對 retriever：`responsible for fetching the most relevant context given a user query`（官方文件；2026-08-03 擷取）
- Haystack 3.0：`Build pipelines using reusable components`（官方文件；2026-08-03 擷取）

### 本報告概括／推論

- 「FRUS 一日版先直寫」是依據固定管線、TEI 客製規則與 LanceDB 原生能力做出的工程推論，不是上述框架官方的推薦。
- 「未來需求達到多工具／分支／持久狀態時導入 LangGraph」是根據官方描述的 graph／workflow 能力做出的適用性判斷。

## 信心

**高（能力事實）／中高（本案取捨）。**各框架能力均由 2026-08-03 可存取的官方文件核對；對 FRUS 的建議是由已定義的一日限制、固定 hybrid pipeline 和底層 API 能力推導，尚未以同 corpus 的框架對照 benchmark 驗證。
