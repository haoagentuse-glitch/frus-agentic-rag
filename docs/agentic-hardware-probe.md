# FRUS Agentic RAG：硬體與架構探測

研究日期：2026-08-03  
目標環境：WSL Ubuntu、RTX 4060 Laptop 8GB、32GB RAM、`jobshift:latest`、FRUS 全量資料

## 結論

推薦採用 **LangGraph 的受控式 Adaptive/Corrective RAG**，而不是無上限 ReAct 或多代理辯論。它保留 Agentic RAG 真正有價值的能力——查詢規劃、檢索路由、證據充足度判斷、一次自我修正與可觀察狀態——同時把正常路徑限制在 3 次 LLM 呼叫、重試路徑限制在 4 次，才符合 8GB VRAM 的延遲預算。

重要判斷：

- **VERIFIED**：LangGraph 官方提供 state、conditional edge、持久化、checkpoint、fault recovery 與 human-in-the-loop；v1 是穩定性導向版本。
- **VERIFIED**：`jobshift:latest` 可直接使用 CUDA 與 BGE-M3，但必須掛載 WSL 主機上的完整 Hugging Face `hub` 目錄並指定現有 snapshot。
- **REFUTED**：BGE-M3 權重不在 `jobshift:latest` 映像內；只掛到映像預設 cache 路徑也未成功離線載入。
- **UNVERIFIED**：目前沒有在 Windows、WSL CLI、現有 Docker 映像或已檢查的 Hugging Face cache 找到 Qwen/Ollama；「本機 Qwen 已可用」不能當成既定事實。

## 現實量測

| 項目 | 結果 | 狀態 |
|---|---:|---|
| 專案目錄 | `/home/haoche_nitro_v15/FRUS_agenticRAG` 存在 | VERIFIED |
| GPU | RTX 4060 Laptop，8188 MiB | VERIFIED |
| `jobshift:latest` | image `8a059a1e84d0`，content size 3.53GB | VERIFIED |
| Python / uv | Python 3.12.13、uv 0.12.1 | VERIFIED |
| Torch CUDA | Torch 2.6.0+cu124，CUDA 可用 | VERIFIED |
| 已有套件 | Sentence Transformers、Transformers、Pydantic、FastAPI | VERIFIED |
| 缺少套件 | LangGraph、LangChain、LangChain-Ollama、LanceDB、Ollama client | VERIFIED |
| BGE-M3 cache | 兩份，各約 4.3GB | VERIFIED |
| BGE-M3 GPU 測試 | 2 queries：load 7.703s、encode 0.941s、peak 2.125 GiB | VERIFIED，單次量測 |
| BGE-M3 CPU 測試 | 2 queries：load 3.648s、encode 0.428s | VERIFIED，單次量測 |
| Qwen/Ollama | 未找到 | 未查到 |

BGE-M3 成功載入方式是唯讀掛載：

```text
host: /home/haoche_nitro_v15/.cache/huggingface/hub
container: /cache:ro
snapshot: /cache/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181
```

這是快取內部 snapshot hash，不能永遠寫死在程式；執行 Agent 應先檢查 `refs/main`，驗證 snapshot 包含 `config.json`、tokenizer、pooling 與實際權重，再把解析後路徑寫進 `.env`。

## 三個候選

| 目標 | 架構 | 正常／最差 LLM calls | 4060 估計 | 主要失敗模式 |
|---|---|---:|---|---|
| 最快 walking skeleton | LangChain `create_agent` + `hybrid_search` tool + Ollama | 2–4／受 `max_iterations` 限制 | 最快搭好，但延遲與路徑不穩定 | 小模型漏呼叫工具、重複工具、回答未檢索；只是在 retriever 外包一層 ReAct |
| **最成熟可維護（推薦）** | LangGraph `StateGraph`：plan → retrieve → grade → conditional rewrite → answer → deterministic citation check | **3／4** | 適合 8GB；所有節點、狀態與停止條件可測 | grader 誤判、structured output 失敗、retry 沒提升、上下文過長 |
| 最強技術展示 | LangGraph supervisor + historian/retriever/critic 多代理、並行子問題與交叉檢查 | 6–12／更多 | 7B Q4 可跑但互動時間很可能過長 | local 7B 的 agent coordination 脆弱、VRAM 雖不因 call 數線性增加但延遲線性增長、錯誤傳播、難做可信 eval |

### 為何選推薦案

LangChain 官方把 Agentic RAG 定義為由模型決定何時、如何檢索；其 Hybrid RAG 則加入 query enhancement、retrieval validation、answer validation 與迭代。FRUS 的問題通常必須檢索，所以「是否檢索」不是最有價值的決策；真正值得展示的是：

1. 是否拆成多個年代／人物／事件子問題。
2. 應走 BM25、dense、metadata filter，還是三者並行。
3. 證據是否足以支持回答。
4. 不足時如何只重寫一次，避免無限迴圈。
5. 每句主張能否對回 FRUS canonical document。

LangGraph 官方 Agentic RAG 教學本身就是 retrieve → grade → rewrite → answer；官方也明確提供 durable execution、persistence 與 conditional graph。這比自由 ReAct 更容易測試，也比多代理更符合本機 7B 模型能力。

## 推薦圖與呼叫預算

```text
START
  -> normalize（程式，語言/年份/人名）
  -> plan_and_route（LLM #1，Pydantic JSON）
  -> parallel_retrieve（程式：BM25 + BGE-M3 dense + metadata）
  -> RRF / optional rerank（程式/模型，首版不載 reranker）
  -> grade_evidence（LLM #2，Pydantic JSON）
       ├─ sufficient -> answer（LLM #3）
       └─ insufficient 且 retry=0 -> rewrite（LLM #3）
             -> retrieve -> grade 可改用同一套規則或 LLM
             -> answer（LLM #4）
  -> verify_citations（程式：chunk_id、document_id、URL、quoted span）
  -> END
```

硬限制：

- `max_retrieval_attempts = 2`。
- `max_subqueries = 3`。
- 每一路 top-k 先取 20，融合後送入答案的 evidence 最多 6–8 chunks。
- planner、grader、rewriter 強制 JSON Schema／Pydantic，temperature 0。
- 單次送給 Qwen 的 context 先限 6K–8K tokens；不要因模型宣稱 32K 就把整份文件塞入。
- citation verifier 不用 LLM；引用不存在、超出 retrieved set 或 span 對不上就拒絕輸出該引文。

### 呼叫次數與延遲

- 正常：planner 1 + grader 1 + answer 1 = **3 calls**。
- 自我修正：再加 rewrite 1 = **4 calls**；第二輪 evidence grade 可先用 deterministic threshold，以免增到 5 calls。
- 自由 ReAct 的 calls 取決於模型行為，必須設 `max_iterations`；不能宣稱固定延遲。
- 多代理方案至少 6 calls，且通常更多；8GB 不會因順序 calls 自動 OOM，但 wall-clock latency 近似累加。
- **UNVERIFIED**：Qwen 7B 在這台機器每一節點的實際 TTFT、tokens/s 與完整 graph p95；必須裝好模型後測量，不可預估成 SLA。

## 模型與 GPU 分時

### 生成模型

首選 kill-test：`qwen2.5:7b-instruct-q4_K_M`。Ollama 官方標示約 4.7GB、32K context；Qwen 官方 model card標示 7.61B、多語與 JSON/structured output 能力。實際執行仍先把 `num_ctx` 限 8192，避免 KV cache 吃掉 8GB 餘量。

若 7B 的整體 p95 太慢或工具/JSON 成功率未達門檻，對照測 `qwen2.5:3b-instruct-q4_K_M`（約 1.9GB）。3B 不是預設勝者；只有實測品質與延遲共同達標才改用。

### GPU 分時策略

1. **離線建索引**：停止／unload Qwen，BGE-M3 在 CUDA FP16 分批 embedding；每卷 checkpoint，OOM 就降 batch。
2. **線上查詢**：Qwen 7B Q4 獨占 GPU；BGE-M3 常駐 CPU 做單 query embedding。現場單次量測中 CPU 2-query encode 0.428s，足以成為首版候選，但須以 30-query p95 重測。
3. **不要首版同時常駐** BGE-M3 GPU + Qwen 7B + reranker。BGE-M3 單次已量到 2.125 GiB peak，加上 Qwen 4.7GB 權重、KV cache 與 runtime overhead，餘量太小。
4. reranker 先關閉。若 hybrid Recall@20 達標但 nDCG@10 不足，再以 CPU 或「Qwen unload → rerank → reload」測；後者通常有明顯載入延遲。

## 最便宜 kill-test（先於完整搭建）

在 20 個雙語 FRUS 問題與一個小型已知索引上跑推薦 graph；其中至少 5 題需要 query decomposition、5 題有年份或人物 filter、5 題故意無答案。比較固定 2-step baseline 與 agentic graph。

預登記門檻：

| 指標 | 通過條件 | 不通過時改判 |
|---|---:|---|
| Graph 可終止 | 20/20，且無 run 超過 4 LLM calls | 超過即停用自由 tool loop，改成全顯式 conditional edges |
| Structured output | planner + grader schema valid ≥ 95% | 先補一次 parser retry；仍不足則換 3B/7B 對照或用 deterministic router |
| Citation validity | 100% 引用 ID 存在 retrieved set；quote span 可對回原文 | 未達即禁止自然語言引用，改由程式生成 citation block |
| Retrieval 改善 | agentic Recall@10 相對 baseline 增加，且不是只靠更高 top-k | 無改善則保留 graph 做觀察，不宣稱 agentic 提升 |
| Latency | 7B graph p95 ≤ 30s，或至少未超 baseline 2.5 倍 | 超標先合併 planner+grader／改 3B router；仍超標則回 constrained hybrid pipeline |
| 無答案行為 | 5 題中 ≥ 4 題拒答且附檢索摘要 | 未達則 answer 前加 deterministic sufficiency floor |

最重要的 kill 條件：**若 20 題上 Agentic 版本沒有比固定 hybrid baseline 提升 retrieval/citation 指標，便不能把「更多 LLM calls」當技術含量；作品集應誠實呈現負結果。**

## UNKNOWN 登記表

| UNKNOWN | 為何現在未知 | 最廉價接觸現實的方法 | 決策影響 |
|---|---|---|---|
| Qwen/Ollama 實際位置 | CLI、images、已查 cache 均未找到 | 執行 Agent 先找 compose、volume、11434；找不到才建立 Ollama service | 阻擋 LLM benchmark，不阻擋 parser/index |
| Qwen 7B tool/JSON 成功率 | 官方支援不等於此 prompt 穩定 | 20 題 × planner/grader schema 測試 | 決定 7B、3B 或 deterministic router |
| 完整 graph p95 | calls、prompt tokens 與 GPU server 未實測 | 20 題記錄 node-level TTFT/tokens/s/total | 決定是否合併節點 |
| BGE-M3 CPU p95 | 現在只有一次 2-query 測量 | warm model 後跑 30 次中英 queries | 決定線上 CPU embedding 是否可接受 |
| Agentic 是否真的提升 FRUS | 沒有 qrels 就無法知道 | 固定同一批 20 題做 baseline/agentic A/B | 決定是否保留 rewrite/grade |
| 全量 LanceDB index RAM/latency | 套件尚未安裝、全量尚未建 | 先用 10 卷外推，保留原始數據 | 決定單 table、partition 或 index 參數 |

## 三個最可能的失敗條件

1. 本機 7B 可以生成答案，但 planner/grader 的 JSON 與工具選擇不穩；這會讓 graph 看起來 agentic，實際不可測。
2. Agentic 版本只增加延遲，沒有改善 Recall、citation validity 或拒答；這會直接削弱面試故事。
3. 操作 Agent誤以為 BGE-M3 在映像內，或只掛 snapshot 目錄導致相對 symlink 斷裂；必須掛完整 `hub` 並驗證一次 encode。

會推翻推薦案的證據：

- Haystack 或 LlamaIndex 在相同 20 題、相同模型與索引上，以更少程式碼達到相同可觀察性，且 latency/quality 更好。
- 受控 LangGraph 在 20 題無法穩定終止或 structured output <95%，而簡單 Haystack Agent 可穩定通過。
- 2-step baseline 已達到相同 retrieval/citation 指標，agentic graph 無顯著收益且 p95 超過 2.5 倍。

## 為何暫不選其他框架

- **Haystack**：官方已有 stable Agent、Tool、loops、AsyncPipeline 與 Ollama integration，是可信替代案；但 FRUS 現有檢索預計仍由 LanceDB/custom code 主導，使用 Haystack 仍需包 custom component。若團隊重視組件式 production pipeline 多於 graph state/debug，應列入 A/B。
- **LlamaIndex**：文件索引與 RouterRetriever 很強，但官方舊 Query Pipeline 已進入 feature-freeze/deprecation 並建議改用 Workflows。一天內同時引入其資料抽象與既有 LanceDB pipeline，轉換成本大於收益。
- **多代理**：不是技術含量的必要條件；在本機 7B 上，多個角色常只是把同一模型多呼叫幾次，缺乏獨立能力來源。

## 官方來源

- [LangGraph Agentic RAG tutorial](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph v1 release](https://docs.langchain.com/oss/python/releases/langgraph-v1)
- [LangChain retrieval architectures](https://docs.langchain.com/oss/python/langchain/retrieval)
- [ChatOllama integration](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
- [Ollama tool support](https://ollama.com/blog/tool-support)
- [Qwen2.5 7B official model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Ollama Qwen2.5 tags](https://ollama.com/library/qwen2.5/tags)
- [BGE-M3 official model card](https://huggingface.co/BAAI/bge-m3)
- [Haystack Agent](https://docs.haystack.deepset.ai/docs/agent)
- [Haystack Pipelines](https://docs.haystack.deepset.ai/docs/pipelines)
- [LlamaIndex Query Pipeline deprecation notice](https://docs.llamaindex.ai/en/stable/module_guides/querying/pipeline/)
