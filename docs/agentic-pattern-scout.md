# FRUS Agentic RAG Pattern Scout

研究日期：2026-08-03  
問題：哪一種 Agentic RAG pattern 已有論文與官方實作依據、能在一天內以本機模型搭建，而且不讓 FRUS 史料被一般網路內容污染？  
動機：作品必須展示「檢索失敗後會做不同動作」的真正代理決策，同時維持可重現、可評估、可追溯到 FRUS canonical document 的史料紀律。

## 結論

**已確立：選擇「Bounded Adaptive Corrective Multi-hop RAG」**，以 LangGraph typed state graph 實作下列流程：

```text
START
  -> plan_query
       |-- social_or_system -> direct_response
       |-- simple_fact     -> hybrid_retrieve
       `-- complex         -> decompose(max 3) -> hybrid_retrieve
  -> grade_evidence
       |-- sufficient                         -> synthesize
       |-- missing + retrieval_attempts < 2   -> rewrite_missing -> retrieve
       `-- exhausted                          -> abstain
  -> verify_citations
       |-- valid                              -> END
       |-- invalid + repair_attempts < 1       -> repair_from_evidence
       `-- invalid again                      -> abstain
```

它不是把固定 RAG 放進一個聊天 Agent：`plan_query` 會改變檢索次數與查詢，`grade_evidence` 會決定生成、針對缺口重搜或拒答，`verify_citations` 會阻擋無有效 FRUS 引用的答案。所有 factual/history 問題一律要檢索；只有問候、操作說明等非史實問題可直接回答。

**已確立：不採 unrestricted ReAct、一般 web fallback 或無上限反思迴圈。** FRUS 是封閉、具 canonical citation 的史料庫；外部網頁只能在未來作為明確標籤的第二來源，不能混成 FRUS 證據。本版的 corrective action 是「在 FRUS 內改寫缺失子問題、調整 metadata/filter、重新執行 hybrid retrieval」，兩次仍不足即拒答。

**已確立：這是工程組合，不宣稱忠實重現 Adaptive-RAG、CRAG、EfficientRAG 或 SELF-RAG。** 那些論文分別使用訓練過的複雜度分類器、fine-tuned retrieval evaluator、訓練過的多跳元件或 reflection tokens；本作品僅採用它們驗證過的設計問題，再用可觀察、有上限的 graph nodes 落地。

## Pattern 比較

| Pattern | 原方法的核心 | 一日實作難度 | FRUS 適合度 | 判定 |
|---|---|---:|---:|---|
| Adaptive routing | 依 query complexity 在 no / one-step / iterative retrieval 間路由；原論文以自動標籤訓練小型分類器 | 低至中；但忠實重現需訓練 | 高；簡單查件與跨卷比較成本不同 | 採用其思想；本版用 typed plan，不稱為原版 Adaptive-RAG |
| Corrective RAG (CRAG) | retrieval evaluator 判 Correct / Incorrect / Ambiguous；不佳時 web search，相關文件再切成 strips 過濾 | 中；忠實重現需 evaluator 與 web | 中；「先評估再修正」很適合，但 web 會破壞 FRUS 來源邊界 | 採 corpus-only corrective loop，明標 CRAG-inspired |
| Query decomposition / multi-hop | 將比較、時間、推論問題拆成多個可檢索子問題，再聚合 supporting evidence | 低；一次 structured output 即可 | 很高；FRUS 常要跨文件、跨年代或跨行政當局比較 | 採用，最多 3 個子問題；只重寫缺失 hop |
| Retrieve-grade-rewrite loop | 檢索後由 grader 判 relevance；不足則 rewrite，再回到 retrieval | 低；LangGraph 有官方逐步教程 | 高；最容易展示真正 conditional edge | 採用，但加 attempt budget、evidence coverage 與拒答 |
| SELF-RAG self-reflection | 訓練模型生成 retrieval / critique reflection tokens，逐段決定 retrieval、support 與 usefulness | 高；不是多寫一個 critique prompt | 中；概念適合引用治理，原方法不適合 8GB／一天 | 不實作；改用外部 citation verifier，名稱不可冒充 SELF-RAG |
| Open-ended ReAct agent | LLM 自由選工具並反覆思考、行動 | 低到中，但行為不穩、難評估 | 低；容易無限重試、錯用來源或以模型記憶補史料 | 不採用 |

## 來源地圖

### 1. Adaptive routing

**已確立。** Adaptive-RAG 依問題複雜度選擇 no-retrieval、single-step 或 iterative retrieval；原方法不是一段 router prompt，而是以自動蒐集標籤訓練的小型分類器。論文亦報告 multi-step 路線顯著慢於 one-step，說明有路由的工程價值；但分類仍有明顯混淆，因此 router 必須獨立評估。

- Soyeong Jeong et al., “Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity,” NAACL 2024；arXiv v1 2024-03-21。原文：[ACL Anthology](https://aclanthology.org/2024.naacl-long.389/)／[arXiv](https://arxiv.org/abs/2403.14403)
- 概括：論文在 classifier、one-step、iterative 策略上提供證據；本案的 `QueryPlan` 是較輕的工程替代，沒有論文等價性。

### 2. Corrective retrieval

**已確立。** CRAG 先以 retrieval evaluator 評估 query-document relevance，再觸發 Correct、Incorrect、Ambiguous；原版對錯誤或模糊 retrieval 會使用 web search，並用 decompose-filter-recompose 精煉內部文件。原文使用 fine-tuned T5-large evaluator，且作者指出 prompt ChatGPT 判 relevance 的比較結果較差，不能假設本機 7B grader 天生可靠。

- Shi-Qi Yan et al., “Corrective Retrieval Augmented Generation,” arXiv v1 2024-01-29、v3 2024-10-07。[論文 PDF](https://arxiv.org/pdf/2401.15884)
- 概括：FRUS 只借用「評估後選 corrective action」；以 corpus-only retry／abstain 取代 web search 是本專案的治理改造，不是原 CRAG。

### 3. Query decomposition 與 multi-hop

**已確立。** MultiHop-RAG 顯示既有 RAG 對需要多份 supporting evidence 的問題表現不足；其資料中約 42% 問題需要兩份證據、約 30% 需三份、約 15% 需四份。EfficientRAG 的實驗顯示，一次 LLM decomposition 的 retrieval recall 優於直接以原問題檢索，而迭代 decomposition 能以更少 chunks 達到相近或更高 recall。

- Yixuan Tang, Yi Yang, “MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries,” arXiv 2024-01-27；COLM 2024。[arXiv](https://arxiv.org/abs/2401.15391)／[OpenReview](https://openreview.net/forum?id=t4eB3zYWBK)
- Ziyuan Zhuang et al., “EfficientRAG: Efficient Retriever for Multi-Hop Question Answering,” EMNLP 2024。[ACL Anthology PDF](https://aclanthology.org/2024.emnlp-main.199.pdf)
- 概括：這些是 open-domain benchmark，不直接保證在 FRUS 上提升；FRUS 必須自建 evidence-level evaluation。

### 4. Retrieve-grade-rewrite 的官方實作

**已確立。** LangGraph 官方教程直接展示：模型決定是否呼叫 retriever、structured grader 以 conditional edge 在 answer 與 rewrite 間分流，rewrite 後回到 retrieval。Graph API 把 state、node、edge 明確分離，能逐節點 trace 與測試，適合本案做有界 retry。

- LangChain, “Build a custom RAG agent with LangGraph,” 存取日期 2026-08-03。[官方教程](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
- LangChain, “Thinking in LangGraph,” 存取日期 2026-08-03。[官方指南](https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph)
- 概括：官方範例沒有強制 retry limit，也只做 binary relevance；本案必須自行補 attempt counter、per-subquestion coverage、citation gate 與 refusal。

### 5. Self-reflection

**已確立。** SELF-RAG 的核心是 end-to-end 訓練 LM 生成特殊 retrieval／critique reflection tokens；不是在一般模型後加一個「請反思」prompt。論文仍明說模型可能產生未被 citation 完整支持的輸出。

- Akari Asai et al., “Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection,” ICLR 2024。[ICLR 論文](https://proceedings.iclr.cc/paper_files/paper/2024/file/25f7be9694d7b32d5cc670927b8091e1-Paper-Conference.pdf)
- 概括：本案使用獨立 verifier 與 deterministic canonical-ID validator，不能在 README 稱為 SELF-RAG。

## FRUS 最小可驗證設計

### Typed state

```python
class AgentState(TypedDict):
    original_question: str
    answer_language: Literal["zh-TW", "en"]
    route: Literal["social", "simple", "complex"]
    subqueries: list[SubQuery]          # max 3
    evidence: list[Evidence]
    missing_subqueries: list[str]
    retrieval_attempts: int             # max 2
    draft_answer: str | None
    citation_errors: list[str]
    repair_attempts: int                # max 1
    outcome: Literal["answer", "abstain"] | None
    trace: list[TraceEvent]
```

### 節點責任

1. `plan_query`：用 Pydantic structured output 產生 route、0–3 個英文 retrieval subqueries、人物／年份／行政當局 filters。中文問題可產生英文檢索詞，但最終回答維持使用者語言。
2. `hybrid_retrieve`：每個 subquery 同時跑 BM25 與 BGE-M3 dense，RRF 合併；保留 retriever、rank、score、chunk_id、volume_id、document_id、canonical_url。
3. `grade_evidence`：逐 subquery 判是否有 supporting evidence，而不是只問「看起來相關嗎」。輸入含 top rerank score 等可觀察訊號；輸出 `supported | partial | unsupported`、使用哪些 evidence IDs、缺少什麼。不得讓 grader 製造新史實。
4. `rewrite_missing`：只重寫 partial／unsupported 的 hop；保留原始 question，禁止把模型猜測當成 filter。最多回到 retrieval 一次。
5. `synthesize`：只從 accepted evidence 寫答案；每個可檢驗 claim 後放 canonical document citation。若證據衝突，並列而不是裁決。
6. `verify_citations`：先以程式檢查 citation ID 存在、URL 格式、引用文件確實在 retrieved evidence；再用 structured verifier 檢查 claim-evidence support。程式檢查不過一律不回傳答案。
7. `abstain`：回報查不到哪些 hop、已嘗試哪些 query，絕不以模型內部知識補完。

### 不使用的工具／路線

- 不提供一般 web search tool。
- 不讓 LLM 直接撰寫任意 URL；citation URL 必須由 `volume_id + document_id` 程式生成。
- 不把 FRUS 的 editor note、person index 或未發布 placeholder 當成同等正文證據；Evidence 必須保留 `section_type`。
- 不做無限 `while grader == no`；圖層計數器是硬上限。
- 不把 BM25／BGE-M3／LanceDB 包在 LangChain abstraction 才能使用；它們可以是一般 Python function，由 LangGraph node 呼叫。

## 評估方案

### Dataset

人工建立 **30 個 gold cases × 中英雙語 = 60 runs**：

- 10 個 single-document lookup。
- 10 個跨文件 comparison／temporal／causal multi-hop；每題標 2–4 個 gold evidence document IDs。
- 5 個初始詞彙與 FRUS 用語不匹配、預期 rewrite 後命中的問題。
- 5 個 corpus 中不可回答或越界的問題，gold outcome=`abstain`。

每筆保存：`question_zh`、`question_en`、`route_gold`、`gold_doc_ids`、`required_hops`、`answerable`、`reference_notes`。中文與英文視為同一個 semantic case，避免把翻譯題灌水成獨立證據。

### Baseline 與 ablation

固定同一 generator、同一 hybrid retriever、同一 top-k：

1. `B0`：2-step hybrid RAG（原問題一次檢索後直接生成）。
2. `B1`：B0 + adaptive plan/decomposition。
3. `B2`：B1 + evidence grade/rewrite/abstain。
4. `B3`：B2 + citation verifier/repair（完整系統）。

### Metrics

- Retrieval：single-hop Recall@10、multi-hop **All-Evidence Recall@10**、MRR@10；multi-hop 不只算「至少命中一份」。
- Routing：social/simple/complex route macro-F1、與不必要 decomposition 比率；abstention 另計 precision／recall。
- Correction：first-pass miss 中第二次檢索新增 gold evidence 的比例；rewrite semantic drift 比率。
- Answer：人工 answer correctness、citation precision、claim citation coverage、正確 refusal rate。
- Reliability：100% runs 在 hard budget 內終止；invalid canonical citation rate。
- Cost：LLM calls、retrieval calls、tokens、p50/p95 latency、峰值 VRAM；均按 B0–B3 比較。

## 預登記 kill-tests

1. **Agentic value kill-test**：完整 B3 在 10 個 multi-hop cases 的 All-Evidence Recall 或 answer correctness，相對 B0 兩者皆未提升至少 10 個百分點，則不得宣稱 Agentic RAG 改善品質；README 必須報 negative result，產品預設退回 B0 或只保留有實證的節點。
2. **Router kill-test**：route macro-F1 < 0.80，或 simple cases 有 >20% 被不必要拆解，停用自由 LLM router；改用明確規則先分 simple/complex，LLM 只產生 subqueries。
3. **Grader safety kill-test**：5 個 unanswerable cases 任一產生無證據的肯定史實，或 false-accept rate >10%，禁止 grader 直接放行；改為 `reranker threshold + required-hop coverage + LLM grade` 三者聯合，否則一律 abstain。
4. **Rewrite drift kill-test**：rewrite 後新增證據卻與原問題人物／年份／行政當局衝突的比例 >10%，停用自動 rewrite，改回詢問使用者澄清。
5. **Citation kill-test**：任何輸出含不存在、未檢索或非 canonical 的 FRUS citation，視為阻斷性失敗；不得只顯示 warning，必須 repair 或 abstain。
6. **Latency kill-test**：B3 的 p95 latency 超過 B0 的 3 倍，或本機 p95 >45 秒，將 max subqueries 由 3 降為 2、停用完整答案反思，保留一次 evidence correction；若仍超標，UI 預設 B1 並將完整 agentic 模式列為研究選項。
7. **Termination kill-test**：任何測試超過 2 次 retrieval 或 1 次 answer repair，視為圖設計失敗；先修 state/conditional edge，不以更高 recursion limit掩蓋。

## 三態總結

- **已確立**：複雜度路由、multi-hop decomposition、retrieval grading 與 corrective action 均有 2024 論文或 LangGraph 官方教程支持；SELF-RAG 需要訓練過的 reflection-token 模型。
- **發展中**：以一般 instruction model 擔任 grader／router 是常見工程做法，但不等價於各論文的 trained components，且 7B 級本機模型在 FRUS 上的 decision accuracy 必須實測。
- **未查到**：未查到 FRUS 專屬的 Agentic RAG benchmark、gold multi-hop QA set，亦未查到任何來源能保證 BGE-M3 + 本機 7B grader 在 RTX 4060 上達到上述品質或延遲門檻；這些只能由專案 benchmark 回答。

## 最終建議名稱

README 建議使用：

> **A bounded, citation-governed Agentic RAG for FRUS, combining adaptive query planning, multi-hop hybrid retrieval, corpus-only corrective retries, and deterministic canonical citation validation.**

避免使用「完整實作 SELF-RAG／CRAG／Adaptive-RAG」；更精確的說法是「research-inspired, independently evaluated engineering graph」。
