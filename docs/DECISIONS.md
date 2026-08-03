# Decisions

- 2026-08-02｜履歷維持兩頁骨架，第一頁以 EventSignal 與 SpiderForge 為主，第二頁補資料研究與設計／製程證據｜官方履歷指南均要求職缺對位、個人責任與可驗證成果；本人的差異化在工程交付與設計訓練的組合｜若 TDRI 證實職務主要是使用者研究或供應商管理，則提高研究與協作經歷比重。
- 2026-08-02｜履歷 bullet 省略「我／我們」，第一人稱視角改由本人責任與動作動詞呈現｜Harvard、MIT 履歷指南均不建議在 bullet 使用人稱代名詞，且現稿重複「我」會產生 AI 自傳感｜若投遞的是另附自傳或求職信，該文件可恢復自然的第一人稱敘事。
- 2026-08-02｜未驗證數字保留為內部待補標記，不直接當成正式成果｜4 萬篇、13 個來源、生成時間與研究責任範圍尚缺統計或交付證據｜取得 log、來源清單、測試或報告分工後可改為完成式。
- 2026-08-02｜以「跨域 AI 工程師」作為主定位，將設計需求轉譯為技術原型的能力放在單一技術清單之前｜展示研究、製造改善、課程開發、API 契約與全端整合共同證明同一種需求轉譯能力，且直接對應 TDRI 的橋樑角色｜若投遞職缺改為純模型研究或單一後端開發，則改回以對應技術深度為首要定位。
- 2026-08-03｜P3 由跨期職缺追蹤改定位為單期職缺市場語意分析｜來源平台防爬使後續快照無法穩定取得；新 repository 已提供 64,643 筆資料、98 個群集、技能分析與公開展示｜若未來取得合法且穩定的多期資料，另立跨期研究版本，不回頭把現有單期結果包裝成趨勢。
- 2026-08-03｜RAG 題材改為全已發布 FRUS historical-document；694 個 XML 全進 manifest，但 planned placeholder、索引與 front/back matter 不混入主正文向量索引｜官方 snapshot 實測為 694 XML、約 3.15 GiB，且含 planned stub；官方 canonical 單位是 released volume/document｜若新版官方 schema 或 release manifest 提供不同 published-document 邊界，依 pinned snapshot 重建 inventory。
- 2026-08-03｜單機首選 LanceDB＋BM25／multilingual E5＋RRF，E5 與 Qwen 分時使用 RTX 4060 8GB｜本機有 31.7 GiB RAM；內嵌 disk-first 架構能降低 Windows 上 Docker/JVM 維運與 GPU 同時常駐風險｜若 50k gate 的 ANN Recall@10 <0.90 或 p95 >1.5 秒且調校無效，改 Qdrant on-disk。
- 2026-08-03｜一天交付 MVP、可恢復全量 pipeline 與實測 ETA，不預先保證一天完成全庫 dense index｜全量 published documents/chunks 與本機 throughput 尚未實測；10k／50k benchmark 能廉價回答｜若實測含 30% buffer 的全量 ETA仍在一天內，則把全量完成提升為 Day 1 驗收項目。
- 2026-08-03｜FRUS MVP 先衍生 WSL 的 jobshift:latest，唯讀沿用 BGE-M3 cache，不修改 Jobshift image／資料｜實測 image 為 Python 3.12.13、Torch 2.6.0+cu124、Sentence Transformers 3.4.1，能看見 RTX 4060，且 BGE-M3 cache 約 4.3GB｜若 10k benchmark 吞吐不足或依賴衝突，保留全量 BM25並改用獨立 multilingual-e5-small image。
- 2026-08-03｜FRUS 問答由固定 2-step 升級為 LangGraph 的 bounded Adaptive/Corrective Multi-hop graph，但保留 2-step 作 baseline｜LangGraph 官方 Agentic RAG pattern 支援 plan/retrieve/grade/rewrite conditional path；FRUS 的跨卷比較與 citation governance 能形成可驗證 agent decisions｜若 B3 對 multi-hop 品質未比 B0 提升 10 percentage points，或 p95 超過 B0 2.5 倍，產品預設退回通過實證的最小 graph。
- 2026-08-03｜第一版不採 web fallback、無限 ReAct 或 multi-agent supervisor；正常最多 3、修正最多 4 次 LLM calls｜FRUS 是封閉 canonical corpus；本機 8GB 上多代理只會累加延遲，且外部網頁會污染史料來源邊界｜若單一 graph 已通過品質/延遲門檻，且新增異質工具確有需求，再另立 multi-agent 實驗。
