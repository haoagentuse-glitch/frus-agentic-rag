now: FRUS bounded agentic RAG 已在本機跑通端到端。全量語料 552/552 卷、723,557 chunks 已建 BM25 並載入 LanceDB；LangGraph B0–B3 圖、確定性 citation gate、Phoenix 追蹤、FastAPI (:8010) 與 Streamlit trace UI (:8511) 皆實測可用；ruff/mypy/29 unit tests 全綠。
next: 全量 dense embedding 尚未執行（實測 32.4 chunks/s、ETA 6.2 小時，指令已交付使用者自行啟動）；30 題 × 雙語 × 5 系統的 ablation 正在背景跑（300 runs，逐筆 checkpoint 到 reports/ablation_runs.jsonl，可 --resume）。
blocked: agentic 品質增益尚未有數字，README 目前明文不宣稱提升；gold cases 為機器草擬、未經史學專家覆核，multi-hop recall 只能當回歸訊號；dense 未建前所有 RRF 數字僅反映 BM25 單臂。
