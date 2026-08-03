now: 全量語料 552/552 卷、723,557 chunks 已建 BM25 並載入 LanceDB；LangGraph B0–B3 圖、確定性 citation gate、Phoenix 追蹤、FastAPI (:8010) 與 Streamlit trace UI (:8511) 皆實測可用；ruff/mypy/40 tests 全綠；目錄已重整為 corpus/retrieval/agent/evaluation 四個套件。
next: 使用者手動啟動全量 dense embedding（實測 32.4 chunks/s、ETA 6.2 小時）；完成後跑乾淨的 B0–B3 雙語 ablation 與 route stability，再用實測數字更新 README。
blocked: agentic 品質增益尚無數字，README 明文不宣稱提升；dense 未建前所有檢索都是 BM25 單臂，先前 75 筆 ablation 結果因此作廢（retriever fingerprint 會自動丟棄）；gold cases 為機器草擬未經史學覆核，只能當回歸訊號。
