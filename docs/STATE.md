now: 全量 FRUS 552/552 卷、723,557 chunks 的 BM25/BGE-M3 索引與 B0–B3 graph 已完成；v1 350-run ablation 已分離 candidate/union/accepted/citation recall；README 已改為 evaluation-driven 求職敘事；ruff、mypy、51 tests 通過，但 graph-smoke 的 unanswerable path 仍錯誤作答。
next: 跑完整 score distribution，比較 fixed top-k／relative margin selector；測短序號 citation protocol；以對等 retrieval budget 重跑 B0 vs B1，另跑 R0–R3 retrieval-stage ablation，最後取得乾淨 latency。
blocked: deterministic selector 門檻未校準，使 B2/B3 correction 尚未有效啟動；gold cases 未經史學覆核且缺 timeline／真正 vocabulary mismatch；v1 latency 受 Phoenix exporter 汙染不可引用。
