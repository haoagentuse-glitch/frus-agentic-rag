# eval-v0 — 已作廢，不得引用

300 runs，5 個系統 × 30 題 × 雙語，全量 hybrid 索引。
執行完成，但下列缺陷使其結論無效。作為紀錄保留，不進入 README。

## 使結論無效的缺陷

1. **recall 指標名不副實。** `all_evidence_recall` 讀的是 `Answer.evidence`，
   該清單已被 grader 的 `accepted_evidence_ids` 過濾並截至 `top_k=10`。
   它量的是「檢索 ∩ 評分」，不是檢索召回，因此無法區分責任歸屬。

2. **檢索預算不對等。** B0 每題 1.00 次檢索，B1 1.83 次，B2 3.50 次。
   B1 相對 B0 的 recall 提升無法區分來自查詢規劃或來自更多次檢索。

3. **answer correctness 不可採信。** judge 僅收到 gold 文件 ID 字串，
   未收到文件內容或 reference answer，無從判斷答案是否被該文件支持。
   預登記門檻 1 使用此欄位，判定連帶失效。

4. **gold 錨點頻率統計只掃各文件首個 chunk**（`ordinal == 0`），
   且僅限卷內，非全語料文件頻率。

5. **correction 題保留原始錨點詞**，僅改寫外圍虛詞，
   測到的是全域歧義而非 query correction。

## 檢索器本身的已知缺陷（於此輪之後才診斷出）

以精確標題查詢 10 題 lookup，3 題完全無法召回。該三題標題逐字唯一，
但構成詞均為全語料高頻詞（`Memorandum by Secretary of State Dulles`）。
`head` 雖已前綴進 `text` 而被 FTS 索引，但無獨立欄位與權重。
