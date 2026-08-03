# FRUS 全量資料集查證與 ingest 可行性（截至 2026-08-03）

## 先給決策

**結論：有條件採用。** `HistoryAtState/frus` 很適合做 AI Engineer 面試作品：資料專業、公開、可直接下載、具穩定文件 ID 與 TEI 結構，而且規模確實大到需要 hybrid RAG。全量是指目前已發布的 **552 卷**，不是把 repo 裡 694 個 XML 全部當正文；其中 **142 個沒有 `<div type="document">`**，大多是約 2.5 KB 的未發布 placeholder／索引入口。

但不能承諾 RTX 4060 8 GB 在一個 8 小時工作日內完成全量 dense embedding。完整 lexical/BM25 ingest 與可運作系統能在一天內完成；全量 dense 能否同日完成，必須先用相同模型、chunk 長度與 batch 對 10,000 chunks 做實機 benchmark。以本報告的規劃範圍 **65–100 萬 chunks** 計算，8 小時內跑完要求端到端至少 **23–35 chunks/s**，並預留解析、索引與除錯時間。

---

## 狀態標記

- **VERIFIED（官方）**：官方頁面或官方 GitHub 直接明示。
- **VERIFIED（快照實測）**：對固定 commit 的唯讀快照做可重現計數；不是官方宣稱的統計。
- **UNVERIFIED（工程估算）**：依明列假設推估；須用實機 benchmark 或正式 parser 改判。

## 官方直接引述（與下文概括分開）

- Office of the Historian 稱 FRUS 為「**official documentary record of U.S. foreign policy**」。來源：[About FRUS](https://history.state.gov/historicaldocuments/about-frus)，頁面未標發布日，查閱日 2026-08-03。
- GitHub README 說明卷檔為「**one file per volume**」，並稱 repo 檔案「**are in the public domain**」。來源：[HistoryAtState/frus](https://github.com/HistoryAtState/frus)，本次固定快照 commit 2026-07-20。

其餘內容皆為來源概括或本次量測，不是原文引述。

---

## 1. 資料本身與授權

### 已確立

- **VERIFIED（官方）** FRUS 自 1861 年起出版，是美國國務院主要外交政策決策與重要外交活動的正式文獻紀錄；範圍含雙邊／區域關係、恐怖主義、毒品、健康、環境、國安與外交經濟政策。[Office of the Historian — About FRUS](https://history.state.gov/historicaldocuments/about-frus)，頁面未標發布日，查閱日 2026-08-03。
- **VERIFIED（官方）** 官方 ebook 頁面目前明列 **552 volumes**；這也與快照中含正文文件的 552 個 XML 完全相符。[Office of the Historian — Ebooks](https://history.state.gov/historicaldocuments/ebooks)，頁面未標發布日，查閱日 2026-08-03。
- **VERIFIED（官方）** GitHub 提供線上版的 digital master source files；格式為 **TEI P5 XML**，`volumes/` 原則上一卷一檔，並附專案 ODD、Schematron 與 Relax NG schema。[官方 README](https://github.com/HistoryAtState/frus)，本次快照 commit 日期 2026-07-20。
- **VERIFIED（官方）** repo README 明示檔案為 public domain，可不經許可複製與散布。[官方 README — License](https://github.com/HistoryAtState/frus#license)，本次快照 commit 日期 2026-07-20。

### RAG 所需的 TEI 結構

- 卷：根 `<TEI xml:id="{volume_id}">`，檔名 stem 即官方 URL 的 `VOLUME_ID`。
- 文件：`<div type="document" xml:id="…">`；常見屬性有 `n`、`subtype`、`frus:doc-dateTime-min/max`。
- 上下文階層：`section`、`compilation`、`chapter`、`document`，可保留為 metadata 與 breadcrumb。
- 其他可用欄位：`teiHeader` 書名與出版資訊、`note`、`pb`、人物與術語參照。
- **VERIFIED（官方）** 已發布卷的 volume ID 與 document ID 被視為 canonical，不會任意更改；person／term ID 與 page-break ID 尚非 canonical。[官方 README — Canonical Identifiers](https://github.com/HistoryAtState/frus#canonical-identifiers)，本次快照 commit 日期 2026-07-20。

建議主鍵：`{volume_id}:{document_xml_id}:{chunk_no}:{source_commit}`。人物、術語與頁碼只能作可更新 metadata，不能當跨版本永久主鍵。

---

## 2. 全量規模：官方值與快照實測

固定快照：[`4b4c402f0cce25144ded2198ba9566b6c37c7c49`](https://github.com/HistoryAtState/frus/commit/4b4c402f0cce25144ded2198ba9566b6c37c7c49)，commit 日期 **2026-07-20**。GitHub repository metadata 顯示 `pushed_at=2026-07-27`，但 default branch 本次抓得的 HEAD 為上述 SHA；因此所有實測都以 SHA 為準，不以「最新」字樣保證未來可重現。[GitHub Repository API](https://api.github.com/repos/HistoryAtState/frus)，資料查閱日 2026-08-03。

| 項目 | 數字 | 狀態與方法 |
|---|---:|---|
| 官方已提供卷數 | **552 卷** | VERIFIED（官方）；官方 ebooks 頁面，查閱日 2026-08-03 |
| `volumes/*.xml` | **694 檔** | VERIFIED（快照實測）；Git tree 與 Windows checkout 均為 694 |
| 含 `<div type="document">` 的 XML | **552 檔** | VERIFIED（快照實測）；逐檔掃描 start-tag |
| 不含 document div 的 XML | **142 檔** | VERIFIED（快照實測）；其中 141 檔小於 10 KB，多為 placeholder；另有一個大型 supplement |
| document div 總數 | **314,483** | VERIFIED（快照實測）；計數所有 `<div … type="document" …>`，包含 historical document、editorial note 與 placeholder document |
| Git blob 中 volume XML 原始大小 | **3,337,854,663 bytes = 3.109 GiB** | VERIFIED（快照實測）；GitHub recursive tree 的 blob `size` 加總 |
| Windows checkout 的 volume XML | **3,379,455,993 bytes = 3.147 GiB** | VERIFIED（快照實測）；CRLF checkout 較 blob 多約 39.7 MiB |
| 全 repository 當前 tree（所有 blob） | **3,357,659,184 bytes = 3.127 GiB** | VERIFIED（快照實測）；GitHub tree，未含 Git history |
| 淺層 clone `.git` pack | **684,455,024 bytes = 652.82 MiB** | VERIFIED（快照實測）；`git clone --depth 1` 後 `git count-objects -vH` |
| GitHub API repository `size` | **14,445,907 KB ≈ 13.78 GiB** | VERIFIED（官方 metadata）；此值反映 GitHub repo size，不等同工作樹或一次下載量，不建議為 ingest 拉完整歷史 |
| 最新正式 release | **v1.0.18，2026-03-13** | VERIFIED（官方）；[release](https://github.com/HistoryAtState/frus/releases/tag/v1.0.18) |
| v1.0.18 `frus.xar` 下載大小 | **626,304,024 bytes = 597.29 MiB** | VERIFIED（官方 release asset metadata）；SHA-256 `653da8…2586e` |
| 去標籤後 `<text>` 可見字元 | **1,428,518,126 chars** | UNVERIFIED（近似量測）；regex 去標籤，未做完整 XML entity／XInclude 語意解析 |
| word-like token | **235,013,378** | UNVERIFIED（近似量測）；Unicode 字母數字 regex，不是 embedding tokenizer token |

來源：[固定 commit 的 Git tree API](https://api.github.com/repos/HistoryAtState/frus/git/trees/4b4c402f0cce25144ded2198ba9566b6c37c7c49?recursive=1)，tree 產生／查閱日 2026-08-03；[Repository API](https://api.github.com/repos/HistoryAtState/frus)，查閱日 2026-08-03；[v1.0.18 release](https://github.com/HistoryAtState/frus/releases/tag/v1.0.18)，發布日 2026-03-13。

### 重要解讀

1. **不要把 694 當成 694 卷已發布正文。** 正確 ingest manifest 是 552 個含 document div 的卷；142 個入口保留 status，但不送 embedding。
2. **不要 full clone。** 完整 repo metadata 約 13.78 GiB；單一 release asset 約 597 MiB，或 shallow clone 約 653 MiB，下載與磁碟差距很大。
3. 314,483 是 TEI document div，不等同最終 chunks。平均約 747 個 word-like token/document，但長度分布很可能長尾。

---

## 3. chunks、embedding 與索引磁碟估算

### 假設

- 正文不跨 document 邊界切塊。
- chunk = **512 embedding tokens**，overlap = **64**，stride = 448。
- 英文歷史文獻的 embedding tokenizer token 暫以 word-like token 的 **1.15–1.35 倍**規劃；此比例必須由實際 tokenizer 重算。
- 全量包含前言／索引等可檢索區塊時，另保留少量 section chunks。

每份文件的正確公式：

`chunks_doc = max(1, ceil((tokens_doc - 512) / 448) + 1)`

**UNVERIFIED（工程估算）**：全量約 **650,000–1,000,000 chunks**。下界近似總 token/stride；上界納入 314,483 次文件邊界取整、front/back matter 與 tokenizer 誤差。最終數字必須在正式 tokenizer pass 後覆寫。

| 儲存項 | 65 萬 chunks | 100 萬 chunks | 公式／說明 |
|---|---:|---:|---|
| 768-d float32 原始 vectors | 1.86 GiB | 2.86 GiB | `N × 768 × 4` |
| 1024-d float32 原始 vectors | 2.48 GiB | 3.81 GiB | `N × 1024 × 4` |
| 1024-d float16 原始 vectors | 1.24 GiB | 1.91 GiB | 僅在 vector DB 真正支援相同精度時成立 |
| 1024-d float32 vector DB + HNSW | 約 3.0 GiB | 約 6.1 GiB | UNVERIFIED；以原始 vectors 的 1.2–1.6× 規劃，實作相依 |
| lexical/BM25 index | 約 0.7 GiB | 約 2.2 GiB | UNVERIFIED；以清理後文本的約 0.5–1.5× 規劃，依 analyzer/positions/store-text 而異 |
| 清理文本＋metadata payload | 約 1.5–3.0 GiB | 同左 | UNVERIFIED；是否重複存全文影響最大 |

**整體磁碟預算：**

- 不保留 Git history：source XML 3.15 GiB + release/clone pack 約 0.6 GiB + payload 1.5–3.0 GiB + dense 3.0–6.1 GiB + lexical 0.7–2.2 GiB + checkpoint/log 約 0.5–1 GiB = **約 9.5–16 GiB**。
- 建議實際預留 **20–25 GiB**；若要保留 snapshot、雙索引備份或用較重的 Elasticsearch/OpenSearch，預留 **30–40 GiB**。
- 若保留完整 Git history，另加約 13.78 GiB repository metadata 所代表的量級；面試作品沒有必要。

RTX 4060 的 **8 GB VRAM 不影響磁碟估算**；它主要決定 embedding batch size 與吞吐。模型是否能同日完成應以 benchmark 判斷，不應由 VRAM 容量直接推論。

---

## 4. 一天內的全量／試跑分階段

### Phase A — 先取得完整但可重現的來源（15–45 分）

1. 最穩定方案：下載 v1.0.18 `frus.xar`（597.29 MiB）並驗 SHA-256。
2. 要追最新 master：`git clone --depth 1`，立刻記錄 `git rev-parse HEAD`；不要抓完整 history。
3. 產 manifest：694 XML 全列入，但只把 552 個含 document div 的檔標為 `released=true`。

### Phase B — 3 卷 dry run（30–60 分）

選小／中／大各一卷；跑 TEI-aware parser、metadata、chunk、BM25、dense、引用 URL。驗收：

- URL 可回到 `https://history.state.gov/historicaldocuments/{volume_id}/{document_id}`。
- 不跨 document；footnote 保留在所屬文件或具反向連結。
- 三卷 XML parse success 100%，chunk 不空、主鍵不重複。

### Phase C — 全量 parse + lexical first（1–2 小時，需實跑確認）

先 ingest 552 卷到 lexical index。這一步完成後，即使 dense 還在跑，系統已能對完整 FRUS 做 BM25 搜尋與有來源回答。142 個未發布／無文件 XML 只進 manifest，不進檢索語料。

### Phase D — 10,000 chunks 實機 benchmark（15–30 分）

使用最終模型、512/64、相同資料清理與 batch；記錄：

- chunks/s（P50 與持續 10 分鐘值）
- GPU peak VRAM、CPU RAM
- tokenizer 與 model encode 各自耗時
- 寫入 vector DB 的吞吐

8 小時 dense 預算的最低門檻：

- 65 萬 chunks：`650000 / 28800 = 22.6 chunks/s`
- 100 萬 chunks：`1000000 / 28800 = 34.7 chunks/s`

若持續吞吐低於 **35 chunks/s**，就不能對外承諾高端估算能在 8 小時內完成。

### Phase E — 全量 dense、可續跑與驗收（剩餘時間）

- 按 volume checkpoint；每卷完成即寫 manifest checksum、chunk count、model revision。
- upsert key 固定，重跑不得倍增資料。
- dense 未完成時 UI 顯示 coverage，例如 `dense: 327/552 volumes`；BM25 仍是全量。
- 全量完成後以 30–50 題雙語 evaluation 驗證 lexical、dense、RRF hybrid 三組，不只展示聊天畫面。

---

## 5. 更新策略

- **VERIFIED（官方）** README 說 repository 會隨 history.state.gov 更新，包含既有卷修正、新出版與舊卷數位化；`master` commit 會觸發 release automation。[官方 README — Release Schedule](https://github.com/HistoryAtState/frus#release-schedule)，本次快照日期 2026-07-20。
- 每次 ingest 儲存 `source_commit`、每個 volume XML 的 SHA-256、parser version、embedding model revision。
- 更新時只比對 `volumes/*.xml` checksum：新增／修改卷重新 parse、刪除舊 chunk 後 upsert；未變卷不重算。
- volume/document ID 可作穩定 join；person/term/page-break ID 不可假設跨版穩定。
- release v1.0.18（2026-03-13）與本次 master HEAD（2026-07-20）存在時間差。作品要選擇「release 可重現」或「master 較新」並在 UI/README 顯示，不能混稱同一版本。

---

## 6. 前三個失敗條件與預登記改判證據

1. **Dense 同日完工失敗**
   - 失敗條件：10,000-chunk benchmark 的持續端到端吞吐 `<35 chunks/s`，或推算總時數超過可用 8 小時窗口。
   - 改判證據：實測 tokenizer 後總 chunks 明顯低於 65 萬，或同設定吞吐穩定高於 35 chunks/s。
   - 處置：維持全量 BM25；dense 按卷背景續跑，或改較小的 multilingual embedding model／768-d index，但不得謊稱 dense 已全量。

2. **Parser 忠實度失敗**
   - 失敗條件：TEI schema／XML parse error；含 document 的卷數不等於 552；解析後 document 數相對 `<div type="document">` 基準 314,483 少超過 0.1%；引用 URL 抽查失敗。
   - 改判證據：TEI-aware `iterparse` 驗證 552/552 卷成功、document count 可解釋一致、50 筆隨機 URL 全可定位。
   - 處置：禁止 regex 當 production parser；處理 namespace、XInclude、entity、note 與 nested div，再重跑失敗卷。

3. **磁碟／索引估算失敗**
   - 失敗條件：tokenizer 後 chunks `>1,000,000`，或 10 萬 chunks 的實際索引線性外推後總 operational footprint `>20 GiB`。
   - 改判證據：正式 tokenizer count 與 10 萬 chunk 索引 snapshot 顯示總量落回上表範圍。
   - 處置：避免重複存全文、選 768-d 或量化 vector storage、調低 HNSW／payload 冗餘；仍保留完整 552 卷 lexical coverage。

---

## 7. 可重現測量方法

環境：Windows 11、Git、Node.js v24.18.0；測量日 2026-08-03。

```text
git clone --depth 1 https://github.com/HistoryAtState/frus.git <TEMP>/frus
git -C <TEMP>/frus rev-parse HEAD
git -C <TEMP>/frus count-objects -vH
```

檔案樹另以官方 API 固定 SHA 取得，確認 `truncated=false`：

```text
GET https://api.github.com/repos/HistoryAtState/frus/git/trees/4b4c402f0cce25144ded2198ba9566b6c37c7c49?recursive=1
```

實測演算法：

1. 篩 `volumes/*.xml`，對 Git tree blob `size` 加總。
2. checkout 逐檔讀取；計數每個 `<div ...>` start-tag 中 `type="document"` 的數量與有無。
3. 近似文字量取 `<text>...</text>`，去 XML tag／entity、正規化空白後計 Unicode 字母數字 word-like token。
4. 第 3 步不是 TEI 語意 parser，因此文字／token 數標 UNVERIFIED；production 應以 namespace-aware streaming XML parser 與最終 embedding tokenizer 重算。

### 尚未查到／尚未實測

- 官方沒有公布「全系列總 documents／tokens／chunks」統計；314,483 與 235,013,378 都是本次快照量測。
- 未在這台 RTX 4060 上跑最終 embedding 模型，因此無法 VERIFIED 全量 embedding 時間。
- 未建立最終 lexical/vector DB，因此 HNSW 與 BM25 磁碟倍率仍是工程估算。

## 最終建議

**採用 FRUS 全量 552 卷，但以 lexical-first、dense checkpointed 的方式交付。** 一天作品的可信說法應是：「全量 FRUS 已 parse 並有 BM25；dense 覆蓋率可觀測且可續跑」，只有 benchmark 與 manifest 證明 552/552 dense 完成後，才把敘述改為「全量 hybrid index」。
