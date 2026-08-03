# FRUS 全量 RAG：資料治理與全庫代表性評估 kill-test

研究基準日：2026-08-03  
任務：判斷「全 FRUS 都索引」是否值得，並提供可直接納入實作計畫的 provenance、取樣、評估與停損規則。

> 引述政策：本文沒有逐字引述來源；所有來源內容均為**概括**。未查到頁面原始發表日期時明列「未查到」，不以存取日冒充發表日。

## 結論

**VERIFIED：索引全部「已公開 FRUS historical-document」有研究價值，但盲目索引 `volumes/` 裡每個 XML 沒有價值。** 全庫真正能展示的是跨卷、跨年代的政策變化與來源追溯；單一事件問答用少數卷就能完成，不能證明全庫成本合理。[S1][S3]

**REFUTED：不能把 694 個 XML 等同 694 卷可檢索全文。** 2026-08-03 讀取官方 GitHub contents API 得到 694 個 entries、總計 3,337,854,663 bytes；其中 141 個小於 10 KB。抽查 `frus1993-00v48.xml` 僅 2.4 KB，`revisionDesc status="planned"`，正文只有狀態說明，沒有歷史文件。[S2][S4]

**VERIFIED：作品集應承諾「全已發布文件的可重現 ingest pipeline」，不是承諾一天內一定完成 3.34 GB 的地端 dense embedding。** 先跑全庫 lexical／metadata index 與代表性 dense build；只有通過本文預登記的跨年代收益、provenance 與 coverage 門檻，才把 dense index 擴至所有 published documents。

## 來源與時間

| ID | 官方來源 | 發表／更新時間 | 本文用途 |
|---|---|---|---|
| S1 | [HistoryAtState/frus README](https://github.com/HistoryAtState/frus) | current master commit `4b4c402f0cce25144ded2198ba9566b6c37c7c49`，2026-07-20；查閱 2026-08-03 | TEI P5、每卷一檔、URL／ID 規則、canonical 範圍、release 行為 |
| S2 | [GitHub contents API: volumes](https://api.github.com/repos/HistoryAtState/frus/contents/volumes?per_page=1000) | API 無個別發表日；對 S1 commit 狀態查閱 2026-08-03 | 694 entries、總 bytes、小檔案數；數量由回傳 JSON 現場計算 |
| S3 | [About FRUS](https://history.state.gov/historicaldocuments/about-frus) | 法規 1991 制定、2021 修法；頁面發表時間未查到；查閱 2026-08-03 | FRUS 的官方定位、選錄性質、年代與主題範圍 |
| S4 | [`frus1993-00v48.xml`](https://github.com/HistoryAtState/frus/blob/master/volumes/frus1993-00v48.xml) | S1 snapshot；查閱 2026-08-03 | planned placeholder 實例 |
| S5 | [FRUS repository license](https://github.com/HistoryAtState/frus/blob/master/LICENSE.md) | 發表時間未查到；S1 snapshot；查閱 2026-08-03 | 美國政府作品／CC0 聲明與不保證、不得暗示背書 |
| S6 | [Citing the FRUS series](https://history.state.gov/historicaldocuments/citing-frus) | 發表時間未查到；查閱 2026-08-03 | document number、早期卷 page number、canonical URL 引用規則 |
| S7 | [Repository search config `volumes.xconf`](https://github.com/HistoryAtState/frus/blob/master/volumes.xconf) | 發表時間未查到；S1 snapshot；查閱 2026-08-03 | 官方 Lucene 索引單位、date min/max、volume/type/subtype facets |
| S8 | [FRUS 1969–76 v24 landing page](https://history.state.gov/historicaldocuments/frus1969-76v24) | 該卷出版 2008；查閱 2026-08-03 | volume metadata、TOC、download、tags、front/back matter 實例 |
| S9 | [FRUS 1969–76 v24 Document 176](https://history.state.gov/historicaldocuments/frus1969-76v24/d176) | 文件日期 1969-12-04；卷出版 2008；查閱 2026-08-03 | document／footnote／person／term 與網頁呈現實例 |
| S10 | [Status of the Series](https://history.state.gov/historicaldocuments/status-of-the-series) | 頁面發表時間未查到；內容列至 2026；查閱 2026-08-03 | planned、research、clearance、publication、partial chapters |
| S11 | [Subject Taxonomy](https://history.state.gov/tags) | 頁面發表時間未查到；查閱 2026-08-03 | 511 subjects、560 tagged volumes、20,982 tag assignments；建立 topic sampling frame |
| S12 | [FRUS ebooks FAQ](https://history.state.gov/historicaldocuments/ebooks) | 發表時間未查到；查閱 2026-08-03 | footnote link、舊卷索引限制、document number 優先 |

來源優先序均為 Office of the Historian 與其官方 GitHub；未使用第三方摘要。

## 「全庫」的可執行定義

### 納入

1. 固定 S1 的 commit SHA，保存 `snapshot_sha`、ingest timestamp 與每檔 SHA-256；更新採增量重建，不直接覆寫無版本 index。
2. 解析每個 volume XML，但只把 `tei:div[@type='document'][@subtype='historical-document']` 納入主文件 corpus。S1 明示 document `<div>` 與 `xml:id` 的關係；S7 也把 `type=document` 作為官方全文索引單位。[S1][S7]
3. planned stub 沒有 historical-document node，自然得到零 documents；另保留 volume catalog 記錄，讓系統能回答「尚未出版」，但不得將 placeholder 文字送進回答證據。[S4][S10]
4. 分章發布的卷不可只看 volume-level status 一刀切；已存在的 historical-document 節點可進 corpus，但要保留 `volume_status`、`partial_release=true` 與 chapter path。S10 明示有 volumes with chapters outstanding。[S10]
5. `frus:attachment` 可切成獨立 chunks，但繼承父 `volume_id + document_id`；它沒有經 S1 保證為系列級 canonical ID，不自行創造官方 URL。

### 不納入主文件向量索引

| 區域 | 處理 | 理由／證據 |
|---|---|---|
| `teiHeader`、`facsimile` | 只存 metadata／影像 URL，不 embedding | 樣本含出版、內容日期、tags、revision 與大量 page image metadata；混入正文會污染召回。[S8][S9] |
| TOC、list of papers | 轉成 hierarchy／document order，不全文 embedding | 內容高度重複文件標題；早期卷尤其有長 TOC。S7 官方也以 div type/subtype 分 facet。[S7] |
| Preface、Editorial Note、Sources | 放 `editorial` 獨立 index，預設低權重 | 對「如何編纂／來源檔案」很重要，但不是歷史文件聲音；必須讓 UI 清楚標示 editorial evidence。[S8] |
| Persons glossary | 放 `reference_person` index | 可提供姓名與卷內角色；S1 明示 person IDs 非 canonical、目前只適用單卷且可能改變。[S1][S8] |
| Terms & Abbreviations | 放 `reference_term` index | 適合 acronym expansion；term IDs 同樣只作卷內 join，不作全庫 identity。[S1][S8][S9] |
| Back-matter index | 不進 dense；保留為 lexical navigation／hard-negative 來源 | 它大量重複 names、topics 與 document refs，會造成「索引頁比原文更相似」的假召回。[S8] |
| General index XML（如 `frus1861-99Index.xml`） | catalog-only，與 volume corpus 分開 | 它不是一卷 historical-document；名稱與 S2 directory listing 可辨識。[S2] |
| Appendices | `appendix` 獨立 index，需顯式 filter 才與正文混搜 | 某些 appendix 有實質資料，但證據層級與 document 不同；不可一概刪除或一概當原始文件。[S8] |
| Errata／placeholder／status text | governance store，不作答案證據 | S1 說明 boundary 修正與 placeholder 作法；S4 是 planned placeholder 實例。[S1][S4] |

## TEI provenance 保留契約

### Canonical 與非 canonical

- `volume_id`：檔名 basename 與 root `TEI/@xml:id`；S1 保證 released volume ID canonical。[S1]
- `document_id`：historical-document `div/@xml:id`，例如 `d176`；S1 保證 released document ID canonical。[S1]
- `canonical_url`：`https://history.state.gov/historicaldocuments/{volume_id}/{document_id}`。[S1][S6]
- `document_number`：`div/@n`；從 1955–57 subseries 起，官方建議以 document number 引用。更早卷的網路版雖補上 document ID／number，正式引用仍須保留 page range。[S6]
- `footnote_id`：例如 `d176fn1`，只作 volume/document 內定位；回答顯示 note number 與 parent document URL。S1 只保證 volume/document ID canonical，不保證 footnote ID。[S1][S9]
- `person_ref`／`term_ref`：保存原始 `@corresp`／`@target` 與 surface text，但 key 必須是 `(volume_id, local_id)`；禁止把相同 local ID 或姓名直接當系列級同一人／同一詞。[S1]
- `page_id`：保存 `pb/@n`、`@xml:id`、`@facs` 以服務早期卷引用；S1 明示 page break ID 非 canonical、可能修正。[S1][S6]

### 最小 metadata schema

```text
snapshot_sha, source_file, source_blob_sha256, ingested_at
volume_id, volume_title_complete, subseries, volume_number, volume_title
volume_status, partial_release, publication_year, content_date_min, content_date_max
document_id, document_number, document_title, document_date_min, document_date_max
document_date_precision, document_order, hierarchy_path, compilation_id, chapter_id
chunk_id, chunk_index, section_kind, attachment_index, text
page_start, page_end, page_ids[]
footnotes[{local_id, n, type, text}], source_note_text
persons[{surface, local_ref, role}], terms[{surface, local_ref}]
volume_topic_tags[], administration_tags[]
canonical_url, rights='US-PD+CC0', parser_version, validation_errors[]
```

`chunk_id` 是本系統 ID，例如 `{volume_id}:{document_id}:body:0004`，不得描述為官方 canonical ID。

### 日期

S7 官方 search config 直接索引 `frus:doc-dateTime-min/max`；本系統應保留 interval，而不是強迫每份文件只有單日。[S7] 1861 樣本存在整年範圍的文件，1969 樣本則是單日範圍；因此：

- exact day：min/max 同日，`precision=day`。
- month/year/range：保留 min/max，`precision` 記 month/year/range。
- 查詢「before/after/between」使用 interval overlap；不可只取 midpoint。
- `publication_year` 與 `document_date` 分開，禁止把卷出版年當事件年。

### footnote 與 source note

1. 解析 note 時保留 `@n`、`@type`、`@xml:id` 與所在段落 offset。
2. `type=source` 的 source note 同時存 structured field 與 parent chunk context；其他註腳預設附在鄰近 chunk 尾端。
3. 長註腳可成獨立 `section_kind=footnote` chunk，retrieval score 降權，永遠帶 parent canonical URL。
4. 生成答案必須區分「文件正文稱」與「編者註腳說明」。S9 顯示 Document 176 的來源與「attached but not printed」均在註腳。[S9]

## Metadata filters

### 可作 hard filter

- `volume_id`, `document_id`, `document_number`
- `subseries`, `administration_tags`
- `document_date_min/max` interval
- `section_kind in {document, attachment, footnote, editorial, appendix, reference_person, reference_term}`
- `volume_status`, `partial_release`
- `hierarchy_path`, `compilation_id`, `chapter_id`
- `publication_year`

### 只能先作 coarse／soft filter

- `volume_topic_tags`：S11 的 taxonomy 是套在 volume 資源的成長中標記，不是每份 document 的 gold topic；用它做候選路由，不能用「某卷沒有 tag」推導文件沒有該主題。[S11]
- `persons`、`terms`：local IDs 不穩定且只保證卷內；跨卷搜尋要用 normalized surface aliases，結果仍回報原始字串。[S1]
- `document_genre`（telegram、memorandum、minutes 等）：可從 head 規則推導，但未查到 repo 對全庫保證一致的官方 genre 欄位，標 `derived=true`。
- classification／archive collection：常藏在 source note 自由文字；未查到全庫一致 schema，不可先承諾精準 hard filter。

## 全庫代表性評估集

### 取樣框架

評估不是從熱門冷戰卷手挑 30 題，而是先建 `document_inventory`，再做三維分層：

1. **Era（6 層）**：1861–1899；1900–1918；1919–1945；1946–1960；1961–1976；1977–最新已發布年份。FRUS 官方列表橫跨 Lincoln 至 Clinton administration；最新 publication 狀態須由 S10 取得，不從 planned filename 猜測。[S3][S10]
2. **Topic（6 層）**：bilateral/regional；war/crisis；security/arms/intelligence；economic/trade/energy；international organizations/global issues；foreign-policy organization/public diplomacy。這些範圍由 S3 與 S10 的官方分類支持。[S3][S10]
3. **Volume**：每個 era-topic cell 至少抽 2 卷；同一卷最多占 retrieval gold documents 的 10%。以 S11 volume tags 建候選框，但人工確認 document-level relevance。[S11]

最小正式集為 **60 題**：30 題 single-document、12 題 within-era multi-document／timeline、12 題 cross-era comparison、6 題 adversarial／no-answer。另保留 20 題 development set，不與測試集共享 gold documents。

### 至少 36 種題型

下列是出題模板；每題完成時必須人工填 `gold_document_ids`、`gold_spans`、`time_constraint`、`required_notes` 與 hard negatives，不可讓 LLM 自產 gold。

| ID | 題型／模板 | 必測能力 |
|---|---|---|
| Q01 | 指定卷與 document number，找文件主旨 | ID／metadata lookup |
| Q02 | 給一段短引文，找 canonical document | exact lexical |
| Q03 | 用現代同義改寫問歷史措辭 | semantic retrieval |
| Q04 | 用中文問英文 FRUS 文件 | cross-lingual dense／query translation |
| Q05 | 給日期＋寄件／收件角色找文件 | date + person filter |
| Q06 | 問某 acronym 在指定卷的全名 | term reference index |
| Q07 | 問某人在該卷所任職位與時間 | person reference index |
| Q08 | 問正文主張，答案必須排除編者註 | section discrimination |
| Q09 | 問來源檔案／classification，答案只可引用 source note | footnote retrieval |
| Q10 | 問「附件未刊／已附」等編者資訊 | footnote vs body |
| Q11 | 長文件中找單一小節 | intra-document chunking |
| Q12 | 表格／list 中找精確項目 | TEI structure preservation |
| Q13 | attachment 內容找回 parent document | attachment provenance |
| Q14 | 指定 page range 找早期卷內容 | early-volume page support |
| Q15 | 同名人物在同卷的 role disambiguation | local person refs |
| Q16 | 相同 acronym 在兩卷的不同／相同展開 | volume-scoped term refs |
| Q17 | 指定起訖日列出事件文件並排序 | interval filter + chronology |
| Q18 | 找某事件最早被討論的已發布文件 | first-hit + date precision |
| Q19 | 找某政策最後一次出現於指定 subseries | last-hit + coverage |
| Q20 | 重建同一卷內決策的提案→討論→決定 | within-volume multi-hop |
| Q21 | 比較同一 administration 不同機構的立場 | multi-document comparison |
| Q22 | 比較兩國／兩位官員在同一危機的主張 | actor contrast |
| Q23 | 找被拒絕的替代方案及拒絕理由 | negative/contrast evidence |
| Q24 | 區分文件中的事實、推測與未證實說法 | epistemic language |
| Q25 | 比較事件前後政策措辭變化 | before/after timeline |
| Q26 | 比較兩個 administrations 對同一 topic 的延續與改變 | cross-era synthesis |
| Q27 | 比較戰時與戰後對同一國的政策文件 | era + bilateral comparison |
| Q28 | 追蹤某人物跨多卷的角色變化 | alias + cross-volume timeline |
| Q29 | 追蹤某組織／術語跨年代稱呼變化 | term drift |
| Q30 | 比較同一政策在 regional 卷與 functional 卷的描述 | cross-volume routing |
| Q31 | 問某項主張有哪些相互衝突的文件 | contradiction retrieval |
| Q32 | 問索引頁提到的 topic，要求回到原文件而非 back index | index leakage defense |
| Q33 | 問 planned／未出版範圍的問題 | abstain + publication status |
| Q34 | 給不存在的 document ID／錯卷號 | invalid canonical lookup |
| Q35 | 日期超出 corpus／要求現代政策結論 | temporal abstention |
| Q36 | 問「FRUS 是否完整代表所有政策資料」 | scope-aware answer；須說 FRUS 是官方選編紀錄，不把無收錄當無事件 [S3] |

### Gold record

```text
query_id, query_zh, query_en, era, topic, question_type
allowed_filters, required_filters, forbidden_sections
gold_documents[{volume_id, document_id, canonical_url, relevance_grade}]
gold_spans[{document_id, text_hash, footnote_n, page_n}]
answer_facts[], comparison_axes[], chronology_order[]
hard_negatives[], unanswerable, abstention_reason
```

hard negatives 優先從同卷、相鄰日期、同人物、同 abbreviation 的錯文件抽取，才能測出 metadata 與 provenance，而非只測 topic matching。

## Retrieval 與回答評估

### 必跑 baselines

1. BM25／lexical，document metadata prepend。
2. dense only。
3. BM25 + dense fusion。
4. hybrid + reranker。
5. hybrid + metadata router（era／volume／section）。

### Retrieval metrics

- document-level Recall@5、Recall@10、MRR@10、nDCG@10。
- `filter_compliance`：結果是否全部符合 date／volume／section hard filter。
- `canonical_resolution_rate`：chunk 能否解析成存在的 released volume/document URL。
- `timeline_coverage`：多文件題的必要時間節點召回比例。
- 分層報表：每個 era、topic、question type 各自分數；只報總平均不合格。

### Answer metrics

- claim citation coverage：可驗證主張有 citation 的比例。
- citation entailment：引用 span 是否支持緊鄰主張。
- provenance correctness：volume、document、note／page、URL 是否一致。
- chronology correctness：時間順序與 interval 不矛盾。
- comparison completeness：gold comparison axes 覆蓋率。
- abstention precision／recall：planned、超時代、無證據題能否拒答。

LLM judge 只能補充，不可取代人工 gold document／span 與 deterministic URL 檢查。

## 引用契約

每個回答至少包含：

```text
[FRUS {subseries}, {volume_number}, {volume_title}, Document {document_number},
{document_date}; {canonical_url}]
```

- 1955–57 以後：以 document number 為主要 locator。[S6]
- 更早卷：加 `pp. {page_start–page_end}`；網路 `document_id` 仍保留，因官方說舊卷的網頁 number 不一定能讓紙本讀者定位。[S6]
- 用到註腳：加 `note {n}`，URL 仍指 parent document；不把 `footnote_id` 說成官方永久 ID。
- 用到 editorial／persons／terms：標籤分別寫 `[Editorial material]`、`[Persons glossary]`、`[Terms glossary]`，不得假裝是歷史文件正文。
- 比較題每個主要 actor／年代至少各有一個直接支持 citation；不能用一份文件概括全時期。
- citation resolver 發現 volume/document 不在 snapshot manifest、status planned、URL 不一致時，回答 fail closed，不顯示無法追溯的生成內容。

## 三個預登記失敗條件

### F1：全庫 dense 成本超過作品集價值

**觸發證據**：在目標機器上，代表性 10% published-document build 推估全庫超過 24 小時、peak disk 超過預留上限，或 parser／embedding failure 超過 1%；同時 12 題 cross-era set 的 full-candidate hybrid Recall@10 相較 era-routed subset 提升少於 5 percentage points。

**改判**：全 published documents 保留 BM25 + metadata；dense 只索引代表性／熱門卷並按 query route，README 不再聲稱「全庫 dense」。

### F2：provenance 不可靠

**觸發證據**：100 個分層抽樣 chunks 中，低於 99% 能解析到正確 released `volume_id + document_id + canonical_url`；任何 planned placeholder 成為答案證據；或 footnote／attachment 被錯掛父文件超過 1%。

**改判**：停止 demo release，先修 TEI parser 與 manifest；在通過前只展示已人工驗證卷，不能用生成品質掩蓋引用錯誤。

### F3：全庫品質只在冷戰熱門卷有效

**觸發證據**：任一 era 的 Recall@10 比最佳 era 低超過 15 percentage points，或任一 era-topic cell 少於 2 卷／5 題；早期卷 page citation correctness 低於 95%；時間序列題 timeline coverage 低於 0.80。

**改判**：不發布單一「全庫準確率」；針對早期卷採結構／拼字與 page-aware chunking，補齊該 strata 後重測。若仍失敗，產品明確標示支援年代，撤回全庫品質主張。

## 改判條件與實作順序

**值得全庫的唯一證據**不是「成功建立很多 vectors」，而是：

1. 12 題 cross-era comparison 中 hybrid 相較單卷／單 era baseline 有可重現提升。
2. 六個 era strata 都通過 provenance 與 retrieval 門檻。
3. 使用者實際能從每個主張跳到 Office of the Historian canonical document。

建議順序：先 snapshot／manifest → TEI parser + 6-era fixture tests → 全庫 inventory／BM25 → 60 題 gold eval → 10% stratified dense → 跑 F1–F3 → 通過才擴全 published dense → 增量更新與 regression eval。

## 已確立／未查到

- **已確立**：volume 與 document IDs 是官方 canonical；person、term、page IDs 不是。[S1]
- **已確立**：repo 含 planned placeholders，directory entry 不等於 released historical content。[S2][S4][S10]
- **已確立**：官方本身以 div type/subtype、volume 與 date min/max 建搜尋欄位，支持本文的 metadata-first 設計。[S7]
- **未查到**：全庫一致、可直接使用的 document genre 欄位。
- **未查到**：人物／術語的系列級穩定 authority IDs；S1 反而明示目前尚未 canonical 化。
- **未查到**：在使用者目標硬體上，3.34 GB XML 的實際 parsing／embedding 時間；必須跑 10% 分層 benchmark，不可估成既定事實。
- **未查到**：所有 appendix 是否都應視為歷史文件；因此採獨立 index、逐題明示 evidence type。
