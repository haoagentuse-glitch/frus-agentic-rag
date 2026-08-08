# 交接：目前狀態與已知問題

最後更新 2026-08-05。這份文件記錄的是**尚未解決**的事，以及每件事目前量到什麼。
已完成且穩定的部分在 `README.md`。

---

## 0. 現在可以直接跑的三件事

```bash
# 1. cross-encoder 分數分佈（純檢索，無 LLM，數十秒）
FRUS_GPU=1 ./scripts/dev.sh python scripts/score_distribution.py
```

```bash
# 2. 完整消融（350 runs，循序，約 3-4 小時）
FRUS_GPU=1 ./scripts/dev.sh frus eval 2>&1 | tee reports/ablation_run.log
```

```bash
# 3. 拒答成因探查（需要 reports/agent_ablation.json 存在）
FRUS_GPU=1 ./scripts/dev.sh python scripts/_abstain_probe.py 24
```

**`FRUS_GPU=1` 不能省。** 少了它容器看不到 CUDA，cross-encoder 掉到 CPU，
同一批 50 對從 1.0s 變成 62.6s。句子過濾的 pair 數是重排的 6-15 倍，所以在 CPU 上
`focus_active()` 會直接停用自己並在 trace 記下原因——這是刻意的，不是退化。
`frus eval` 開跑前會印出實際裝置。

---

## 1. 已知問題（依嚴重度）

### 1.1 中文問題有約 40% 拒答，成因是 evidence id 抄寫失敗 —— **最高優先，未修**

上一輪 350 runs：可回答題中 zh-TW 拒答 61/150（40.7%），en 18/150（12.0%）。
其中 57/79 是「有 claim、每個 claim 都帶 id、但引用數 0」，即 id 全被
`agent/citations.py` 的閘門擋掉。

smoke 測試 4/4 完全照語言分：兩題中文都是
`no claims: answer carries no citable statements` 且 `repaired: true`
（B3 的 `repair_claims` 把 claim 全數丟光，代表模型寫的 id 沒有一個有效），
兩題英文都正常作答。

推論：qwen3:4b 在中文生成模式下無法逐字複製英文 chunk id。**尚未直接驗證**——
需要看模型實際吐出的 id 長什麼樣（是拼錯、是截斷、還是整個編造），
`scripts/_abstain_probe.py` 會把 `ids_not_retrieved` 印出來，但還沒跑過完整一輪。

可能的修法，由便宜到貴：
1. 在閘門前做一次 id 的模糊比對（編輯距離 1-2 內對回真實 id）。風險是把幻覺救成引用。
2. 把 id 換成短序號（`[1]`、`[2]`），程式端維護序號到 chunk_id 的映射。
   模型只需要抄一個數字。這大概是最有效且最不失真的做法。
3. 讓中文答案的 claim 部分用英文生成，只有 `answer_text` 翻成中文。

另外 `citations.py` 的閘門是**全有全無**：8 個 claim 裡 1 個 id 壞掉，
整篇答案作廢，其餘 7 個有效引用一起丟。只有 B3 有一次 `repair_claims` 補救。
不論上面選哪條路，這個行為都該改成部分接受。

### 1.2 LLM grader 已被移除，但取代它的門檻還沒定 —— **機制已就位，參數未定**

`config.grader_mode` 預設 `"score"`，`agent/nodes.py::grade_evidence` 走
`retrieval/select.py::select_evidence`，不再有 LLM 呼叫。B2/B3 的預算因此
從 4 次降到 3 次。

**但 `score_keep_absolute` 和 `score_keep_margin` 都是 `None`，所以現在等於全留。**
這是刻意的：cross-encoder 的 logit 沒有校準，憑直覺挑門檻是在猜一個沒人看過的分佈。

為什麼移除 LLM grader（兩個方向都失敗的紀錄）：

| 版本 | 行為 | 結果 |
|---|---|---|
| 原始（點名保留） | schema `max_length=8`，視窗 10 片 | 砍掉 40% 文件，含 15 份 gold；單 hop 時有 2 片純粹被 schema 砍掉 |
| 反轉（點名拒絕） | 兩版 prompt，交集規則 | 4/4 smoke 全部 `rejected=[]`，完全空轉 |

verdict 本身正常（partial/supported 有變化），所以模型有在判斷，只是不肯點名任何一片該丟。

**已量到的分佈（僅 5 cases / 7 gold rows，樣本太小，只能看形狀）：**

| 切片 | gold 均值 | 非 gold 均值 | 保留 100% gold 的門檻 | 該門檻保留全體比例 |
|---|---:|---:|---:|---:|
| 整體 | +0.392 | −3.298 | −1.963 | 27.5% |
| zh-TW | +0.157 | −3.522 | −1.963 | 30.0% |
| en | +0.705 | −3.077 | −0.527 | 9.0% |
| lookup | **+3.158** | −0.810 | +2.894 | 5.0% |
| multihop | **−0.885** | −3.686 | −1.963 | 10.0% |
| correction_matched | −0.459 | −4.284 | −0.527 | 5.0% |

**關鍵限制：分數尺度隨題型大幅移動。** lookup 的 gold 落在 +3.2，
multihop 的 gold 落在 −0.9，差了四個單位。一個為 lookup 設的絕對門檻會把
multihop 的 gold 全部殺光。

所以下一步應該是：
1. 跑完整的 35 cases（`scripts/score_distribution.py` 不帶 `--limit`），
   讓每個切片有足夠的 gold row。
2. **相對 margin 比絕對門檻重要**，因為它隨題型自動平移。絕對門檻只該當作
   一個很低的地板（例如 −4），擋掉明顯無關的。
3. `score_min_per_hop`（預設 3）是最後的保險，確保多跳題不會整跳消失。

`select_evidence` 在門檻未設時會把 `score_range` 寫進 trace，
所以就算不設門檻，跑一輪也會累積分佈資料。

### 1.3 B3 的糾正迴圈在門檻未設時不會觸發

`select.py::missing_hops` 靠 `score_keep_absolute` 判斷一個 hop 有沒有找到東西。
門檻是 `None` 時它回傳空陣列，所以 **B3 現在等於 B2**。
`tests/test_graph_budget.py` 用 `FRUS_GRADER_MODE=llm` 固定成舊模式，
才還能覆蓋糾正路徑——那是目前唯一還在測 LLM grader 的地方。

### 1.4 上一輪的延遲數字被 Phoenix 汙染 —— **已修，但舊數字要作廢**

span payload 原本是每題約 200KB（4 次 LLM 呼叫各帶 12000 字元的 prompt 與
completion、每次檢索 20 份 × 2000 字元）。Phoenix 吞不下，exporter 每批逾時 10 秒。

同一題 B3：**追蹤開 789s，關掉 42s。**

已把上限降到 4000 / 500 字元 / 5 份文件（`observability.py`，可用
`PHOENIX_MAX_*` 環境變數調回），重跑後匯出錯誤 0 次。

**因此上一輪報告的 gate 5（B3 p95 122.70s、4.2× B0）不可信**，
其中相當比例是儀器開銷而非 agent。舊結果留在 `reports/v1-chunk-grader/`，
延遲欄位不要引用。

### 1.5 路由 macro-F1 0.4477，遠低於 0.80 門檻

規則式路由更差（0.2258），所以 SPEC 裡「不到 0.80 就退回規則式」的預案
反而會讓結果變糟。這是產品決策，尚未做。

### 1.6 `timeline_search` 完全沒有測試覆蓋

gold set 裡沒有任何 timeline 題。三類該補的題型：
高扇出多跳（10-14 份 anchor）、有時間範圍的編年題、過寬而應拒答的題。

### 1.7 基礎映像檔重複的 torch 層 —— **已解決**

成因不是「重複的 layer」而是**被遮蔽的 venv**：`jobshift:latest` 與本專案都把
virtualenv 放在 `/opt/venv`，本專案的 `uv sync` 覆寫了它，而 Docker layer 是疊加的，
覆寫不會回收先前 layer。映像因此帶著一套 6.4 GB 完全無法存取的 torch 2.6.0+cu124。

改以 `python:3.12-slim-bookworm` 為基底、依 `uv.lock` 安裝：**19.2 GB → 9.56 GB**，
`torch.cuda.is_available()` 仍為 True。不需要 CUDA 基底映像，torch wheel 自帶 runtime。
`scripts/dev.sh` 改用 `frus-agentic-rag:latest`（`FRUS_IMAGE` 可覆寫），不再依賴 jobshift。

Ollama 已拆成獨立的 compose 專案 `Ollama/compose.yaml`，與 Phoenix 同樣的做法。
理由：它是多個專案共用的模型伺服器，掛在本專案底下時，`compose down` 會把大家的
服務一起停掉，而 2.3GB 的權重存在一個以本專案命名的 volume 裡。

```bash
docker compose -f Ollama/compose.yaml up -d
docker compose -f Ollama/compose.yaml exec ollama ollama pull qwen3:4b-instruct
```

權重改放在 external volume `ollama-models`（不帶專案前綴，因為它比任何專案都長壽），
遷移時已從舊 volume 複製過去，不需要重新下載。本專案透過 host gateway 的 11434 連它。

映像清理總計：

| | 前 | 後 |
|---|---:|---:|
| `frus-agentic-rag` | 19.2 GB | 9.56 GB |
| `jobshift:latest` | 10.6 GB | 已刪除 |
| build cache | 11.2 GB | 0 |
| Docker 總計 | 約 39 GB | **17.9 GB** |

### 1.8 WSL 記憶體未設定

主機 32GB，WSL 預設只拿 15.5GiB。建議 `.wslconfig` 設 `memory=24GB, swap=8GB`。未套用。

---

## 2. 這一輪改了什麼

| 改動 | 檔案 | 狀態 |
|---|---|---|
| 移除 runtime LLM grader，改用 cross-encoder 分數確定性過濾 | `retrieval/select.py`（新）、`agent/nodes.py`、`config.py` | 機制完成，門檻未定 |
| 每份 evidence 記錄 raw score / rank / 是否保留 / gold 標記 | `retrieval/select.py::score_rows`、`models.Evidence.rerank_score` | 完成 |
| 分數分佈分析工具（中英 × 題型 × gold 標記） | `scripts/score_distribution.py`（新） | 完成，已跑過 5 cases |
| 句子級 cross-encoder 過濾 | `retrieval/focus.py`（新） | 完成，保留 63-74% 文字，成本 2-4s |
| grader schema 上限 8 → 反轉為拒絕清單 | `agent/schemas.py`、`agent/prompts.py` | 完成但證實空轉，已被 1.2 取代 |
| CPU 上自動停用句子過濾並記錄原因 | `retrieval/focus.py::focus_active` | 完成 |
| Phoenix payload 降到 1/10 | `observability.py` | 完成 |
| `dev.sh` 轉發任意 `FRUS_*` 環境變數 | `scripts/dev.sh` | 完成，配對實驗靠這個 |
| retriever fingerprint 納入 `+focus` | `evaluation/ablation.py` | 完成，舊 run 會自動丟棄重跑 |

測試 35 passed，ruff / mypy 全過。新增 `tests/test_focus.py`（4 個），
其中一個抓到 `Mr. Nolting` 被句子切分器切開的真 bug。

---

## 3. 建議的下一步順序

1. **跑完整的 `score_distribution.py`**（無 `--limit`），並補齊 3.1 的指標。
   便宜、無 LLM、可重複，而且是所有下游決策的前提。
2. **比較 3.2 的三種篩選策略**，A 為基線。
3. **修 1.1 的 id 抄寫問題**（建議走短序號那條）。這是目前唯一還在丟掉正確答案的缺陷。
4. 設好門檻後跑一輪完整消融，這時候的延遲數字才第一次乾淨。
5. 補 1.6 的三類題型，才有辦法驗證多文件合成與拒答行為。

### 3.1 決定門檻前必須先量的東西

目前只有均值與「保 100% gold 的地板」，不足以定規則。以下為缺口，
`scripts/score_distribution.py` 需擴充：

| 指標 | 現況 | 為什麼需要 |
|---|---|---|
| gold / non-gold 的 P5、P25、P50、P75、P95 | **已有**（`quantiles()`） | 看重疊區有多寬，而非只看均值差 |
| `recall@1 / @2 / @3 / @5 / @10` | 缺 | 若 recall@5 已接近 recall@10，固定 top-5 就夠，不需要任何自適應規則 |
| **每題**保留比例的分佈 | 缺（目前只有全體比例） | 全體 27.5% 可能是「多數題留 10%、少數題留 90%」。整體數字會藏住尾巴 |
| `top_score − gold_score` 的分佈 | 缺 | **這就是 `score_keep_margin` 的分佈本身**，是決定該參數的直接證據 |
| 最佳 gold 的 rank 分佈 | 缺 | 決定 `score_min_per_hop` 的下界；若 gold 常落在 rank 8，min-3 會殺掉它 |
| 每個 hop 是否至少留住一份 gold | 缺 | `score_min_per_hop` 唯一要保證的性質。**注意**：目前的分佈腳本走單跳檢索（`hop="main"`），要量這項得先過 planner |
| 中英文分開的 answer success rate | 上一輪已有（40.7% / 12.0%） | 見 1.1，成因已知 |
| 過濾前後的 context **token** 數與 synthesis 延遲 | 只有字元數（保留 63–74%）與節點耗時 | 字元不等於 token，中文尤其失真；synthesis 延遲才是 context reduction 的真正回報 |

**優化目標不是把保留比例壓低。** 要同時看四項：

```
gold recall（不能掉）
+ 最終回答品質（judge correctness）
+ context reduction（token 數）
+ latency reduction（synthesis 節點耗時）
```

保留比例本身只是手段。一個把保留比例壓到 5% 但 gold recall 掉 10pp 的設定是失敗的，
而一個保留 40% 卻讓 synthesis 快一倍、正確率不變的設定是成功的。

### 3.2 三種篩選策略，A 是基線

`retrieval/select.py` 已能表達三者，差別只在 config：

| 版本 | 設定 | 說明 |
|---|---|---|
| **A. 固定 top-k** | `min_per_hop = max_per_hop = k`，margin/absolute 皆 None | 最單純的基線。**先跑這個** |
| **B. min-k + relative margin** | `min_per_hop = k`、`keep_margin = m` | 隨題型自動平移，不受 1.2 的尺度問題影響 |
| **C. min-k + margin + max-k** | B 再加 `max_per_hop` | 加上天花板，擋住「整池都高分」時 context 爆掉 |

預期 C 最穩定，但**不要跳過 A**。若 B 或 C 相對 A 沒有顯著提升 gold recall 或降低 context，
就直接留 A——簡單問題不值得用複雜規則解，而這個專案已經有過一次為了聰明而失效的階段（見 1.2）。

實作備註：`score_max_per_hop` 已加入 `config.py` 與 `select_evidence`，
天花板在兩個地板之後套用，所以 `max` 永遠勝過 `min`，設定衝突時不會反而留得更多。
**此改動尚未跑過 lint / mypy / pytest**（額度用盡時中斷），合併前請先跑一次。

### 3.3 關於重新定義 B0–B3 的提案

有一個提案是把系統重新定義為：B0 基礎檢索、B1 hybrid + cross-encoder rerank、
B2 加確定性分數篩選、B3 加句子過濾。

**不建議直接套用在 B0–B3 這組名字上。** 理由：現有的 B0–B3 是 SPEC 預登記的
**agent 圖**消融（規劃分解、grading、糾正迴圈），第 2 節的五條門檻全部以此定義為準。
改掉語意會讓跨輪次的數字失去可比性，而且會失去量測「規劃分解」的能力——
那是目前唯一通過的門檻（multi-hop recall +24.2pp）。

建議改為**新增一組獨立的檢索階段消融**（例如 R0–R3），與 B0–B3 並存：

```
R0：hybrid 檢索，無重排
R1：+ cross-encoder 重排
R2：+ 確定性分數篩選
R3：+ 句子級過濾
```

這樣兩個問題各自可答：agent 圖值不值得（B 系列），檢索後處理值不值得（R 系列）。

不過該提案有一點是對的且需要處理：**句子過濾目前對所有系統一律套用**
（我刻意這樣做，讓各系統的檢索保持可比），因此在現有 ablation 中**量不到它的效果**。
要量它必須跑配對的兩輪，靠 `dev.sh` 的環境變數轉發：

```bash
FRUS_GPU=1 FRUS_FOCUS_SENTENCES=false ./scripts/dev.sh frus eval
```

fingerprint 會記為 `hybrid+rerank`（無 `+focus`），與開啟的那輪自動分開，不會被混算。

---

## 4. 不要重蹈的覆轍

這個專案反覆出現同一種故障：**元件安靜地失效，而儀器回報成功。**
已經發生過至少五次——reranker 因為執行緒競爭從未真正執行過卻回報有跑、
planner 捏造 volume_id 導致 52% 子查詢回傳零筆而 `errors: []`、
grader 把沒看過的證據當成拒絕、Ollama 停機期間寫下 42 筆假拒答、
Phoenix 把延遲灌水 4-19 倍而看起來像 agent 慢。

所以新增任何一個「可選」階段時，trace 必須記錄**它實際做了什麼**，
而不是「它被呼叫了」。`focus_active()` 回傳停用原因、
`select_evidence` 在未設門檻時記下 `skipped` 與 `score_range`、
fingerprint 問的是「有沒有真的跑」而不是「有沒有設定」——都是為了這件事。
