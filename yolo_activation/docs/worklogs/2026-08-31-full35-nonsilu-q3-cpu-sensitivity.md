# 2026-08-31：Full35 全非 SiLU Q3 純 CPU 續跑與 CUDA 防護

## 任務與範圍

- 依使用者要求恢復 Full35 Q3 的 11-region zero-shot sensitivity queue。
- 只執行 CPU 驗證，不重新訓練、不執行 backward、不更新權重。
- 從既有完整結果續跑，不重算已完成案例。
- 強制保證本實驗不建立 CUDA context，也不干擾其他人的 GPU 工作。

## 資料與權重契約

- COCO 使用既有 COCO2017 完整驗證集。
- BBAT5 使用不可變的 `/home/uxin/yolo/original/pose/derived/bbat5-v1/` 正式資料。
- BBAT5 Detect 使用 `configs/detect.yaml`，Pose 使用 `configs/pose.yaml`。
- 資料比例為 100%，沒有抽樣、重新切分、替換來源、修改影像或修改標註。
- qSiLU 基底權重為
  `artifacts/runs/full35/short-recovery-v2-lr01-uniform-qsilu-pq-seed1/inference/best_joint.pt`。
- 本階段只做逐 region activation 替換後的 zero-shot validation；epoch、optimizer 與梯度更新均不適用。

## 發現的 GPU 問題

原先四個 Q3 程序雖然在 Ultralytics validation 明確指定 `device="cpu"`，系統層
`nvidia-smi` 仍顯示每個程序建立約 498 MiB CUDA context，合計約 1.99 GiB。這代表
「tensor 推論在 CPU」仍不足以保證「完全不碰 GPU」。

確認下列四個 PID 都屬於本次 Q3 shard 後，已依使用者要求以 SIGTERM 停止：

- `2171314`：LeakyReLU 1/8 shard A。
- `2171384`：Hardswish shard B。
- `2171418`：Hardswish shard A。
- `2171642`：LeakyReLU 1/8 shard B。

停止後重新查詢，這四個 PID 均已消失；GPU 清單只剩其他使用者的 MambaPose 程序
`1670604`，本工作沒有停止或修改該程序。

## 變更內容與原因

修改 `scripts/full35_nonsilu_q3.py`：

1. 在匯入 PyTorch 與可能間接匯入 PyTorch 的套件之前，無條件設定
   `CUDA_VISIBLE_DEVICES=""`。
2. 新增 fail-closed CPU guard，主流程開始時必須同時滿足：
   - `CUDA_VISIBLE_DEVICES` 為空。
   - `torch.cuda.is_available()` 為 `False`。
   - `torch.cuda.device_count()` 為 0。
3. 任一條件不符即丟出例外並停止，不允許悄悄退回 GPU。
4. 將三項 CPU safety 狀態寫進輸出 `summary.json`，方便後續稽核。

修改 `tests/test_full35_nonsilu_q3.py`：

- 驗證 CUDA 隱藏設定出現在 `import torch` 之前。
- 驗證 runner 的 CPU guard 回傳 CUDA 不可用且裝置數為 0。

## 驗證方式與結果

- `pytest -q tests/test_full35_nonsilu_q3.py tests/test_full35_adapter.py`：9 項通過。
- `ruff check scripts/full35_nonsilu_q3.py tests/test_full35_nonsilu_q3.py`：通過。
- 測試程序從啟動時設定空的 `CUDA_VISIBLE_DEVICES`；測試後 `nvidia-smi` 沒有出現本專案 PID。
- 先恢復一個實際 shard，確認已進入 COCO 5000 張影像、157 batches 推論後再次查詢
  `nvidia-smi`；本專案 PID `2241745` 不在 GPU 程序清單。
- 恢復四個 shard 並再次查詢；GPU 清單仍只有既有 MambaPose，Q3 GPU 占用為 0 MiB。
- Ultralytics 日誌明確顯示 `CPU (Intel Core Ultra 9 285K)`、`0 gradients`。

## Queue 配置與完成狀態

四個 shard 都以 `nice -n 10`、每個 `torch-threads=4` 執行，合計最多 16 個
PyTorch inference threads：

- Hardswish A：前 6 個 regions。
- Hardswish B：後 5 個 regions。
- Dyadic LeakyReLU 1/8 A：前 6 個 regions。
- Dyadic LeakyReLU 1/8 B：後 5 個 regions。

四個 shard 全部以 exit 0 結束，22/22 candidate-region 結果完整。停止前已完成的 8 個
案例通過快取契約後重用；停止時未完成的 4 個案例則從案例開頭重新驗證。最終合併程序只
讀取 22 個完整快取並重建總表與圖，不重新推論、不訓練。

最終輸出：

- [單一權威報告](../../reports/full35-q3-cpu-final-report.md)
- [相對 uniform qSiLU 圖](../../reports/q3-nonsilu-cpu/nonsilu-q3-11-region-local.png)
- [相對 accepted SiLU gate 圖](../../reports/q3-nonsilu-cpu/nonsilu-q3-11-region-global.png)
- [機器可讀 summary](../../reports/q3-nonsilu-cpu/summary.json)
- [逐列 CSV](../../reports/q3-nonsilu-cpu/rows.csv)

## 最終結果

- uniform recovered qSiLU 相對 accepted SiLU 的八項最差差值為 −0.008810，通過 −0.015 gate。
- Hardswish 的 deployment regions 中，`backbone_attention`、`masf` 與
  `neck_attention` 通過正式 gate；其餘 deployment regions 不通過。
- Dyadic LeakyReLU 1/8 的 deployment regions 中只有 `masf` 通過，八項最差差值為
  −0.007597；其他 deployment regions 精度下降明顯，不適合直接替換。
- `detect_one2many`、`pose_flow`、`pose_one2many` 是 training-only regions，
  zero-shot 指標不變不代表部署推論可節省運算。
- 量化階段可優先保留 qSiLU 作 fallback，再分別驗證 Hardswish 的
  `neck_attention`／`masf`／`backbone_attention`；多區組合仍需另做完整驗證。

## 困難與解法

- 困難：`device="cpu"` 沒有阻止框架初始化 CUDA context。
  - 解法：必須在 Python 匯入 PyTorch 前隱藏 CUDA，再於執行期 fail-closed 稽核。
- 困難：內建 `apply_patch` 因 `bwrap: loopback: Failed RTM_NEWADDR` 無法讀取檔案。
  - 解法：使用 `git apply --check` 驗證後才套用相同補丁。
- 困難：Git 根目錄位於上一層，第一次備援補丁使用子專案相對路徑而被 Git 明確標為
  `Skipped patch`。
  - 解法：改用 repo-relative 的 `yolo_activation/...` 路徑，看到
    `Applied patch ... cleanly` 後再核對檔案內容。
- 其他困難：無。

## 未解事項與風險

- CPU 與 GPU 浮點結果不保證 bitwise 相同；本報告的比較會全部使用同一 CPU baseline。
- 這是單一 region 的一階 zero-shot sensitivity，不能直接代表多 region 混合 policy。
- LeakyReLU 1/8 目前是 float graph 的精確 1/8；完整 INT8、PTQ/QAT 與 RTL bit-exact
  驗證留給後續量化階段。

## GitHub 發布整理

- 將 baseline、兩種 11-region 表格、運算 proxy、CPU/GPU 證據、錯誤修復、限制與量化交接
  合併為 `reports/full35-q3-cpu-final-report.md`，不再另建第二份人類可讀 Q3 報告。
- 附件只包含兩張高解析圖、`summary.json`、`rows.csv` 與 190-site manifest；沒有上傳
  checkpoint、weights、cache、run、optimizer state 或未完成案例。
- `rows.csv` 原始輸出使用標準 CRLF；發布副本只將行尾正規化為 LF，數值、欄位與 22 列內容
  均未改動，使 `git diff --check` 可通過。
- 同步更新 `yolo_activation/README.md`、`reports/README.md` 與工作紀錄索引。
- 以最新 `origin/main` 建立隔離的臨時 worktree，只準備本次 Q3 報告範圍，不混入主工作樹
  其他未追蹤內容。
- 22 列 Markdown 表格已由 `summary.json` 自動重建並逐列比對，sites、四項 delta、
  八項最差 global delta 與 gate 全部一致。
- 第一次互動式 Python 比對在最後一個 `for` 區塊少空白結束行而出現 SyntaxError；
  修正後完整重跑並得到 `report_rows=22`、`table_match=true`、`figure_links=2`。
- 臨時 worktree 初次展開 112,581 個檔案時，狀態查詢曾落在 checkout 尚未完成的瞬間；
  確認沒有殘留 Git 程序後再次查詢，worktree 與 `origin/main` 完全乾淨。
- 發布授權：使用者指定 commit subject 為 `5090 Done 0831`，並核准 push 成功後清除 C01、C02。

其他困難：無。
