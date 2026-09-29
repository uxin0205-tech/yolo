# 2026-09-08：方向 1 第一輪執行紀錄（分析暫停）

## 最新結果與狀態

使用者已授權開始實作與 GPU 驗證。第一輪目前為 `paused_for_analysis`，不是完成優化：原生對照完成 1/5 epochs，HOG 正式訓練為 0/10，沒有已驗收的精度改善。所有 GPU 工作已結束；新 E1 權重只供診斷，不替代原 BEST。

完整數字、架構 ASCII 圖、五組診斷及推導見[第一輪執行報告](<../../proposals/integrated-roadmap/first-round-execution.md>)。報告由 `scripts/report_recovery.py` 讀取既有 JSON 產生，已在 terminal 顯示。README、主計畫、optimizer policy 與工作紀錄索引已同步；machine-readable 計畫通過 JSON 格式檢查，最後修正測試總數與單變因對照命名。

## 變更內容與原因

原始來源固定為 `/home/uxin/yolo/yolo_combine/final/full35`，只讀使用其程式、source bundle、inference EMA 與 paired full-resume criterion state。所有新程式、logs、checkpoint、cache 與報告都放在 `yolo_optimize`。

新增 runtime、HOG auxiliary、訓練入口、supervisor、batch benchmark、報告生成器、state audit、五組診斷及安全短驗證。HOG 從 raw pre-MASF P3 接訓練專用 1×1 head，部署時移除；本輪尚未搬動 MASF、改 BinaryQK 或替換 RepConv。

資料維持完整 COCO80。BBAT5 使用不可變的 canonical `bbat5-v1`，runtime View 保留完整 assignment 與 labels，不重切、抽樣或修改影像。來源 JPEG 修復、disk.npy 與 cache 隱式寫回已有唯讀 guard，cache 只允許放在本專案。

已依 `diagnosing-bugs` 建立真實 AP 失敗回饋，做單變因 state 對照；將已證實的程式 bug 與未證實的精度原因分開，不用測試通過替代 AP 驗證。

## 環境、BEST 與 batch

- 環境：RTX 5090，32607 MiB；torch 2.11.0+cu128、Ultralytics 8.4.90。啟動前無其他 compute job，磁碟約 1 TB 可用。
- J3 best_joint、best_pose 均完成完整 COCO validation 5000 張與 BBAT5 validation 683 張的 Float／BitTrue 重驗，八項 internal AP 重現歷史數值。
- BitTrue joint：best_joint 0.7111747389752653；best_pose 0.7111415212904801。差距約 0.00003322，不宣稱顯著優勝；按既有 joint 規則保守沿用 best_joint EMA。
- 原 BEST SHA-256 在交付前再次確認未變：`d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- 實體 batch 32／64 的 native、HOG 皆成功；128 皆 OOM。短測每組 2 warmup + 5 timed macros，包含反向及真正 optimizer step，所有更新丟棄。
- 32 約 167–171 images/s，peak reserved 約 10.5 GiB；64 約 170–172 images/s，約 20.6 GiB。自動吞吐推薦 64，工程選 32：僅約 1–2% 差距，32 保留較大顯存餘裕且沿用原 J3 BN batch 行為。

Detect logical batch 為 128（32×4），但每個 optimizer macro 包含兩個 Detect logical batches，即 Detect 256 張 + Pose 16 張；reference batch 64，task weights Detect=1、Pose=0.25。

## 實際執行設定

兩臂規劃從同一 parent EMA 建立 fresh AdamW，原生預算 5 epochs；HOG 最多 10 epochs、patience 4、min_delta 0.0001。共用 10-epoch cosine horizon，warmup=1 epoch，起始 factor=0.1、最終 factor=0.5。

Neck LR=1e-5，Detect/Pose head LR=2.5e-5，HOG auxiliary LR=3e-4，betas=(0.948,0.999)、weight decay=0.00027、clip=10、AMP。可訓練 Neck 與 heads；backbone、attention、既有 MASF 凍結。Shared BN 統計凍結，head BN 維持 train。

E2E criterion 延續 paired snapshot：Detect horizon=120、updates=51、one-to-many 權重=0.5；Pose horizon=128、updates=59、權重=0.4748031496062992。fresh optimizer 不等於重新開始 criterion。

HOG μ 尚未正式校準；預計用 train-only raw P3 梯度比 5% 校準，不是 μ=0.05。HOG E1 關閉、E2 ramp、E3–8 啟用、E9–10 關閉，提前停止不補造尾段結果。

## Native E1：完整跑完後暫停

2026-09-08 18:18:53（Asia/Taipei）啟動，18:29:33 結束；完整完成 463/463 macros，而非先前進度快照的 macro 260。Detect 完整一遍 118287 張；Pose 7404 張，包含循環重用。每 600 秒 GPU 監測已於 18:28:53 記錄，工作結束後確認沒有 compute process。

AMP 共 7 次 retry：macro 0 為 4 次、macro 6 為 2 次、macro 51 為 1 次，scale 最終由 65536 降至 512。不能寫成後續沒有 retry。E1 full snapshot 與 inference 權重均已保存，原始 parent 未動。

| BitTrue 指標 | 原 BEST | E1 | 差值 |
|---|---:|---:|---:|
| Joint | 0.71117474 | 0.70756101 | -0.00361372 |
| BBAT Box | 0.63003552 | 0.62317960 | -0.00685592 |
| Ball Box | 0.50743704 | 0.49878115 | -0.00865588 |
| Bat Box | 0.75263401 | 0.74757805 | -0.00505596 |
| Ball Pose | 0.85990872 | 0.85457640 | -0.00533231 |

四項 AP 比 parent 下降超過 0.005，觸發保存後暫停。已經有 E1 AP 結果，只是沒有精度改善或完整 5-epoch 結論。run-local best 不等於跨 run 勝出；E1 退步也不證明後續 epochs 必定不能恢復。

## 五組診斷與證據界線

完整 BBAT5 validation 已跑完：E1 原樣、E1 參數+原 head BN、原參數+E1 head BN、原 state+E1 Pose head 參數、原 state+E1 Neck 參數。輸出在 `artifacts/direction1-20260908/diagnostics/`，各組不新增訓練、不保存部署模型。

E1 原樣重驗完全重現下降；換回 BN 並未補回 Box AP。Neck 與 Pose head 參數各自替換也會讓部分 AP 下降，不能歸因單一 BN 設定。混合 state 可能破壞共同適應，差值不可相加或視為唯一因果。Float／BitTrue 同向下降，不支持新 backend 差異為主因；HOG 尚未正式訓練，不能歸因 HOG。

CPU state audit 的兩份 EMA state 均 1238 tensors、keys 相同。frozen scope 無超過 1e-5 的 material drift，但有微小 EMA 舍入差異；head BN 統計有變動。EMA 非浮點 BN counters 不更新，不代表 live BN 沒更新。詳細表見 `state-audit.json`，不把 state diff 當 AP 因果證明。

## 已修正的程式問題與驗證

1. 原生 AMP retry 未回滾失敗 forward 的 BN/RNG。新增 RetrySafeMacroStepEngine，每次 retry 回到同一 macro 起點，不改 scaler/optimizer 順序。
2. 原生 EMA 仍對固定浮點 state 做平滑，造成舍入漂移。新增 FixedStateEMA，固定參數、硬體 buffers、凍結 BN 統計 exact-copy；可訓練 state 沿用原 EMA。未改 EMA age/decay/tau。
3. HOG 日誌累計失敗 attempts。實際 router+engine CPU fixture 先重現 1／4 retries 將 Detect 4 images 記成 8／20；每次 attempt 清統計後，images、valid cells、physical sizes 與成功參考一致，μ/criterion/梯度不變。

訓練契約 17、HOG 12、資料保護 8、安全回歸 8，共 45 項唯一 CPU tests 已通過。AMP／EMA 整合後相關 24 項重跑通過；最後日誌修補由 8 項安全測試驗證，沒有不必要地重跑全套。

GPU correctness smoke：native 與 HOG 各成功 1 macro，均遇 4 次 AMP retries，optimizer/EMA 各只更新 1 次，480 個固定 state 完全不變，BN counters 正確。HOG 實際 auxiliary 參數更新及 AdamW 動量有限非零。更新全部丟棄，不增加正式 epochs、不保存部署權重。最後日誌修補僅做 CPU 回歸，沒有追加 GPU。

GPU 產物：`safety-integration/native.json`、`safety-integration-hog/summary.json`。這些證明程式修補正確，不證明修補後 AP 回升；原始 AP 失敗仍未解除。

## 困難與解法

- 預設沙盒 bwrap 故障：提升必要命令權限，仍限制來源唯讀、輸出留在本專案。
- Auxiliary 不適合直接套 final stage regex：使用 training wrapper 與獨立 optimizer role。
- 原生 JPEG 修復／cache 隱式寫回風險：加入已測試的 readonly guard，未改原始影像。
- COCO 附加 evaluator 缺 annotations runtime 路徑：補唯讀 symlink；internal AP 與外部 COCO API AP 分開，不混用。
- 首次 HOG smoke 在 native zero_grad 後檢查 .grad 而失敗：修正驗證工具，改核對實際參數更新與動量，只重跑 HOG；原 failed summary 保留，不偽裝首次即通過。
- 文件代理同步延遲：主代理接手最後狀態與紀錄，保留已完成研究；沒有 commit、push 或刪除原始權重。

## 未解事項、下一步與停止邊界

EMA age 重置仍是待驗假說：parent updates=26597，E1 fresh EMA updates=463，舊 EMA 保留係數約 0.999898／0.206637，候選更接近 live。這尚未證實是 AP 主因。

若追加，建議修補後兩組各 1 epoch，只比較 fresh EMA age 與延續 parent age；其餘 LR、BN、criterion、資料與 optimizer 相同，並驗證 live/EMA。不能把修補前 E1 當完全等價 control，也不能把強平滑掩蓋退步當成學習改善。這項追加診斷尚待確認、未排程，沒有擅自修改配方。

視覺失敗案例的圖片／影片路徑尚未收到，尚未驗證使用者關注的主觀視覺改善。目前工具無法讀取 /status weekly 配額，無法確認是否已達 73% remaining；不宣稱符合用量門檻，本輪不再追加 GPU 實驗。

MASF、BinaryQK、單層 RepConv、MuSGD 等仍為條件式後續，第二輪與 person-only 不納入。現階段保留原 BEST；training_enabled=false，AP_recovery_verified=false。
