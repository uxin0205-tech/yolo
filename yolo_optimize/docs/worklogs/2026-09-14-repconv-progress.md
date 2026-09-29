# 2026-09-14：BinaryQK E5 後接 RepConv 進度

## 2026-09-14T00:21:27+08:00

使用者查詢狀態；只查新 queue 狀態、交接事件、已完成摘要、回合指標、checkpoint 檔名與當前 heartbeat，未讀訓練 log、未查 GPU、未更改排程。

舊起點 Rep17 已完成第 1 回合存檔／驗證後收束，原產物保留。新 Rep17 smoke 已完成，來源 SHA 為 c8dcaf439b68c77d45a4f213972e57d51912028b558418511badee55fafbf681，確認是 BinaryQK scale_bias E5，不是原生 QK B5。新 queue PID 2589417、train PID 2596282 存活；heartbeat 約 1.4 秒前更新。已完成 E1～E3 完整驗證與 checkpoint，目前尚未有 E4 完整回合產物。Rep20、Rep17＋20 尚待接續。

| AP50–95 (%) | BinaryQK E5 起點 | Rep17 E3 | 差值 (pp) |
| --- | ---: | ---: | ---: |
| coco/box/map50_95 | 50.5248 | 50.5437 | +0.0189 |
| coco/person/box/map50_95 | 62.6208 | 62.7068 | +0.0860 |
| bbat/box/map50_95 | 60.8534 | 61.1373 | +0.2838 |
| bbat/pose/map50_95 | 88.8678 | 89.0361 | +0.1683 |
| bbat/ball/box/map50_95 | 49.3118 | 49.6400 | +0.3282 |
| bbat/ball/pose/map50_95 | 85.7615 | 86.2080 | +0.4464 |
| bbat/bat/box/map50_95 | 72.3951 | 72.6346 | +0.2395 |
| bbat/bat/pose/map50_95 | 91.9740 | 91.8642 | -0.1097 |

初步多項改善，但 bat Pose 下降 0.1097 pp，尚未恢復原 MASF B5 的 bat Pose；這是中途結果，不判定最終採用。依固定 5E 計畫繼續，結束後獨立 BitTrue／Float 驗證，背景每 600 秒監測。

變更：僅保存狀態核對工作紀錄與索引。驗證：以上程序／事件／JSON 核對通過。困難及新錯誤：無。未解事項：三組最終精度與採用門檻尚待完成，不追加訓練、不覆寫已完成結果。

## 2026-09-14T02:16:53+08:00 狀態更新

Rep17 完成 5E 及獨立驗證，gate_passed=false，未取代正式模型。新 queue 已自動接續 Rep20，E1～E3 完成驗證；Rep17＋20 尚待接續。queue PID 2589417／train PID 2650235 存活，當前 heartbeat 約 0.1 秒。只核對狀態／摘要／已完成指標，沒有讀取訓練 log 或查 GPU。

| AP50–95 (%) | BinaryQK E5 起點 | Rep17 E5（完成） | Rep20 E3（中途） |
| --- | ---: | ---: | ---: |
| coco/box/map50_95 | 50.5248 | 50.5584 | 50.5327 |
| coco/person/box/map50_95 | 62.6208 | 62.7191 | 62.6945 |
| bbat/box/map50_95 | 60.8534 | 60.9242 | 60.8882 |
| bbat/pose/map50_95 | 88.8678 | 89.0121 | 88.7122 |
| bbat/ball/box/map50_95 | 49.3118 | 49.2785 | 49.3029 |
| bbat/ball/pose/map50_95 | 85.7615 | 85.7911 | 85.3881 |
| bbat/bat/box/map50_95 | 72.3951 | 72.5700 | 72.4735 |
| bbat/bat/pose/map50_95 | 91.9740 | 92.2331 | 92.0362 |

Rep17 多數指標對 BinaryQK 起點略有恢復，但尚未通過事先設定的改善／雙起點保護門檻；bat Pose 92.2331% 仍低於 MASF B5 92.4968%。Rep20 E3 尚非最終结果，不作採用判定。變更：只追加本次狀態紀錄。驗證：queue／child 存活及 JSON 結果正常。困難／新錯誤：無。未解事項：Rep20 及雙層最終驗證待完成。維持原 5E 計畫與 600 秒背景監測，不新增回合。

## 全部完成後的完整狀態彙整

使用者要求把東西全部列出。核對新 queue ALL_DONE（台北 04:49:57）、程序退出、三組 summary completed／Float＋BitTrue 驗證與所有 E1～E5 checkpoint。以 CPU SHA 核對三份推論權重、逐項比對 E5 與 BitTrue 指標、固定 scale/bias 整數碼，全部通過。新增 [完整狀態報告](<../../experiments/post_binary_rep_v1/STATUS-20260914.md>)，合列原生 QK E2、MASF B5、BinaryQK E5 與 Rep 三組，列成本已知／未測邊界與權重入口；同步 README 最新結果。困難／錯誤：無。三組未過採用閘，未自動升版；未解事項為板端量測、多 seed／獨立 test 等。本次未使用 GPU、未追加訓練、未推送 Git。
