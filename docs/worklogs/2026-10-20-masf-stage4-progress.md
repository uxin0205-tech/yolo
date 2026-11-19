# 2026-10-20：第四階段延長訓練進度核對

最新狀態：本流程已於 09:50 完成並停止，GPU 已釋放。以下為過程紀錄，最終採用建議及交付見 [完成紀錄](2026-10-20-masf-stage4-results.md)。

## 任務與範圍

使用者詢問目前狀況。本次於台灣時間 00:03～00:05 唯讀核對背景 queue、supervisor、monitor、正式逐輪 AP、Pose 重載開關驗證與 GPU 程序；沒有修改配方、重啟訓練或啟動第五階段。

## 驗證方式與結果

Run 為 `yolo_masf/artifacts/stage4/extension-20261019-v1`。15 個 jobs 中 10 個成功，Detect L1 正在執行，其餘 4 個等待依賴。Supervisor PID 801908 與 worker PID 801920 正常；最近兩次監測相隔約 600 秒，沒有異常動作。RTX 5090 當時 GPU 使用率 69%，本訓練程序使用 7,276 MiB 顯存，沒有其他 compute process。

Pose L1 追加 6 輪後依 patience 停止，累計 11 輪；最高關鍵點 AP50–95 為追加第 1 輪（累計第 6 輪）的 0.920154750。Pose L2 追加 5 輪後停止，累計 10 輪；最佳仍為本次續訓起點（原第 5 輪）的 0.919994127。兩者均低於原 Pose parent 的 0.920709389，暫建議保留原 parent。兩組最佳 checkpoint 均完成全量重載與 alpha-on/off，開啟減關閉的 primary AP 分別為 +0.000001260736、+0.000001551426，差距極小。

Detect L1 已完成追加 3 輪，正在追加第 4 輪（累計第 9 輪），核對時完成 200／463 個更新步驟。當前最佳官方 COCO AP50–95 為 0.514675024，較 parent 0.514620791 增加 0.000054233，約 0.0054 個百分點。該輪 sports ball AP 由 0.513224007 升至 0.514632328，但 baseball bat AP 由 0.487594089 降至 0.485806714；球棒退步超過原訂 .001 保護門檻，因此不能僅依整體 AP 略升就採用。Detect L2 尚未開始。

已完成的 Pose 兩組與 Detect L1 三輪均無 AMP retry；目前 Detect 第 4 輪回報 loss／梯度有限，仍無 AMP retry。整個延長研究尚未結束，因此目前沒有最終比較報告，這不是執行異常。

## 完成時間估計與後續

Detect L1 前三輪平均 1,778.11 秒／輪。由目前最佳在追加第 3 輪及 patience 5 推算，L1 最早到追加第 8 輪才會停止；接著 L2 至少 5 輪、最多 15 輪。包含後續完整驗證，現估整體約於 **2026-10-20 05:00～13:30（台灣時間）** 完成，實際依新最佳、早停與 GPU 資源變化。既有背景流程持續 600 秒監測，完成後整理第四階段結果並停止。

## 困難與解法

無新執行困難。最初 03:30 的樂觀下限已因 Detect 取得新最佳、patience 起算點延後而不再適用；依已完成輪數及實測速度更新估計。

## 未解事項或風險

Detect L1 尚未完成，L2 尚未開始；AP 小幅波動不能視為顯著收益。最終需比較分類別 AP 與已訓開關結果。本次只做狀態檢查，不新增測試或額外 GPU 工作。

## 00:44：球／球棒效果與原因核對

使用者詢問加入 MASF 是否應提升球／球棒等小物件。重新唯讀檢查目前最佳 Detect L1（追加第 3 輪）、第一輪已訓 E5 alpha-on/off、實際 MASF／task bridge／訓練 scope，並先參照 optimize 的 `docs/research/2026-09-10-masf-paper-small-objects.md`、`2026-09-01-yolo26-p3-masf-no-gain-diagnosis.md` 與原 bridge／continue 腳本。沒有更改正在執行的程式與配方。

目前最佳 Detect 對比原 parent，sports ball AP50–95 從 0.513224007 升至 0.514632328（+0.1408 個百分點），baseball bat 從 0.487594089 降至 0.485806714（−0.1787 個百分點），全類別 APsmall 從 0.353835064 升至 0.354208945（+0.0374 個百分點）。這是 head 與 MASF 共同適應的結果，不能全數歸因 MASF；COCO sports ball 也不等於只有棒球，類別 AP 不能當作該類小物件專用 AP。

同一份第一輪 E5 權重只切換 MASF，baseball bat AP50–95 為 off 0.484687021、on 0.485757785，開啟提高 0.1071 個百分點；sports ball 開啟則小降 0.0040 個百分點。這證明該 checkpoint 的開關效果依類別不同，不支持「MASF 完全無效」或「小物件一律改善」。球棒 alpha-on 仍低於原 parent。這不是目前 L1 最佳 checkpoint 的開關結果，後者待排定驗證。

BBAT5 Pose 原五輪中 ball／bat bbox AP50–95 都較 parent 提高，但 keypoint AP 下降；同權重 alpha-off 幾乎保留 bbox 改善，因而不能把整輪 bbox 收益直接歸因新增 MASF。L1 延長後 keypoint 最佳仍低於 parent，L2 最佳停留在續訓起點。

已確認的結構／訓練限制：本地 Full35 在同一 P3 上以 DW3＋DW5、1×1、qSiLU 做鄰域混合，640 輸入時仍為 stride 8 的 80×80 特徵，不增加新的高解析度輸入；僅更新 head 與 MASF，backbone／neck／BN running 固定；一個 alpha 共用所有位置與通道，沒有專為小球與細長球棒分別選分支；Detect 選模主指標仍是 COCO80 整體 AP。這些限制可能影響收益，但不能單憑結構判定本次唯一根因。FPN 原論文以跨層連接建立各尺度語意特徵，不能把其效果當作本地同解析度 DW3／DW5 的保證。[FPN 原論文](https://arxiv.org/abs/1612.03144)

原五輪 Detect／Pose alpha 約 0.00322／0.000509；從零初始化會使第一步 context 梯度為零，之後梯度仍受 alpha 因子影響。但不能只由 alpha 小推斷特徵擾動幅度，尚須量測 context／原特徵的相對幅值。本輪已有真實梯度與參數更新證據，L2 正是測試提高 context LR、降低 head LR 是否改善適應。

optimize 舊版的 P3 改動傳到 P4／P5、one2one detach 阻斷 MASF 梯度，本輪均已透過 head P3 fork、detach-before-MASF bridge 及 CPU 驗證處理，不能再當作目前根因。球棒當前最佳 AP50 略升但 AP50–95 降，提供定位品質檢查方向；背景混入、細節損失或過擬合都尚未被本輪錯誤分析證實。後續若需進一步診斷，應先沿用完整驗證輸出分尺寸及檢查漏檢／誤檢／定位，不在 GPU 訓練中臨時改配方。此說明沒有新增 GPU 工作。

本次困難：最初使用 optimize 根層腳本路徑未命中，已依既有 reference manifest 定位 `experiments/studies/pre-fusion-full35-b100/scripts/`；CVF 網頁存取被拒，改查 arXiv 原論文。未解事項仍是當前 Detect 兩組完整結果與尺寸／錯誤類型歸因。

## 01:08：背景進度核對

使用者再次詢問狀況。本次只讀取 queue、supervisor、monitor、逐輪正式指標及 GPU 程序，沒有修改程式、配方或新增 GPU 工作。Detect L1 已完成追加 5 輪，正在追加第 6 輪（累計第 11 輪），最近狀態為 280／463 個更新步驟，約 60%。新增第 4、5 輪官方 AP 分別為 0.514540255、0.514509924，未超過追加第 3 輪最佳 0.514675024。

Pose L1／L2 已完成的結果不變，仍未超越原 Pose parent。Detect L2 尚待 L1 與重載開關驗證完成；15 個 jobs 中 10 個成功、1 個執行中、4 個等待依賴。Supervisor 與 worker 存活，600 秒監測正常，本訓練程序使用 7,276 MiB 顯存，沒有其他 compute process。單次 nvidia-smi 使用率為 0%，而剛完成的週期監測為 66%，進度持續前進，不能將單次快照判定為停滯。

Detect 追加第 4 輪曾有 1 次 AMP retry，該輪已成功完成，後續第 5 輪與目前第 6 輪沒有 retry，沒有訓練中斷或 queue 失敗。無新困難或需介入處理的異常。若 L1 不再取得新最佳，依 patience 預計追加第 8 輪結束，約台灣時間 02:20；整批仍估 05:00～13:30 完成。最終 AP 與採用建議仍待 Detect L2 及驗證；工作停在第四階段。
