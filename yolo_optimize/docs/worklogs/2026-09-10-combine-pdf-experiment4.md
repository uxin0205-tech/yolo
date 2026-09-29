# 2026-09-10：閱讀 combine 報告實驗 4

使用者上傳 `combine/碩一_陳宥炘_0831.pdf`，要求參考實驗 4。依 research 技能由主代理閱讀第 9–11 頁，核對 PDF 表格與圖、原 final／training 配置及目前 smoke 梯度，形成[研究報告](<../research/2026-09-10-combine-report-experiment4.md>)。

關鍵修正：完整独立 Pose 的 P1–P3（MuSGD）與融合 J0–J3（AdamW）是兩段，不可只做 head-only；原圖存在 J1 初期暫時 AP 下降，不能把最終驗收門檻直接當首 epoch 停止門檻。本輪 0.25 權重後 Pose 梯度仍為 Detect 5.22–7.75 倍，需擴充 train-only 校準。PDF P2 scope 誤植以原 training.yaml 補足；報告0.02與 final0.08 gate 差異明記，不混稱。

新增 `calibrate_task_weight.py`，已啟動 16 個校準 macro＋8 個後續確認 macro。每 macro 恢復同一 checkpoint、optimizer LR=0、不使用 validation；依前 16 個加權共享 norm ratio 中位數提出固定 Pose weight，再以後 8 個確認 .5–2 的目標比例帶。總讀取 6144 Detect／384 Pose training images，未建立新 split。這是訓練期校準，不更改推論硬體；结果尚待 job 完成。

`merge-j1-smoke-v1` 已正常完成並通過真實更新／α 梯度／固定硬體契約；正式 `merge-j1-v1` **尚未啟動**，閱讀期間未跑正式 J1。此前文件說將進入 J1，是計畫而非訓練已執行。

驗證：PDF 文字頁碼、表格與梯度／AP／loss 圖直接核對；完成狀態由 JOB_DONE 與 smoke summary 確認；梯度加權口徑對照 joint_loss.py。沒有修改原 PDF 或舊模型。

困難：一般 view_image 遭 bwrap 初始化錯誤，改唯讀轉送已渲染 PNG 供視覺檢查。報告與程式版本的 patience／epoch／gate 不同，保留各自來源不擅自合併。未解：新固定 Pose 權重及分離訓練／最終 gate 尚待實作驗證；不宣稱已有新的正式訓練成果。
