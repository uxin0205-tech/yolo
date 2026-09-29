# 方向 1：MASF 真實 loss 校準與訓練專用梯度橋接

## 內容與原因

接續融合前 Full35-B100 方向 1。上一輪 MASF head LR 提高後沒有穩定 AP 收益，本次不盲目增加相同配方的 epoch，也不重跑已完成的 native fork。依診斷流程一次只改一個變因：讓部署使用的 one-to-one loss 直接監督 MASF，但仍阻止此新增路徑向 Backbone 傳梯度。

候選原因依序為：部署分支缺少直接監督、兩分支梯度方向衝突、訓練 scope 不足。本輪只檢驗前兩者；不變更 Backbone、資料集、推論精度或融合後模型。完整 AP 才能判定增準，梯度連通不等於找到精度退化的全部原因。

新增腳本：`calibrate_masf_task.py`、`masf_task_bridge.py`、`train_masf_task_bridge.py`、`run_masf_task_bridge.py`。既有 `continue_masf_head.py` 與完成結果均未修改。

## 真實 loss 校準結果

以 native fork E5 live checkpoint、同一完整 train loader 的前 8 個 batch（256 張）作 FP32 校準，不更新參數、不選 validation 樣本。第一個 macro 的資料雜湊與已完成 native fork E6 一致。

8 個 batch 中，3 個的 one-to-many／one-to-one MASF 梯度 cosine 為負。未縮放的 one-to-one 梯度有時遠大於 one-to-many，因此不直接以係數 1 加入。

固定係數取：

```text
lambda = min(0.25, min_batch(0.25 × ||g_many|| / ||g_one||))
       = 0.012076444778011642
```

兩種梯度均包含當前原生 loss 分支權重。此規則使校準的每個 microbatch 中新增梯度 norm 不超過原 MASF 梯度的 25%；不是對全訓練或累積後梯度的全域保證。lambda 在訓練時固定，不需要每張圖片重新計算，也不加入部署圖。

確認 one-to-one boxes、scores 與真實 loss 前向完全相同；新增梯度能到 MASF、不能到 raw P3/P4/P5。所有模型 state、來源 checkpoint 雜湊皆保持不變。校準產物：`artifacts/masf-task-calibration-v1/summary.json`。

## 架構與訓練設定

```text
訓練：p3_raw ─────────→ MASF → one-to-many loss
           └→ detach → 同一個 MASF → gradient-scale(lambda) → one-to-one loss

推論：p3_raw → 單次 MASF → 原生 Detect one-to-one
```

gradient-scale 的 forward 為 identity，backward 才乘固定 lambda。只重算訓練中的 MASF，不重算 Backbone。BN 統計必須固定，程式對此設有 assertion。推論分支直接呼叫原本 `P3MASFDetect`，沒有額外 MASF pass；尚未量測硬體 latency。

從同一 native fork E5 恢復 optimizer／EMA／scaler，跑 E6–E10。對照直接使用已完成 `masf-head-fork-v1`，資料 seed 20260927、warmup 與排程相同，不重跑對照。使用 physical batch 32 × accumulation 4 = logical 128、imgsz 640、完整 COCO train 118287、每 epoch 925 次更新。

AdamW betas=(0.948, 0.999)、eps=1e-8、weight decay=0.00027（原有 decay 分组）、clip norm=10；head LR=1e-5、context LR=1e-5、alpha LR=1e-4；沿用原 10-epoch cosine、原始 warmup 1 epoch，不在 E6 重啟。全 BN 統計固定；只訓練 P3 head／MASF，alpha clamp=[-0.25, 0.25]。每 epoch 以 EMA 與 live 完整驗證 COCO 5000，主要決策看 overall／person，ball／bat 僅輔助觀察。

## 實際更新與推論驗證

128 張 smoke 已通過，optimizer step=4626、EMA updates=12026、criterion age=5。首 macro 原生 loss=237.9498634338379，與 native fork 完全相同；P3 head 首梯度 norm 和更新幅度亦相同，MASF 的梯度則按預期改變。

context 相對更新約 1.0214e-5、alpha 約 7.4441e-4，通過原有安全檢查。固定參數與 EMA state 檢查通過。CPU160 對照原生推論完全相同、Bit-True 匯出重新載入完全相同。smoke 匯出只作格式與數值驗證，尚未完整 AP 重驗，不能作正式 best。

## 已啟動工作與監測

正式 `masf-task-bridge-v1` 於 UTC 2026-09-09 17:31:11（台灣 2026-09-10 01:31:11）啟動；啟動時主 queue session=47321。工作狀態保存於 `artifacts/masf-task-queue-v1-state.json`。

queue 沿用最多 600 秒的 `child.wait`，正常不读 log 或 GPU；失敗保存事件並停止該階段，完成後產生 E6–E10 與 native fork 的配對 AP 差值，進入 awaiting_analysis。它不會自動把候選升格，也不能據此宣稱已部署模型自動喚醒通道。

## 困難與處理

首次校準啟動誤用相對路徑，而 monitor 的 cwd 是 study root，因此子程式尚未開始就找不到腳本。只讀失敗 log 最後一行後，改用絕對路徑，以獨立 retry 事件檔重新啟動；舊事件保留，沒有覆寫成果。重啟後校準正常完成。其餘校準與 smoke 困難：無。

## 尚未完成與風險

正式 E6–E10 尚在執行，方向 1 尚未結束。256 張校準不能代表全 COCO；梯度衝突與缺少直接監督只是可檢驗機制，尚未證明是 AP 未提高的原因。

完成後須比較相同 epoch 的 overall／person，排除只挑最好 epoch 造成的偏差；若有足夠收益，再作獨立匯出、完整 COCO 重驗與固定場景推論。若仍無增益，不宣稱成功、不自動無限延長相同配方；依結果再評估是否值得改訓練 scope。HOG／RepConv 舊結果保留，不處理第二輪、Pose 或融合後資料。
