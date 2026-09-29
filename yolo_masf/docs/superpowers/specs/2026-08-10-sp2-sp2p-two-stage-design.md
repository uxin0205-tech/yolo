# SP2 與 SP2P 兩階段訓練設計

## 目標

在不停止目前 `formal_m0` 的前提下，調整 Phase 1 pipeline：移除 SP2 ONNX 匯出；將 SP2 正式訓練改為 10 epochs 凍結 backbone 0–10 層、再 90 epochs 全模型解凍；由 validation 選出的 M2/M3 與完成的 SP2-B 組成唯一一個 SP2P，並讓 SP2P 同樣執行 10+90 epochs。

## 已選方案與未採用方案

採用「雙來源繼承」：SP2P 的共用網路與 Selective P2 head 取自 SP2-B，P2 partial-MFAM 僅取自 `BEST_PARTIAL` 的 M2 或 M3 canonical checkpoint。這能直接組合兩個已訓練元件，而且只訓練一個混合模型。

不採用以下方案：

- 只繼承 SP2-B、partial-MFAM 隨機初始化：沒有完整沿用勝出的 M2/M3。
- 同時訓練 SP2M2 與 SP2M3：會額外增加 103 epochs，且違反只建立一個最佳混合模型的需求。

## 模型定義

SP2 維持現有硬體友善設計：P2 使用 hidden channels 32 的 Ball-only depthwise-separable box/class towers；P3/P4/P5 保留 Ball/Bat 兩類別主 head；推論時 P2 Bat score 固定為零；不使用 Top-K、資料 Gather 或不規則 sparse tile。

SP2P 是 SP2 加上選定 partial-MFAM：

- 若 `BEST_PARTIAL=M2`，P2 有 1/2 channels 通過 3×3、5×5 MFAM。
- 若 `BEST_PARTIAL=M3`，P2 有 1/4 channels 通過 3×3、5×5 MFAM。
- 不加入 7×7 或 9×9 分支。
- Selective P2 head 與 loss/decode 行為和 SP2 相同。
- 對外實驗名稱固定為 `SP2P`；checkpoint manifest 額外記錄實際 parent `M2` 或 `M3` 及其設定雜湊，確保嚴格重載不會混淆架構。

## 權重來源與訓練

SP2：

1. `smoke_sp2`：沿用既有 3 epochs 結構 smoke，不作正式比較。
2. `sp2_a`：從 B1-B canonical 建立 SP2；匹配權重轉移，SP2 專屬 head 維持其初始化；凍結模型 indices 0–10，訓練 10 epochs，`lr0=0.01`。
3. `sp2_b`：載入 `sp2_a/best.pt`；解除全部凍結，訓練 90 epochs，`lr0=0.001`；其 best 轉成 SP2 canonical checkpoint。

SP2P：

1. 完成 M2、M3、SP2-B 後，只用固定 validation manifest 評估 M2/M3。
2. 依既有規則凍結 `selection.json`；test 資料不得參與選擇。
3. 建立選定比例的 SP2P；先載入 SP2-B 的所有可匹配權重，再只從 BEST_PARTIAL canonical 載入 P2 partial-MFAM slot，禁止覆寫其他層。
4. `smoke_sp2p`：由上述雙來源合併權重跑 3 epochs，結果不進正式比較。
5. `sp2p_a`：重新由相同雙來源權重建立；凍結 indices 0–10，訓練 10 epochs，`lr0=0.01`。
6. `sp2p_b`：載入 `sp2p_a/best.pt`；解除全部凍結，訓練 90 epochs，`lr0=0.001`；其 best 轉成 SP2P canonical checkpoint。

SP2P 是序列式組合實驗，先前 parent 已各受訓 100 epochs，因此報告必須標示它的訓練預算與一般從 B1-B 開始的單段架構消融不同，不把提升全部歸因於架構。

## Pipeline 順序

保留已完成階段與目前執行中的 M0。後續關鍵順序為：

1. 完成 M0、M1、M2、M3、P3M 正式訓練。
2. 執行 `sp2_a`、`sp2_b`，取代原 `formal_sp2`。
3. 執行 `val_partial`，只評估 M2/M3 validation。
4. 執行 `selection`，產生不可覆寫的 `selection.json`。
5. 執行 `smoke_sp2p`、`sp2p_a`、`sp2p_b`。
6. 執行 B0 baseline、全部模型 validation、test、硬體 profiling、final audit 與報告。

完全移除 `export_sp2` 與最終稽核中的 ONNX artifact 要求。`val_all` 可重用 `val_partial` 已產生且 hash 相符的 M2/M3 指標，其他模型正常評估。test 一律在 selection 凍結後執行。

## 評估矩陣

最終 validation、test 與 hardware profile 包含：B0、B1、M7、M0、M1、M2、M3、P3M、SP2、SP2P。只有 M2/M3 具備 `selection_eligible=true`；SP2P 不得反向影響 BEST_PARTIAL。

報告需同時列出 Ball/Bat 指標、mAP50、mAP50-95、small-object 指標、Params、MACs、GFLOPs、peak activation、P2 activation、operator list、feature traffic 與 GPU latency，並保留 data-exposed 警告。

## 失敗、續跑與產物規則

- 每一段都有獨立 stage、run directory、best/last 與 canonical checkpoint，避免 10/90 epochs 被誤判為同一個 100-epoch resume。
- 已完成 stage 只有在 config/data/environment 與直接 predecessor hashes 相符時重用。
- SP2P 必須記錄 SP2-B canonical hash、BEST_PARTIAL canonical hash、selection hash 與實際 partial ratio。
- 雙來源轉移若出現非預期 missing key、shape mismatch、錯誤 parent 或覆寫非 P2 slot，立即 fail closed。
- pipeline 仍維持單一 systemd 服務，不新增平行 GPU 訓練。

## 測試與完成判定

實作採 TDD，至少涵蓋：

- SP2-A/SP2P-A 精確 freeze 0–10、10 epochs、`lr0=0.01`；B 段無 freeze、90 epochs、`lr0=0.001`。
- SP2-B 僅由 SP2-A best 初始化。
- SP2P 雙來源轉移只從 BEST_PARTIAL 覆寫 P2 partial-MFAM。
- M2/M3 selection 在 SP2P 之前且 test 在 selection 之後。
- workflow 不含 `export_sp2`，final audit 不要求 ONNX。
- SP2P forward、finite loss、AMP backward、save/reload、官方 NMS 與 canonical strict reload。
- 最終矩陣完整包含 SP2P，且選模候選仍只有 M2/M3。
- CPU 測試與 compileall 通過後，pipeline 才能沿既有 ID 安全接續；目前執行中的 M0 不得被停止或重啟。
