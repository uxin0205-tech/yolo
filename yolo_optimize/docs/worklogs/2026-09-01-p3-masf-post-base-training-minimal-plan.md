# 2026-09-01：P3 MASF 主訓練後最小實驗計畫

## 工作目標

把 Detect-entry MASF 定位為 BASE-FP 主訓練完成後、量化與部署轉換前的小範圍 recovery，並依使用者
要求刪除非必要實驗，只保留能公平判定部署價值的最小比較。

## 設計依據與變更內容

本輪使用 `codebase-design` 的 seam／interface 原則：Detect adapter 位於 `model.23` seam，外部
interface 維持 `[P3, P4, P5] → Detect outputs`，MASF routing 留在 implementation 內；測試只跨
同一 interface 驗證可觀察行為，不讓 owner index 散落到呼叫端。

1. 將原 C0/C1/C2 三個新訓練 arm 縮成 `CTRL-P3` 與 `MASF-P3` 兩臂；舊 shared-seam 不重跑。
2. 兩臂都從同一無 MASF `BASE-FP` 開始並使用相同 P3 predictor recovery，排除額外 epoch 的混淆。
3. 明定流程位置：BASE-FP 主訓練後、BinaryQK／QAT／PTQ／calibration／export 前。
4. 保留四個工程 gate、一組 paired-seed 初篩，以及只有通過才補到三組 paired seeds 的策略。
5. 從本方向移除 learned selector、dynamic gate、P2 lateral、RepConv 與 BinaryQK。
6. 同步更新方向 README、架構報告、總索引與根 README 的舊 C0/C1/C2 說明。

## 驗證方式與結果

- 自動 assertion 確認 `plan.md` 只有 `CTRL-P3`／`MASF-P3` 兩個新 training arms，沒有舊
  C0/C1/C2/C3 matrix；初篩與通過後總成本分別為 2／6 jobs。
- 掃描本次相關 Markdown 共 35 個本機相對連結，結果 `missing_local_links=0`；行尾空白檢查無發現。
- 自動 assertion 確認順序是 BASE-FP 主訓練完成 → Detect-entry MASF recovery →
  BinaryQK／QAT／PTQ／calibration／export。

## 資料集影響

無。沒有讀寫、重建或重新切分 dataset；計畫保留 BBAT5 immutable dataset 與正式
`configs/detect.yaml` 契約。

## 困難與解法

無。

## 未解事項與風險

- Detect-entry production implementation 與 GPU recovery 尚未執行。
- recovery epoch 沿用正式 Stage A schedule；實際執行前仍須把 resolved config snapshot 寫入 artifact。
- 若無 MASF 的 BASE-FP parent 不存在，必須先恢復／完成該 baseline，不能用 shared-seam checkpoint
  直接替代。
