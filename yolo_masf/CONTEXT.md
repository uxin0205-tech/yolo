# MASF-YOLO 實驗

本 context 定義 BBT5 球棒／球偵測架構消融、訓練與證據管理使用的共同語言。

## Language

**來源資料集（Source Dataset）**:
`bbt5-detect-baseline/dataset/` 中只讀的 BBT5 detection view，是所有固定切分的來源。
_Avoid_: 原始 split、正式 split

**固定資料集（Locked Dataset）**:
經 group/hash leakage audit 後，由 manifest 鎖定的 train、validation、test 集合。
_Avoid_: 隨機資料集、臨時 split

**來源初始化權重（Source Initializer）**:
`yolo11m_bat_detect_init.pt`；由 pose 權重轉成 detect 模型，且已接觸 BBT5。
_Avoid_: 官方乾淨權重、無洩漏 baseline

**乾淨初始化權重（Clean Initializer）**:
與 Ultralytics v8.3.0 官方 release SHA-256 完全一致的 COCO80 `yolo11m.pt`，未經 BBT5 fine-tuning；轉入兩類模型時，相容 backbone/neck/head tensors 可複製，80 類輸出 tensors 必須重新初始化。
_Avoid_: `yolo11m_bat_detect_init.pt`、pose-derived initializer

**資料暴露（Data Exposed）**:
initializer 已在 BBT5 相關訓練中看過資料，因此結果只能用於同源操作性比較。
_Avoid_: data leakage-free、unseen generalization

**正式變體（Formal Variant）**:
使用鎖定設定完成正式訓練、validation、test、profile 與 final audit 的模型。
_Avoid_: smoke model、preflight model

**B1**:
加入標準 P2 detection head 的四尺度 baseline，採凍結 10 epochs 後全解凍 90 epochs。

**B0-Fair**:
保留 B0 原始 P3/P4/P5 三尺度 Detect graph，從與第二輪 P3 正式變體相同的來源初始化權重出發，並使用完全相同的全模型 100 epochs 訓練設定；用來隔離 training budget 差異，但不消除來源初始化權重的資料暴露。
_Avoid_: clean baseline、unseen baseline

**嚴格公平組（Strict Fair Tier）**:
使用同一乾淨初始化權重、固定資料集、direct 100 epochs 與完全相同 resolved training profile 的 B0-Clean、P3-PaperFormula-Clean、P3-Partial25-Clean、P2-Direct-Clean。

**歷史測試集（Historical Test）**:
現有固定 test；雖然 group/hash audit 通過，但結果已被研究者反覆查看並影響後續設計，只能用於歷史對照，不可再支撐 unseen claim。
_Avoid_: final test、unseen test

**最終保留集（Final Holdout）**:
來自新影片、比賽或攝影機，建立後在 validation 選模與所有訓練決策凍結前不得查看的資料；目前尚未建立。
_Avoid_: 現有 test、重新切一次相同 frames

**MFAM**:
在指定 feature slot 以多個 depthwise kernel 聚合尺度資訊，再用 1×1 fusion 的模組。
_Avoid_: Hadamard attention、binary basis

**Partial MFAM**:
只有固定比例 channels 經過 MFAM，其餘 channels 保留 identity 的變體。
_Avoid_: sparse MFAM、dynamic routing

**Selective P2（SP2）**:
只為 Ball 提供輕量高解析 P2 prediction head，Bat 仍由 P3/P4/P5 主 head 預測。
_Avoid_: 移除 P2 feature map、動態 Top-K P2

**Canonical Checkpoint**:
由正式 native best EMA 轉成、由 repo 契約擁有且可 fresh-process strict reload 的 CPU float32 state dict。
_Avoid_: best.pt、last.pt、smoke checkpoint

**Native Checkpoint**:
Ultralytics run 直接產生的 `best.pt` 或 `last.pt`，連同 run CSV 與設定保存。

**選模凍結（Selection Freeze）**:
只用 validation 產生並鎖定 `selection.json`；完成後才能讀取 test 結果。
_Avoid_: test 選模、事後改選

**正式證據（Formal Evidence）**:
正式 checkpoint、metrics、predictions、profiles、selection、stage manifests、final audit 與 report。
_Avoid_: training preview、cache、smoke checkpoint

**可重建中間物（Disposable Artifact）**:
pipeline 完成後可依白名單刪除的 smoke/preflight/gate checkpoint、cache 或訓練預覽圖。
_Avoid_: 垃圾權重、所有 artifacts
