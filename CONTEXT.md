# YOLO 棒球視覺研究

本脈絡定義整個工作區共用的棒球資料與實驗血緣語言，讓不同模型專案使用相同資料口徑。

## Language

**BBAT5 原始來源（BBAT5 Source Archive）**:
保留最初影像、Pose 標註與舊 split 的唯讀資料資產；只用於追溯與重建正式版本。
_Avoid_: 正式訓練集、目前資料集、canonical dataset

**BBAT5 v1 正式資料版本（Canonical BBAT5 v1）**:
所有新棒球實驗共同使用的不可變版本容器；它同時包含 Pose 與二類 Detect Task View，本身不屬於單一任務。
_Avoid_: Pose dataset、Detect dataset、basic split、raw dataset、latest dataset、BBT5 copy

**BBAT5 任務 View（BBAT5 Task View）**:
同一正式資料版本針對 Pose 或 ball/bat 二類 Detect 提供的標註表示；兩者共用影像身分與 split，但標註列格式不同。
_Avoid_: Pose dataset copy、Detect source dataset

**COCO80 Detect 資料**:
供 80 類 Detect head 訓練與評估的通用資料；它與 BBAT5 二類 Detect Task View 是不同任務資料。
_Avoid_: BBAT5 Detect、ball/bat dataset

**Runtime Dataset View**:
為隔離 cache 而從正式資料集產生的可重建本機 View，不擁有 split 或標註語意。
_Avoid_: derived dataset version、canonical dataset、training source

**Portable GitHub Snapshot**:
供 clone 與下載的物化發行副本，內容必須等價於正式資料集，但不作本機訓練權威來源。
_Avoid_: canonical local dataset、runtime cache

**Legacy Basic Split**:
沿用原始 train/valid 歸屬且包含同源群組 leakage 的歷史 split，只能解釋既有結果。
_Avoid_: validation-safe split、formal split、current dataset
