# 已訓練模型驗證

## COCO 官方 `COCOeval`

來源／年：[COCOeval](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)；程式未標版本年，COCO 基準為 2014。規則：IoU 0.50–0.95／0.05，recall 0–1／0.01，`maxDets=[1,10,100]`，AP 用 100。detection 依 `score`/confidence 穩定降序截取；AP 掃全 precision-recall 曲線，非單一 conf 閾值。`iscrowd` 進入 IoU，crowd GT 可多匹配，ignore 參與統計。

適用範圍：上述摘要針對 bbox/segm 評估；不可直接拿其 IoU／maxDets 規則當作 Pose 或影片抖動指標。score 不代表框穩定。`maxDets`、area、類別或 crowd/ignore 不一致，AP 即不可比。

## TIDE

來源／年：[TIDE 實作頁](https://github.com/dbolya/tide)；[原論文](https://dbolya.github.io/tide/paper.pdf)，ECCV 2020。六類：Cls、Loc、Cls+Loc、Duplicate、Bkgd（對所有 GT 最大 IoU ≤ 閾值）、Missed GT；oracle 修正並估 AP 影響，可讀 prediction file。

不可外推：輸入是單幀 prediction/GT，以 score 與靜態 IoU，無 frame-id、track identity、keypoint trajectory。故 Loc/Cls/Missed 不能證明框抖動或 pose 時序，也不取代逐幀、軌跡、關鍵點評估；結果受 IoU 及漏標／crowd 品質影響，不能宣稱本案必升 AP。

## 子任務 worklog

- 原因：為 warm-start 驗證固定官方口徑。
- 驗證：讀程式／論文，核對 score、AP、`maxDets`、`iscrowd`、六類；未作模型／資料操作。
- 困難：COCO 無版本年，註明基準年；TIDE 無時序證據，明列限制。
- 風險：本地 evaluator、標註、輸出格式待主代理核對；不替代重跑。
