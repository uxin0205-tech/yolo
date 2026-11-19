# 2026-10-17：補充全階段硬體友善、特徵圖形狀分析與清理要求

## 任務與範圍

使用者要求所有 RepConv 候選使用 qSiLU、PWL 等硬體友善方法，並在每個階段檢查除法等運算是否有可行替換；最後加入長方形及二次冪特徵圖的分析、完整模型複查，以及無用資料夾的實際移除。使用者表示稍後才會指定開始，本次只修訂正式計畫與入口。

所有研究階段維持「待指定」，沒有啟動模型實作、載入 checkpoint、GPU 工作、正式 AP 評估或資料夾清理。採單一主代理，未提交或推送 Git。

## 變更內容與原因

- 更新[正式整合計畫](../research/yolo-research-plan-2026-09-21.md)修訂日期為 2026-10-17，保留九個階段與既有 40 個候選／工作項目，以及 2026-10-03 的啟動方式與 COCO 球棒／球評估目標。
- 增加逐階段硬體檢查表與運算清單要求。固定二次冪除法、其他固定除法、動態分母與隱含非線性分別評估，明記 shift 的數值契約、常數折疊、reciprocal、誤差與剩餘成本；訓練專用與離線運算分開計算。
- 加強第 3C、V6 與 R1～R3 的完整硬體契約：全部 RepConv 候選使用共同 qSiLU parent 與相同 Attention／PWL／reciprocal 設定，核對訓練前向、folding、重載及匯出後實際算子，禁止靜默回退。
- 將第 9 階段細分為 9A 全模型部署運算檢查、9B 長方形／二次冪特徵圖分析、9C 最終成果與清理。先 trace 真實形狀與分組，再篩選少數候選，重查位置 bias、mask、token／分母位寬、座標還原與匯出，不預設二次冪更快，也不重跑全部訓練矩陣。
- 補充 shape profiles、runtime padding、位寬與部署算子等 manifest 欄位。尺寸只作執行時處理；canonical `/home/uxin/yolo/original/pose/derived/bbat5-v1/`、registry、來源、影像、標註與 split 均保持不變。
- 明確記錄使用者對無用資料夾的清理要求：對指定階段內完成證據抽取、替代驗證且不再被引用的內容實際刪除，留下清單與空間變化；受保護資產與尚有依賴的內容保留。此次沒有將未盤點資料夾直接判定為無用。
- 同步更新[根 README](../../README.md)、[`yolo_cleanup.md`](../../yolo_cleanup.md) 的接續標示與[工作紀錄索引](README.md)。

## 驗證方式與結果

五份文件的靜態檢查通過：原有九個階段及 40 個候選／工作項目的順序均保留，包含 3A～3C 與新增 9A～9C 的 15 列狀態全部維持「待指定」。核對逐階段硬體要求、三組 RepConv 的共同契約、形狀驗證與清理條件均已寫入。

74 個本機 Markdown 連結均有效，code fence 成對閉合，新增行無行尾空白或衝突標記。逐份比對修改前內容與預定替換，確認沒有額外改寫；人工檢視四份既有文件的差異。

2026-10-03 的啟動說明與 COCO 球棒／球評估段落、舊任務書正文均完整保留。`AGENTS.md`、`CONTEXT.md`、BBAT5 registry、COCO 設定及 2026-10-03 工作紀錄共五份受保護文件的 SHA256 未變，既有 Git 工作區狀態保留。

已查閱 [Ultralytics 官方尺寸說明](https://docs.ultralytics.com/modes/predict/#fixed-shape-vs-minimum-rectangle-rect)，核對 predict 的長方形尺寸與 stride padding，以及 training／predict／export 介面差異。文件中的 `640×640 → 40×40`、`512×1024 → 32×64` 均明示為累積 stride 16 及對應 padding 下的條件式算例，不當成本專案實測。

此次只驗證文件，不新增程式測試。未執行 CPU 模型 smoke、形狀前向、數值 reference、正式 AP、訓練、RTL／HLS 或板端量測。

## 困難與解法

一般沙箱命令仍遇到 `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`；改由 `require_escalated` 執行必要文件讀寫。寫入限定五份指定文件，使用修改前 hash 與唯一文字定位檢查，保留既有未提交內容。

修改前四份文件的短期比對資料位於 `/tmp/yolo-hardware-plan-20261017-yrkq6bt_/`，另記錄規則、資料 registry、COCO 設定與 2026-10-03 紀錄的 hash 以驗證未被改動；未在研究資料夾增加 legacy／archive／backup。其他困難：無。

## 未解事項或風險

實際模型的 shape 限制、各算子部署可行性、目標硬體與尺寸收益均待對應階段開始後確認。官方通用介面不保證本專案自訂 Attention、融合模型與 trainer 已支援所有形狀。

目前未指定新的輸入解析度、shape 候選訓練預算或刪除清單；尚未盤點的資料夾不因名稱被刪除。硬體友善補充不自動啟動整網 PTQ／QAT，未量測項目不宣稱加速。
