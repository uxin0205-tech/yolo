# 2026-09-08：以已訓練權重為起點的視覺驗證與 recovery 規畫

## 變更內容與原因

使用者補充：提供的模型基本上已經訓練過，主要不滿意實際視覺效果，希望整理規畫、驗證流程與超參數。據此將當前優先流程改為「實際checkpoint／案例重現 → 設定與完整指標核對 → 一對matched recovery」，不再把重建乾淨FP／J0列為所有優化的必要前置。

新增 `optimizations/trained-model-recovery/README.md`、`plan.md`、`hyperparameters.md`，完整說明HOG暖啟動、MASF必要bridge與Detect-only架構、BinaryQK challenger契約限制、選配layer17 RepConv，以及視覺／AP兩條驗收路徑。舊研究保留供追溯，不刪除、不冒稱失效實驗。

依使用者指定，機械式盤點、一手評估來源整理與既有入口同步交由既有 Luna/max 子代理；主要判斷、超參數與流程由主代理負責。使用 `diagnosing-bugs` 的可重現問題前置；因尚無具體案例，診斷未進入已驗證根因階段。`research` 用於COCO/TIDE一手依據，`finish-work` 用於交付索引與結果邊界同步。

## 文字與 metadata 核對結果

- Full35 `j3_best_joint` 的 inference/full-resume檔案存在；J2 rollback與best_detect/best_pose也存在。僅stat，未讀checkpoint內容。
- J3實際README記Detect physical32×4=logical128；存檔YAML有microbatch64及舊stage欄位，故標明來源差異，不以舊YAML冒充最終resolvedcfg。
- 主代理以Python3標準庫串流唯讀 `outputs/training/logs/macro.csv`；6個role的`.decay`欄位各5,093筆、step23,819–28,911。觀察峰值：backbone3.8e-6、Neck1.9e-5、Detect/Pose各5e-5、MASF3.8e-5、attention5e-7。只據此提出更保守的recovery起點，不說已找出最優LR。
- 推論原始碼預設640/conf0.25/iou0.7/max_det300並有end2end分支；尚無使用者實際指令，不把default當作案例設定。
- `_joint_config_impl.py` 明確拒絕baseline `qk_ste=true`。新計畫將QK微調標為需獨立challenger契約，不建議繞過assertion。
- COCO/TIDE研究結果寫於 `docs/research/2026-09-08-trained-model-validation-evidence.md`；只支持評估方法，不支持本模型已有精度改善。

## 驗證方式與結果

本次只做檔案存在性、原始碼／配置閱讀、CSV日誌彙整及Markdown交付一致性檢查。文件連結／必要內容檢查結果於本紀錄末段補記。沒有GPU、模型載入、forward/backward、training、模型validation、calibration或profiling；未產生新的AP、可視化結果或checkpoint。

資料集入口保持canonical bbat5-v1與COCO80；未建立新split、抽樣、修改影像或labels。未修改訓練／推論程式，未動GPU上的既有工作，未提交或推送Git，未刪除歷史資料。

## 困難與解法

1. 使用者尚未提供實際視覺失敗案例／當時指令：已提出非阻塞詢問；先完成可執行的未來驗證規格，根因維持未確認。
2. 存檔配置與最後J3執行記錄有差異：以README與macro.csv分別記錄可證實事項；正式啟動前仍須凍結完整resolvedmanifest。
3. 現有模型含shared MASF與固定BinaryQK契約，不能直接套舊cleanFP流程：將HOG改成保留原圖的warm-start，MASF relocation加入有條件bridge，QK訓練保留契約前置。
4. 沙箱預設命令發生bwrap網路namespace問題：讀取命令使用審核允許的相同唯讀範圍；文件修改仍以apply_patch完成，不擴大資料操作。
5. 無法讀取使用者所指weekly limit的可靠數值：未假裝已確認額度；控制本輪範圍，完成文件即收尾。

其他困難：無。

## 未解事項與風險

- 使用者實際影片／圖片、權重與推論設定仍待確認，因此未找到已驗證的視覺問題根因。
- 所有新超參數、HOG收益、MASF bridge可恢復性、部署開銷與QK梯度可學性均未經模型實驗。
- 首輪單seed只能提供工程訊號；開發案例與validation有模型選擇偏差，不宣稱獨立test泛化。
- 實際跑GPU與新增任何模型契約／訓練程式需後續授權；本輪不因計畫完成自動啟動。

## 文件驗證補記

主代理以Python3標準庫唯讀檢查本次三份新計畫及本紀錄：4/4檔案可讀，9個Markdown連結逐一解析，所有本地目標存在，fenced code block均成對；檢查結束碼0。另核對兩份Full35 YAML的mosaic／flip記錄及gate CSV的Pose欄位，與文件所述來源一致。這是文件／文字驗證，不是模型validation。

入口同步子代理完成7份既有README／plan／索引的連結、圍欄、空白與關鍵標記檢查：連結錯誤0、格式錯誤0、pending目標0。主代理另唯讀複核最新優先入口、歷史／另案標示、資料集規範及工作紀錄入口。研究子代理已修正COCO摘要的適用範圍措辭，避免誤稱COCOeval僅支援bbox/segm；無額外模型或資料操作。
