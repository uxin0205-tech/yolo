# 2026-09-08：方向1、BEST選擇與AdamW／MuSGD訓練整合

## 變更內容與原因

使用者要求整合方向1其他優化與optimizer／超參數，並澄清「BN較大」是模型／參數變動較大，不是batch或BatchNorm；必須先比較哪個BEST可用，再決定後續處理。

新增integrated-roadmap/direction1-master-plan.md、optimizer-policy.md、direction1-plan.json。B0固定為原J3best_joint歷史錨點，PSEL則由BEST比較選出且目前為null。整合MASF/HOG/BinaryQK、RepConv、衝突投影、scale、部署效率與条件式MuSGD長訓；R2-REGION/BASIS保持第二輪。不將成熟parent、重建FP/J0或一定MuSGD當強制前提。

研究skill委派一手資料查核；finish-work用於交付和入口同步。依使用者先前指定，已有Luna/max子代理分別處理本地optimizer唯讀盤點、一手資料紀錄與入口同步；主代理負責方法選擇、比較契約與超參數設計。

## 已核對的證據

- 既有candidate CSV與逐epoch validation CSV：J3best_joint joint0.7111747389752653、J3best_pose0.7111415212904801；best_detect Pose未適應。best_pose selector是BBAT box/pose平均，非keypoint AP單項最高。
- final builder實際支援AdamW與MuSGD，config parser也允許兩者；原trainer只建一次optimizer，stage更新LR/scheduler，沒有自動class切換。更正任何把MuSGD說成被正式AdamW guard禁止的暫時推測；Q/K STE guard是另一個仍存在的限制。
- 本地MuSGD在yolo_combine/.venv的Ultralytics 8.4.90，不在yolo根.venv。只讀METADATA／原始碼，沒有import套件。
- formal MuSGD是momentum0.948、nesterovtrue、muon0.2、sgd1.0，role×decay分組；decay matrix/conv走hybrid，no_decay走SGD。權重不需加總為1；decay語意與AdamW不同，不能照搬LR或將兩種state硬轉。
- 超參數中MuSGD neck1e-3／heads2.5e-3是未執行的probe初值，正式roleLR仍null；須通過同training trace更新幅度、scalar/matrix群組與scheduler覆寫檢查。

## 驗證方式與結果

本次僅讀取既有CSV、程式、版本metadata、規範與一手來源，並整理Markdown/JSON。文件連結、JSON停用旗標、候選／分數來源和入口檢查於交付前補記。沒有GPU、模型載入、forward/backward、訓練、模型validation、calibration、profiling或依賴升級。

## 困難與解法

1. 使用者「BN」歧義：收到澄清即改成模型變動，停止以大batch解釋optimizer選擇。
2. BEST檔名容易被當全面最優：列五個現有候選及分數公式；先篩選兩個主要候選，其他依弱項追加，不把所有checkpoint都訓練。
3. MuSGD存在實作但缺混合階段handoff：區分builder可用與切換未實作；fresh/fresh配對、相同live/EMA/criterion/loader狀態，避免state reset或更多epochs混淆。
4. 通用MuSGD預設與本地formal builder不同：以本地8.4.90有效group設定為契約，官方main只作背景，明記版本差異。
5. sandbox bwrap loopback限制：唯讀操作使用審核允許的相同範圍；檔案修改使用apply_patch或互動式apply_patch，不以其他工具覆寫。

其他困難：無。

## 未解事項與風險

PSEL、完整checkpoint載入、具體視覺症狀、MuSGD有效LR／handoff與新模組相容性尚未實驗；不宣稱任何新AP或必然優越。training_ready/training_enabled/gpu_authorized皆false。普通optimizer階段切換不是已證明創新。

資料保留COCO80與canonical bbat5-v1原assignment及labels，person-only暫緩；final、weights、archives與logs未修改或刪除；沒有commit/push。沒有可刪除清理候選，所有來源保留。

## 交付驗證補記

主代理標準庫檢查4份新Markdown、17個本地連結，錯誤0；新JSON解析、三個disabled旗標、selected_parent及MuSGD正式roleLR為null、builder可用但自動切換不可用的斷言通過。B0分數與原candidate CSV一致，4個合法候選舊gate為true、best_detect為false。舊continuation manifest新增最新整合JSON入口及BEST待選標示，不改來源或權重。optimizer文件補記O只回答handoff後的比較，不宣稱AdamW前綴必要，20ep不是充分長訓證明。

入口同步子代理完成15份既有入口／索引檢查：缺失連結0、未配對fence0、尾端空白0、舊誤導措辭殘留0，新目標檔均已存在；並修正conflict-safe的person-only前置與scale-codebook強制重建FP入口。主代理交付前再檢查新Markdown與兩份JSON，保持所有執行旗標關閉；僅文件準備，不代表模型驗證。
