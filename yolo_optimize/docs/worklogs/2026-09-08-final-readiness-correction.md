# 2026-09-08：final接續準備與訓練充分性假設修正

## 變更內容與原因

使用者明確指定從 `yolo_combine/final` 接續，但指出其中許多東西可能沒有train好；本輪只先整理，列出待處理內容。修正前版把已有checkpoint當作成熟可直接微調基準的假設：接續來源確定為final/full35，品質與必要補訓範圍仍需驗證。

新增 `optimizations/trained-model-recovery/final-readiness.md` 與停用的 `continuation-manifest.json`，列清楚可沿用來源、已知未適應的best_detect、Q/K未進行sign-decision學習的契約限制、正常early-stop與品質不佳的差別。順序改為來源盤點→重現→必要原生loss修復→可接受基準→新優化。原三份recovery文件加最新前置，HOG/位置/QK配方延後，不自動從零重訓，也不保證10epochs足夠。

使用finish-work完成有限來源稽核與入口交付；diagnosing-bugs只用於限制未重現問題的根因宣稱。本次沒有足以斷言當前視覺問題根因的模型回饋迴路，不進行假設測試或模型修復。依使用者既有委派要求，Luna/max子代理負責有界文字盤點、既有入口機械同步；主要分類與順序由主代理決定。

## 驗證方式與結果

- 主代理以Python3標準庫唯讀解析final/full35/MANIFEST.json：407筆全部存在且stat大小相符，缺失0、大小不符0、unsafe path0、manifest symlink0。
- 僅對README.md、run.py、verify.py、configs/joint.yaml、configs/experiment-joint.yaml、code/project/src/yolo_combine/joint_cli.py這6個小型文字檔計算SHA256，6/6與manifest相符。完整hash存入新continuation manifest。
- 未讀任何.pt內容，沒有執行原verify.py的全包雜湊，也沒有檢查manifest外所有檔案覆蓋；有限metadata稽核不叫完整權重驗證。
- 子代理核對發布分析／RELEASE_STATUS／gate CSV／events.jsonl：J0 8/8、J1 20/20、J2 25/80、J3 11/20；本次事件級證據限J3尾端，正常patience5停止。J0–J2來自發布報告，不冒稱全程逐事件重建。
- 主代理核對final的run.py載入包內code/project/src，CLI支援J3/microbatch覆寫；YAML的enable_j3=false與microbatch64不等於J3没跑。final的stage policy和原分析顯示J3 warmup3，修正前版通用YAML warmup1可能誤導之處。
- best_detect Pose尚未適應／舊gate下限-0.08皆有原發布報告支持。尚無證據判定其他模組就是欠訓練或本次視覺問題根因。

文件、JSON及入口驗證於交付前補記。未使用GPU、torch、模型載入、forward/backward、模型validation、training、calibration或profiling。

## 困難與解法

1. 「final」容易混淆交付完成、原實驗完成、品質合格：拆成三種狀態，不把使用者疑慮反過來寫成所有stage沒跑。
2. 可攜YAML不是完整J3有效設定：記錄CLI/stage覆寫與待resolved config事項；不直接改原始final。
3. 部分歷史事件未在當前JSONL覆蓋：用發布報告註明證據等級，未擴大尋找整個工作區所有歷史run。
4. 預設apply_patch／exec沙箱遇bwrap loopback錯誤：使用允許範圍內的提升權限唯讀命令與互動式apply_patch，只改yolo_optimize文件。

其他困難：無。

## 未解事項與風險

完整checkpoint完整性、載入／resume狀態、完整resolved config、使用者具體視覺症狀、是否要補訓及補哪些層仍待後续驗證。所有超參數是候選，training_ready=false；不會自動啟動。canonical資料與labels/split不變、person-only暫緩。未修改final、未刪權重／archive／cache、未提交或推送。所有來源與歷史結果均保留。

## 交付驗證補記

主代理標準庫檢查：5份本輪Markdown、19個本地連結，錯誤0；JSON解析及training_enabled/training_ready/gpu_authorized皆false、accepted parent為null的斷言通過，來源／資料路徑存在，6個小型文字雜湊再次吻合。子代理檢查7份入口（含未改的研究索引）：缺失連結0、未配對圍欄0、尾端空白0、舊入口誤導措辭殘留0。主代理複核入口後只修正README圖的縮排及回退到P-READY，無新增連結或資料／模型操作。
