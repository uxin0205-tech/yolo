# 2026-09-13：MASF 完成後，硬體友善 Attention 與 Rep17／20

## 授權及已完成前置

使用者取消額外未改架構加訓，要求先完成MASF、報告與分析，再處理Q/K task gradient、scale/bias、原生QK與layer17/20 RepConv。MASF B5已ALL_DONE且通過權重稽核；先核對結果，未延長B回合。研究共同parent固定B5，正式default不換。

B5 Pose AP88.9071%，較原E2增加.2128pp，但B5關Pose alpha後88.9096%，不能宣稱MASF獨立帶來Pose改善。先前native QK已訓練E2，B5基於它，重用既有完整驗證，不重跑原生對照或解除舊pause。

## 本次變更

新目錄 experiments/post_masf_hardware_v1 保存研究報告、三候選模型、CPU檢查、雙任務適應訓練及queue。三者由共同B5獨立開始：scale_bias→Rep17→Rep20，各5epoch/warmup1；沒有不改模型加訓對照、沒有自動延長或同時加入兩層Rep。

新bias採signed16-bit m/1024，scale合計16個固定m/1024常數，逐圖片不重新估計。保留PWL[-10,0]20段、qSiLU、兩套MASF。Rep理由、成本、stride2無identity與fold公式均寫入README，並引用舊Rep17負面結果與原論文。

## 診斷與驗證

採research主來源流程（單一主代理，不用子代理）核對STE原論文、RepVGG原論文／作者實作與本機code。舊QK score零梯度存在直接證據；先前新scale_bias真實smoke已有Q/K更新，故不能籠統稱目前全部沒有梯度。

此次先執行preflight.py，bias梯度檢查紅燈；依diagnosing-bugs做小範圍範圍／normalizer／零表對照。PiecewiseLinearSoftmax正確，既有bias約−5至+6.5，初始±2clamp使部分offset梯度0；擴充固定16bit格式後通過。另修正Source.variant與父類arm欄位撞名，避免BitTrue模板建錯；新queue.py改名run_queue.py，避免遮蔽PyTorch所需標準queue。

結果：三候選CPU preflight全部通過，scale與兩軸bias可得梯度、score訓練／推論前向一致、Rep17/20零分支初始等價及fold等價，兩種PWL模板重建通過。run_arm.py --help於CUDA_VISIBLE_DEVICES空字串下成功；4個Python AST通過。未宣稱CPU合成目標等於真實task loss，GPU必先smoke才進入該組正式訓練。

## 限制及安全

資料固定COCO完整train118287/val5000與BBAT5-v1完整train5964/val683，沒有重切或新增版本。雙任務macro沿原Attention設定256Detect+16Pose，非Pose-only B組的128。每组5E是方法改動適應，不是加回未改模型control；缺少control仍有收益歸因限制。

部署常數約2KiB是解析值，PyTorch仍展開相對bias做reference；不稱全Attention無乘除、INT8上板或實測硬體加速。target latency/energy尚未量測。沒有刪除舊權重、結果或資料；沒有自動發布Git。排程狀態以本機events/summary為準。

## 實際啟動與 GPU smoke 通過

新佇列 PID2517588，2026-09-13 20:05:40（Asia/Taipei）開始scale_bias smoke，20:06:02完成並自動進入該組正式工作（先E0全量驗證，再5epoch，PID2517983）。兩處Q/K、scale與兩軸bias均有真實task-loss梯度与參數更新，固定state通過。Q STE覆蓋率範圍0.4216～0.4479，K為0.3549～0.4609；峰值allocated 9.125 GiB。這不代表AP已回升；部署整數碼是否改變及最終精度須等待逐epoch／獨立驗證。Rep17與Rep20仍排於後方。
