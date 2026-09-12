# 2026-09-12：全階段精度與部署成本比較

## 需求與範圍

使用者要求各階段加入 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU latency、GPU latency、target hardware latency 與 energy/frame，納入詳細報告並上傳 GitHub，commit 名稱沿用 `5090 Done 0912`。

使用 finish-work 技能進行來源盤點、補測、結果整合、驗證及發布；由主代理執行，不使用子代理。不重新訓練、不改 COCO／BBAT5 v1 split、影像或標註，也不覆寫 checkpoint。

## 變更與方法

新增 `experiments/benchmark/`，固定 26 組既有模型代表／配對：FP／A0／B100、HOG control／HOG、Conv／折疊 RepConv、P3 control／shared／fork／bridge、P2 control／MASF、舊新融合、Pose 恢復、SiLU／qSiLU、Hardswish／PolyShift zero-shot、雙教師 KD、Pose-head KD、關鍵點重組及 Pose one2one／one2many。

manifest 保存 checkpoint SHA、指標來源 SHA、既有 AP 與 task。AP 來自既有完整驗證，不用合成輸入重算 AP。source snapshot 大小和部署 FP32 tensor payload 分別列出，Params 使用重建後唯一 registered parameters。

固定 B1、640×640、FP32、seed 50900912 合成輸入，CPU 4 threads／interop 1，eager eval inference_mode，TF32 關閉，不使用 autocast／compile／額外 fuse。CPU 暖機 2＋10 次，GPU 暖機 10＋50 次，保留逐次樣本、median、mean、P90 及同步 wall time。

GPU peak allocated／reserved 在暖機後重置，包含 resident 模型與輸入；CPU RSS 是 worker 高水位，包含載入。GPU energy 由 NVML 累計 mJ 差計算，每組 3 個至少 2 秒、至少 10 frames 區段；不是 TDP 推估，也不是 CPU／整機／目標板能量。

各 GPU 工作以順序 queue 執行，程序每次最多等待 600 秒；工作完成立即接續。正常不重讀 log／GPU，只有錯誤才診斷。此次沒有訓練 queue。

## 困難、修正與驗證

1. CPU→GPU 時融合 wrapper 未替 head 普通屬性移動 decode cache，產生 device mismatch。量測工具明確移動 stride／anchors／strides 並清除 shape，保留權重，qSiLU 短測 v2 通過。
2. 靜態 pickle 稽核發現 P3 fork 類別需加入白名單。只加入已檢視的 `masf_p3.P3MASFDetect`；所有載入維持 `weights_only=True`、固定 checkpoint SHA 與有限 safe globals，不動態信任未知類別。
3. inference_mode 使 dispatcher 漏掉部分矩陣乘法；已由已知矩陣尺寸重現，成本改用獨立 no_grad CPU accounting-v2。浮點矩陣、整數矩陣與分組卷積共 3 項測試通過；latency 仍用 inference_mode。
4. CPU accounting 曾與 4 組時延測試重疊。依工作起迄保守圈定 j3_joint、pose_recovery、silu_e9、qsilu_e2，僅這 4 組另以 isolated-recheck-v1 補測，原值保留，其他 22 組不重跑。

MAC subtotal 計 Conv 與矩陣乘法，另記 BinaryQK 位元乘積；FLOPs 不包含 BN／activation／PWL／reciprocal／pooling／NMS 等，不能稱全算子或硬體 cycles。CPU 算量補正不產生新 AP。

原始 26 組 queue 已完成；完整隔離重測、報告數值與發布檢查以後續交付紀錄及 `reports/performance/` 的驗證檔為準。每個 checkpoint 在量測前後核對 SHA，RepConv 折疊另做 CPU160 數值對照。

## 報告與發布

完整結果放 `reports/performance/`，提供 CSV／JSON、八個階段分表與量測限制；主報告及各階段入口同步加上九項比較。失敗紀錄與原測量保留，不為了表格完整而編造數字。

發布目的地沿用 `uxin0205-tech/yolo` 的 `main`，提交名稱 `5090 Done 0912`。只發布報告、數值證據、設定與研究程式，不上傳模型、資料集、PDF、cache 或大封存；保留其他專案與未提交修改。若平台再次拒絕 push，必須如實回報，不能繞過審查。

## 未解事項與風險

已詢問目標板卡與功耗設備，尚未指定／連接時 target latency 及 target energy/frame 保留缺值。NVML 整卡遙測含 idle／桌面活動，沒有獨占 GPU 或鎖時脈；小幅時間／能量差異不作因果結論。

量測是 core-only，排除影像解碼、H2D 與外部 NMS；one2many 仍需 NMS，不能把其 core latency 宣稱為完整 pipeline。未做板端等價、能耗 rail 量測、獨立 test 或多輪隨機順序統計驗證。沒有刪除任何原始模型或正常結果。

## 量測與報告完成

26 組正式量測、26 組 CPU 算量補正及 4 組隔離時延／能量補測全部完成。最終報告分為 8 個階段，另有 15 組明記比較口徑的差值。Target 欄位保持未量測；沒有新 AP、訓練或資料修改。

## 發布前檢查與審查限制

完整表格驗證通過：26 組 AP 與 manifest 精確一致，所有可量測成本為有限正值，target 缺值有明確理由；174 份 Python AST 與 914 個現行文件連結通過。成本公式測試再次 3／3 通過。

GitHub CLI 未登入，無法用 gh 查詢可見性；使用現有 origin 與 Git fetch 核對目標，不讀取或輸出任何憑證。

一次準備操作嘗試更新 README 的前次發布狀態、排除本機 PUBLISH_STATUS.json 並清除未使用計數器，被 auto-review 拒絕。拒絕理由是不能移除先前 push 受阻的明確證據。該命令未執行；README、PUBLISH_STATUS.json 與發布排除規則維持原狀，未採替代通道繞過。現行 measure.costs 已使用 accounting 修正版，保留的舊未使用類別不參與報告計數。

本輪成本報告完成與前次 push 尚未成功是兩個不同狀態。後續是否發布成功，須以實際遠端 hash 驗證為準，不能由本機 commit 推定。
