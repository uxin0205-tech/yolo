# 2026-09-09：MASF bridge 未過，接續 BinaryQK 單點 score 診斷

## Bridge 結果

`masf-off-parent-bridge` 完成 E1–E4，quiet supervisor 正常 exit0，status paused_for_analysis；E4 Ball Box=0.5013680048，比原 BEST 低 0.0060690314，觸發預設安全線。各 epoch EMA joint：0.7102720871、0.7103233237、0.7102358449、0.7099530038。最佳 E2 仍低於原 BEST，也略低於零步關閉診斷 0.7103965838，沒有建立可採用的 no-MASF bridge。

決策：本輪保留原 shared MASF，跳過 Detect-only relocation，不直接搬權重、不放寬 gate、不接退化 E4 加訓。單臂停止不結束流程，轉入 BinaryQK 的既定前置診斷。原 BEST、bridge checkpoints 與資料集均保留。

## QK 診斷契約與限制

從原 BEST 分別執行 `qk-fp10-parent-diagnostic`（FP dot site10／binary site22）與 `qk-fp22-parent-diagnostic`（binary site10／FP dot site22）。只換該 site 的 score 為 `(q/sqrt(D)).T @ k`，保留既有 QKV、relative bias、PWL、value path、fixed coefficients、heads 與另一個 binary site。完整 COCO80 val5000／canonical BBAT5 val683，以原 BitTrue-normalizer evaluator 評分。

此為 **eval-only FP-dot + retained bias/PWL 的干預診斷**，不是 VariantConfig 的純 FP P0 baseline，不是從未 binary-trained 的 teacher，更不是已通過 export／hardware contract 的訓練 challenger。原 FP P0 config 禁止 relative bias；本診斷不修改或繞過該 config assertion，而是明確另列 score 介入來控制 bias 不變。不能據此將 YAML 改成 FP＋bias 就作正式部署。

`qk_diagnostic.py` 以 Source proxy 只對新 materialized task 圖掛入 eval-only score hook；不改原 bundle／baseline config，也不啟用 STE 或 Q/K 訓練。Hook 檢查 training mode 並拒絕訓練。原 binary score 仍先執行再替換，因此診斷完全不能拿來宣稱 latency／energy 收益。`last_scores` 不作蒸餾梯度節點。

Float／BitTrue backend 名稱在這裡只描述其餘保留的實作與 normalizer；整個 hybrid 不再是純 binary graph。候選不輸出可部署權重。變好表示值得再研究該 site 的誤差，變差可能是已訓練權重對 binary 分布的適應，不能把一次替換當成完整 FP／Binary 因果實驗。

## 修改與驗證

新 proxy、runner candidate 與 quiet stage 均在本 workspace；run_recovery 退出前確認每個預期 Detect／Pose hook 有實際呼叫，provenance 記錄 site、backend、hit 次數與非部署限制。

`tests/test_qk_diagnostic.py`：2 passed，0.58 秒；涵蓋 FP dot 公式、只替換選定 site、錯誤 shape／site 拒絕、training-mode 拒絕。額外實際 Full35 CPU 整圖 160×160 驗證：兩個 site 各自 materialize Detect／Pose，四個介入均呼叫一次，前後每個 state tensor 完全相同；不作 AP／效能證據。

## 接續與風險

依序完成兩個免訓練診斷，原 as-is 分數重用原 BEST 重驗；正常只作最多 600 秒 blocking wait，不讀 log 或額外查 GPU。完成後比較八 AP／joint，只有有用訊號才擴展；QAT／KD 必須另有合法 challenger、真實可回傳梯度及更合格 teacher，不強開 baseline STE。

困難：無。沒有刪除、commit、push、改資料版本或產生新的 split；原 BEST 不覆寫。尚無新的 QK AP 結果，不預判哪個 site 勝出。
