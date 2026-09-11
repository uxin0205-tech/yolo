# final 接續準備、基準修復與後續優化

更新：2026-09-08。狀態：**計畫完成，模型實驗未執行；現在不使用 GPU。**

使用者最新指定從 `yolo_combine/final` 接續，但其中很多部分可能未 train 好。因此不把已有 checkpoint 視為成熟基準；先盤點、重現，必要時以原生 loss 補訓／修復，再開新優化。不預設全部從頭重訓，也不預設 5–10 epochs 足夠。最新優先入口是[final 接續現況與待處理清單](<final-readiness.md>)，`training_ready=false`。尚不能斷言當前視覺問題由 MASF、HOG 缺失或 BinaryQK 引起。

> 本輪方向1唯一當前整合順序見[方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)。`B0` 是原 J3 `best_joint` 的歷史分數參照；先同口徑比較 J3 `best_joint`／J3 `best_pose`／J2 `best_joint` 後選 `PSEL`，尚未選定，`best_detect` 不直接作 task=both。必要原生 loss 修復／小範圍 AdamW 適應後，才做單因子方向1消融。正式 baseline 維持 AdamW guard；MuSGD 有 builder 支援但只作另案 paired challenger，中途切換的 state／LR 校準尚待驗證。「BN 較大」是模型／參數改動較大，不是 batch 或 BatchNorm。`training_ready=false`，GPU 0。

## 入口

- [本輪唯一當前順序：方向1 master plan](<../integrated-roadmap/direction1-master-plan.md>)
- [最新：final 接續現況、基準修復與實驗順序](<final-readiness.md>)
- [停用的接續 manifest（不是 trainer 設定）](<continuation-manifest.json>)
- [驗證流程、實驗順序與架構](<plan.md>)
- [歷史設定與本輪建議超參數](<hyperparameters.md>)
- [最新工作紀錄](<../../docs/worklogs/2026-09-08-final-readiness-correction.md>)、[前版規畫紀錄](<../../docs/worklogs/2026-09-08-trained-model-recovery-plan.md>)
- [COCO／TIDE 一手評估依據](<../../docs/research/2026-09-08-trained-model-validation-evidence.md>)
- [原三方向完整研究](<../integrated-roadmap/README.md>)：保留研究背景，不等於本輪必跑清單。

## 已有什麼，還缺什麼

接續來源已固定為 `/home/uxin/yolo/yolo_combine/final/full35/`；`j3_best_joint` 只是首個候選，還不是品質已驗收的共同 parent。已透過檔案 metadata 確認下列權重存在，**沒有載入 checkpoint 或跑推論**：

| 用途 | Full35 下的路徑 | 本輪注意事項 |
| --- | --- | --- |
| 重現現有候選 | `weights/combined/inference/best_joint.pt` | 先核對品質，不預設已 train 好 |
| 新 repair 的來源候選 | `weights/combined/full-resume/best_joint.pt` | 先選 resume 或新 run；不能把 fresh optimizer 稱 exact resume |
| 診斷既有取捨 | `weights/combined/inference/best_pose.pt`、`best_detect.pt` | best_detect 的 Pose 未適應、舊 gate 未過，不直接用於 task=both |
| 既有回退對照 | `weights/rollback/j2/inference/best_joint.pt` | 可用來判斷後期 joint 的變化，不等於乾淨 FP-QK 對照 |

使用者實際視覺案例仍待補，但來源已明確，不阻擋此次整理。取得案例前不能確認其精確視覺問題根因；準備完成不代表模型修復完成。

```text
final/full35 候選 P0（訓練充分性待驗）
    │
    ├─> 固定案例重現 + 完整既有 validation
    │       └─> 核對 threshold／前後處理／EMA／backend／task
    │
    └─> 必要時先以原生 loss 補訓／修復 → 驗收 P-READY
            │
            └─> 再依錯誤證據選一組 matched optimization
                    ├─ 訓練監督：W-CTRL10 vs W-HOG10
                    ├─ MASF 位置：必要時 no-MASF bridge → Detect-only 對照
                    └─ BinaryQK：診斷來源；獨立 challenger 過契約才訓練
                         ↓
             視覺改善 + 指標不退化 + 部署版本複驗
                         ↓
                保留 winner，否則回退 P-READY
```

RepConv 僅在證據支持時測 layer17 一處；person-only 暫緩。蒸餾、更多 scale／basis 與多方法組合列第二輪，不同時堆入首輪。
