# P3 bridge Activation 配對結果

2026-09-11。已完成四臂零樣本、SiLU／qSiLU 各 10 epoch 配對及 qSiLU 選定權重独立 Float／Bit-True 重驗。qSiLU 通過 activation 精度淘汰線並有較高 joint score，可作後續 KD 學生候選；不是所有指標都優於 SiLU，也不是原融合 gate 全部通過。

## 配對結果

同一 optimize P3 bridge + Pose recovery E5 起點，seed1、10 epoch／各 4,630 macro、完整 COCO／BBAT5 v1、相同 optimizer／augmentation／microbatch，選各自 activation-relative best_joint。SiLU 最佳 E9、qSiLU 最佳 E2；不是 E10 結束權重。

| Bit-True AP50–95 | SiLU E9 | qSiLU E2 | qSiLU−SiLU |
| --- | ---: | ---: | ---: |
| COCO box | 0.504309 | 0.503885 | -0.000423 |
| person box | 0.626783 | 0.625887 | -0.000896 |
| BBAT box | 0.612832 | 0.618008 | +0.005175 |
| BBAT pose | 0.889444 | 0.891329 | +0.001885 |
| ball box | 0.493900 | 0.505192 | +0.011293 |
| ball pose | 0.854713 | 0.859649 | +0.004936 |
| bat box | 0.731765 | 0.730823 | -0.000942 |
| bat pose | 0.924174 | 0.923008 | -0.001166 |

Joint score：SiLU 0.704562、qSiLU 0.706088，差 +0.001525。qSiLU 有 ball 收益，但 person／bat 小幅下降；單 seed 不能宣稱普遍優勝。相對原獨立模型的嚴格融合 gate，BBAT pose、bat box、bat pose 仍未全過。

## 設定與硬體邊界

重用原 `activation_lab` 的 qSiLU-PQ：|x| 節點 0、1、2、4、8，固定 dyadic 分段二次係數，無逐圖 scale、無可訓練 activation 參數。替換完整 SiLU references（包括兩個 head），保留 P3 bridge MASF／固定 PoT BinaryQK／PWL [-10,0] 20 段。
AdamW、backbone3.8e-7／neck1.9e-6／MASF3.8e-6／合法attention5e-8／heads5e-6、betas(0.948,0.999)、WD0.00027、warmup1、cosine final0.5、AMP、clip10。共享 BN running 固定、affine 可訓練；Detect／Pose weight1／0.25。兩臂 Detect logical128、physical16，每 macro256 Detect＋16 Pose，Pose physical16。
Physical32 的 qSiLU 反向 OOM，兩臂一同改16後真實 smoke 通過；峰值 allocated：SiLU10.81GB／qSiLU19.15GB。原始OОM與setup失敗輸出保留，不覆寫。qSiLU forward/backward 在本機 GPU 比原生SiLU耗時較長；不能以多項式硬體結構直接宣稱 GPU 加速。queue事件時間：SiLU約2小時30分，qSiLU約6小時37分，包含訓練／驗證／存檔，不是純推論 latency。

Zero-shot qSiLU 八項降幅都小於0.015；Hardswish最大下降約0.158，PolyShift ball box下降0.024，未擴大其短訓。沒有為候選重跑整個 Pose／融合階段，亦未執行多seed、20epoch finalist、PTQ／QAT或上板驗證；後續先依使用者要求銜接KD，不能將本結果稱完整INT8部署驗收。

## 選定權重與載入

`artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt`

SHA256：`1bbbbe0f87cb853827af1bb625e7066178e3fb2c7e14c410d997efb24e94834a`。
這是 activation-relative selector，不是最初獨立模型 gate 全過的融合權重。只含 state dict；載入必須重建 qSiLU 架構，不能當原 SiLU graph 載入後忽略 activation。`verify_selected.SelectedSource` 已獨立重建並精確重現八項 Bit-True AP。完整 optimizer續訓檔在同 run 的 `checkpoints/best_joint.pt`。

原始證據：`artifacts/zero-shot-v1/summary.json`、`artifacts/runs/*/summary.json`、`artifacts/paired-queue-v1/events.jsonl`、`artifacts/selected-and-teacher-probe-v1/summary.json`。全部保留；本階段不刪除、commit 或發布。

## KD 前置結果

直接把選定學生的 score 切成真正 FP qᵀk/√d（保留已學相對bias／PWL，非官方P0）只是teacher候選，不是假称已完成FP訓練。CPU確認實際FP算式；全量驗證結果比學生差：COCO -0.011906、person -0.007406、BBAT box -0.016675、pose -0.016762、ball box -0.024243、ball pose -0.031314、bat box -0.009107、bat pose -0.002211。不能拿這個候選直接啟動蒸餾。
下一步需要先補齊合格teacher，或明確改成任務分流教師方案。見 [KD 接續說明](<../../kd/dual_task_v1/README.md>)。

## AP 與部署成本補充（2026-09-12）

- [Activation：九項指標與比較](<../../../reports/performance/stage-6.md>)

包含 AP50–95、Params、Model size、MAC／FLOPs、Peak memory、CPU／GPU latency、target latency 及 energy/frame。target 未量測明記缺值；不將 core-only 時間當完整 pipeline。
