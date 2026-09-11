# 先前 combine 報告：實驗 4 核對與目前訓練修正

來源為使用者提供的《碩一_陳宥炘_0831.pdf》（本機／歷史參照：`../../combine/碩一_陳宥炘_0831.pdf`；未隨本次報告發布），實驗 4 位於 PDF 第 9–11 頁。文字完整讀取並視覺核對第 10、11 頁；本次由主代理直接執行 research 技能，不使用子代理。

## 報告實際方法

第 9 頁：共享 layers 0–22 backbone／Neck，分離 COCO80 Detect 與 ball／bat Pose head；兩個 dataloader，每 macro 兩個 Detect logical batches 加一個 Pose batch，Pose 用完 cycle。loss 先各除以實際影像數，再以 Detect／Pose 權重 1／0.25 加權，除以權重和，再乘 reference batch size。這不是直接相加兩個 raw loss。

第 9 頁：先前觀測 Pose 共享梯度約為 Detect 的 19.7 倍，因此降低 Pose 權重。joint score 為 0.2 COCO overall + 0.2 person + 0.2 BBAT box + 0.4 BBAT pose，全部使用 AP50–95；報告最終差值門檻為 0.02。

第 10 頁：先建立独立 Pose baseline，再聯合融合，並非只有 J0 head-only：

| 階段 | 報告 epochs／patience | 範圍與 optimizer |
| --- | --- | --- |
| P1 | 17／10 | Pose head，MuSGD |
| P2 | 22／12 | 表格 scope 欄誤植為 LR，須以程式補足 |
| P3 | 100／20 | full graph except attention，MuSGD |
| J0 | 8 | Pose head，AdamW |
| J1 | 20／5 | Neck＋heads，AdamW |
| J2 | 40／10 | backbone layer9+、Neck、MASF、heads，AdamW |
| J3 | 20／5 | full low-LR，AdamW |

P1–P3 報告 lr0=0.00038、lrf=0.5。J1 Neck/head LR 7.5e-5／2e-4；J2 backbone/Neck/MASF/head LR 1.5e-5／7.5e-5／1.5e-4／2e-4。以上是報告數值，不代表目前所有配置完全一致。

原[training.yaml](<../../../yolo_combine/variants/full35/configs/training.yaml>) 補足 P2：Neck、Pose26 head、MASF 與 attention 可訓練部分；P3 是 full graph，鎖定 Q/K sign path、gamma、PWL/PoT constants。報告「except attention」比實作粗略，不能因此推斷所有 attention 參數都不可訓練。正式 Pose logical batch128，P1 physical128×1、P2 64×2、P3 32×4；目前使用 native joint Pose16 的獨立適應不是原 P1–P3 配方的完整重現。

第 10 頁報告參數數量從 45,580,762 降至 26,529,701（約 -41.8%），COCO overall／person 下降 0.0084／0.0058。這是原架構報告的參數減幅，不是本輪硬體延遲或 storage bytes 的實測。

## 第 11 頁的重要證據

person AP50–95 曲線在 J0 約 0.626，J1 初期降至約 0.600，再逐步回到約 0.620。數值是讀圖約值；不是從原 CSV 精確重算。它證明原成功流程也存在暫時退化，因此最終精度門檻不能直接當作首 epoch 立即停止門檻。

梯度圖：J1 初期 Pose／Detect norm ratio 仍約 4–5，後期下降；cosine 多為弱正或近零，少數負值。幅度失衡和方向衝突是不同問題，不能因比例大就直接上 PCGrad。第 10 頁將 Conflict-aware PCGrad 列為後續可能方向，不是已採用方法。

最下圖標題是 **Batch-normalized training losses**：指每影像 loss 歸一化，不能把中文「BN 後訓練損失」解讀成已證明 BatchNorm 層校準效果。

## 與目前工作比對

1. 完整 Pose 前置：現已補上。head-only 40 epoch 最佳 Pose AP 0.852475；完整 Pose AdamW 低 LR 適應 42 epoch 平台，最佳 0.897997。原独立 Pose 0.912161，仍有差距。不能稱 MuSGD 無用，因原完整 Pose 使用 MuSGD 而且已有實績；當前選 AdamW 只表示此次受控適應有效。
2. 原 final 配置與報告版本不同：[final joint.yaml](<../../../yolo_combine/final/full35/configs/joint.yaml>) 與 [stage_policy.py](<../../../yolo_combine/src/yolo_combine/stage_policy.py>) 目前 J1 patience8、J2 80／17、gate0.08；PDF 為 J1 patience5、J2 40／10、最終門檻0.02。必須分開標示，不能說兩者完全相同。
3. 本輪較嚴格的 COCO 0.005 是額外研究保護，不是 PDF 門檻。舊 runtime 把它每 epoch 立即 safety stop，可能阻止正常適應。下一輪應分成「最終候選仍須通過 COCO 保護」與「訓練中的災難性退化／持續惡化停止」，保存最佳與逐 epoch 結果，不因第一輪暫時下滑就判定失敗。
4. 本輪使用 10% Pose trunk 插值初始化是新增開發診斷，不是報告原方法，不保證優於原 Detect trunk 初始化。原独立兩模型都保留。

## 目前實際梯度證據

聯合 `merge-j1-smoke-v1` 已完成，正式 J1 尚未啟動。兩個 macro 的共享梯度：

| macro | Detect norm | Pose norm | Pose／Detect | cosine |
| --- | ---: | ---: | ---: | ---: |
| 1 | 303.5163 | 2353.3150 | 7.7535 | 0.03343 |
| 2 | 327.1384 | 1706.5215 | 5.2165 | 0.01146 |

來源：[smoke summary](<../../combine/bridge_v1/artifacts/fusion/merge-j1-smoke-v1/summary.json>)；[joint_loss.py](<../../../yolo_combine/src/yolo_combine/joint_loss.py>) 確認統計在 task backward weighting 與 AMP unscale 之後、global clip 之前，因此**已含 0.25 Pose 權重，不能再乘一次 0.25 解讀**。

由 g = 64/(1+w)·(g_detect + w·g_pose) 可知，共同因子不影響比例；固定 checkpoint 下加權 norm ratio 約正比 w。僅兩個 macro 不能代表全資料，先擴充 train-only 梯度校準，不用 validation 搜權重。若要調整固定 Pose weight，可由 w_new = w_old × target_ratio / measured_median_ratio 提出候選，再用不同後續 training batches 確認；此為待驗證推導，不是報告已驗證參數。

## 決策

後續校準已完成：16個macro中位數5.8558，固定Pose weight0.045在後8個training macro確認中位數1.2763；沒有使用validation。真實更新smoke通過後，新`balanced-j1-v1`已啟動，最終與訓練安全門檻正式分離，見[工作紀錄](<../worklogs/2026-09-10-balanced-j1-start.md>)。下段「先暫緩」是閱讀當時決策，現已完成前置條件。

保留已完成模型及 10% 初始化適應；正式 J1 先暫緩，先校準當前共享梯度，並將早期安全停止與最終驗收拆開。AdamW 仍是聯合主線，MuSGD 不因新 PDF 而未經校準直接切換。暫不加入 PCGrad，activation 與方向2 仍待融合驗收。
