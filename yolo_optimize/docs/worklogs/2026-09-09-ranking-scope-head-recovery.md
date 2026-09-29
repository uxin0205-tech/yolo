# 2026-09-09：排序拆解、共享特徵責任定位與 B-HEAD 接續

## 新證據與推導

先前 E5 Ball Box AP 回退 −0.0050611186，固定 conf=0.25 的漏球卻未增加。本次新增 `analyze_pose_localization.py`，CPU 重播已保存 683 張預測：IoU50／75 與官方計數全部相符；Ball 在 IoU80／85／90 的 TP 分別 187／110／77，兩個模型相同。不支持高 IoU 命中數大幅崩壞的假說。

原記錄只保留 conf≥0.25 的完整座標，因此不足以重建全 AP。明確補做一次雙模型全量驗證，使用更新的 `audit_pose_errors.py --output artifacts/direction1-20260909/pose-ranking-audit` 保存全 conf、class 與官方十門檻 TP flags，以及 runtime 影像入口路徑。沒有覆寫前次產物。

`analyze_pose_ranking.py` 用相同官方 `ap_per_class` 重建 4 個 class/task AP，與正式 AP50–95 全部相等（容差 1e-12）。令 A 為實際 AP，U 為相同 TP 候選、但用標註把 TP 排最前面的 oracle AP，則 `ΔA = ΔU − Δ(U−A)`。Ball Box：ΔA=−0.0050611、ΔU=+0.001、排序差增加0.0060611；Ball Pose：ΔA=−0.0035890、ΔU=0。這是數值分解，不是訓練唯一根因；oracle 用到 GT，絕不能作部署、teacher 或宣稱可得增準。

## 免訓練雙向 scope 診斷

`probe_pose_classifier.py` 對 classifier、完整 Pose head 分別雙向交換；每個組合完整驗證 canonical BBAT5 val683。每次核對所有 state keys，僅明列 seam 可改，其他 tensor 完全不變；原 checkpoint 不修改。分支含自身 BN buffers，不是 BN／weight 單因子拆解。

| 記憶體中的組合 | Ball Box AP | Ball Pose AP |
| --- | ---: | ---: |
| 原 BEST | 0.507437 | 0.859909 |
| Native E5 EMA | 0.502376 | 0.856320 |
| E5 + parent classifier | 0.502710 | 0.855018 |
| Parent + E5 classifier | 0.506125 | 0.860288 |
| E5 + parent 完整 Pose head | 0.503199 | 0.855018 |
| Parent + E5 完整 Pose head | 0.506155 | 0.860288 |

分類分支單獨還原不足以恢復精度；保留 parent 共享特徵比保留 E5 共享特徵的組合更接近 parent。這支持先限制 Neck 更新，不等於證明 head-only 訓練一定成功，也不是把混接模型升格部署。先前早期 E1 params-only／BN probes 與本次 E5 雙向完整 branch swaps 不同，未重複當成新證據。

產物：`artifacts/direction1-20260909/pose-error-audit/localization-analysis.json`、`pose-ranking-audit/ranking-analysis.json`、`pose-classifier-probe/summary.json`、`pose-pose_head-probe-v2/summary.json`。

## 依原計畫 B 分支接續的設定

新增獨立 `heads` variant，從原 BEST inference EMA 與配對 full-resume criterion 開始，fresh AdamW；不接 E5、不加 HOG、不搬 MASF、不開 STE，也不切 MuSGD。唯一學習變因是 Neck LR=0、requires_grad=false，共享 BN 仍 eval；Detect/Pose heads 各 2.5e-5。原 B-HEAD 提案 5e-5 在此不照抄，為與已完成 native 對照維持同 LR，明示採 2.5e-5，不同時改 scope 與 LR。

上限 5 epochs；warmup1、cosine horizon10、末因子0.5；Detect physical32、logical128、每 macro256 張，Pose16；463 macros/epoch；AdamW betas=(0.948,0.999)、wd=0.00027、clip10；parent EMA age26597；每 epoch 保存完整快照並驗證 EMA／live，仍用原每項 AP −0.005 安全線與原驗收門檻。無 shared 可訓練參數時 shared cosine 不適用，明示關閉該統計，不偽造零衝突值；head 梯度／更新檢查保留。

## 實作與驗證

`recovery_scope` 同時用於 builder 與每次 `training_mode()`，避免 optimizer 起初凍結、下一 epoch 卻重新解凍。CLI 與 quiet supervisor 增加 heads variant。`verify_recovery_safety.py` 增加真正 head 更新與非 head 不可訓練檢查。

GPU smoke `heads-only-gpu-smoke`：exit0，1 optimizer step、1 EMA update，585 個固定 state 在 live／EMA 逐值不變；Detect／Pose gradients 存在、shared 不存在；AMP 重試4次後成功，更新丟棄。此 smoke 使用 fresh EMA age1 作單步正確性，不冒稱正式 parent-age 訓練。

CPU scope＋既有 trainer 合約：24 passed，1.61 秒。新測試涵蓋反覆 train mode、非 head group LR0／不可訓練、optimizer step 後 Neck 不變。正式完整訓練結果尚未產生，不宣稱 AP 通過。

## 困難與解法

新 CPU 分析腳本一處字典括號語法錯誤，修正後成功。新測試最初誤以為 frozen role 不在 optimizer groups；原 builder 為 stage 切換有意保留，實際判準改成 LR0＋requires_grad=false＋step 無變動。ToyBase 原先沒有 heads，補上 head seam，避免空集合斷言。第一版測試失敗訊息回傳時，同一 orchestration call 已啟動 smoke；沒有啟動正式訓練，smoke 保留完整原 fixed-state guard，正常完成後才修正並重新通過 CPU 測試。

額外發現正式 shared 統計遇空 shared group 會拒絕，已於 heads 專用 adapter 關閉不適用統計。其他困難：無。未使用子代理，未改資料 split／labels、來源、部署 export；未 commit／push／刪除。新 GPU 工作持續最多 600 秒 blocking wait，正常不讀 log、不查 GPU。
