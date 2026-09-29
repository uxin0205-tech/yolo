# P2 MASF 結果、無 MASF 融合與 Pose 專項接續

## 變更與原因

P2 MASF 最後 5 epoch 配對試驗已完成，未達同 epoch overall／person 均不退步且至少一項提升 0.001 的門檻。因此依使用者先前授權，融合主線不採用 MASF；全部舊模型與結果保留。

| Epoch | Control overall | Control person | P2 overall | P2 person |
| --- | --- | --- | --- | --- |
| 1 | 0.508197 | 0.627790 | 0.508221 | 0.627766 |
| 2 | 0.508330 | 0.627858 | 0.508239 | 0.627913 |
| 3 | 0.508318 | 0.627800 | 0.508255 | 0.627880 |
| 4 | 0.508394 | 0.627617 | 0.508225 | 0.627912 |
| 5 | 0.508420 | 0.627698 | 0.508271 | 0.627797 |

融合選 control E2 EMA：在 overall／person 皆不低於上一個已驗收 parent 的完成 epoch 中，取 overall 最大者。不是單獨挑最高 overall 而忽略 person。來源與 SHA256 見 `combine/artifacts/selected-source-v1.json`。

`combine/local_source.py` 使用新 Detect trunk／head 與原完整訓練 Pose head；Pose 模板的 trunk 僅於記憶體替換，不平均舊新 trunk、不修改原 Pose checkpoint。保留原 graph audit、factory、資料與版本檢查。Softmax PWL 固定 [-10, 0]、20 段，表值不是本輪可訓練參數；未宣稱完全無除法的硬體部署。

## 驗證結果

CPU 組裝完整性、Detect 初始化輸出完全相同、Pose head 權重完全保留、兩任務共用一次 stem、Float／BitTrue materialization 均通過，見 `combine/artifacts/no-masf-assembly-v1.json`。

完整 COCO val 5000 張與 canonical BBAT5 Pose val 683 張已重新驗證；BitTrue COCO overall 0.5083300413、person 0.6278584228，與選定來源差值均為 0。原独立 Pose 與共享初始模型分別驗證，未用退化後的初始值取代正式比較基準。

| BitTrue 指標 | 原獨立模型 | 新共享初始模型 |
| --- | --- | --- |
| BBAT box AP | 0.630964 | 0.346039 |
| BBAT keypoint AP | 0.912161 | 0.584586 |
| ball box AP | 0.510747 | 0.287191 |
| bat box AP | 0.751180 | 0.404886 |
| ball keypoint AP | 0.876263 | 0.639635 |
| bat keypoint AP | 0.948058 | 0.529536 |

結論：Detect 轉接保持精度，但 Pose 對新特徵分布明顯不適應，尚不能稱融合成功。完整數據見 `combine/artifacts/baseline-v1/summary.json`。

## J0 設定與保護

沿用原 combine 的 J0：8 epoch，AdamW，Pose head LR 2e-4，beta 0.948／0.999，weight decay 0.00027，warmup 1 epoch，AMP、gradient clip 10，Pose physical batch 16，640 輸入。完整 canonical 訓練集 5964 張，不另切分。COCO 不參與 J0 backward；128 logical／32 physical 是後續 Detect 批次設定，不宣稱 J0 在訓練 COCO。

固定共享 trunk、Detect、共享 BN 統計與 attention 硬體參數；本地 Pose-only EMA 只更新 Pose head 浮點狀態，避免固定 Detect 的加權浮點漂移。沿用 combine 八項評分與 checkpoint 選擇，額外要求 J0 完整 COCO overall／person 變動不超過 1e-8，若違反即報错，不把 -0.08 Pose gate 解讀成允許 COCO 大幅退化。

`smoke_j0.py` 先從完整 loader 取前兩個 batch 做真實原生 loss 更新，檢查 live／EMA 固定狀態、非零 Pose 更新與記憶體；這不是抽樣精度實驗。正式入口 `train_j0.py` 僅在 smoke 通過後可執行，結束後先分析，不自動解凍 J1/J2。

GPU smoke 已通過：Pose 參數更新 L2=0.07138594，live／EMA 的固定 trunk／Detect 完全相同，峰值已配置 CUDA 記憶體 2,980,215,296 bytes。正式 J0 於 UTC 2026-09-10 03:01:19 啟動；run `combine/artifacts/fusion/j0-no-masf-v1`，監測採 child.wait(timeout=600)，正常不讀 log 或反覆查 GPU。完成狀態以該 run summary 與事件檔為準，不假稱具備未實作的 stalled 心跳偵測。

## 使用者新增的 Pose MASF 分支

使用者本次允許沿用 combine 的配置與指標，並要求在 Pose 資料集檢查 MASF 是否提升 ball／bat。新增獨立研究規格 `combine/pose-masf/README.md`；COCO MASF 的失敗結論不能直接外推 Pose。此分支仍需成對訓練，不能只比較來源不同的 checkpoint 或將 box AP 當 keypoint AP。目前尚未執行此成對訓練。

## 困難、解法與風險

使用者最新停止點：Pose MASF 比較完成後，與 COCO 結果一起呈現並停止，由使用者決定是否採用 MASF。已排 `run_pose_pair.py`：pidfd 等待现有 control J0，不重跑；之後只執行 Pose P3 MASF 真實更新 smoke 與同起點 8 epoch candidate。CPU alpha=0 完全等價及 Float／BitTrue 還原已通過。保留原生 detach，檢查 alpha／context 實際非零且有限梯度，不把 weight decay 更新當有效梯度。queue 不包含 J1/J2、activation 或方向 2。Pose P3-only 結論與既有 COCO P2-shared 結論必須分開標示。

原來源 layer16 不一致：使用新 Detect trunk 加完整原 Pose head 的明確適配，再重新測量原／新 Pose，不繞過 audit。設定載入要求 `full35/configs/` 層級：CPU 最小重現失敗後修正本地路徑，原 preflight 已通過。既有檔案更新遇 bwrap 限制，使用經審核的明示 apply_patch；未改兄弟專案。

TensorBoard 未安裝，沿用 auto 的 JSONL／CSV／PNG，不影響訓練。Pose 精度尚未恢復；GPU smoke／正式 J0 結果以後續紀錄與實際 summary 為準。尚未開始 activation 或方向 2。無刪除、覆寫舊結果、commit 或 push。
