# Activation 優先與雙資料集 KD 接續

## 使用者指示與變更

依最新指示，改為先 activation，再依方向 2 進行 KD；撤回前一輪「先繼續共享 affine／Neck 恢復」的執行優先序。保留全部 BBAT 差距與未驗收狀態，不因改階段而抹除。
新建 `activation/bridge_v1/`，以 optimize 的 Pose head recovery E5 為同起點，固定 SHA256，safe weights_only 載入。採用原 activation recipe 的 qSiLU-PQ 優先、Hardswish／PolyShift 評估，不把舊模型的結果當新模型已通過。

## 研究技能與依據

使用 research 技能核對本機一手程式與既有規格，依使用者規則不啟動子代理。依據：`yolo_activation/training/full35/activation-recipe.yaml`、`activation_lab/activations.py`、`optimizations/round2-innovation/region-ranking-full-spec.md`。研究與雙資料集接續契約記在新分支 README。

## 驗證流程

先 CPU strict weights／四臂參數相同／雙 head finite／完整 activation reference 盤點；再全量 COCO 5,000／BBAT683，四臂 Float／Bit-True，SiLU 必須重現起點。短恢復與 KD 不在未確認零樣本／teacher 條件下盲啟動。
KD 必須 task-routed，COCO 用 Detect 原生監督，BBAT5 用 Pose 原生監督；原 R2-REGION 規格已包含此情境，但 teacher 與有效梯度仍待 activation 後查核。不能用 Float-BinaryQK 冒充 FP-QK teacher。

## 結果、困難與未解事項

程式與接續文件已新增，CPU／GPU 結果待執行。困難：無。風險：當前融合仍有五項 BBAT gate 未過；activation 相對 gate 與原融合 gate 必須分開列出；KD teacher 尚未驗證。GPU 一旦啟動即以 600 秒 blocking monitor 監測。

CPU 四臂 preflight 已通過：全部參數一致、雙 head 輸出有限，記錄在 `activation/bridge_v1/artifacts/preflight-v1.json`。全量 GPU zero-shot 已啟動，monitor 名稱 `bridge-activation-zero-shot-v1`，PID 708876，UTC 2026-09-10T19:36:37；目前未啟動短訓練或 KD。

## 零樣本完成與配對短訓準備

四臂 Float／Bit-True 全量驗證完成，SiLU 八項 AP 精確重現。qSiLU 最大下降為 ball pose -0.008974，全部在 activation 0.015 範圍；Hardswish COCO -0.102849／bat box -0.157533，PolyShift ball box -0.023975，先不擴大短訓候選。只準備同起點 SiLU／qSiLU 的 10 epoch 配對。
新增 `activation/bridge_v1/train.py`，沿用原 activation short recovery：seed1、patience0、warmup1、J3 LR×0.1、Pose weight0.25、shared BN affine 可訓練而 running 固定、FP32 bbox IoU 避免 AMP 比例運算問題。相對 gate0.015 與原融合 gate 分開，不把本階段 best_joint 誤稱原融合驗收。
共用 smoke 改為遵循 config.shared_bn_affine_trainable，先前 false 分支行為不變；新短訓必須先以同設定完成兩個真實 macro 與固定硬體／MASF BN 檢查。

## 實際修復與啟動

初次 SiLU smoke 在 GPU 更新前發現 BN affine 是 config property 而非 dataclass 欄位；改用明確 subclass property，設定不變，重啟後通過。qSiLU physical32 反向 OOM，只將兩臂 physical microbatch 一同降為16，logical Detect128、每macro256 Detect＋16 Pose不變；新 smoke 分支保留舊結果。兩臂各兩個真實 macro 均通過，峰值 allocated：SiLU10,809,923,072 bytes、qSiLU19,154,190,336 bytes。MASF梯度／更新、固定 live／EMA 硬體與 MASF BN 已檢查。
串列程式原命名 queue.py 遮蔽 Python queue 標準庫，CPU預檢即發現，改名 run_pair.py；實際 config 序列化及語法檢查隨後通過，未因此重跑 GPU 驗證。
`run_pair.py` 已於 UTC 2026-09-10T19:50:51 啟動 SiLU正式短訓，PID717115；完成10epoch且正常summary才自動接qSiLU10epoch。正常只child.wait(timeout=600)，JOB_DONE／ERROR／ALL_DONE事件才介入。KD仍需配對完成後的teacher與梯度查核，未提前启动。

SiLU 已於 UTC 2026-09-10T22:20:43 正常完成10epoch／4,630macro。最佳activation-relative joint為E9：COCO0.504309／person0.626783／BBAT box0.612832／pose0.889444／ball box0.493900／ball pose0.854713／bat box0.731765／bat pose0.924174。這是SiLU控制，不是qSiLU收益；原融合gate仍需另報。Queue確認summary後自動接qSiLU，PID780558；未中斷、未重跑SiLU。持續600秒blocking monitor。
