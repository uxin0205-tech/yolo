# 原生 5-epoch、LR×0.25 與 BN 對照結果

更新：2026-09-08。原生 E1–E5 與 LR×0.25 的 E1–E2 已完成並保存；兩者各自觸發安全暫停。追加 BN-only 評分也未通過必要 AP。沒有新 BEST，GPU 已全部停止。HOG、MASF 移位、BinaryQK 與 RepConv 本輪尚未正式訓練。

## 最新結論

LR 變小可減輕 live 退化，head BN 也有影響，但兩者都不是單獨足夠的解法。目前沒有證明唯一根因，更沒有證據把此次退化歸因於凍結中的 MASF／BinaryQK 權重改動。原 BEST 仍是可用且已重驗的起點，不能把退化 checkpoint 升格為後續 parent。

| 同一 E2、BitTrue | EMA joint | Live joint | Live Ball Box |
| --- | ---: | ---: | ---: |
| 原 LR | 0.711067 | 0.705217 | 0.480829 |
| LR×0.25 | 0.710557 | 0.707566 | 0.491465 |
| LR×0.25 live + parent head BN | 未評估 | 未評估 | 0.495485 |

BN-only 只評完整 BBAT5 val，不重跑 COCO，也不能拼湊 joint。它在相同 live 參數上換回 252 個 head BN buffers；介入前後參數 SHA 完全相同。Ball Box 恢復 0.004020，但仍比原 BEST 低 0.011952；因此不直接採用「凍結 BN 就會解決」的結論。

LR 候選於 21:57:10 開始、22:19:20 正常以 exit code 0 保存後安全暫停。E2 EMA 全部通過 parent −0.005，但 live BBAT Box／Ball Box／Ball Pose 分別下降 0.009120／0.015972／0.005987，故 live safety 阻止 E3。它沒有完成 5 epochs，只能與原 LR 的共同 E1／E2 比較，不宣稱同預算勝出。BN probe 評分耗時 3.15 秒，exit code 2 表示必要 AP 未過，不是程式崩潰。

來源：`artifacts/direction1-20260908/native-quarter-lr-control/summary.json`、`artifacts/direction1-20260908/diagnostics/quarter-e2-live-parent-bn/result.json`。兩組的 parent、初始 state hash、資料、macro、seed、EMA age、criterion、scope 和 scheduler horizon 相同；優化器 LR 是訓練變因，額外 live safety 只影響停止時機。

補查原 parent：預設 batch 設定是 64，但 snapshot 的 `loader_state.runtime_batch_plan`、`provenance.runtime_batch_plan` 與 `resolved_config.runtime_overrides` 都記錄实际 physical32，與本輪一致；沒有 batch 不一致的證據。Detect／Pose mosaic 都為 0，fliplr 分別 0.5／0，也與原正式 loader 相同。

## 結果：目前沒有精度恢復證據

以下為同一固定驗證集的 BitTrue mAP50–95，單位為 0–1，不是百分比。

| 狀態 | Joint | Ball Box | BBAT Pose |
| --- | ---: | ---: | ---: |
| 原 BEST（PSEL，EMA） | 0.711175 | 0.507437 | 0.903717 |
| E1 EMA | 0.711226 | 0.506927 | 0.904019 |
| E2 EMA | 0.711067 | 0.506417 | 0.903831 |
| E3 EMA | 0.711019 | 0.505924 | 0.903329 |
| E4 EMA | 0.710052 | 0.503708 | 0.901613 |
| E5 EMA | 0.710116 | 0.502376 | 0.902266 |
| E5 live | 0.705387 | 0.494349 | 0.893315 |

E5 EMA Ball Box delta = −0.0050611186，超過预先設定的 −0.005 暫停線，超出幅度只有 0.0000611186。照既定規則暫停，不事後放寬；這不是統計顯著性的判定。E1 是 run-local best，但比原 BEST 的 joint 只高 0.0000513，不升格新 BEST。

| 必要 AP | 原 BEST | E5 EMA | E5 live |
| --- | ---: | ---: | ---: |
| COCO Box | 0.498022 | 0.498065 | 0.496644 |
| Person Box | 0.620381 | 0.620505 | 0.619674 |
| BBAT Box | 0.630036 | 0.627479 | 0.623988 |
| BBAT Pose | 0.903717 | 0.902266 | 0.893315 |
| Ball Box | 0.507437 | 0.502376 | 0.494349 |
| Bat Box | 0.752634 | 0.752583 | 0.753626 |
| Ball Pose | 0.859909 | 0.856320 | 0.844766 |
| Bat Pose | 0.947526 | 0.948212 | 0.941863 |

Live joint 依 E1–E5 為 0.707199／0.705217／0.705987／0.702684／0.705387。有波動但全部低於原 BEST。EMA age 延續可以平滑評分，不能據此宣稱 live 已學得更好。

## 實驗可追溯性

原 BEST：`/home/uxin/yolo/yolo_combine/final/full35/weights/combined/inference/best_joint.pt`；SHA-256 `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`，保持不變。

結果：`artifacts/direction1-20260908/native-parent-ema-control-adopted/summary.json`。E1 明示採用已完成 EMA 診斷的 continued full snapshot，SHA-256 `44d78885d839fa7e2a8626b9ee0785ef7b530a5d09c05678fc973102562a395d`；E2–E5 在新程序訓練。不是宣稱前一個未保存的失敗 run 已逐 tensor 證明等同。

20:59:05 啟動，21:42:48 由獨立監測器觀察訓練程序退出；因中途對話中斷，原 supervisor 已退出，無法取得非 child 的退出代碼，不能寫成已知 exit code 0。最終 `summary.status=paused_for_analysis`、完整 epoch checkpoint 與退出事件均存在，GPU 回到 443 MiB／0% 使用率。

原生程式的 0.08 gate 是對獨立模型基準；本輪 0.005 safety gate 是對同一 PSEL。兩者不同，不能把 0.08 gate 通過寫成精度恢復。

## 原本與這次實際修改的架構

```text
layer16：raw P3 → shared MASF → p3_shared ─┬─> layer17 → P4 → layer20 → P5
                                         ├─> Detect([p3_shared,P4,P5])
                                         └─> Pose  ([p3_shared,P4,P5])

本次 native 對照：圖不變，只更新 Neck／Detect head／Pose head。
Backbone、attention、MASF 與 shared BN 統計凍結。
HOG 沒有啟用；沒有直接搬動已訓練的 MASF。
```

後續 HOG 候選若恢復啟動，只增加訓練期分支，不改目前 Detect／Pose 的輸入圖：

```text
layer16：raw P3 ──┬─> shared MASF → 原本 P3/P4/P5 → Detect／Pose
                 └─> HOG 1×1 head → HOG loss（僅訓練，部署移除）
```

Detect-only MASF fork、RepConv17 仍各自獨立比較，不與 LR／HOG 首次實驗混在一起。

## CPU 診斷：已確認與未確認

執行 `scripts/analyze_native_control.py`，只讀取五個 epoch 的完整快照與既有 macro 日誌，輸出 `artifacts/direction1-20260908/native5-cpu-analysis.json`。沒有新推論或改動資料。

已確認：五個 epoch 的 live／EMA 固定狀態皆零變更，故不是凍結 scope 失效。E5 相對原 BEST 的 group L2 變動如下：

| 可訓練部分 | 相對參數變動 |
| --- | ---: |
| Neck | 0.582% |
| Detect one-to-one head | 0.460% |
| Pose one-to-one head | 1.760% |

Pose head 的相對變動較大，但不同 group 的原始尺度也不同；這只是定位優先順序，不直接證明 LR 過大。Head BN 統計也漂移，Pose one-to-one 約 17.23%；它並非參數變動的同一尺度。先前換回 parent BN 仍使 Ball Box 下降 0.008684，因此「只換回 BN 就解決」已有反證；舊 probe 不能代替本次 E5 的因果實驗。

24 筆既有 shared-gradient 樣本中，負 cosine 比例 62.5%，但中位数只有 −0.012162，範圍 −0.057529 至 +0.096956；E4 live 最差時，該 epoch 的 cosine 中位數反而為正。負向投影幅度的整體中位數約 1.22%，低於既定 2% 候選條件，且樣本稀疏。暫不以「強烈 task conflict」為主因，也不直接加入投影。

所有 macro 的全域裁剪前 norm 都超過 10。日誌名 `clipped_gradient_norm` 實際是 `clip_grad_norm_` 回傳的裁剪前值，不是裁剪失效；E2–E5 AMP retries 都為 0。不能只因裁剪頻繁就認定它造成 AP 下降。

## 為何先驗證 LR，而不是只調 Pose loss weight

Fresh AdamW 第一步在忽略 epsilon／weight decay 時，偏差修正後 `m̂=g`、`v̂=g²`，故 `Δθ≈−η·sign(g)`。把同一梯度乘上一個正的常數，分子與分母大致一起縮放，不等於把該 head 的更新也按比例縮小。此推導只精確對應首步／固定縮放等條件；不同 minibatch、clip 與後續 moments 會改變實際路徑。

因此下一個必要單變因是直接縮小既有 group LR，不先同時改 loss weight、BN、EMA 或加入 HOG。這仍需 AP 驗證；目前沒有證據證明 fresh optimizer moments 重設就是唯一主因。

## 已執行候選：base LR × 0.25（E2 安全暫停）

| 設定 | 已完成對照 | LR 候選 |
| --- | --- | --- |
| 起點 | 原 BEST inference EMA | 同一原 BEST，重新建立，不接退化 E5 |
| Neck LR | 1e-5 | 2.5e-6 |
| Detect／Pose head LR | 2.5e-5 | 6.25e-6 |
| Optimizer | Fresh AdamW | 不變 |
| AdamW betas／weight decay | (0.948,0.999)／0.00027 | 不變 |
| Warmup／horizon | 1／10 epochs | 不變 |
| 上限 | 5 epochs | 5 epochs，live 或 EMA 退化即保存後停止 |
| Physical／logical Detect batch | 32／128 | 不變 |
| Pose batch／task weights | 16／Detect 1、Pose 0.25 | 不變 |
| BN、EMA、架構、資料 | 原設定 | 不變 |

新增 live safety 不改梯度，只決定是否繼續花費預算；若提前停止，只比較共同完成的 epoch，不把不同長度當成同預算勝負。Selector 仍只看 EMA，live 不混入 BEST 分數。

實作 `--base-lr-scale 0.25 --pause-on-live-regression`，預設舊行為不變。相關 4 項 CPU 測試通過（1.41 秒，2 項新增），包含完整 4630-step scheduler 的比例、scope 不變與先存後停止。累計 70 項唯一 CPU tests；CLI help 通過。CPU 測試不代表 LR 候選精度通過。

本次先停止追加原生訓練，不再猜更多 LR。原生對照沒有提升，不代表 HOG 無效，也不要求 control 必須先勝過原 BEST；但 HOG 尚未產生實驗結果。後續若進入 HOG，應從已重驗的原 PSEL 開獨立對照，先做 train-only μ 校準，不能接目前退化的 E2／E5，也不能把未驗收的 LR×0.25 自動視為新 baseline。第二輪、person-only、MuSGD 仍不納入。
