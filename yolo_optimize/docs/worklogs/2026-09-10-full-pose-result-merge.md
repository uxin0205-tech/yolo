# 2026-09-10：完整 Pose 結果與融合初始化適應

## 完整 Pose 結果

`full-pose-gentle-v1` 完成 42 epoch／15666 macro，因六項 AP 平均分數 patience 12 正常平台停止，沒有再次觸發精度 safety stop。最佳 Pose checkpoint 是該 run 第 30 epoch（zero-based 29），SHA256 `68d6be769751aaf7ce15d552f6b8d7f1e6f4bb5e463a530bbc976e655959f117`。

| 指標 | head-only 最佳 | 完整 Pose 最佳 | 原独立 Pose |
| --- | ---: | ---: | ---: |
| box AP | 0.567273450 | 0.612079796 | 0.630963613 |
| pose AP | 0.852474613 | 0.897997109 | 0.912160548 |
| ball box AP | 0.475850818 | 0.486823217 | 0.510746900 |
| ball pose AP | 0.829325532 | 0.862116389 | 0.876263080 |
| bat box AP | 0.658696082 | 0.737336374 | 0.751180326 |
| bat pose AP | 0.875623694 | 0.933877828 | 0.948058016 |

降低 LR 的同起點試驗改善了原首 epoch 退化，並取得顯著適應收益；不能證明所有下降都只有 LR 原因。仍未完全達到原独立 Pose，不切換未驗證 MuSGD。

## 共享相容性驗證

完整 Pose trunk 直接接凍結 Detect head，COCO overall／person 僅 0.374565763／0.474528572。原 Detect checkpoint 未動，其 AP 仍為 0.508211955／0.627664127。因此獨立 Pose 成功不等於融合成功。

執行 `check_assembly.py`，對完整 Pose head 固定，將共享 trunk 設為原 Detect 狀態加 r 倍 Pose 狀態差分。保持固定 attention／BN 等相同狀態，完整 COCO5000／BBAT683 BitTrue 驗證：

| Pose trunk 比例 r | COCO overall | person | BBAT box | BBAT pose | ball box | ball pose | bat box | bat pose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | .50821196 | .62766413 | .51547872 | .82923396 | .39868834 | .79846965 | .63226911 | .85999827 |
| .1 | .50739824 | .62596915 | .53471286 | .84745105 | .42243528 | .82123333 | .64699044 | .87366877 |
| .25 | .50107617 | .61966462 | .55925208 | .86609786 | .44687836 | .84471591 | .67162581 | .88747981 |

單純搬 head 明顯丟失完整 Pose 適應收益；25% 超過原 COCO 0.005 保護。10% 未超過但 Pose 仍需重新適應，只作融合初始化，不是升格新模型。此選擇使用 validation，屬開發決策，不是獨立 test 泛化證據；後續不無限掃比例。

## 接續 J0

新增 `merge_pose.py`，鎖定來源 SHA，建立 10% Pose trunk + 完整 Pose E30 head + 原 P3 bridge Detect head。原兩個独立 checkpoint 保留，不直接混成部署輸出。固定共享 trunk／Detect／MASF，只更新 Pose head，按 combine 做融合 J0，最多 20 epoch、AdamW head LR 2e-4、warmup 1、batch16、patience10（六項平均分數、min_delta .0001）。

J0 的 COCO 不變檢查使用該已驗證初始化 .50739824／.62596915；原独立 Pose 六項基準不變。後續 joint 的 COCO 保護仍必須對原 Detect .508211955／.627664127，不可因初始化較低而累積放寬。

真實兩 batch smoke 通過，Pose 參數更新、live／EMA 非 Pose 全部 state 固定、硬體契約通過，見 `artifacts/fusion/merge-j0-smoke-v1/summary.json`。重用 smoke 的 loss horizon 為 40，僅做 epoch0 的更新契約驗證，不作正式 20 epoch schedule 等價宣稱；正式 trainer 依新 stage horizon20 執行。

正式 `merge-j0-v1` 已啟動，事件檔 `combine/artifacts/logs/merge-j0-v1.events.jsonl`，600 秒 blocking monitor。先完成此共享初始化適應與驗收，再決定 J1；activation／方向2 未啟動。

## 困難與未解

主要困難是獨立 Pose 與 Detect 所需特徵不同，直接搬 head 或 trunk 都無法同時保留兩者指標。10% 插值只減輕初始衝突，未證明能收斂至可接受 joint；若後續無法改善，需以已保存獨立模型和對照判斷，不宣稱融合成功。沒有刪除、覆寫、commit 或 push。
