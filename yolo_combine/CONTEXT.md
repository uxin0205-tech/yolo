# YOLO26 Detect–Pose Fusion

2026-10-29 重新建立第六階段，舊 Stage6 實驗權重、數據、報告及 queue 已全部刪除，沒有封存。現行流程與訓練入口見 [README](README.md)，事前 optimize 來源 SHA 與配方見 [protocol](artifacts/stage6/reference-review.json)。

新起點為 Stage5 Detect 與受保護的已訓 Pose head。先在 Detect trunk 的副本上完成 Pose 適應，再以原 Detect anchor 比較 ρ；不混合兩個互不相關的 trunk。共享 trunk 只算一次，再交給 COCO80 Detect 與 ball／bat Pose head。

主線保留 B2、S1 Q10 與 D1 整數 PWL exp／reciprocal。最終 COCO 四項 AP 各下降不得超過 0.005、BBAT 六項各下降不得超過 0.02。訓練安全線與最終驗收分開，精度未達標不能宣稱完成主線採用。

以下詞彙只解釋 optimize 仍依賴的歷史 Full35 程式，不是本輪 parent 或驗收門檻。

This context defines the model forms and acceptance language used to combine person detection with ball/bat pose while preserving a trustworthy independent baseline.

## Language

**Authoritative Full35 Source**:
The immutable accepted Full35-A2 bundle and checkpoint from which fusion experiments are initialized.
_Avoid_: F1 checkpoint, training output, editable source

**Independent Task Pair (F0.5)**:
A Detect model and a Pose model that each own a complete feature extractor and serve as the accuracy baseline and rollback path.
_Avoid_: shared model, fused trunk

**Shared Feature Trunk**:
One live set of feature-extractor weights used by both tasks; it is not two models that merely have the same architecture.
_Avoid_: same backbone architecture, dual weight banks

**Shared Dual-Head Model (F1)**:
One Shared Feature Trunk feeding an independent Detect head and an independent ball/bat Pose head.
_Avoid_: weight averaging, pruning, unified head

**Initialization Equivalence**:
The verified property that a newly constructed F1 produces exactly the same task tensors as its corresponding Independent Task Pair while both task trunks still contain identical tensors.
It ends when task-specific trunk or neck training makes those independent weights diverge.
_Avoid_: permanent equivalence, accuracy guarantee

**Accuracy Gate**:
The acceptance rule that no tracked F1 task metric may fall more than 0.08 mAP50-95 below its corresponding independent baseline.
_Avoid_: average-only gate

**Canonical BBAT5 v1**:
The immutable paired ball/bat Pose and Detect dataset used by every new baseline and fusion experiment; both task views share one leakage-safe assignment.
_Avoid_: BBT5 basic split, raw Pose dataset, latest dataset

**Runtime Dataset View**:
A rebuildable cache-isolation view whose images and labels reference Canonical BBAT5 v1 without owning dataset semantics.
_Avoid_: local dataset version, copied training source

**Legacy Basic Split**:
The historical Roboflow train/valid assignment with source-group overlap; it explains retained checkpoints but cannot be selected for a new run.
_Avoid_: formal split, canonical dataset

**Isolated Architecture Workspace**:
One Full35 or Partial75 experiment folder that locks its architecture from its own
location and owns every mutable dataset view, cache, report, checkpoint, and run output.
It may reference the same immutable source and reuse stateless implementation modules.
_Avoid_: shared artifacts root, caller-selected architecture, cross-workspace checkpoint
output
