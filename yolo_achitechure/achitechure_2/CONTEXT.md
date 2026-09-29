# YOLO26m 融合模型架構簡化

本脈絡定義 architecture_2 如何承接已選定的 Detect–Pose 融合模型，並在不重新選擇融合方法的前提下比較 C3k2 單因子簡化。

## Language

**融合 Winner（Fusion Winner）**:
由 `yolo_combine` 完成實驗後正式選出的單一模型；可能是共享、路由或部分共享形式。
_Avoid_: architecture_1 parent、來源 A/B、暫定模型

**Handoff Revision**:
一組不可變的 winner checkpoint、builder、模型契約、訓練配方、資料與雜湊證據。
_Avoid_: latest checkpoint、可覆寫 handoff

**C0-Handoff**:
未經 architecture_2 訓練或架構修改的 Fusion Winner，必須與上游輸出逐張量等價。
_Avoid_: baseline retrain、C0-Control

**C0-Control**:
從 C0-Handoff 開始，使用與 C1–C3 完全相同恢復預算訓練的公平比較控制組。
_Avoid_: untouched winner、C0-Handoff

**候選因子（Candidate Factor）**:
C1、C2 或 C3 所代表的一個且僅一個 C3k2 變更；第一輪禁止組合。
_Avoid_: fusion method、quantization、multi-factor candidate

**候選區域（Candidate Region）**:
由 handoff 宣告且經 graph audit 驗證、可套用候選因子的模型區域。
_Avoid_: hard-coded layer number、entire model

**解析後候選（Resolved Candidate）**:
候選因子與某個 Candidate Region 的唯一組合；共享 winner 可沿用 C1，路由或部分共享 winner 使用 S-/D-/P- 前綴避免混淆。
_Avoid_: combined candidate、automatic C_best

**主任務契約（Main Task Contract）**:
COCO80 Detect 與 BBAT5 ball/bat Pose 的共同推論責任；介面可選 detect、pose 或 both。
_Avoid_: BBAT5 2-class Detect main task

**BBAT5 診斷 View**:
由權威 Pose labels 前五欄衍生的 ball/bat Detect view，用來檢查棒球類別，不參與主線 C_best 因果排名。
_Avoid_: COCO80 replacement、second source of labels

**固定20%篩選 View（Architecture Screen 20）**:
從既有 COCO train 與 BBAT5 search-train 產生的不可變、run-specific manifest View；只作
C0–C3初篩，不擁有新資料語意，也不使用正式 val。
_Avoid_: fraction=0.2、new canonical split、formal ranking dataset

**QAT-lite（Q2L）**:
eligible candidate 在 Q1 後使用固定20% View進行的200-step W8A8 fake-quant短恢復；只描述
PTQ gap能否部分恢復，不取代正式 Q2。
_Avoid_: official QAT recipe、Bit-True、deployment INT8、automatic acceptance

**量化資格（Quantization Eligibility）**:
Float 結果完成後，由使用者明確指定某候選是否進入 Q0/Q1/Q2L/Q2；C0 固定具資格。
_Avoid_: automatic threshold、C_best-only quantization
