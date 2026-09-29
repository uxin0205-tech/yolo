# 2026-08-30：完成Full35 quantization V0–V3 CPU實作與V4+非GPU準備

## 目標與授權邊界

依使用者確認，整合先前grilling、activation預選、歷史PTQ與新增W5／W7需求後，實作V0–V3；V4之後只可準備非GPU設定，不得啟動GPU、mAP validation、QAT、正式訓練或export。

本輪所有pytest與完整模型分析都以`CUDA_VISIBLE_DEVICES=-1`執行。沒有建立CUDA context、沒有optimizer、backward或epoch，也沒有更動`yolo_combine/final`、`yolo_activation`、site-packages或BBAT5 assignment。

## 變更內容與原因

### V0：血緣整合

- 保留publication commit `bf56249b`（subject `5090 Profile 0829`）與`PUBLICATION_MANIFEST.yaml`。
- 將遠端比本機多出的activation實驗身分、190／124／66 graph邊界、可／不可主張表、驗證與可重建連結整合回`2026-08-29-activation-preselection-report.md`。
- 用SHA-256確認本機報告與publication commit逐位元相同：`89c968ecc47f32f59da0d325f617549de3871f07d20038ce0df5f228bddf0709`。
- 稽核publication manifest：21個檔案中15個仍逐位元相同、6個README／索引因後續內容演進、0個遺失。
- 保留本機歷史`backbone_early`三parent × W8／W4六格PTQ與其原始hash，另以`v0-lineage-reconciliation-v1.yaml`明記它使用舊151-layer catalog與小型probe，不得升格成V2正式基線。
- 沒有對目前ahead 1／behind 52且大量untracked的根工作樹做checkout、reset、merge、commit或push，避免覆蓋任一側證據。

### V1：catalog與dual weight views

- 以測試先重現`pose_head.one2one_cv4_sigma.[0-2]`被錯列為deployment的問題，再改成training-only `pose_sigma`。原因是官方inference fuse會移除sigma支路。
- 修正後catalog為251個Conv／Linear：deployment 148、training-only 99、Binary Q/K protected 4。
- 新增`Full35WeightViewAdapter.build(model)`：deep-copy CPU FP32 master，另建立官方BN-folded deployment，不修改caller模型。
- 官方fused graph在部分`cv2=None`後原始`contract()`不可直接使用，因此只為稽核重建等價inference contract；沒有修改上游final bundle。
- forward parity只允許one-to-many tensor paths消失；one-to-one結構、finite、NRMSE與相對reference peak誤差必須通過。
- 三個parent皆完成179→0 BN fold、148 deployment path parity與inference contract parity。

### V2：靜態格式分析

- 新增`WeightAnalysisPlan`、`WeightFormatAnalyzer`與per-layer measurement資料結構。
- uniform主矩陣實作W8／W7／W6／W5／W4，支援per-tensor、per-output-channel、group32、group64與max／MSE grid scale。
- 新增Fixed SD4：4-bit近對數codebook，支援per-tensor／per-channel與max／MSE grid scale。
- 新增PDF原式Paper-TWN static proxy：`delta=0.7×mean(abs(W))`，`alpha=mean(abs(selected W))`，三值`{-alpha,0,+alpha}`。
- LS-SD4、Channel-TWN、TTQ沒有實作成已完成結果；它們明確留在後續matched QAT。
- 加入unknown granularity fail-closed測試。Red階段證明`group16`原本不會raise，Green階段加入白名單驗證，避免未知名稱被`else`分支靜默當成group64。
- 加入unknown scale method fail-closed測試。Red階段證明Fixed SD4的`median`原本不會raise，Green階段把uniform／SD4都限制為`max|mse`，避免錯誤標籤靜默退回max。
- 新增`prepare_v1_v3.py` CPU入口。每次先重建固定diagnostic manifest，再載入指定activation parent、建立dual views、執行static profile並原子寫入JSON。
- 三個parent各完成2,960筆、總計8,880筆：uniform 1,480、Fixed SD4 1,184、Paper-TWN 296；master與deployment各1,480筆。
- 產生三份約3.3 MB完整JSON、三份dual-view manifest、V0–V3 delivery manifest與Fixed SD4 routing候選manifest。

### V3：diagnostic manifest與metric gate

- 固定seed `20260830`、每task calibration 32與probe 64。
- COCO train／val選樣都強制包含person class 0；calibration 16張person、probe 42張person。
- BBAT5只從canonical formal train／val清單挑選，一個`.rf.` source group只取一個canonical exemplar；calibration 32 group、probe 64 group，train／val overlap為0，ball／bat與pose rows皆有coverage。
- 每個image／label保留絕對路徑與SHA-256；未改split、labels或assignment，未建立30%資料版本。
- 新增八指標`Full35MetricGate`：COCO overall／person、BBAT overall box／pose、ball box／pose、bat box／pose缺一不可。
- 實作total drop `0.04`、W8 incremental `0.01`、QAT sham drift `0.01`與recover floor `0.06`。本輪沒有執行mAP，只驗證gate邏輯。

### V4+非GPU準備

- `v4-plus-prepared-plan-v2.yaml`固定V4為3 activation parents × W8／W7／W6／W5／W4 = 15格。
- V5固定3 parents × 10 regions × 5 bits = 150格。
- 保留successive racing、sentinel audit、三個Pareto角色、QAT matched sham、AdamW control／MuSGD pilot與正式八指標門檻。
- 設定明記`execution_authorized: false`、`gpu_required_steps_must_not_run: true`與`auto_qat: false`。

## 實驗與分析結果

### Dual-view parity

| Parent | Forward NRMSE | Max error/reference peak | 結果 |
|---|---:|---:|---|
| qSiLU＋A8 | 3.390751e-6 | 7.105692e-6 | pass |
| poly_quality＋A8 | 4.142677e-6 | 1.444763e-5 | pass |
| poly_shift＋A8 | 4.893339e-6 | 2.236249e-5 | pass |

### Deployment全網靜態結果

- W8 NRMSE：`0.006267–0.006272`，約3.98×壓縮。
- W7：`0.012509–0.012514`，約4.54×。
- W6：`0.024540–0.024547`，約5.30×。
- W5：`0.049508–0.049521`，約6.35×。
- W4：`0.095830–0.095840`，約7.92×。
- Fixed SD4 per-channel MSE：`0.160325–0.160336`，容量與W4相同，整網不勝W4。
- Paper-TWN：`0.677546–0.677598`，約16×；任何parent／view皆0／148層勝W4。

### SD4逐層routing

- 三parent的deployment winner集合完全一致：37層。
- 三parent的master winner集合完全一致：35層。
- 三parent與兩view共同支持：34層，集中在neck 19、Detect tower 6、backbone_early 4、backbone_deep 4、neck attention-safe 1。
- 保守34層SD4＋其餘W4的deployment NRMSE改善約1.16–1.17%，因此只列候選，不宣告SD4政策winner。

完整數字、十個region worst-case表與限制見`docs/reports/2026-08-30-v1-v3-cpu-implementation.md`。

## 驗證方式與結果

### TDD紀錄

1. `pose_sigma`分類測試先暴露舊deployment誤分類，再修正為training-only。
2. unknown granularity測試Red結果為`Failed: DID NOT RAISE ValueError`。
3. unknown SD4 scale method測試Red結果同樣為`Failed: DID NOT RAISE ValueError`。
4. 加入granularity與scale白名單後，兩個測試皆Green。

### 完整驗證

```text
env CUDA_VISIBLE_DEVICES=-1 /home/uxin/yolo/.venv/bin/pytest -q
53 passed in 37.47s

/home/uxin/yolo/.venv/bin/ruff check src tests scripts
All checks passed!

/home/uxin/yolo/.venv/bin/ruff format --check src tests scripts
32 files already formatted
```

三個parent完整CPU analysis皆exit 0且各回報`gpu_used=false`、`measurements=2960`。diagnostic manifest三次重建後檔案SHA-256仍為`76b4392d2d07232967018b20af82149ba67ee036fad2cb75bb15d2d7565f4356`。

另以唯讀稽核解析12份YAML與7份JSON，核對三份V2檔案bytes／SHA-256、34個SD4候選唯一性、V4 15格、V5 150格及`execution_authorized=false`，全部通過。

根層`README.md`已存在BBAT5規範與工作紀錄入口，因此未在受限子專案工作中重複修改根README。

## 遇到的困難與解法

### GitHub Issue不可讀

- 困難：依規範執行`gh issue view 11 --comments --repo uxin0205-tech/yolo`時，`gh`未登入；公開API／HTML亦不可確認private Issue。
- 解法：明確記錄不可用，不臆測Issue內容；以本對話逐項接受的grilling決策作實作依據。

### 遠端publication與本機工作樹分歧

- 困難：根branch ahead 1／behind 52，子專案在本機為untracked；遠端公開報告較完整，而本機有遠端沒有的PTQ。
- 解法：不checkout／reset；以publication commit與檔案hash整合公開證據，另保留本機PTQ hash與限制，建立V0 reconciliation manifest。

### 官方fuse後contract介面失效

- 困難：官方fuse會將若干training-only head成員設為`None`，原始contract helper不能直接稽核fused graph。
- 解法：在本專案adapter內只重建所需inference contract，並以path parity與deterministic CPU forward共同驗證；不修改上游模型。

### BN fold的絕對誤差判斷

- 困難：大幅值tensor會讓很小相對誤差呈現較大的absolute max error。
- 解法：同時要求全局NRMSE與`max error / reference peak`，不以單一absolute數值誤判；三parent均通過既定門檻。

### Canonical BBAT5 symlink

- 困難：對canonical symlink呼叫`resolve()`會指向歷史原始來源，若以resolved parent判斷會誤報非canonical路徑。
- 解法：保留manifest中的canonical logical path，驗證其parent、可讀性與SHA，不把symlink target升格為訓練入口。

### Full-model static analysis耗時

- 困難：每parent 2,960筆含MSE grid與SD4 codebook projection，CPU約使用22 cores、1.2 GB RAM，每個parent約10–12分鐘。
- 解法：逐parent串行，避免三個模型同時佔用資源；日常pytest只使用tiny tensor，完整main profile按artifact版本執行，不把full ablation放入CI。本輪仍未使用GPU。

### Sandbox的apply_patch啟動問題

- 困難：一般sandbox命令偶發`bwrap: loopback: Failed RTM_NEWADDR`，包含內建patch呼叫。
- 解法：仍使用規定的`apply_patch`，但以受核准TTY行程輸入patch；沒有改用`cat`、Python或shell redirection寫專案檔。

## 未解事項與風險

- 尚無量化後八指標mAP，因此不能宣稱W8、W7或任何SD4政策通過精度門檻。
- V2是weight reconstruction；predictor等語意敏感層不能只靠NRMSE決定。
- LS-SD4、Channel-TWN、TTQ、A-SD4仍未實作或訓練；Fixed SD4與Paper-TWN static結果不能冒充這些方法。
- V4／V5只準備設定，尚未run；GPU未重新授權前不得啟動。
- 未建立30% QAT train view；建立新資料版本仍需使用者明確授權且必須BBAT group-safe。
- 沒有target FPGA／ASIC kernel、bit-true export、latency、power或resource量測；W5／W6／W7不得宣稱硬體加速。
- 根git歷史仍分歧且子專案untracked；本輪沒有commit／push，後續發行前要另做乾淨的finish-work稽核。
