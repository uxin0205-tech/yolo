# 2026-08-29：實作qSiLU、LSQ+與activation intake gate

## 任務與範圍

依使用者指示先檢查`/home/uxin/yolo/yolo_activation/`現況，分析目前領先的`qsilu_pq`後開始實作。本次只在`yolo_quantize`新增CPU可驗證的activation／quantizer primitives、唯讀P0 intake與provenance gate；未修改`yolo_activation`、Full35 final bundle、dataset或checkpoint，也未啟動GPU訓練。

## 變更內容與原因

1. 建立`src/yolo_quantize/`套件、`pyproject.toml`與pytest入口，讓後續量化工作有獨立且可測試的public interface。
2. 新增`ActivationIntake`：掃描所有`training/full35/*queue.yaml`與state，保留completed／failed／blocked／pending job，輸出`eligible`、`control-only`、`rejected`、`provisional`分類、八項gate deltas與三項主要absolute metrics。
3. intake以queue advisory lock辨識runner是否仍活著；pending但lock未持有時明確標成interrupted，避免把GPU空閒誤判成實驗完成。
4. 新增SHA-256 provenance gate：固定檢查qSiLU float／BitTrue source、activation manifest／accepted baseline／recipe，以及Full35 release manifest、release status、joint config與accepted checkpoint；Full35既有checksum可驗證的檔案必須吻合。
5. 新增`QSiLUPQ`與`QSiLUFixedPointConfig`／`emulate_qsilu_pq_legacy`。浮點surrogate保留upstream dyadic piecewise-quadratic、C1 knots與exact ReLU tails；legacy emulator明確標記`legacy_signed_half_away_from_zero`，不冒充後續requant的RNE。
6. 新增`LSQPlusActivationQuantizer`、`LSQPlusSpec`、RNE STE與gradient scaling。bit width是一般整數，第一輪正式接受A3、A4、A5、A6、A7、A8；qSiLU函數近似與activation-output量化保持兩個獨立邊界。
7. 新增`configs/activation/qsilu-pq-lsq-plus.yaml`，保存provisional qSiLU來源hash、已完成short-recovery evidence／checkpoint hash、LSQ+候選位寬、`0.01` W8A8 gate與`0.04`最終task gate。
8. 新增CLI `scripts/activation_intake.py`；blocked回傳exit code 2，ready回傳0。另新增`.gitignore`避免cache、run與checkpoint進入Git。

## 分析結果

- 主activation queue已為`5 completed / 0 pending / 14 blocked`；14個downstream工作因前置候選未過原gate而正確blocked。
- finalist queue為`1 completed / 1 pending / 0 blocked`。qSiLU finalist在epoch 1、macro 106收到`KeyboardInterrupt`，沒有存下`activation-experiment.json`，queue因此仍為`gate=null`與pending；GPU空閒是程序被中止，不是正常完成。
- 已完成qSiLU zero-shot與short recovery均通過原activation `0.015` gate。short recovery worst八項delta為`-0.0086354053`，三項主要metrics為COCO box `0.4952728940`、BBAT box `0.6257507805`、BBAT pose `0.9019860182`。
- Hardswish、`poly_shift`、`poly_quality` short recovery分別以worst delta `-0.0168844225`、`-0.0301380810`、`-0.0209698809`未通過原gate，因此目前不能與qSiLU並列eligible。
- qSiLU不是「8-bit activation」的同義詞：目前upstream BitTrue是Q16.10函數模擬；真正activation output bit-width由獨立LSQ+ policy控制，可測A3至A8。

## 驗證方式與結果

- 依TDD先確認每個新public behavior紅燈，再補最小實作並轉綠。
- `/home/uxin/yolo/.venv/bin/python -m pytest -q`：`26 passed`。
- `/home/uxin/yolo/.venv/bin/ruff check src tests scripts`與`ruff format --check src tests scripts`：通過。
- qSiLU跨專案parity：在`[-16, 16]`均勻取65,537個float64點，量化端與`yolo_activation` upstream輸出`exact_equal=True`、最大絕對誤差`0.0`。
- qSiLU legacy integer emulator以upstream golden vector逐元素exact通過。
- 真實intake CLI：共納入71個source／queue／result／checkpoint SHA-256 evidence，`ready=false`；唯一兩項blocker是finalist queue仍有1個pending job，以及pending時runner lock未持有。
- Full35 `RELEASE_STATUS.json`、`configs/joint.yaml`與`weights/combined/inference/best_joint.pt`均與`CHECKSUMS.sha256`吻合；accepted checkpoint SHA-256為`d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c`。
- 本次未執行training、dataset抽樣、validation或export。

## 困難與解法

- 困難：使用者通知GPU可用，但queue state仍pending。
- 解法：交叉檢查`nvidia-smi`、process、queue lock與訓練log，確認根因是`KeyboardInterrupt`，在intake加入interrupted判斷。
- 困難：Full35的`MANIFEST.json`不會出現在自身`CHECKSUMS.sha256`，最初被誤判為缺少checksum。
- 解法：增加回歸測試，允許release manifest以本次actual SHA-256記錄，但其餘Full35 required files仍強制比對既有checksum。
- 困難：內建patch sandbox偶發`bwrap: loopback: Failed RTM_NEWADDR`。
- 解法：經核准後仍只使用`apply_patch`實際程式更新，沒有改用整檔覆寫或修改外部專案。

## 未解事項或風險

- qSiLU finalist尚未完成；必須由activation工作流正常resume／重跑並生成terminal evidence，或由使用者明確改變gate，才可凍結正式intake snapshot。
- 現在只能稱qSiLU為provisional唯一eligible非SiLU候選，不能宣稱正式winner。
- MSE representative-batch initialization、Full35 adapter、190-site transactional replacement、protected-zone parity、W8A8 matched bring-up與GPU QAT尚未開始。
- 尚無FPGA／ASIC板上latency、power或resource量測；目前僅能報hardware-oriented與BitTrue證據。
- BBAT5資料完全未修改；後續固定30% diagnostic view仍必須沿用bbat5-v1 assignment、group-safe manifest與完整formal validation。

## 後續修正（同日）

後續依使用者接受的量化screening 0.04三任務門檻，將upstream舊0.015 gate與quantization eligibility分離。Hardswish、poly_shift與poly_quality仍是上游gate failed，但completed evidence都在0.04內，因此改列screen candidates；qSiLU不再描述為唯一可進量化screening的候選。Full35 adapter、124-site deployment output quantization與30格GPU smoke也已完成，詳見2026-08-29-activation-coupled-smoke-matrix.md。本段取代本文較早的動態分類與未開始項目。
