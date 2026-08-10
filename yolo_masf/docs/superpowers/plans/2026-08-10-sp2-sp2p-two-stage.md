# SP2 與 SP2P 兩階段訓練 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 SP2 改成 10 epochs 凍結 backbone＋90 epochs 全解凍，建立雙來源繼承的 SP2P 並套用相同 10+90 訓練，同時移除 ONNX pipeline 工作且保留既有訓練成果。

**Architecture:** 保留公開 `SP2P` 實驗名稱，內部使用靜態且可嚴格重建的 `SP2M2`／`SP2M3` variant 表示 selection 決定的實際架構。SP2P 先載入 SP2-B 共用權重，再僅載入 BEST_PARTIAL 的 `model.20.*`，以 lineage manifest 鎖定兩個 parent；workflow 在 SP2P 前先完成 validation-only selection。

**Tech Stack:** Python 3.12、PyTorch 2.11、Ultralytics 8.4.90、pytest、YAML、systemd user service。

## Global Constraints

- 不修改 `configs/static-phase1.yaml`，保持既有 config hash 與 pipeline ID，避免重跑已完成 stages。
- 不停止或重啟正在執行的 `formal_m0`；程式與測試完成後只在安全 checkpoint 重載單一 systemd pipeline。
- SP2-A/SP2P-A：freeze indices 0–10、10 epochs、`lr0=0.01`。
- SP2-B/SP2P-B：無 freeze、90 epochs、`lr0=0.001`。
- SP2P 只可能採用 M2 或 M3；選擇只使用 validation，test 必須在 selection 凍結後。
- SP2P 從 SP2-B 載入共用網路及 Selective P2 head，只從 BEST_PARTIAL 載入 `model.20.*` partial-MFAM。
- SP2P 不加入 7×7、9×9、Top-K、資料 Gather 或不規則 sparse tile。
- 不執行 SP2 ONNX 匯出，也不要求 ONNX artifact；既有 export utility 可保留為非 pipeline 工具。
- 所有 GPU 工作維持同一個 `masf-yolo-phase1.service`，不得平行啟動第二條訓練。

---

### Task 1: 建立可嚴格重建的 SP2P 模型變體

**Files:**
- Modify: `masf_yolo/variants.py`
- Modify: `masf_yolo/models/builder.py`
- Test: `tests/test_variants.py`
- Test: `tests/models/test_selective.py`

**Interfaces:**
- Produces: `SP2P_VARIANTS: Mapping[str, VariantDefinition]`
- Produces: `sp2p_variant_id(selected_partial: str) -> str`
- Produces: `is_selective_variant(variant_id: str) -> bool`
- Consumes: 現有 `VariantDefinition`、`PartialMFAM`、`SelectiveP2Detect`

- [ ] **Step 1: 寫入失敗測試**

```python
def test_sp2p_internal_variants_are_static_partial_selective_models() -> None:
    assert sp2p_variant_id("M2") == "SP2M2"
    assert sp2p_variant_id("M3") == "SP2M3"
    with pytest.raises(ValueError, match="M2 or M3"):
        sp2p_variant_id("M1")
    for parent, ratio in (("M2", 0.5), ("M3", 0.25)):
        definition = get_variant(sp2p_variant_id(parent))
        assert definition.kernel_branches == (3, 5)
        assert definition.processed_ratio == ratio
        assert definition.p2_slot == "partial_mfam"
        assert is_selective_variant(definition.variant_id)

def test_sp2p_builds_partial_mfam_with_selective_head() -> None:
    model = build_model("SP2M3")
    assert isinstance(model.model[P2_SLOT_INDEX], PartialMFAM)
    assert model.model[P2_SLOT_INDEX].processed_channels == 32
    assert isinstance(model.model[-1], SelectiveP2Detect)
    with torch.no_grad():
        output, raw = model.eval()(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 6, 340)
    assert set(raw) == {"ball", "main"}
```

- [ ] **Step 2: 驗證測試因缺少 SP2P API 而失敗**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_variants.py tests/models/test_selective.py -q`

Expected: FAIL，原因為 `sp2p_variant_id`／`SP2M2`／`SP2M3` 尚不存在。

- [ ] **Step 3: 最小實作靜態內部變體**

在 `masf_yolo/variants.py` 加入：

```python
_SP2P_VARIANTS = {
    "SP2M2": VariantDefinition("SP2M2", (3, 5), 0.5, "partial_mfam"),
    "SP2M3": VariantDefinition("SP2M3", (3, 5), 0.25, "partial_mfam"),
}
_VARIANTS.update(_SP2P_VARIANTS)
SP2P_VARIANTS = MappingProxyType(_SP2P_VARIANTS)

def sp2p_variant_id(selected_partial: str) -> str:
    if selected_partial not in {"M2", "M3"}:
        raise ValueError("SP2P parent must be M2 or M3")
    return f"SP2{selected_partial}"

def is_selective_variant(variant_id: str) -> bool:
    return variant_id in {"SP2", "SP2M2", "SP2M3"}
```

保持 `TRAINED_VARIANTS` 不含兩個內部變體，並將公開評估矩陣改為 `EVALUATED_MODELS = ("B0", *TRAINED_VARIANTS, "SP2P")`。在 builder 用 `is_selective_variant(definition.variant_id)` 決定是否替換 Selective head。

- [ ] **Step 4: 驗證模型測試通過**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_variants.py tests/models/test_selective.py -q`

Expected: PASS。

---

### Task 2: 實作 SP2P 雙來源權重轉移與 lineage

**Files:**
- Modify: `masf_yolo/models/transfer.py`
- Modify: `masf_yolo/training/worker.py`
- Test: `tests/models/test_transfer.py`
- Test: `tests/training/test_worker.py`

**Interfaces:**
- Produces: `transfer_sp2p_parents(model, sp2_state, partial_state, *, selected_partial: str) -> SP2PTransferReport`
- Produces: `SP2PTransferReport.to_dict() -> dict[str, Any]`
- Consumes: SP2-B canonical、M2/M3 canonical、`sp2p_variant_id()`、`selection.json`

- [ ] **Step 1: 寫入雙來源轉移失敗測試**

```python
def test_sp2p_transfer_uses_sp2_for_shared_state_and_partial_only_for_slot() -> None:
    sp2 = build_model("SP2")
    m3 = build_model("M3")
    target = build_model("SP2M3")
    with torch.no_grad():
        sp2.state_dict()["model.0.conv.weight"].fill_(0.25)
        m3.state_dict()["model.20.process.cv1.conv.weight"].fill_(0.75)
        m3.state_dict()["model.0.conv.weight"].fill_(0.99)
    report = transfer_sp2p_parents(
        target, sp2.state_dict(), m3.state_dict(), selected_partial="M3"
    )
    assert torch.all(target.state_dict()["model.0.conv.weight"] == 0.25)
    assert torch.all(target.state_dict()["model.20.process.cv1.conv.weight"] == 0.75)
    assert "model.20.process.cv1.conv.weight" in report.partial_keys
    assert all(key.startswith("model.20.") for key in report.partial_keys)
```

另加錯誤 parent、缺少 slot key、shape mismatch 均 fail closed 的測試。

- [ ] **Step 2: 驗證測試因 transfer API 不存在而失敗**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/models/test_transfer.py tests/training/test_worker.py -q`

Expected: FAIL，原因為 `transfer_sp2p_parents` 尚不存在。

- [ ] **Step 3: 實作限制範圍的雙來源轉移**

實作順序固定為：

```python
shared_keys = {key for key in model.state_dict() if not key.startswith("model.20.")}
if any(key not in sp2_state for key in shared_keys):
    raise ValueError("SP2P shared keys do not match SP2 parent")
for key in sorted(shared_keys):
    if model.state_dict()[key].shape != sp2_state[key].shape:
        raise ValueError(f"SP2P shared shape mismatch: {key}")
    model.state_dict()[key].copy_(sp2_state[key])
slot_keys = {key for key in model.state_dict() if key.startswith("model.20.")}
source_slot_keys = {key for key in partial_state if key.startswith("model.20.")}
if slot_keys != source_slot_keys:
    raise ValueError("SP2P partial slot keys do not match selected parent")
for key in sorted(slot_keys):
    if model.state_dict()[key].shape != partial_state[key].shape:
        raise ValueError(f"SP2P partial slot shape mismatch: {key}")
    model.state_dict()[key].copy_(partial_state[key])
```

Report 必須記錄 `selected_partial`、`sp2_shared_keys`、`partial_keys`、missing 與 shape mismatch；partial parent 禁止覆寫 `model.20.*` 以外的 key。

在 worker 對 `smoke_sp2p`／`sp2p_a` 讀取 immutable `selection.json`，載入 `sp2_b/canonical.pt` 與選定 `formal_m2` 或 `formal_m3` canonical，建立對應 `SP2M2`／`SP2M3` 後執行轉移。`sp2p_b` 只能載入 `sp2p_a` best。

- [ ] **Step 4: 驗證雙來源測試通過**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/models/test_transfer.py tests/training/test_worker.py -q`

Expected: PASS。

---

### Task 3: 建立 SP2/SP2P 10+90 profiles 與初始化規則

**Files:**
- Modify: `masf_yolo/training/profiles.py`
- Modify: `masf_yolo/pipeline.py`
- Modify: `masf_yolo/training/worker.py`
- Test: `tests/training/test_profiles.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/training/test_worker.py`

**Interfaces:**
- Produces: `frozen_stage_profile(variant_id, model_path, project, *, name) -> dict[str, Any]`
- Consumes: `formal_profile()`、`b1_a_profile()`、training records

- [ ] **Step 1: 寫入精確 profile 與 parent 失敗測試**

```python
@pytest.mark.parametrize("variant,name", [("SP2", "sp2-a"), ("SP2M3", "sp2p-a")])
def test_frozen_stage_profile_matches_required_ten_epoch_policy(variant, name) -> None:
    profile = frozen_stage_profile(variant, "parent.pt", "runs", name=name)
    assert profile["epochs"] == 10
    assert profile["freeze"] == list(range(11))
    assert profile["lr0"] == 0.01
    assert profile["name"] == name

def test_sp2_and_sp2p_b_profiles_are_unfrozen_ninety_epoch_runs() -> None:
    pipeline = _pipeline_with_training_records()
    for stage, variant in (("sp2_b", "SP2"), ("sp2p_b", "SP2M3")):
        profile = pipeline._profile_for_stage(stage, variant)
        assert profile["epochs"] == 90
        assert profile["freeze"] is None
        assert profile["lr0"] == 0.001
```

Worker 測試必須 assert `sp2_a` 從 B1-B canonical、`sp2_b` 從 SP2-A best、`sp2p_b` 從 SP2P-A best 初始化。

- [ ] **Step 2: 驗證測試因 staged profiles 尚未實作而失敗**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/training/test_profiles.py tests/training/test_worker.py tests/test_pipeline.py -q`

Expected: FAIL，指出 `frozen_stage_profile` 或新 stage profile 缺失。

- [ ] **Step 3: 實作 A/B profile 分派**

```python
def frozen_stage_profile(variant_id, model_path, project, *, name):
    profile = formal_profile(variant_id, model_path, project, epochs=10)
    profile.update({"name": name, "lr0": 0.01, "freeze": list(range(11))})
    return profile
```

讓 `b1_a_profile()` 呼叫此 helper，保持既有行為。`_profile_for_stage()` 對 `sp2_a`／`sp2p_a` 使用 frozen profile；對 `sp2_b`／`sp2p_b` 使用 90-epoch formal profile並明確設 `freeze=None`。其他 formal 變體維持原 100 epochs 全解凍。

- [ ] **Step 4: 驗證 profiles 與 worker 測試通過**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/training/test_profiles.py tests/training/test_worker.py tests/test_pipeline.py -q`

Expected: PASS。

---

### Task 4: 重排 workflow、提前 selection 並移除 ONNX 工作

**Files:**
- Modify: `masf_yolo/workflow.py`
- Modify: `masf_yolo/pipeline.py`
- Test: `tests/test_workflow.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: `_evaluate_variants(split: str, variants: tuple[str, ...], *, reuse: bool) -> StageResult`
- Produces stages: `sp2_a`、`sp2_b`、`val_partial`、`smoke_sp2p`、`sp2p_a`、`sp2p_b`
- Removes stages: `formal_sp2`、`export_sp2`

- [ ] **Step 1: 寫入新 DAG 失敗測試**

```python
def test_sp2_and_sp2p_stages_have_locked_order_without_onnx() -> None:
    names = [stage.name for stage in PHASE1_STAGES]
    assert "formal_sp2" not in names
    assert "export_sp2" not in names
    assert names.index("formal_p3m") < names.index("sp2_a") < names.index("sp2_b")
    assert names.index("sp2_b") < names.index("val_partial") < names.index("selection")
    assert names.index("selection") < names.index("smoke_sp2p")
    assert names.index("smoke_sp2p") < names.index("sp2p_a") < names.index("sp2p_b")
    assert names.index("sp2p_b") < names.index("baseline_b0") < names.index("val_all")
    assert names.index("selection") < names.index("test_all")
```

另測 `val_all` 對 M2/M3 只接受與 `selection.json.val_hashes` 相符的既有 metrics，不覆寫檔案；hash 不符立即失敗。

- [ ] **Step 2: 驗證舊 DAG 與 ONNX stage 使測試失敗**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_workflow.py tests/test_pipeline.py -q`

Expected: FAIL，指出 `formal_sp2`／`export_sp2` 仍存在且新 stages 缺失。

- [ ] **Step 3: 最小修改 stage handlers 與 validation reuse**

將尾段固定為：

```python
"formal_p3m",
"sp2_a",
"sp2_b",
"val_partial",
"selection",
"smoke_sp2p",
"sp2p_a",
"sp2p_b",
"baseline_b0",
"val_all",
"test_all",
"profile_all",
"final_audit",
"report",
```

`val_partial` 呼叫 `_evaluate_variants("val", ("M2", "M3"), reuse=False)`。`val_all` 對 M2/M3 驗證 selection hashes 後重用，對其餘公開模型正常評估並產生完整 aggregate。刪除 `_export_sp2` handler，但保留 `masf_yolo/evaluation/exporting.py`。

- [ ] **Step 4: 驗證 workflow 測試通過**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_workflow.py tests/test_pipeline.py -q`

Expected: PASS。

---

### Task 5: 鎖定 SP2P canonical、lineage 與最終稽核

**Files:**
- Modify: `masf_yolo/pipeline.py`
- Modify: `masf_yolo/artifacts/finalize.py`
- Modify: `masf_yolo/artifacts/strict_reload.py`
- Test: `tests/artifacts/test_finalize.py`
- Test: `tests/artifacts/test_checkpoints.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Produces: SP2P run record fields `display_variant`、`architecture_variant`、`selected_partial`、`parent_hashes`、`selection_hash`
- Consumes: 內部 `SP2M2`／`SP2M3` variant IDs 與 immutable selection

- [ ] **Step 1: 寫入 lineage 與 audit 失敗測試**

建立一份 `selected=M3` 的 fixture，assert：

```python
assert record["display_variant"] == "SP2P"
assert record["architecture_variant"] == "SP2M3"
assert record["selected_partial"] == "M3"
assert record["parent_hashes"] == {"sp2_b": "sp2-hash", "formal_m3": "m3-hash"}
assert record["selection_hash"] == sha256_file(selection_path)
```

Final audit 測試需拒絕錯誤 architecture parent、缺少 lineage、selection hash 不符與 SP2P canonical 不可 strict reload；同時 assert 不再要求 `exports/sp2/manifest.json`。

- [ ] **Step 2: 驗證 audit 因仍要求 ONNX 且不認識 SP2P 而失敗**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/artifacts/test_finalize.py tests/artifacts/test_checkpoints.py tests/test_pipeline.py -q`

Expected: FAIL，指出 SP2P lineage／canonical stages 缺失或 ONNX 仍為必要 artifact。

- [ ] **Step 3: 實作 architecture/display variant 分離**

`_train()` 對 SP2P stages 使用 selection 對應的 `architecture_variant` 呼叫 worker、finalizer 與 strict reload，但 run record 額外寫入固定 `display_variant="SP2P"` 及 lineage。`_best_checkpoint("SP2")` 指向 `sp2_b`，`_best_checkpoint("SP2P")` 指向 `sp2p_b`。

將 `CANONICAL_TRAINING_STAGES` 的 `formal_sp2` 改成 `sp2_a`、`sp2_b`、`sp2p_a`、`sp2p_b`，final audit 驗證 A/B canonical、SP2P architecture 與 selection 一致；刪除全部 ONNX 必要條件。

- [ ] **Step 4: 驗證 canonical 與 audit 測試通過**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/artifacts/test_finalize.py tests/artifacts/test_checkpoints.py tests/test_pipeline.py -q`

Expected: PASS。

---

### Task 6: 更新評估、報告與中文操作文件

**Files:**
- Modify: `masf_yolo/reporting.py`
- Modify: `tests/test_reporting.py`
- Modify: `codex_plan.md`
- Modify: `README.md`
- Modify: `masf_yolo/Readme.md`

**Interfaces:**
- Consumes: 公開 `EVALUATED_MODELS`，包含 `SP2P`
- Produces: 報告中的 SP2-A/B、SP2P-A/B、BEST_PARTIAL parent 與 sequential-training-budget 警告

- [ ] **Step 1: 寫入報告矩陣失敗測試**

```python
def test_report_includes_sp2p_lineage_and_no_onnx_requirement(tmp_path: Path) -> None:
    report = rebuild_report(config_path)
    text = Path(report).read_text(encoding="utf-8")
    assert "SP2P" in text
    assert "SP2-A" in text and "SP2-B" in text
    assert "SP2P-A" in text and "SP2P-B" in text
    assert "序列式訓練預算" in text
    assert "ONNX export" not in text
```

- [ ] **Step 2: 驗證舊報告缺少 SP2P 而失敗**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_reporting.py -q`

Expected: FAIL，指出 SP2P 或 staged runs 缺失。

- [ ] **Step 3: 更新報告與中文文件**

報告 formal stages 改為 `sp2_a`、`sp2_b`、`sp2p_a`、`sp2p_b`，評估表加入 SP2P；明列 SP2P 的 parent、partial ratio 與額外 103 epochs，禁止把結果描述成與 B1-B 起始的單段公平消融。

同步更新 `codex_plan.md` 的訓練矩陣、總 epochs、pipeline 順序、完成判定；更新兩份 README 的資料夾用途與 artifact 路徑，移除「pipeline 將匯出 ONNX」敘述，但可保留既有 CPU export 技術驗證為歷史紀錄。

- [ ] **Step 4: 驗證報告測試通過**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest tests/test_reporting.py -q`

Expected: PASS。

---

### Task 7: 全面 CPU 驗證與安全接續單一 GPU pipeline

**Files:**
- Verify only: repository Python/tests/docs
- Runtime artifacts: `artifacts/static-phase1/stages/`

**Interfaces:**
- Consumes: Tasks 1–6 全部變更
- Produces: 通過測試的程式與載入新版 DAG 的單一 `masf-yolo-phase1.service`

- [ ] **Step 1: 執行針對性測試**

Run:

```bash
CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest \
  tests/test_variants.py tests/models/test_selective.py tests/models/test_transfer.py \
  tests/training/test_profiles.py tests/training/test_worker.py \
  tests/test_workflow.py tests/test_pipeline.py tests/artifacts/test_finalize.py \
  tests/artifacts/test_checkpoints.py tests/test_reporting.py -q
```

Expected: PASS，無 warning/error traceback。

- [ ] **Step 2: 執行全套測試與編譯檢查**

Run: `CUDA_VISIBLE_DEVICES='' ../.venv/bin/python -m pytest -q`

Expected: 全部 PASS。

Run: `../.venv/bin/python -m compileall -q masf_yolo tests`

Expected: exit 0。

- [ ] **Step 3: 檢查 config identity、格式與服務狀態**

Run: `git diff --check`

Expected: exit 0。

Run: `../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml`

Expected: pipeline ID 仍為 `cb7e0392cc32`；已完成 stages 保持 completed。

- [ ] **Step 4: 在安全 checkpoint 重載服務**

先確認 `formal_m0` 已 completed；若下一 stage 已啟動，確認 `last.pt` 與 `results.csv` 最近一個 epoch 均完整寫入，再停止唯一的 `masf-yolo-phase1.service`。重新使用同一 config 執行：

Run: `../.venv/bin/python -m masf_yolo.cli pipeline start --config configs/static-phase1.yaml`

Expected: 只有一個 active MASF service；既有完成 stages 被重用；新版 DAG 從第一個未完成 stage 接續，不重跑 B1、M7、smokes 或 M0。

- [ ] **Step 5: 驗證接續健康度並回報工作量**

Run: `systemctl --user status masf-yolo-phase1.service --no-pager -l`

Run: `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader`

Expected: 單一 worker 使用 GPU、state error 為 null、epoch 持續增加。

最終回報需列出：新版 stage 總數、已完成／執行中／待執行數量、每個模型 epochs、剩餘 GPU epochs、SP2/SP2P parent 關係、測試數量、當前 stage 與 GPU 狀態。
