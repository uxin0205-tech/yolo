# BDCN Research Branch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 將 BDCN、R0/R1/R2 denominator variants 與 fused grouped-value path 加入研究計畫、workflow 與 CPU reference framework，但不執行 GPU 訓練。

**Architecture:** BDCN 數學放在獨立 `bdcn.py`；`normalization.py` 只保留共同 factory。模型轉換器為兩個 YOLO26m Attention 建立一個共享 codebook bank，再依 global/per-Attention/per-head 模式分配 table index，確保 sharing 是真實參數共享而非兩份相同初值。

**Tech Stack:** Python 3.10+、PyTorch、Ultralytics YOLO26m、PyYAML、pytest、Ruff。

## Global Constraints

- 正式模型固定同時轉換 `model.10.m.0.attn` 與 `model.22.m.0.1.attn`，少任一處即 fail closed。
- BDCN 只修改 score-to-P normalization；保留 V、`PV+PE(V)`、projection、FFN、residual、C2PSA/C3k2 與 Detect。
- 第一版固定 16 個等距 distance buckets；bucket boundary 不學習。
- 不使用 KD，不量化 P/V/projection，不執行 GPU 訓練或上板。
- Histogram 與 direct denominator 必須數值等價，不作獨立 accuracy 方法。
- BDCN 不宣稱 division-free；R1 每列使用一次 reciprocal LUT，R2 才是會破壞 exact row sum 的 division-free diagnostic。
- R0 direct `P @ V` 與 grouped-value aggregation 必須數值等價。
- BDCN 是 proposed combination，不宣稱 distance LUT、histogram 或 two-term PoT 為世界首次。
- 所有新行為先寫失敗測試，再寫最小實作。

---

## File map

| File | Responsibility |
|---|---|
| `src/yolo_attention/bdcn.py` | bucket、單調 codebook bank、PoT projection、direct/histogram normalization |
| `src/yolo_attention/config.py` | BDCN enum、typed fields 與不合法組合驗證 |
| `src/yolo_attention/normalization.py` | 將 BDCN 接入既有 factory，不承擔內部數學 |
| `src/yolo_attention/attention.py` | 接受 conversion context 建立的 normalizer |
| `src/yolo_attention/integration.py` | 兩個 sites 共用 bank、table-index 分配、BDCN freeze scope |
| `src/yolo_attention/workflow.py` | D0/D1/D2 與 A-FINAL selection step |
| `src/yolo_attention/profiling.py` | BDCN lookup、shift/add、storage、histogram proxy |
| `tests/test_bdcn.py` | BDCN 數值與硬體表示單元測試 |
| `tests/test_yolo_adapter.py` | 兩個 sites 的真實 parameter-sharing integration |
| `tests/test_config_and_workflow.py` | config round-trip、validation 與研究順序 |
| `plan.md` | 正式研究漏斗、gate、公式、運算量與 claim 邊界 |

---

### Task 1: Formalize the approved research funnel

**Files:**
- Modify: `tests/test_config_and_workflow.py`
- Modify: `src/yolo_attention/workflow.py`
- Modify: `plan.md`
- Modify: `README.md`
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: `ResearchWorkflow.default() -> ResearchWorkflow`
- Produces: workflow steps `bdcn-reference`, `bdcn-learning`, `bdcn-projection`, `bdcn-denominator`, and an A-FINAL step that follows both existing N1 and BDCN.

- [ ] **Step 1: Write the failing workflow test**

Replace positional assertions with key-based lookup and add:

```python
def test_workflow_places_bdcn_before_final_and_after_a0() -> None:
    payload = ResearchWorkflow.default().to_dict()
    steps = {step["key"]: step for step in payload["main"]}
    order = [step["key"] for step in payload["main"]]

    assert steps["bdcn-reference"]["runs"] == ["D0-IDX"]
    assert steps["bdcn-learning"]["runs"] == ["D1-SHARED", "D1-PATTN", "D1-PHEAD"]
    assert steps["bdcn-learning"]["epochs"] == 5
    assert steps["bdcn-projection"]["runs"] == ["D2-FP", "D2-1P", "D2-2P"]
    assert steps["bdcn-denominator"]["runs"] == ["R0-DIV", "R1-RLUT", "R2-PSHIFT"]
    assert order.index("normalization-recovery") < order.index("bdcn-reference")
    assert order.index("bdcn-projection") < order.index("final")
    assert steps["final"]["selection"] == "compare A0, N1 winner, and BDCN winner"
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
../.venv/bin/pytest -q tests/test_config_and_workflow.py::test_workflow_places_bdcn_before_final_and_after_a0
```

Expected: failure because `bdcn-reference` is absent.

- [ ] **Step 3: Add declarative workflow steps**

Add these `WorkflowStep` values immediately before `final`:

```python
WorkflowStep(
    "bdcn-reference",
    ("D0-IDX",),
    0,
    "Fixed exponential distance-LUT prior-art control",
),
WorkflowStep(
    "bdcn-learning",
    ("D1-SHARED", "D1-PATTN", "D1-PHEAD"),
    5,
    "Codebook-only sharing ablation",
    "choose the simplest candidate within 0.001 mAP of the best",
),
WorkflowStep(
    "bdcn-projection",
    ("D2-FP", "D2-1P", "D2-2P"),
    "0 + at most one 5-epoch recovery",
    "Project one learned codebook to hardware formats",
    "retain only candidates within 0.01 mAP of A0",
),
WorkflowStep(
    "final",
    ("A-FINAL",),
    0,
    "Freeze the formal pre-quantization model",
    "compare A0, N1 winner, and BDCN winner",
),
```

Remove the previous duplicate `final` step.

- [ ] **Step 4: Update formal research documentation**

Copy the accepted formulas, D0/D1/D2 tables, 0.001 simplicity rule, 0.01 accuracy gate, single-recovery limit, prior-art-safe claim, and three-parent A-FINAL decision from `docs/superpowers/specs/2026-08-11-bdcn-research-branch-design.md` into `plan.md`. Update the root workflow diagram and README summary without copying implementation details.

- [ ] **Step 5: Verify and commit the independently reviewable funnel**

Run:

```bash
../.venv/bin/pytest -q tests/test_config_and_workflow.py tests/test_trainer_profiling_cli.py
```

Expected: all tests pass and `python -m yolo_attention.cli workflow` lists D0/D1/D2.

Commit only after confirming the repository can safely track this project:

```bash
git add plan.md README.md docs/README.md src/yolo_attention/workflow.py tests/test_config_and_workflow.py
git commit -m "docs: add BDCN research funnel"
```

---

### Task 2: Add typed BDCN configuration

**Files:**
- Modify: `src/yolo_attention/config.py`
- Modify: `src/yolo_attention/__init__.py`
- Modify: `tests/test_config_and_workflow.py`

**Interfaces:**
- Produces: `BDCNCodebookKind`, `BDCNSharing`, `BDCNProjection`, and validated `VariantConfig` fields.
- Consumed by: `build_bdcn_bank()` and `BDCNNormalizer` in Task 3.

- [ ] **Step 1: Write failing enum, round-trip, and validation tests**

```python
from yolo_attention.config import BDCNCodebookKind, BDCNProjection, BDCNSharing


def test_bdcn_config_round_trips_and_rejects_irrelevant_options(tmp_path: Path) -> None:
    config = VariantConfig(
        name="D1-PHEAD",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.PER_HEAD,
        bdcn_projection=BDCNProjection.FLOAT,
        bdcn_levels=16,
        bdcn_step=0.125,
    )
    path = config.to_yaml(tmp_path / "bdcn.yaml")
    assert VariantConfig.from_yaml(path) == config

    with pytest.raises(ValueError, match="only apply to BDCN"):
        VariantConfig(name="bad", normalization=NormalizationKind.EXACT, bdcn_levels=8)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
../.venv/bin/pytest -q tests/test_config_and_workflow.py::test_bdcn_config_round_trips_and_rejects_irrelevant_options
```

Expected: import failure because the BDCN enums do not exist.

- [ ] **Step 3: Implement enums and fields**

Add:

```python
class BDCNCodebookKind(StringEnum):
    FIXED_EXP = "fixed_exp"
    LEARNED = "learned"


class BDCNSharing(StringEnum):
    GLOBAL = "global"
    PER_ATTENTION = "per_attention"
    PER_HEAD = "per_head"


class BDCNProjection(StringEnum):
    FLOAT = "float"
    ONE_POT = "one_pot"
    TWO_POT = "two_pot"
```

Add `NormalizationKind.BDCN = "bdcn"` and fields:

```python
bdcn_codebook: BDCNCodebookKind | None = None
bdcn_sharing: BDCNSharing | None = None
bdcn_projection: BDCNProjection | None = None
bdcn_levels: int = 16
bdcn_step: float = 0.125
bdcn_histogram: bool = False
```

Validation rules:

- BDCN requires all three BDCN enums.
- Non-BDCN variants require all BDCN enums to be `None`, `bdcn_levels == 16`, `bdcn_step == 0.125`, and `bdcn_histogram is False`.
- `bdcn_levels >= 2` and `bdcn_step > 0`.
- `FIXED_EXP` requires `FLOAT` projection and is not trainable.

- [ ] **Step 4: Export enums and verify**

Export the three enums from `src/yolo_attention/__init__.py`, then run:

```bash
../.venv/bin/pytest -q tests/test_config_and_workflow.py
```

Expected: all config/workflow tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/yolo_attention/config.py src/yolo_attention/__init__.py tests/test_config_and_workflow.py
git commit -m "feat: add typed BDCN configuration"
```

---

### Task 3: Implement the BDCN mathematical reference

**Files:**
- Create: `src/yolo_attention/bdcn.py`
- Create: `tests/test_bdcn.py`
- Modify: `src/yolo_attention/normalization.py`

**Interfaces:**
- Consumes: BDCN enums and `VariantConfig` from Task 2.
- Produces:
  - `BDCNCodebookBank(num_tables: int, levels: int, step: float, kind: BDCNCodebookKind, projection: BDCNProjection)`
  - `BDCNNormalizer(bank: BDCNCodebookBank, table_indices: torch.Tensor, step: float, histogram: bool)`
  - `project_one_pot(codebook: Tensor) -> Tensor`
  - `project_two_pot(codebook: Tensor) -> Tensor`

- [ ] **Step 1: Write failing math tests**

```python
def test_bdcn_is_positive_monotonic_and_row_normalized() -> None:
    bank = BDCNCodebookBank(
        num_tables=2,
        levels=16,
        step=0.125,
        kind=BDCNCodebookKind.LEARNED,
        projection=BDCNProjection.FLOAT,
    )
    normalizer = BDCNNormalizer(
        bank=bank,
        table_indices=torch.tensor([0, 1]),
        step=0.125,
        histogram=False,
    )
    scores = torch.randn(2, 2, 7, 7)
    probability = normalizer(scores)
    codebook = bank.codebook()

    assert torch.all(codebook > 0)
    assert torch.all(codebook[:, :-1] >= codebook[:, 1:])
    assert torch.all(probability >= 0)
    torch.testing.assert_close(probability.sum(-1), torch.ones(2, 2, 7))


def test_histogram_and_direct_denominator_are_equivalent() -> None:
    bank = BDCNCodebookBank(1, 16, 0.125, BDCNCodebookKind.LEARNED, BDCNProjection.FLOAT)
    scores = torch.randn(1, 2, 9, 9)
    direct = BDCNNormalizer(bank, torch.tensor([0, 0]), 0.125, histogram=False)
    histogram = BDCNNormalizer(bank, torch.tensor([0, 0]), 0.125, histogram=True)
    torch.testing.assert_close(histogram(scores), direct(scores), rtol=1e-6, atol=1e-7)


def test_two_pot_never_has_more_error_than_one_pot() -> None:
    codebook = torch.tensor([[1.0, 0.73, 0.41, 0.19, 0.07]])
    one_error = (project_one_pot(codebook) - codebook).abs()
    two_error = (project_two_pot(codebook) - codebook).abs()
    assert torch.all(two_error <= one_error + 1e-7)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
../.venv/bin/pytest -q tests/test_bdcn.py
```

Expected: module import failure because `bdcn.py` does not exist.

- [ ] **Step 3: Implement monotonic codebook parameterization**

Use a cumulative ratio parameterization:

```python
ratio = torch.sigmoid(self.raw_ratios).clamp_max(1.0 - 1e-6)
tail = ratio.cumprod(dim=-1)
codebook = torch.cat((torch.ones_like(tail[:, :1]), tail), dim=-1)
```

Initialize every ratio to `exp(-step)` using its inverse-logit. `FIXED_EXP` registers a buffer containing `exp(-arange(levels) * step)`; `LEARNED` registers `raw_ratios` as `nn.Parameter`.

- [ ] **Step 4: Implement bucket and normalization paths**

Use:

```python
distance = scores.amax(dim=-1, keepdim=True) - scores
bucket = torch.round(distance / self.step).to(torch.long).clamp(0, self.bank.levels - 1)
tables = self.bank.codebook()[self.table_indices]
weights = tables.view(1, scores.shape[1], 1, -1).expand(
    scores.shape[0], -1, scores.shape[2], -1
).gather(-1, bucket)
```

Direct denominator is `weights.sum(-1, keepdim=True)`. Histogram denominator uses `torch.nn.functional.one_hot(bucket, levels).sum(dim=-2)` followed by a dot product with the selected table. Return `weights / denominator.clamp_min(torch.finfo(weights.dtype).eps)`.

- [ ] **Step 5: Implement PoT projection with STE**

`project_one_pot` selects the nearest value from `2**(-arange(0, 16))`. `project_two_pot` enumerates every unordered pair including zero as the second term, then gathers the candidate with minimum absolute error. In training, return `float_codebook + (projected - float_codebook).detach()`; in evaluation return `projected`.

- [ ] **Step 6: Add factory seam and verify**

Extend `build_normalizer` with optional keyword-only arguments:

```python
def build_normalizer(
    config: VariantConfig,
    *,
    bdcn_bank: BDCNCodebookBank | None = None,
    bdcn_table_indices: torch.Tensor | None = None,
) -> nn.Module:
```

For BDCN, require both injected objects and construct `BDCNNormalizer`; existing normalizers remain unchanged. Run:

```bash
../.venv/bin/pytest -q tests/test_bdcn.py tests/test_normalization_candidates.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/yolo_attention/bdcn.py src/yolo_attention/normalization.py tests/test_bdcn.py
git commit -m "feat: add BDCN normalization reference"
```

---

### Task 4: Integrate one shared bank across both YOLO26m Attention sites

**Files:**
- Modify: `src/yolo_attention/attention.py`
- Modify: `src/yolo_attention/integration.py`
- Modify: `tests/test_yolo_adapter.py`
- Modify: `tests/test_attention_integration.py`

**Interfaces:**
- Consumes: `BDCNCodebookBank` and `BDCNNormalizer` from Task 3.
- Produces: `_build_bdcn_context(config, paths, heads) -> tuple[BDCNCodebookBank, dict[str, Tensor]]` and injection through `HardwareFriendlyAttention.from_ultralytics`.

- [ ] **Step 1: Write failing true-sharing integration tests**

```python
def test_bdcn_global_bank_is_shared_by_both_yolo26m_sites() -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    config = VariantConfig(
        name="D1-SHARED",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=BDCNSharing.GLOBAL,
        bdcn_projection=BDCNProjection.FLOAT,
    )
    convert_yolo26_model(model, config)
    first = model.get_submodule("model.10.m.0.attn").normalize
    second = model.get_submodule("model.22.m.0.1.attn").normalize

    assert first.bank is second.bank
    assert first.bank.raw_ratios.data_ptr() == second.bank.raw_ratios.data_ptr()
    assert first.table_indices.tolist() == [0, 0, 0, 0]
    assert second.table_indices.tolist() == [0, 0, 0, 0]
```

Add parameterized expectations:

```python
@pytest.mark.parametrize(
    ("sharing", "first_indices", "second_indices", "tables"),
    [
        (BDCNSharing.GLOBAL, [0, 0, 0, 0], [0, 0, 0, 0], 1),
        (BDCNSharing.PER_ATTENTION, [0, 0, 0, 0], [1, 1, 1, 1], 2),
        (BDCNSharing.PER_HEAD, [0, 1, 2, 3], [4, 5, 6, 7], 8),
    ],
)
def test_bdcn_table_assignment(sharing, first_indices, second_indices, tables) -> None:
    model = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False)
    config = VariantConfig(
        name=f"test-{sharing.value}",
        normalization=NormalizationKind.BDCN,
        bdcn_codebook=BDCNCodebookKind.LEARNED,
        bdcn_sharing=sharing,
        bdcn_projection=BDCNProjection.FLOAT,
    )
    convert_yolo26_model(model, config)
    first = model.get_submodule("model.10.m.0.attn").normalize
    second = model.get_submodule("model.22.m.0.1.attn").normalize

    assert first.table_indices.tolist() == first_indices
    assert second.table_indices.tolist() == second_indices
    assert first.bank is second.bank
    assert first.bank.num_tables == tables
```

The body builds YOLO26m, converts it, then asserts both `table_indices` lists and `bank.num_tables == tables`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
../.venv/bin/pytest -q tests/test_yolo_adapter.py -k bdcn
```

Expected: failure because conversion cannot inject a BDCN bank.

- [ ] **Step 3: Build the model-level sharing context**

After validating exactly two official paths and equal `num_heads`, allocate tables as follows:

```python
if config.bdcn_sharing is BDCNSharing.GLOBAL:
    num_tables = 1
    indices = {path: torch.zeros(num_heads, dtype=torch.long) for path in paths}
elif config.bdcn_sharing is BDCNSharing.PER_ATTENTION:
    num_tables = len(paths)
    indices = {
        path: torch.full((num_heads,), site, dtype=torch.long)
        for site, path in enumerate(paths)
    }
else:
    num_tables = len(paths) * num_heads
    indices = {
        path: torch.arange(num_heads, dtype=torch.long) + site * num_heads
        for site, path in enumerate(paths)
    }
```

Create one `BDCNCodebookBank` and pass the same object to both normalizers. Registering the shared module under both normalizers is allowed; add a checkpoint round-trip test to ensure state loading preserves equal values and shared object identity in a newly converted model.

- [ ] **Step 4: Add normalizer injection to Attention**

Change constructors to accept:

```python
def __init__(
    self,
    source: Attention,
    config: VariantConfig,
    *,
    normalizer: nn.Module | None = None,
) -> None:
```

Use `self.normalize = normalizer if normalizer is not None else build_normalizer(config)`. Mirror the keyword in `from_ultralytics`. Non-BDCN conversion remains byte-for-byte behaviorally equivalent.

- [ ] **Step 5: Add codebook-only freeze scope**

In `freeze_for_stage`, support `bdcn_codebook`: freeze everything, then visit each `BDCNNormalizer` and set only `module.bank.parameters()` to trainable. Extend `TrainingRecipe` valid stages and add `configs/training/bdcn-codebook.yaml` with 5 epochs, the existing COCO path, `weights/yolo26m.pt`, seed 0, and the same optimizer/LR as normalization recovery.

- [ ] **Step 6: Verify integration and commit**

Run:

```bash
../.venv/bin/pytest -q tests/test_yolo_adapter.py tests/test_attention_integration.py tests/test_recipes_and_artifacts.py
```

Expected: all tests pass, including two-site sharing and P0 equivalence.

```bash
git add configs/training/bdcn-codebook.yaml src/yolo_attention/attention.py src/yolo_attention/integration.py src/yolo_attention/run_config.py tests/test_yolo_adapter.py tests/test_attention_integration.py tests/test_recipes_and_artifacts.py
git commit -m "feat: integrate BDCN with both YOLO26 attention sites"
```

---

### Task 5: Add hardware proxies, documentation, and full verification

**Files:**
- Modify: `src/yolo_attention/profiling.py`
- Modify: `tests/test_trainer_profiling_cli.py`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `src/README.md`
- Modify: `configs/README.md`
- Modify: `tests/README.md`

**Interfaces:**
- Consumes: final BDCN config and shape.
- Produces: `BDCNCostReport` fields suitable for JSON and the A-FINAL comparison table.

- [ ] **Step 1: Write failing operation-count tests**

For two sites with `N=400`, `H=4`, `L=16`, assert:

```python
def test_bdcn_cost_report_counts_both_attention_sites() -> None:
    report = estimate_bdcn_operations(
        tokens=400,
        heads=4,
        levels=16,
        sites=2,
        projection=BDCNProjection.TWO_POT,
    )
    assert report.bucket_lookups == 2 * 4 * 400 * 400
    assert report.histogram_bins == 2 * 4 * 400 * 16
    assert report.shift_ops == 2 * report.bucket_lookups
    assert report.add_ops == report.bucket_lookups
    assert report.total_codebook_entries == 2 * 4 * 16
```

- [ ] **Step 2: Run test and verify RED**

Run:

```bash
../.venv/bin/pytest -q tests/test_trainer_profiling_cli.py::test_bdcn_cost_report_counts_both_attention_sites
```

Expected: import/name failure for `estimate_bdcn_operations`.

- [ ] **Step 3: Implement explicit cost accounting**

Create frozen `BDCNCostReport` and compute:

- bucket lookups: `sites * heads * tokens * tokens`.
- histogram bins processed: `sites * heads * tokens * levels`.
- one-PoT: one shift per lookup, no projection add.
- two-PoT: two shifts plus one add per lookup.
- storage entries: global `levels`, per-attention `sites * levels`, per-head `sites * heads * levels`; report all three instead of assuming one sharing mode.
- reciprocal count: `sites * heads * tokens`.
- `PV` MAC unchanged from the existing report.

- [ ] **Step 4: Synchronize all maintenance documentation**

Document:

- D0/D1/D2 commands and config values.
- true parameter-sharing semantics.
- fixed 16-entry bucket table and non-learned boundaries.
- direct/histogram equivalence.
- D2-FP/1P/2P distinction.
- result fields: bucket occupancy, saturation rate, codebook values, projection error, row-sum error, KL divergence, mAP and cost report.
- BDCN does not reduce dense `PV`.
- no COCO/GPU result exists until training actually runs.

- [ ] **Step 5: Run complete verification**

Run:

```bash
../.venv/bin/pytest -q
../.venv/bin/ruff check src tests scripts
../.venv/bin/python -m yolo_attention.cli workflow --json
../.venv/bin/python -m yolo_attention.cli smoke --model yolo26m.yaml --variant configs/variants/h-screen.yaml --imgsz 64
```

Expected:

- all tests pass;
- Ruff prints `All checks passed!`;
- workflow includes D0/D1/D2 before A-FINAL and keeps Q0/Q1/Q2 optional;
- smoke reports exactly the two expected converted paths.

- [ ] **Step 6: Commit**

```bash
git add AGENTS.md README.md docs/architecture.md src/README.md configs/README.md tests/README.md src/yolo_attention/profiling.py tests/test_trainer_profiling_cli.py
git commit -m "docs: complete BDCN CPU research scaffold"
```

## Final self-review checklist

- Every accepted spec requirement maps to one task above.
- `BDCNCodebookKind`, `BDCNSharing`, and `BDCNProjection` names are consistent in all tasks.
- Global sharing uses one actual parameter object across both sites.
- No task starts GPU training, quantizes P/V/projection, or claims FPGA speed.
- D0 does not duplicate N0-LUT when formulas/configurations are identical.
- Only one D2 hardware candidate may receive recovery training.
- No placeholder implementation or undefined downstream interface remains.
