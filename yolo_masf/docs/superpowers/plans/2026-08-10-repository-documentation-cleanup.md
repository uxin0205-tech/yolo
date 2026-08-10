# Repository 文件整理與實驗產物清理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立分層中文 README、安全且可稽核的 artifact cleanup 工具，並清除 smoke/preflight/gate checkpoints 與明確垃圾，同時保證所有正式權重與研究證據的 SHA-256 不變。

**Architecture:** `masf_yolo.cleanup` 集中定義 fail-closed gate、刪除白名單、永久保留規則、dry-run plan 與原子 manifest；CLI 只負責載入 config 並呼叫此深層模組。文件採 root overview＋重要子目錄 local README；生成型 leaf directory 不重複放文件。實際刪除只在單元測試、dry-run、正式權重 hash snapshot 與人工清單核對後執行。

**Tech Stack:** Python 3.12、pytest、pathlib、hashlib、既有 `atomic_write_json`、Markdown、systemd user service read-only status。

## Global Constraints

- 不啟動 GPU、不重新訓練、不修改 dataset 或來源權重。
- 保留所有正式 `best.pt`、`last.pt`、formal/staged canonical checkpoints、evaluation、profiles、selection、final audit 與 reports。
- 只刪除設計規格列出的 exact whitelist；所有 resolved paths 必須在 `/home/uxin/yolo/yolo_masf` 內。
- `yolo26n.pt` 的引用掃描只看 runtime-relevant Python/YAML/TOML/shell files，排除 Markdown 與 `artifacts/`。
- Git author identity 不得由 agent 設定；commit 失敗時只取消本 task 的 staged files，保留工作樹內容並回報。
- 既有使用者修改與 parent repo 其他未追蹤目錄不得更動。

---

### Task 1: 以 TDD 建立 fail-closed cleanup domain module

**Files:**
- Create: `masf_yolo/cleanup.py`
- Create: `tests/test_cleanup.py`
- Modify: `masf_yolo/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `CleanupTarget(relative_path: str, category: str, reason: str, size_bytes: int, sha256: str, status: str)`。
- Produces: `CleanupPlan(repo_root: Path, artifact_root: Path, pipeline_id: str, targets: tuple[CleanupTarget, ...], preserved_checkpoint_hashes: dict[str, str])`。
- Produces: `build_cleanup_plan(repo_root: Path, artifact_root: Path, *, service_active: Callable[[str], bool]) -> CleanupPlan`。
- Produces: `apply_cleanup(plan: CleanupPlan, manifest_path: Path) -> dict[str, object]`。
- Produces CLI：`python -m masf_yolo.cli cleanup --config configs/static-phase1.yaml [--apply]`；預設 dry-run。

- [ ] **Step 1: 寫 gate 與 whitelist 的失敗測試**

在 `tests/test_cleanup.py` 建立最小 artifact tree，包含 report completed、final audit PASS、pipeline metadata、正式 best/last/canonical、smoke/preflight/gate checkpoint、cache、preview image、evaluation metrics 與 symlink escape。測試必須先證明以下尚未實作：

```python
def test_cleanup_plan_selects_only_disposable_artifacts(tmp_path: Path) -> None:
    plan = build_cleanup_plan(tmp_path, tmp_path / "artifacts/static-phase1", service_active=lambda _: False)
    selected = {target.relative_path for target in plan.targets}
    assert "artifacts/static-phase1/smoke_runs/m2-smoke/weights/best.pt" in selected
    assert "artifacts/static-phase1/preflight/m2.pt" in selected
    assert "artifacts/static-phase1/runs/m2/weights/best.pt" not in selected
    assert "artifacts/static-phase1/training/formal_m2/canonical.pt" not in selected
```

另加 active service、report 未完成、audit failure、symlink escape 與 preserved/whitelist conflict 的 fail-closed tests。

- [ ] **Step 2: 執行 RED**

Run: `../.venv/bin/python -m pytest tests/test_cleanup.py -q`

Expected: FAIL，原因為 `masf_yolo.cleanup` 或所需 interfaces 尚不存在。

- [ ] **Step 3: 實作最小 cleanup plan builder**

在 `masf_yolo/cleanup.py`：

- 使用固定 pattern table，逐一 `glob` 後 resolve。
- 拒絕 resolved path 不在 repo root、symlink target 在 repo 外或與 preserve predicate 相符。
- 讀取 `pipeline.json`、`stages/report.json`、`final_audit.json`。
- 透過 injected `service_active` 查 `pipeline.json["unit"]`。
- 對 target 與 preserved checkpoint 以 chunked SHA-256 計算 hash。
- runtime reference scan 只讀 `*.py`、`*.yaml`、`*.yml`、`*.toml`、`*.sh`，排除 `artifacts/`。

- [ ] **Step 4: 執行 cleanup tests，確認 GREEN**

Run: `../.venv/bin/python -m pytest tests/test_cleanup.py -q`

Expected: 全部 PASS。

- [ ] **Step 5: 為 apply 與 manifest 寫第二輪 RED tests**

測試 dry-run 不寫 manifest、不刪檔；apply 先寫 planned manifest，再逐檔 unlink 並更新 status；正式 checkpoint hash 保持不變；第二次 apply 對已刪項目為 idempotent。

- [ ] **Step 6: 實作 apply_cleanup**

使用既有 `atomic_write_json`。manifest 至少包含：

```python
{
    "schema_version": 1,
    "pipeline_id": plan.pipeline_id,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "total_planned_bytes": sum(target.size_bytes for target in plan.targets),
    "total_deleted_bytes": 0,
    "preserved_checkpoint_hashes": plan.preserved_checkpoint_hashes,
    "targets": [asdict(target) for target in plan.targets],
}
```

每刪除一檔後更新對應 status 與 `total_deleted_bytes`；只移除已清空且位於 whitelist 下的 cache/weights directories。

- [ ] **Step 7: 加入 CLI contract 與 tests**

在 `build_parser()` 加入 `cleanup --config ... --apply`；`main()` 將 config resolve 成 repo/artifact root，dry-run 印 JSON，`--apply` 執行後印 manifest summary。更新 `test_cli_exposes_only_documented_operator_commands` 並新增預設 dry-run assertion。

- [ ] **Step 8: 執行 Task 1 regression**

Run: `../.venv/bin/python -m pytest tests/test_cleanup.py tests/test_cli.py -q`

Expected: 全部 PASS。

- [ ] **Step 9: 嘗試提交 Task 1**

```bash
git add yolo_masf/masf_yolo/cleanup.py yolo_masf/masf_yolo/cli.py yolo_masf/tests/test_cleanup.py yolo_masf/tests/test_cli.py
git commit -m "feat: add audited experiment cleanup"
```

若缺少 Git author identity，執行精確 `git restore --staged`，不設定 identity。

---

### Task 2: 重建 root 與 top-level 導覽文件

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `bbt5-detect-baseline/README.md`
- Create: `configs/README.md`
- Create: `artifacts/README.md`
- Create: `artifacts/static-phase1/README.md`
- Create: `docs/README.md`
- Create: `field_check/README.md`

**Interfaces:**
- Consumes: `EXPERIMENT_RESULTS_ZH.md`、正式 artifact tree、approved design。
- Produces: 從 root 到資料、程式、設定、結果、權重與操作文件的單一路徑導覽。

- [ ] **Step 1: 重寫 root README**

以中文加入：研究範圍、十個模型差異、dataset/initializer provenance、訓練預算、完整 pipeline、結果摘要、主要命令、資料夾樹、正式權重與 metrics 路徑、data-exposure 與 single-seed 限制、清理後 archive 狀態。

- [ ] **Step 2: 整理 AGENTS.md**

保留單一 `## Agent skills` block；Issue tracker 與 Domain docs 各出現一次，指向已確認的 `docs/agents/*.md`。

- [ ] **Step 3: 建立 top-level local READMEs**

每份文件只描述該目錄的責任、輸入、輸出、重要檔案、不可修改內容與上/下游，不複製完整 root README。`artifacts/static-phase1/README.md` 必須含十模型對照：

```text
模型 → runs/<name>/weights → training/<stage>/canonical.pt
     → evaluation/{val,test}/<name>/metrics.json
     → profiles/<name>/profile.json
```

並標明 smoke/preflight/gate checkpoints 已由 cleanup manifest 記錄後刪除。

- [ ] **Step 4: 執行 Markdown 靜態檢查**

Run: `rg -n '[[:blank:]]+$' README.md AGENTS.md configs/README.md artifacts/README.md artifacts/static-phase1/README.md docs/README.md field_check/README.md bbt5-detect-baseline/README.md`

Expected: 無輸出。

- [ ] **Step 5: 嘗試提交 Task 2**

只 stage 本 task 文件；若 Git identity 缺失，精確取消暫存。

---

### Task 3: 標準化 Python package 與 test documentation

**Files:**
- Create: `masf_yolo/README.md`
- Delete: `masf_yolo/Readme.md`
- Create: `masf_yolo/artifacts/README.md`
- Create: `masf_yolo/data/README.md`
- Create: `masf_yolo/evaluation/README.md`
- Create: `masf_yolo/models/README.md`
- Create: `masf_yolo/training/README.md`
- Create: `tests/README.md`

**Interfaces:**
- Consumes: 各 package 的 public modules 與 tests tree。
- Produces: 就地說明「這層做什麼、怎麼使用、依賴什麼、輸出到哪裡」。

- [ ] **Step 1: 將 `Readme.md` 內容拆成 package overview**

建立標準大小寫 `masf_yolo/README.md`，保留仍正確的 module map、variant definitions、pipeline data flow 與 commands；移除過時的未完成狀態。刪除舊 `Readme.md`，避免 Linux 上同名混淆。

- [ ] **Step 2: 建立五個 subpackage README**

每份列出本目錄 Python files、主要 interfaces、資料流與 safety invariants。`training/README.md` 說明正式 last.pt 保留而 smoke checkpoints 已清理；`artifacts/README.md` 區分 package code 與 root runtime artifacts。

- [ ] **Step 3: 建立 tests README**

說明 test tree 對應 package、純 CPU suite、需 CUDA 的 integration boundaries、執行指令與不應使用正式 artifact 作 mutable fixture 的原則。

- [ ] **Step 4: 驗證 package 文件與實際檔案一致**

Run: `rg --files masf_yolo tests | sort`

逐項核對 README 未列出不存在檔案，也未遺漏 `selective.py`、`cleanup.py` 等重要模組。

- [ ] **Step 5: 嘗試提交 Task 3**

只 stage 本 task 文件；若 Git identity 缺失，精確取消暫存。

---

### Task 4: Dry-run、實際清理與不可變證據驗證

**Files:**
- Create at runtime: `artifacts/static-phase1/cleanup_manifest.json`
- Delete only: approved cleanup target files recorded in the manifest
- Modify: `codex_plan.md`

**Interfaces:**
- Consumes: Task 1 cleanup CLI、已完成 pipeline、正式 checkpoint tree。
- Produces: 已清理 repo、manifest、前後容量與 hash verification evidence。

- [ ] **Step 1: 確認 terminal gates**

Run:

```bash
systemctl --user is-active masf-yolo-phase1.service
../.venv/bin/python -m masf_yolo.cli pipeline status --config configs/static-phase1.yaml
```

Expected: service inactive；state 為 `report/completed`；`final_audit.json` 為 PASS。

- [ ] **Step 2: 記錄清理前容量與正式 checkpoint hashes**

將正式 preserve patterns 的 SHA-256 記錄到 `/tmp/masf-formal-checkpoints-before.sha256`，並保存 `du -sb` 總容量。不得把 `/tmp` snapshot 當最終證據；apply manifest 內也要保存同一份 hashes。

- [ ] **Step 3: 執行 dry-run**

Run: `../.venv/bin/python -m masf_yolo.cli cleanup --config configs/static-phase1.yaml`

核對所有 target 都屬於設計白名單，且輸出不含任何 formal best/last/canonical、evaluation、profile、dataset 或 source weight。

- [ ] **Step 4: 執行 apply**

Run: `../.venv/bin/python -m masf_yolo.cli cleanup --config configs/static-phase1.yaml --apply`

Expected: manifest 寫入後只刪除 planned targets；輸出 deleted count 與 bytes。

- [ ] **Step 5: 比對清理後正式 hashes**

重新產生 `/tmp/masf-formal-checkpoints-after.sha256`，以 `diff -u` 比較。Expected: 無差異。

- [ ] **Step 6: 驗證 preserve artifacts 與 link targets**

以 Python read-only checker：

- 解析所有 repo-owned README 的 relative Markdown links，忽略 HTTP(S) 與 code spans，確認 local targets 存在。
- 確認十模型正式 weights、metrics、profiles、selection、final audit、中文結果報告都存在。
- 確認 cleanup manifest 的 deleted targets 不存在、preserved hashes 與實際 hashes 相同。

- [ ] **Step 7: 更新 codex_plan 完成狀態**

只更新已由 artifact 證明完成的 GPU/checkpoint/evaluation/final audit tasks，記錄 cleanup manifest 與節省容量；不得刪除歷史決策或改寫結果。

- [ ] **Step 8: 執行完整 verification**

Run:

```bash
../.venv/bin/python -m pytest -q
../.venv/bin/python -m compileall -q masf_yolo tests
git diff --check
```

Expected: pytest 0 failures、compileall exit 0、diff check 無輸出。

- [ ] **Step 9: 回報結果並嘗試提交**

回報：清理前後 bytes、刪除檔案數、節省容量、正式 hash 比對、README 清單、tests 結果、pipeline/final audit 狀態。只 stage 本次 repo 文件、cleanup code/tests 與 manifest；不加入 dataset、正式 weights 或其他大型 artifacts。Git identity 缺失時精確取消暫存。
