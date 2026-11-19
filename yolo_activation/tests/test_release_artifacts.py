import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_final_results_keep_completed_and_provisional_states_distinct() -> None:
    results = json.loads(
        (ROOT / "reports/full35-activation-results.json").read_text(encoding="utf-8")
    )
    recoveries = {
        row["id"]: row for row in results["static_queue"]["completed_results"]
    }
    assert recoveries["short-recovery-uniform-qsilu-pq"]["passed"] is True
    assert recoveries["short-recovery-uniform-hardswish"]["passed"] is False
    assert recoveries["short-recovery-uniform-poly-shift"]["passed"] is False
    assert recoveries["short-recovery-uniform-poly-quality"]["passed"] is False

    provisional = results["finalist_queue"]["provisional_stopped_results"]
    assert len(provisional) == 1
    assert provisional[0]["completion_marker"] is False
    assert provisional[0]["publish_as_final"] is False


def test_csv_contains_all_selector_phases() -> None:
    with (ROOT / "reports/full35-activation-results.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["phase"] for row in rows} == {
        "baseline",
        "uniform_zero_shot",
        "short_recovery",
        "finalist_seed1",
    }


def test_weight_manifest_stays_below_github_blob_limit() -> None:
    manifest = json.loads(
        (ROOT / "release/weights/weights.json").read_text(encoding="utf-8")
    )
    assert set(manifest["weights"]) == {
        "accepted-silu-best-joint",
        "qsilu-pq-short-recovery-best-joint",
    }
    for entry in manifest["weights"].values():
        assert sum(part["bytes"] for part in entry["parts"]) == entry["bytes"]
        for part in entry["parts"]:
            assert part["bytes"] < 100_000_000
            assert (ROOT / "release/weights" / part["path"]).stat().st_size == part[
                "bytes"
            ]
