from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hardswish_static_analysis_is_complete_and_hash_bound() -> None:
    plan = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/full35-quantization-plan-v3.yaml"
        ).read_text(encoding="utf-8")
    )
    revision = plan["history_and_revision"]["replacement_parent_analysis"]
    view_path = PROJECT_ROOT / revision["view_manifest"]
    analysis_path = PROJECT_ROOT / revision["analysis"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))

    assert _sha256(view_path) == revision["view_manifest_sha256"]
    assert _sha256(analysis_path) == revision["analysis_sha256"]
    assert analysis["gpu_used"] is False
    assert analysis["parent"]["activation"] == "hardswish"
    assert analysis["parent"]["checkpoint_sha256"] == (
        "79e0e4f615a7d8b82da4fd244165d2c035d7a523681833392d30b66e57177731"
    )
    assert len(analysis["measurements"]) == revision["measurements"] == 2960
    assert Counter(item["family"] for item in analysis["measurements"]) == {
        "uniform": 1480,
        "fixed_sd4": 1184,
        "paper_twn": 296,
    }


def test_q3_regional_hardswish_parity_manifests_are_hash_bound() -> None:
    plan = yaml.safe_load(
        (
            PROJECT_ROOT / "configs/experiments/full35-quantization-plan-v3.yaml"
        ).read_text(encoding="utf-8")
    )
    manifests = plan["activation_policies"]["q3_regional_hardswish"][
        "cpu_parity_manifests"
    ]

    for region, expected in manifests.items():
        path = PROJECT_ROOT / expected["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert _sha256(path) == expected["sha256"]
        assert payload["parent"]["region_assignments"] == [
            {"region": region, "activation": "hardswish"}
        ]
        assert payload["parent"]["activation_counts"] == expected["activation_counts"]
        assert sum(payload["parent"]["activation_counts"].values()) == 190
        assert payload["parity"]["forward_parity_passed"] is True
        assert payload["parity"]["deployment_path_parity"] is True
        assert payload["parity"]["inference_contract_parity"] is True


def test_sd4_v2_uses_active_parents_and_preserves_historical_v1() -> None:
    active = yaml.safe_load(
        (
            PROJECT_ROOT / "artifacts/manifests/fixed-sd4-routing-candidates-v2.yaml"
        ).read_text(encoding="utf-8")
    )
    historical = yaml.safe_load(
        (
            PROJECT_ROOT / "artifacts/manifests/fixed-sd4-routing-candidates-v1.yaml"
        ).read_text(encoding="utf-8")
    )

    assert active["comparison"]["required_parents"] == [
        "qsilu_pq",
        "hardswish",
        "poly_shift",
    ]
    assert historical["comparison"]["required_parents"] == [
        "qsilu_pq",
        "poly_quality",
        "poly_shift",
    ]
    assert (
        len(active["stable_candidates"])
        == len(set(active["stable_candidates"]))
        == active["counts"]["both_view_cross_parent_winners"]
        == 34
    )
    for source in active["source_artifacts"]:
        assert _sha256(PROJECT_ROOT / source["path"]) == source["sha256"]
