import importlib
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "field,value",
    [
        ("checkpoint_sha256", "wrong"),
        ("locked_qat_parent", {}),
        ("activation_recalibrated", True),
        ("activation_quantizers", 123),
    ],
)
def test_build_mismatch_rejected(monkeypatch, field, value):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    verify = importlib.import_module("audit_continuous_evidence").verify_build
    parent = {"path": "parent", "sha256": "parent-hash"}
    build = {
        "locked_qat_parent": parent,
        "checkpoint_sha256": "export-hash",
        "activation_recalibrated": False,
        "activation_quantizers": 124,
    }
    verify(build, parent, {"sha256": "export-hash"})
    build[field] = value
    with pytest.raises(ValueError):
        verify(build, parent, {"sha256": "export-hash"})
