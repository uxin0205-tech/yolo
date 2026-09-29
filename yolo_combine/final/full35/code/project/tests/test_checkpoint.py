from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from yolo_combine.checkpoint import load_checkpoint, save_checkpoint
from yolo_combine.source import SourceBundle


class TinyContractModel(nn.Module):
    def __init__(self, model_kind: str = "tiny") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2, 3))
        self.model_kind = model_kind

    def contract(self) -> dict[str, object]:
        return {"model_kind": self.model_kind, "shape": [2, 3]}


def _source_bundle(tmp_path: Path) -> SourceBundle:
    checkpoint = tmp_path / "bundle" / "weights" / "float" / "full35-a2.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"immutable-source-fixture")
    return SourceBundle(tmp_path / "bundle")


def test_state_dict_checkpoint_round_trip_is_contract_checked(tmp_path):
    source = _source_bundle(tmp_path)
    model = TinyContractModel()
    target = tmp_path / "combined.pt"

    saved = save_checkpoint(
        target,
        model,
        source,
        training={"epoch": 3, "ratio": "2:1"},
    )
    with torch.no_grad():
        model.weight.zero_()
    metadata = load_checkpoint(saved, model)

    assert torch.equal(model.weight, torch.ones_like(model.weight))
    assert metadata["training"] == {"epoch": 3, "ratio": "2:1"}
    assert metadata["source"]["architecture"] == "full35"
    assert not target.with_name(f".{target.name}.tmp").exists()
    payload = torch.load(saved, map_location="cpu", weights_only=True)
    assert "state_dict" in payload
    assert "model" not in payload


def test_checkpoint_rejects_different_model_contract(tmp_path):
    source = _source_bundle(tmp_path)
    target = save_checkpoint(tmp_path / "combined.pt", TinyContractModel(), source)

    with pytest.raises(ValueError, match="contract mismatch"):
        load_checkpoint(target, TinyContractModel(model_kind="different"))


def test_checkpoint_requires_a_contract(tmp_path):
    source = _source_bundle(tmp_path)

    with pytest.raises(TypeError, match="contract"):
        save_checkpoint(tmp_path / "invalid.pt", nn.Linear(2, 2), source)


@pytest.mark.integration
def test_real_full35_shared_checkpoint_round_trip(
    tmp_path,
    source_bundle,
    shared_model,
):
    expected = {
        name: tensor.detach().cpu().clone()
        for name, tensor in shared_model.state_dict().items()
    }
    checkpoint = save_checkpoint(
        tmp_path / "full35-shared.pt",
        shared_model,
        source_bundle,
        training={"stage": "interface-regression", "ratio": "1:1"},
    )
    rebuilt = source_bundle.build_shared()
    metadata = load_checkpoint(checkpoint, rebuilt)

    assert metadata["contract"]["model_kind"] == "shared_dual_head"
    assert metadata["training"]["ratio"] == "1:1"
    actual = rebuilt.state_dict()
    assert actual.keys() == expected.keys()
    for name, tensor in actual.items():
        assert torch.equal(tensor.detach().cpu(), expected[name]), name
