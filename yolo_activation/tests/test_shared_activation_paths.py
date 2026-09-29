from __future__ import annotations

import torch
from torch import nn

from activation_lab.training import (
    DatasetContract,
    RegionRule,
    SearchConfig,
    StaticPolicy,
    TrainingArchitecture,
    TrainingConfig,
)


class _SharedActivationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        shared = nn.SiLU()
        self.first = nn.Sequential(nn.Conv2d(2, 2, 1), shared)
        self.second = nn.Sequential(nn.Conv2d(2, 2, 1), shared)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(value))


def _architecture() -> TrainingArchitecture:
    dataset = DatasetContract(
        dataset_id="bbat5-v1-detect",
        source_kind="canonical_bbat5_v1",
        yaml_path=(
            "/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml"
        ),
        task="detect",
        num_classes=2,
    )
    return TrainingArchitecture(
        TrainingConfig(
            architecture_id="shared-activation-test",
            datasets=(dataset,),
            search=SearchConfig(max_changed_regions=2),
        )
    )


def test_inspection_and_replacement_keep_every_shared_module_path() -> None:
    architecture = _architecture()
    model = _SharedActivationModel()
    manifest = architecture.inspect(
        model,
        model_id="shared-activation",
        region_rules=(
            RegionRule(r"^first\.1$", "first"),
            RegionRule(r"^second\.1$", "second"),
        ),
    ).approve()

    assert [site.module_path for site in manifest.sites] == ["first.1", "second.1"]
    applied = architecture.apply(
        model,
        manifest,
        StaticPolicy("uniform--poly_shift", default_activation="poly_shift"),
        clone_model=False,
    )

    assert applied.changed_paths == ("first.1", "second.1")
    assert not isinstance(model.first[1], nn.SiLU)
    assert not isinstance(model.second[1], nn.SiLU)
    output = model(torch.randn(1, 2, 4, 4))
    assert torch.isfinite(output).all()
