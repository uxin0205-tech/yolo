from __future__ import annotations

from types import SimpleNamespace

from torch import nn

from yolo_quantize.qat_validation import QATJointValidatorAdapter


class _FakeValidator:
    def __init__(self, source, **kwargs) -> None:
        self.source = source
        self.kwargs = kwargs

    def validate(self, model, *, epoch: int, kind: str):
        return (self.source, model, epoch, kind, self.kwargs)


def test_qat_validator_materializes_each_backend_at_validation_boundary() -> None:
    calls: list[nn.Module] = []

    def build_view(**kwargs):
        calls.append(kwargs["shared_ema"])
        return SimpleNamespace(model=nn.Identity(), source="prepared-source")

    adapter = QATJointValidatorAdapter(
        "base-source",
        activation_policy=object(),
        weight_policy=object(),
        expected_catalog={},
        upstream_validator_cls=_FakeValidator,
        view_builder=build_view,
        detect_data_yaml="detect.yaml",
        pose_data_yaml="pose.yaml",
        output_root="output",
        settings="settings",
    )
    ema = nn.Linear(2, 2)

    results = adapter.validate_backends(ema, epoch=3, kinds=("bittrue",))

    assert tuple(results) == ("bittrue",)
    source, model, epoch, kind, kwargs = results["bittrue"]
    assert source == "prepared-source"
    assert isinstance(model, nn.Identity)
    assert (epoch, kind) == (3, "bittrue")
    assert kwargs["detect_data_yaml"] == "detect.yaml"
    assert calls == [ema]
