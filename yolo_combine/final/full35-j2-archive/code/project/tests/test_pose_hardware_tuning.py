import pytest
import torch

from yolo_combine.baselines import _PoseHardwareGuard
from yolo_combine.source import SourceBundle


@pytest.mark.integration
def test_pose_guard_locks_bittrue_state_but_allows_masf_and_attention_tuning(
    source_bundle: SourceBundle,
) -> None:
    model, _ = source_bundle.build_pose_model()
    guard = _PoseHardwareGuard.capture(model)
    parameters = dict(model.named_parameters())

    masf = [value for name, value in parameters.items() if ".p3_masf." in name]
    attention_tunable = [
        value
        for name, value in parameters.items()
        if ".attn." in name
        and ".qkv.q." not in name
        and ".qkv.k." not in name
        and not name.endswith("score.gamma")
    ]
    immutable = [
        value
        for name, value in parameters.items()
        if ".attn.qkv.q." in name
        or ".attn.qkv.k." in name
        or name.endswith("score.gamma")
    ]
    assert masf and all(value.requires_grad for value in masf)
    assert attention_tunable and any(value.requires_grad for value in attention_tunable)
    assert immutable and all(not value.requires_grad for value in immutable)

    with torch.no_grad():
        masf[0].add_(0.001)
    guard.assert_unchanged(model)

    target_name = next(
        name for name in guard.paths if name.endswith("normalize.values")
    )
    with torch.no_grad():
        dict(model.named_buffers())[target_name].add_(0.001)
    with pytest.raises(AssertionError, match="hardware-contract state changed"):
        guard.assert_unchanged(model)

