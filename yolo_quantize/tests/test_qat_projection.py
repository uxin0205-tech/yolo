import pytest
import torch

from yolo_quantize.qat_projection import replay_parent_state
from yolo_quantize.qat_weights import TrainableWeightFakeQuantizer
from yolo_quantize.weight_quantization import UniformWeightSpec


def test_replay_uses_ema_scale_and_full_blend_not_export_weights():
    q = TrainableWeightFakeQuantizer(
        UniformWeightSpec(4, "max"), initial_scales=torch.tensor([0.25])
    )
    q.set_blend_ratio(1)
    weight = torch.tensor([[0.3, -0.7]])
    ema = {
        "a.weight": weight,
        "a.bias": torch.tensor([2.0]),
        "a.weight_quantizer._scale_unconstrained": q.scale_parameter.detach(),
    }
    export = {"a.weight": torch.zeros_like(weight), "a.bias": torch.tensor([0.0])}
    state = replay_parent_state(
        export, ema, [{"path": "a", "format_id": "w4", "encoded_bits": 4}]
    )
    torch.testing.assert_close(state["a.weight"], q(weight), rtol=0, atol=0)
    assert not torch.equal(state["a.weight"], export["a.weight"])
    assert state["a.bias"].item() == 2


def test_unknown_format_fails_closed():
    with pytest.raises(ValueError, match="unsupported"):
        replay_parent_state({}, {}, [{"path": "a", "format_id": "unknown"}])
