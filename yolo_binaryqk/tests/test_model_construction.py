from pathlib import Path
import torch

from binary_attention.model import build_student
from binary_attention.variants.definitions import get_variant
from binary_attention.verification.checkpoint import binary_modules, save_checkpoint, strict_reload

TEMPLATE = Path(__file__).parents[1] / "configs/yolo11m-binaryqk-template.yaml"

def test_variant_yaml_constructs_binary_modules_before_forward():
    model = build_student(TEMPLATE, get_variant("T2"))
    modules = binary_modules(model)
    assert modules and all(type(m).__name__ == "UltralyticsBinaryAttention" for _, m in modules)
    # A real model forward is intentionally exercised at low resolution only.
    model.eval(); model(torch.zeros(1, 3, 64, 64))
    assert sum(int(m.binary_forward_count) for _, m in modules) > 0

def test_model_checkpoint_rejects_variant_mismatch(tmp_path):
    variant = get_variant("T2"); first = build_student(TEMPLATE, variant)
    checkpoint = tmp_path / "model.pt"; save_checkpoint(checkpoint, first, variant, model_yaml=TEMPLATE)
    restored = build_student(TEMPLATE, variant); strict_reload(checkpoint, restored, variant)
    try: strict_reload(checkpoint, build_student(TEMPLATE, get_variant("T1")), get_variant("T1"))
    except RuntimeError: pass
    else: raise AssertionError("variant mismatch must not be reloadable")


def test_e2_dual_constructs_matched_basis_without_training():
    variant = get_variant("E2-DUAL")
    model = build_student(TEMPLATE, variant).eval()
    modules = binary_modules(model)
    assert modules and all(type(module).__name__ == "ResidualDualMatchedBasisAttention" for _, module in modules)
    before = [(int(module.binary_qk_count), int(module.softmax_count), int(module.pv_count)) for _, module in modules]
    model(torch.zeros(1, 3, 64, 64))
    after = [(int(module.binary_qk_count), int(module.softmax_count), int(module.pv_count)) for _, module in modules]
    assert all(tuple(end - start for start, end in zip(left, right)) == (2, 1, 1) for left, right in zip(before, after))
