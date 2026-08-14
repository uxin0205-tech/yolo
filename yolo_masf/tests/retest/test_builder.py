from pathlib import Path

import torch

from masf_yolo.retest.builder import build_retest_model


def test_p2_and_p3_retest_builders_keep_their_intended_detection_scales():
    p2 = build_retest_model("P2", "PaperFormula-Full")
    p3 = build_retest_model("P3")
    assert p2.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    assert p3.stride.tolist() == [8.0, 16.0, 32.0]
    assert p2.model[20].__class__.__name__ == "PaperFormulaMFAM"
    assert p3.model[16].__class__.__name__ == "C3k2"


def test_b0_fair_builder_keeps_the_unmodified_three_scale_graph():
    model = build_retest_model("B0")
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert model.masf_variant == "P3-Base-Original"
    assert model.model[16].__class__.__name__ == "C3k2"


def test_p3_variant_has_formula_module_only_at_p3():
    model = build_retest_model("P3", "Partial50-35")
    assert model.model[16].__class__.__name__ == "Sequential"
    assert model.model[16][1].__class__.__name__ == "PartialPaperFormulaMFAM"
    assert isinstance(model.model[19], torch.nn.Module)


def test_p2_source_initializer_has_b1r_transfer_contract():
    model = build_retest_model(
        "P2",
        "PaperFormula-Full",
        source_weights=Path("bbt5-detect-baseline/weights/yolo11m_bat_detect_init.pt"),
    )
    assert model.masf_transfer_report["matched"]
    assert not model.masf_transfer_report["shape_mismatch"]
    assert not model.masf_transfer_report["unexpected"]
