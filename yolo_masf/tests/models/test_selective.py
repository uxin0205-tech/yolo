from __future__ import annotations

import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

from masf_yolo.artifacts.checkpoints import load_canonical_checkpoint, save_canonical_checkpoint
from masf_yolo.models.builder import P2_SLOT_INDEX, build_model
from masf_yolo.models.mfam import PartialMFAM
from masf_yolo.models.selective import (
    SP2_AUXILIARY_LOSS_WEIGHT,
    SP2_HIDDEN_CHANNELS,
    SelectiveP2Detect,
    ball_only_batch,
)
from masf_yolo.models.transfer import transfer_b1_canonical
from masf_yolo.training.runner import run_training
from masf_yolo.variants import get_variant


def _mixed_batch() -> dict[str, torch.Tensor]:
    return {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0, 0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.45, 0.45, 0.08, 0.08], [0.55, 0.55, 0.2, 0.2]]),
    }


def test_sp2_uses_a_real_single_class_lightweight_p2_head() -> None:
    b1 = build_model("B1")
    sp2 = build_model("SP2")
    head = sp2.model[-1]

    assert isinstance(head, SelectiveP2Detect)
    assert head.hidden_channels == SP2_HIDDEN_CHANNELS == 32
    assert head.auxiliary_loss_weight == SP2_AUXILIARY_LOSS_WEIGHT == 1.0
    assert head.ball_cv3[-1].out_channels == 1
    assert len(head.main_cv2) == len(head.main_cv3) == 3
    assert sp2.stride.tolist() == [4.0, 8.0, 16.0, 32.0]

    standard = b1.model[-1]
    standard_p2_params = sum(p.numel() for module in (standard.cv2[0], standard.cv3[0]) for p in module.parameters())
    selective_p2_params = sum(p.numel() for module in (head.ball_cv2, head.ball_cv3) for p in module.parameters())
    assert selective_p2_params < standard_p2_params


def test_sp2p_builds_partial_mfam_with_selective_head() -> None:
    model = build_model("SP2M3")

    assert isinstance(model.model[P2_SLOT_INDEX], PartialMFAM)
    assert model.model[P2_SLOT_INDEX].processed_channels == 32
    assert isinstance(model.model[-1], SelectiveP2Detect)
    with torch.no_grad():
        output, raw = model.eval()(torch.zeros(1, 3, 64, 64))

    assert output.shape == (1, 6, 340)
    assert set(raw) == {"ball", "main"}


def test_sp2p_loss_backward_canonical_reload_and_official_nms(tmp_path) -> None:
    variant = get_variant("SP2M3")
    source = build_model(variant).train()
    loss, components = source(_mixed_batch())
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()

    checkpoint = tmp_path / "sp2m3.pt"
    save_canonical_checkpoint(
        source,
        checkpoint,
        variant,
        data_hash="d" * 64,
        config_hash="c" * 64,
        environment_hash="e" * 64,
    )
    loaded = build_model(variant)
    manifest = load_canonical_checkpoint(loaded, checkpoint, variant)
    assert manifest.variant_id == "SP2M3"

    with torch.no_grad():
        output, _ = loaded.eval()(torch.zeros(1, 3, 64, 64))
    detections = non_max_suppression(output, conf_thres=0.25, iou_thres=0.7, nc=2)
    assert output.shape == (1, 6, 340)
    assert len(detections) == 1


def test_sp2_routes_only_ball_targets_to_p2_auxiliary_loss() -> None:
    batch = _mixed_batch()
    filtered = ball_only_batch(batch)

    assert filtered["cls"].tolist() == [[0.0]]
    assert filtered["batch_idx"].tolist() == [0]
    assert filtered["bboxes"].tolist() == [batch["bboxes"][0].tolist()]
    assert batch["cls"].tolist() == [[0.0], [1.0]]


def test_sp2_mixed_targets_produce_finite_loss_and_both_head_gradients() -> None:
    model = build_model("SP2").train()
    loss, components = model(_mixed_batch())

    assert loss.shape == components.shape == (3,)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()
    head = model.model[-1]
    assert head.ball_cv3[-1].weight.grad is not None
    assert head.main_cv3[0][-1].weight.grad is not None


def test_sp2_bat_only_batch_keeps_empty_ball_loss_finite() -> None:
    model = build_model("SP2").train()
    batch = {
        "img": torch.rand(1, 3, 64, 64),
        "batch_idx": torch.tensor([0]),
        "cls": torch.tensor([[1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
    }

    loss, components = model(batch)

    assert torch.isfinite(loss).all() and torch.isfinite(components).all()
    loss.sum().backward()


def test_sp2_decode_maps_p2_to_ball_and_never_emits_p2_bat_score() -> None:
    model = build_model("SP2").eval()
    with torch.no_grad():
        merged, raw = model(torch.zeros(1, 3, 64, 64))

    # 64px input: P2=16*16=256, P3-P5=8*8+4*4+2*2=84.
    assert merged.shape == (1, 6, 340)
    assert raw["ball"]["scores"].shape == (1, 1, 256)
    assert raw["main"]["scores"].shape == (1, 2, 84)
    assert torch.count_nonzero(merged[:, 5, :256]) == 0


def test_sp2_validation_loss_accepts_eval_decoded_raw_tuple() -> None:
    model = build_model("SP2").eval()
    batch = _mixed_batch()
    with torch.no_grad():
        predictions = model(batch["img"])
        loss, components = model.loss(batch, predictions)

    assert loss.shape == components.shape == (3,)
    assert torch.isfinite(loss).all() and torch.isfinite(components).all()


def test_sp2_one_epoch_cpu_trainer_validation_regression(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        images = dataset / split / "images"
        labels = dataset / split / "labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        Image.new("RGB", (64, 64), "black").save(images / "frame.jpg")
        (labels / "frame.txt").write_text(
            "0 0.4 0.4 0.1 0.1\n1 0.6 0.6 0.2 0.2\n",
            encoding="utf-8",
        )
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset}\ntrain: train/images\nval: val/images\nnc: 2\nnames: [ball, bat]\n",
        encoding="utf-8",
    )
    result = run_training(
        build_model("SP2"),
        {
            "model": "sp2.pt",
            "data": str(data_yaml),
            "project": str(tmp_path / "runs"),
            "name": "sp2-validation-regression",
            "epochs": 1,
            "imgsz": 64,
            "batch": 1,
            "workers": 0,
            "device": "cpu",
            "optimizer": "SGD",
            "amp": False,
            "plots": False,
            "exist_ok": True,
            "seed": 42,
        },
    )

    assert result.best.is_file()
    assert result.last.is_file()


def test_b1_transfer_maps_p3_p5_and_leaves_sp2_ball_towers_new() -> None:
    b1 = build_model("B1")
    sp2 = build_model("SP2")
    with torch.no_grad():
        for tensor in b1.state_dict().values():
            if tensor.is_floating_point():
                tensor.fill_(0.375)
    before = sp2.state_dict()["model.31.ball_cv3.0.conv.weight"].clone()

    report = transfer_b1_canonical(sp2, b1.state_dict())

    assert torch.all(sp2.state_dict()["model.31.main_cv3.0.0.0.conv.weight"] == 0.375)
    torch.testing.assert_close(sp2.state_dict()["model.31.ball_cv3.0.conv.weight"], before)
    assert "model.31.ball_cv3.0.conv.weight" in report.missing
    assert not report.shape_mismatch


def test_sp2_canonical_checkpoint_strictly_reloads(tmp_path) -> None:
    variant = get_variant("SP2")
    source = build_model(variant)
    checkpoint = tmp_path / "sp2.pt"
    save_canonical_checkpoint(
        source,
        checkpoint,
        variant,
        data_hash="d" * 64,
        config_hash="c" * 64,
        environment_hash="e" * 64,
    )
    destination = build_model(variant)

    manifest = load_canonical_checkpoint(destination, checkpoint, variant)

    assert manifest.variant_id == "SP2"
    for key, tensor in source.state_dict().items():
        torch.testing.assert_close(destination.state_dict()[key], tensor)


def test_sp2_native_checkpoint_reloads_through_official_yolo_and_nms(tmp_path) -> None:
    checkpoint = tmp_path / "sp2-native.pt"
    torch.save(
        {
            "model": build_model("SP2").half(),
            "ema": None,
            "epoch": -1,
            "train_args": {"task": "detect", "model": "sp2.pt", "imgsz": 64, "epochs": 1},
        },
        checkpoint,
    )

    loaded = YOLO(str(checkpoint), task="detect").model.float().eval()
    with torch.no_grad():
        output, _ = loaded(torch.zeros(1, 3, 64, 64))
    detections = non_max_suppression(output, conf_thres=0.25, iou_thres=0.7, nc=2)

    assert output.shape == (1, 6, 340)
    assert len(detections) == 1
