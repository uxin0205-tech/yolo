"""Isolated workers for each resumable P2 study stage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from p2_study import ARTIFACTS, ROOT
from p2_study.models import A2_HEAD_FREEZE, build_model, prepare_initial_weights
from ultralytics import YOLO, __version__
from ultralytics.data.augment import LetterBox
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import get_flops


def load_config(path: str | Path) -> dict:
    """Load and resolve a study configuration."""
    config_path = (ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    data = yaml.safe_load(config_path.read_text())
    data["config_path"] = str(config_path)
    data["study"]["dataset"] = str((ROOT / data["study"]["dataset"]).resolve())
    data["study"]["dataset_root"] = str((ROOT / data["study"]["dataset_root"]).resolve())
    data["study"]["annotation_root"] = str((ROOT / data["study"]["annotation_root"]).resolve())
    data["study"]["pretrained"] = str((ROOT / data["study"]["pretrained"]).resolve())
    return data


def write_json(path: Path, data: dict | list) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def initial_checkpoint(experiment: str, config: dict) -> Path:
    """Return the immutable initial checkpoint for an experiment."""
    if experiment == "A0":
        return Path(config["study"]["pretrained"]).resolve()
    if experiment == "A2":
        experiment = "A1"
    return ARTIFACTS / "weights" / f"{experiment.lower()}_initial.pt"


def preflight(config: dict) -> None:
    """Validate code, hardware, dataset, evaluator, weights, and disk before GPU work."""
    from importlib.metadata import PackageNotFoundError, version

    study = config["study"]
    dataset = Path(study["dataset_root"])
    train = dataset / "images/train2017"
    val = dataset / "images/val2017"
    annotations = Path(study["annotation_root"])
    checks = {
        "root": str(ROOT),
        "ultralytics_file": str(Path(__import__("ultralytics").__file__).resolve()),
        "ultralytics_version": __version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "pretrained": str(Path(study["pretrained"]).resolve()),
    }
    if ROOT not in Path(checks["ultralytics_file"]).parents:
        raise RuntimeError(f"Editable ultralytics does not point to this worktree: {checks['ultralytics_file']}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    checks["gpu"] = torch.cuda.get_device_name(int(study["device"]))
    checks["gpu_memory_bytes"] = torch.cuda.get_device_properties(int(study["device"])).total_memory
    if study["expected_gpu"] not in checks["gpu"]:
        raise RuntimeError(f"Expected {study['expected_gpu']}, found {checks['gpu']}")

    checks["train_images"] = sum(1 for path in train.iterdir() if path.is_file())
    checks["val_images"] = sum(1 for path in val.iterdir() if path.is_file())
    if checks["train_images"] != study["expected_train_images"] or checks["val_images"] != study["expected_val_images"]:
        raise RuntimeError(f"COCO image counts differ: {checks['train_images']=}, {checks['val_images']=}")
    for name in ("instances_train2017.json", "instances_val2017.json"):
        if not (annotations / name).is_file():
            raise FileNotFoundError(f"Required COCO annotation is missing: {annotations / name}")
    annotation = json.loads((annotations / "instances_val2017.json").read_text())
    checks["classes"] = len(annotation["categories"])
    if checks["classes"] != study["expected_classes"]:
        raise RuntimeError(f"Expected {study['expected_classes']} classes, found {checks['classes']}")

    usage = shutil.disk_usage(ROOT)
    checks["disk_free_bytes"] = usage.free
    if usage.free < study["minimum_free_gb"] * 1_000_000_000:
        raise RuntimeError(f"Less than {study['minimum_free_gb']} GB free")
    if not Path(study["pretrained"]).is_file():
        raise FileNotFoundError(f"Pretrained checkpoint is missing: {study['pretrained']}")
    try:
        checks["faster_coco_eval"] = version("faster-coco-eval")
    except PackageNotFoundError as error:
        raise RuntimeError("Install faster-coco-eval>=1.6.7 in the study environment") from error
    write_json(ARTIFACTS / "preflight.json", checks)


def static_tests(config: dict) -> None:
    """Run focused tests, model accounting, and a synthetic detection loss/backward for every model."""
    subprocess.run([sys.executable, "-m", "pytest", "tests/test_p2_study.py", "-q"], cwd=ROOT, check=True)
    info = {}
    for experiment in ("A0", "A1"):
        yolo = build_model(experiment)
        model = yolo.model
        detect = model.model[-1]
        model.train()
        model.args = DEFAULT_CFG
        batch = {
            "img": torch.rand(1, 3, 128, 128),
            "batch_idx": torch.zeros(0),
            "cls": torch.zeros((0, 1)),
            "bboxes": torch.zeros((0, 4)),
        }
        loss, items = model.loss(batch)
        loss.sum().backward()
        if not torch.isfinite(loss).all():
            raise RuntimeError(f"{experiment} synthetic loss is not finite")
        p2_gradients = [
            parameter.grad
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and (experiment == "A0" or name.startswith(("model.23", "model.24", "model.25")))
        ]
        if experiment != "A0" and not any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in p2_gradients):
            raise RuntimeError(f"{experiment} P2 branch has no finite non-zero gradient")
        info[experiment] = {
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "gflops": get_flops(model, imgsz=config["study"]["imgsz"]),
            "detect_nl": detect.nl,
            "stride": detect.stride.tolist(),
            "synthetic_loss": float(loss.detach().sum()),
            "loss_items": [float(value) for value in items.detach().flatten()],
        }
    info["A2"] = {**info["A1"], "training_strategy": "staged"}
    write_json(ARTIFACTS / "model_info.json", info)


def prepare(config: dict) -> None:
    """Create and verify the immutable A1 initial weights."""
    paths = prepare_initial_weights(config["study"]["pretrained"], ARTIFACTS / "weights")
    write_json(ARTIFACTS / "weights" / "initial_checkpoints.json", {key: str(value) for key, value in paths.items()})


def autobatch(config: dict) -> None:
    """Calibrate one global batch on A1 using the requested VRAM fraction."""
    study = config["study"]
    model = YOLO(initial_checkpoint("A1", config)).model.cuda(int(study["device"]))
    torch.backends.cudnn.benchmark = False
    batch = int(
        check_train_batch_size(
            model,
            imgsz=study["imgsz"],
            amp=True,
            batch=float(study["vram_fraction"]),
            dataset_size=study["expected_train_images"],
        )
    )
    if batch < 1:
        raise RuntimeError(f"AutoBatch returned invalid batch={batch}")
    write_json(
        ARTIFACTS / "batch_manifest.json",
        {"batch": batch, "calibrated_on": "A1", "imgsz": study["imgsz"], "vram_fraction": study["vram_fraction"]},
    )


def _validate_training_outputs(run_dir: Path) -> None:
    """Require finite CSV metrics plus last/best checkpoints."""
    csv_path = run_dir / "results.csv"
    last = run_dir / "weights/last.pt"
    best = run_dir / "weights/best.pt"
    if not csv_path.is_file() or not last.is_file() or not best.is_file():
        raise RuntimeError(f"Incomplete training outputs in {run_dir}")
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise RuntimeError(f"No epochs recorded in {csv_path}")
    for row in rows:
        for value in row.values():
            if value and not math.isfinite(float(value)):
                raise RuntimeError(f"NaN/Inf metric in {csv_path}")


def train(config: dict, experiment: str, phase: str, resume: str | None = None) -> None:
    """Train one experiment phase with the globally fixed arguments."""
    study = config["study"]
    batch = json.loads((ARTIFACTS / "batch_manifest.json").read_text())["batch"]
    epochs = {"gate1": 1, "health": study["health_epochs"], "formal": study["formal_epochs"]}[phase]
    fraction = study["gate_fraction"] if phase == "gate1" else 1.0
    run_dir = ARTIFACTS / "runs" / phase / experiment
    checkpoint = Path(resume) if resume else initial_checkpoint(experiment, config)
    yolo = YOLO(checkpoint)
    kwargs = {
        "data": study["dataset"],
        "imgsz": study["imgsz"],
        "epochs": epochs,
        "batch": batch,
        "seed": study["seed"],
        "deterministic": study["deterministic"],
        "amp": True,
        "workers": study["workers"],
        "cache": False,
        "device": study["device"],
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "exist_ok": True,
        "fraction": fraction,
        "save": True,
        "save_period": -1,
        "plots": False,
        "verbose": True,
    }
    if resume:
        yolo.train(resume=True)
    else:
        yolo.train(**kwargs)
    _validate_training_outputs(run_dir)


def train_staged(config: dict, stage: str, resume: str | None = None) -> None:
    """Adapt only new P2 parameters first, then unfreeze the full A2 model at a lower learning rate."""
    study = config["study"]
    batch = json.loads((ARTIFACTS / "batch_manifest.json").read_text())["batch"]
    if stage == "head":
        run_dir = ARTIFACTS / "runs/staged/A2_head"
        checkpoint = initial_checkpoint("A2", config)
        epochs = study["staged_head_epochs"]
        freeze = list(A2_HEAD_FREEZE)
        lr0, close_mosaic = 0.01, 0
    else:
        run_dir = ARTIFACTS / "runs/formal/A2"
        checkpoint = ARTIFACTS / "runs/staged/A2_head/weights/best.pt"
        epochs = study["staged_full_epochs"]
        freeze = None
        lr0, close_mosaic = study["staged_full_lr0"], 10
    yolo = YOLO(Path(resume) if resume else checkpoint)
    if resume:
        yolo.train(resume=True)
    else:
        yolo.train(
            data=study["dataset"],
            imgsz=study["imgsz"],
            epochs=epochs,
            batch=batch,
            seed=study["seed"],
            deterministic=study["deterministic"],
            amp=True,
            workers=study["workers"],
            cache=False,
            device=study["device"],
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            optimizer="MuSGD",
            lr0=lr0,
            warmup_epochs=3.0,
            freeze=freeze,
            close_mosaic=close_mosaic,
            save=True,
            save_period=-1,
            plots=False,
            verbose=True,
        )
    _validate_training_outputs(run_dir)


def coco_validate(config: dict, experiment: str) -> None:
    """Validate a formal best checkpoint and independently collect COCO area metrics."""
    study = config["study"]
    best = (
        initial_checkpoint("A0", config)
        if experiment == "A0"
        else ARTIFACTS / "runs/formal" / experiment / "weights/best.pt"
    )
    output = ARTIFACTS / "validation" / experiment
    results = YOLO(best).val(
        data=study["dataset"],
        imgsz=study["imgsz"],
        batch=1,
        device=study["device"],
        save_json=True,
        project=str(output.parent),
        name=output.name,
        exist_ok=True,
        plots=False,
    )
    prediction = output / "predictions.json"
    annotation = Path(study["annotation_root"]) / "instances_val2017.json"
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError:
        from faster_coco_eval.core import COCO, COCOeval_faster
    ground_truth = COCO(str(annotation))
    evaluator = COCOeval_faster(ground_truth, ground_truth.loadRes(str(prediction)), "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = [float(value) for value in evaluator.stats]
    metrics = {
        "AP50-95": stats[0],
        "AP50": stats[1],
        "AP75": stats[2],
        "AP_small": stats[3],
        "AP_medium": stats[4],
        "AP_large": stats[5],
        "AR_1": stats[6],
        "AR_10": stats[7],
        "AR_100": stats[8],
        "AR_small": stats[9],
        "AR_medium": stats[10],
        "AR_large": stats[11],
        "ultralytics": {key: float(value) for key, value in results.results_dict.items()},
    }
    write_json(output / "coco_metrics.json", metrics)


def _percentile(values: list[float]) -> dict[str, float]:
    """Return median and p95 milliseconds."""
    return {"median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))}


def benchmark(config: dict, experiment: str) -> None:
    """Benchmark FP16/FP32 PyTorch batch-one latency on a fixed 500-image subset."""
    import cv2

    study = config["study"]
    best = (
        initial_checkpoint("A0", config)
        if experiment == "A0"
        else ARTIFACTS / "runs/formal" / experiment / "weights/best.pt"
    )
    images = sorted((Path(study["dataset_root"]) / "images/val2017").glob("*.jpg"))[: study["benchmark_images"]]
    device = torch.device(f"cuda:{study['device']}")
    all_results = {}
    for precision in ("fp16", "fp32"):
        model = YOLO(best).model.to(device).eval()
        model.half() if precision == "fp16" else model.float()
        dtype = torch.float16 if precision == "fp16" else torch.float32
        dummy = torch.zeros(1, 3, study["imgsz"], study["imgsz"], device=device, dtype=dtype)
        for _ in range(study["benchmark_warmup"]):
            model(dummy)
        torch.cuda.synchronize(device)
        inference, end_to_end = [], []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(study["benchmark_repeats"]):
            for image_path in images:
                total_start = time.perf_counter()
                array = cv2.imread(str(image_path))
                array = LetterBox(new_shape=(study["imgsz"], study["imgsz"]), auto=False)(image=array)
                tensor = torch.from_numpy(array[..., ::-1].copy()).permute(2, 0, 1).unsqueeze(0)
                tensor = tensor.to(device=device, dtype=dtype).div_(255)
                torch.cuda.synchronize(device)
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    prediction = model(tensor)[0]
                torch.cuda.synchronize(device)
                inference.append((time.perf_counter() - inference_start) * 1000)
                non_max_suppression(prediction.clone())
                torch.cuda.synchronize(device)
                end_to_end.append((time.perf_counter() - total_start) * 1000)
        all_results[precision] = {
            "inference": _percentile(inference),
            "end_to_end": _percentile(end_to_end),
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
            "samples": len(inference),
        }
        del model, dummy
        torch.cuda.empty_cache()
    model = YOLO(best).model
    all_results.update(
        {
            "params": sum(parameter.numel() for parameter in model.parameters()),
            "gflops": get_flops(model, imgsz=study["imgsz"]),
            "model_size_bytes": best.stat().st_size,
            "device": torch.cuda.get_device_name(device),
            "images": [path.name for path in images],
        }
    )
    write_json(ARTIFACTS / "benchmark" / f"{experiment}.json", all_results)


def parse_args() -> argparse.Namespace:
    """Parse one isolated stage invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("preflight", "static", "prepare", "autobatch", "train", "train_staged", "validate", "benchmark"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=("A0", "A1", "A2"))
    parser.add_argument("--phase", choices=("gate1", "health", "formal"))
    parser.add_argument("--stage", choices=("head", "full"))
    parser.add_argument("--resume")
    return parser.parse_args()


def main() -> None:
    """Dispatch a worker command."""
    args = parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        preflight(config)
    elif args.command == "static":
        static_tests(config)
    elif args.command == "prepare":
        prepare(config)
    elif args.command == "autobatch":
        autobatch(config)
    elif args.command == "train":
        train(config, args.experiment, args.phase, args.resume)
    elif args.command == "train_staged":
        train_staged(config, args.stage, args.resume)
    elif args.command == "validate":
        coco_validate(config, args.experiment)
    elif args.command == "benchmark":
        benchmark(config, args.experiment)


if __name__ == "__main__":
    main()
