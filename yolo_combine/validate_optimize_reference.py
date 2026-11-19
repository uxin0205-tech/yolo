"""唯讀載入 optimize 交付包，使用本輪相同 dataset／evaluator 完整驗證。"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "yolo_optimize/model"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    protocol = read(output / "protocol.json")
    for name, digest in protocol["sources_sha256"].items():
        if sha(ROOT / name) != digest:
            raise ValueError(f"reference changed: {name}")
    if sha(output / "datasets.json") != protocol["dataset_config_sha256"]:
        raise ValueError("dataset snapshot changed")
    # 必須在其他研究模組之前載入舊模型的配套 runtime，避免 Python 同名模組混用。
    sys.path[:0] = [str(PACKAGE), str(PACKAGE / "runtime")]
    import torch
    from load_model import load_model

    torch.set_num_threads(2)
    torch.manual_seed(20261003)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not args.preflight:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
        if gpu:
            raise RuntimeError(f"GPU occupied: {gpu}")
    device = "cpu" if args.preflight else "cuda:0"
    model = load_model(device=device, kind="bittrue")
    paths = {}
    for name in (
        "yolo_attention.attention",
        "yolo_attention.normalization",
        "yolo_attention.binary_basis",
        "yolo_combine.fusion_model",
        "activation_lab.activations",
    ):
        path = Path(sys.modules[name].__file__).resolve()
        if not path.is_relative_to(PACKAGE / "runtime"):
            raise ValueError(f"historical runtime shadowed: {name} {path}")
        paths[name] = {"path": str(path), "sha256": sha(path)}
    classes = [type(m).__name__ for m in model.modules()]
    if classes.count("ShiftPiecewiseQuadraticSiLU") != 190 or "SiLU" in classes:
        raise ValueError("optimize qSiLU contract changed")
    if classes.count("BitTruePiecewiseLinearSoftmax") != 2 or "RepConv" in classes:
        raise ValueError("optimize PWL/folded contract changed")

    class TaskView(torch.nn.Module):
        def __init__(self, shared, task):
            super().__init__()
            self.shared, self.task = shared, task
            self.names = shared.detect_names if task == "detect" else shared.pose_names
            self.end2end = True

        def forward(self, images):
            return self.shared(images, task=self.task)[self.task]

    if args.preflight:
        from load_model import tensors

        with torch.inference_mode():
            for task in ("detect", "pose"):
                values = TaskView(model, task)(torch.zeros(1, 3, 160, 160))
                if not all(torch.isfinite(v).all() for v in tensors(values)):
                    raise ValueError("nonfinite preflight output")
        result = {
            "status": "passed",
            "gpu_used": False,
            "runtime": paths,
            "qSiLU": 190,
            "bittrue_pwl_sites": 2,
            "rep_folded": True,
        }
        write(output / "optimize-preflight.json", result)
        print("optimize isolated-runtime CPU preflight passed", flush=True)
        return
    from binaryqk.data_safety import install_readonly_data_guard
    from binaryqk.evaluate import evaluate
    from masf_yolo.stage4.evaluate import evaluate_pose
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    install_readonly_data_guard(
        cache_workspace=ROOT / "yolo_combine/artifacts/stage6/datasets"
    )
    outputs = {}
    started = time.time()
    for task, spec in read(output / "datasets.json").items():
        cfg = get_cfg(overrides=spec["args"])
        data = check_det_dataset(cfg.data, autodownload=False)
        ds = build_yolo_dataset(
            cfg, data["val"], 8, data, mode="val", rect=True, stride=32
        )
        image_sha = hashlib.sha256("\n".join(ds.im_files).encode()).hexdigest()
        if (
            image_sha != spec["ordered_images_sha256"]
            or ds.batch_shapes.tolist() != spec["batch_shapes"]
            or len(ds) != spec["images"]
        ):
            raise ValueError("dataset order/count/rectangular padding differs")

        def progress(record, task=task):
            write(
                output / "optimize-progress.json",
                {"time_unix": time.time(), "task": task, **record},
            )

        progress({"kind": "evaluation-start"})
        evaluator = evaluate if task == "detect" else evaluate_pose
        outputs[task] = evaluator(
            TaskView(model, task).eval(),
            ds,
            data,
            output / "optimize-evaluation" / task,
            batch=8,
            progress=progress,
        )
        write(output / f"optimize-{task}.json", outputs[task])
        del ds
    result = {
        "status": "passed",
        "checkpoint_sha256": protocol["optimize_selected"]["sha256"],
        "runtime": paths,
        "same_dataset_order_and_padding": True,
        "batch": 8,
        "imgsz": 640,
        "dtype": "FP32",
        "end2end": True,
        "seconds": time.time() - started,
        "detect": outputs["detect"],
        "pose": outputs["pose"],
    }
    write(output / "optimize-result.json", result)
    print("optimize full matched-evaluator validation passed", flush=True)


if __name__ == "__main__":
    main()
