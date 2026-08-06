"""Deterministic COCO manifests; this module never silently downloads data."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import tempfile
import shutil
from ultralytics.data.converter import convert_coco
import yaml

def ensure_yolo_layout(coco: Path) -> None:
    """Expose official COCO folders through Ultralytics' images/labels layout."""
    image_dir = coco / "images"
    image_dir.mkdir(exist_ok=True)
    for split in ("train2017", "val2017"):
        link, target = image_dir / split, coco / split
        if link.exists():
            continue
        if target.exists():
            link.symlink_to(target.resolve(), target_is_directory=True)

def make_manifest(coco: Path, output: Path, size: int | None, seed: int = 42) -> Path:
    if size is not None:
        raise ValueError("subset manifests (5k/30k) are disabled by the full-COCO plan")
    ensure_yolo_layout(coco)
    images = coco / "images" / "train2017"; val = coco / "images" / "val2017"
    required = [images, val]
    missing = [str(p) for p in required if not p.exists()]
    if missing: raise FileNotFoundError("COCO 2017 is incomplete: " + ", ".join(missing))
    filenames = sorted(path.name for path in images.glob("*.jpg"))
    lines = [str(images / filename) for filename in filenames]
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text("\n".join(lines) + "\n")
    meta = {"seed": seed, "count": len(lines), "image_ids": filenames,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "validation": str(val), "source": str(coco)}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return output

def convert_coco_annotations(coco: Path) -> Path:
    """Convert only instances JSON (the COCO folder also contains captions/keypoints)."""
    target = coco / "labels"
    if target.exists() and any(target.rglob("*.txt")):
        return target
    with tempfile.TemporaryDirectory(prefix="coco_instances_") as tmp:
        source = Path(tmp)
        for name in ("instances_train2017.json", "instances_val2017.json"):
            (source / name).symlink_to((coco / "annotations" / name).resolve())
        output = source / "converted"
        convert_coco(labels_dir=str(source), save_dir=str(output), cls91to80=True)
        generated = next(output.parent.glob("converted*")) / "labels"
        shutil.copytree(generated, target)
    return target

def write_dataset_yaml(output: Path, train: Path, val: Path, names: list[str] | None = None) -> Path:
    if names is None:
        names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                 "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                 "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                 "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                 "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
                 "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                 "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
                 "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                 "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump({"path": str(Path.cwd()), "train": str(train.absolute()), "val": str(val.absolute()),
                                      "names": names}, sort_keys=False))
    return output
