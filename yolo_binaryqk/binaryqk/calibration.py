"""在共同 native parent 校準固定 FP32 scale；只讀取 COCO 原影像。"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox

from .basis import BASES, calibration_moments
from .model import PARENT_SHA, ROOT, SITES, load_parent, sha256

COCO = ROOT / "coco2017"
SPLITS = {
    "train2017.txt": (118287, "521d40ccd3df3ac9ba48c1be569828ed7e03b7b2d698186e830ae236364a985b"),
    "val2017.txt": (5000, "4d010d0d7200a7b9176acd164d36c9769350a2e41661152f2cf8ba16c9e2aa73"),
}


def verify_splits():
    for name, (count, digest) in SPLITS.items():
        path = COCO / name
        if sha256(path) != digest or len(path.read_text().splitlines()) != count:
            raise ValueError(f"canonical COCO split mismatch: {name}")
    return {name: {"count": count, "sha256": digest} for name, (count, digest) in SPLITS.items()}


def image_tensor(path, size):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"unreadable image; refusing to skip: {path}")
    image = LetterBox((size, size), auto=False, scaleup=True)(image=image)
    return torch.from_numpy(np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255


def calibrate(output, count=128, size=640):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if count < 1 or count > 118287 or size < 32 or size % 32:
        raise ValueError("invalid calibration count or input size")
    splits = verify_splits()
    paths = (COCO / "train2017.txt").read_text().splitlines()[:count]
    parent = load_parent()
    sums, shapes, calls = {}, {}, {}

    def observer(site, attn):
        def hook(module, args, output):
            batch, _, height, width = output.shape
            q, k, _ = output.reshape(batch, 4, 128, height * width).split((32, 32, 64), 2)
            value = calibration_moments(q, k).double()
            sums[site] = sums.get(site, torch.zeros_like(value)) + value
            shapes[site] = [height, width]
            calls[site] = calls.get(site, 0) + 1
        return hook

    handles = [parent.get_submodule(site).qkv.register_forward_hook(observer(site, parent.get_submodule(site)))
               for site in SITES]
    hashes = []
    try:
        with torch.inference_mode():
            for index, relative in enumerate(paths):
                path = COCO / relative
                if not path.resolve().is_relative_to(COCO.resolve()):
                    raise ValueError("calibration image escaped canonical dataset")
                hashes.append({"path": relative, "sha256": sha256(path)})
                parent(image_tensor(path, size))
                if (index + 1) % 16 == 0:
                    print(f"CPU calibration {index + 1}/{count}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if calls != {site: count for site in SITES}:
        raise ValueError(f"not all parent sites observed once per image: {calls}")
    basis_indices = {"sign": 0, "residual": 1, "hadamard": 2}
    coefficients = {
        variant: {site: (sums[site][:, [basis_indices[b] for b in bases]] / count / len(bases))
                  .float().tolist() for site in SITES}
        for variant, bases in BASES.items()
    }
    report = {"schema": "binaryqk-calibration-v1", "parent_sha256": PARENT_SHA,
              "count": count, "imgsz": size, "device": "cpu", "split": "train2017",
              "policy": "first-N-in-immutable-list; calibration-only; no training/evaluation resplit",
              "splits": splits, "images": hashes, "site_shapes": shapes, "coefficients": coefficients,
              "formula": "mean_images(mean_abs_Q * mean_abs_K) / sqrt(key_dim) / number_of_bases",
              "preprocess": "BGR-to-RGB; LetterBox(auto=False,scaleup=True); FP32/255"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def read_calibration(path):
    report = json.loads(Path(path).read_text())
    if report.get("schema") != "binaryqk-calibration-v1" or report.get("parent_sha256") != PARENT_SHA:
        raise ValueError("calibration source contract mismatch")
    if report.get("count") != 128 or report.get("imgsz") != 640 or report.get("splits") != verify_splits():
        raise ValueError("formal stage1 initialization requires common 128-image, 640px calibration")
    if set(report.get("coefficients", {})) != set(BASES):
        raise ValueError("incomplete candidate coefficients")
    return report
