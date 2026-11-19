"""固定 train prefix 的 coverage 與梯度校準；不以 val 選 μ。"""

import hashlib
import math
import statistics
import time

import torch
from binaryqk.data import dataset
from binaryqk.run import loader, seed, transfer

from . import checkpoint
from .common import ARTIFACTS, review, write_json
from .hog import build_hog_targets
from .models import configure, criterion, parts


def coverage():
    data = review()
    cfg = data["protocol"]
    ds, _, _ = dataset("train", cfg["batch"], epochs=cfg["epochs"])
    seed(cfg["seed"])
    categories = {"all80": None, "person": 0, "sports_ball": 32, "baseball_bat": 34}
    records = {
        name: {
            "instances": 0,
            "no_cell_center": 0,
            "no_valid_energy_cell": 0,
            "short_side_lt8px": 0,
            "valid_cells": 0,
        }
        for name in categories
    }
    seen, trace, started = 0, hashlib.sha256(), time.time()
    with torch.inference_mode():
        for raw in loader(ds, cfg["batch"], cfg["seed"]):
            if seen >= cfg["coverage_images"]:
                break
            batch = transfer(raw, "cpu")
            images, boxes = batch["img"], batch["bboxes"]
            ids = batch["batch_idx"].long().flatten()
            targets = build_hog_targets(images, boxes, ids)
            h, w = targets.energy.shape[-2:]
            xs, ys = (torch.arange(w) + 0.5) / w, (torch.arange(h) + 0.5) / h
            lo, hi = boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2
            mask = (
                (xs[None, None, :] >= lo[:, 0, None, None])
                & (xs[None, None, :] < hi[:, 0, None, None])
                & (ys[None, :, None] >= lo[:, 1, None, None])
                & (ys[None, :, None] < hi[:, 1, None, None])
            )
            cells = mask.sum((1, 2))
            valid = (mask & (targets.energy[ids] > 1e-6)).sum((1, 2))
            small = (boxes[:, 2] * images.shape[-1] < 8) | (
                boxes[:, 3] * images.shape[-2] < 8
            )
            for name, category in categories.items():
                select = (
                    torch.ones(len(boxes), dtype=torch.bool)
                    if category is None
                    else batch["cls"].flatten() == category
                )
                record = records[name]
                record["instances"] += int(select.sum())
                record["no_cell_center"] += int(((cells == 0) & select).sum())
                record["no_valid_energy_cell"] += int(((valid == 0) & select).sum())
                record["short_side_lt8px"] += int((small & select).sum())
                record["valid_cells"] += int(valid[select].sum())
            seen += len(images)
            trace.update("\n".join(batch["im_file"]).encode())
            for key in ("img", "bboxes", "batch_idx", "cls"):
                trace.update(batch[key].contiguous().numpy().tobytes())
            if seen % 256 == 0:
                print(f"coverage {seen}/{cfg['coverage_images']}", flush=True)
    if seen != cfg["coverage_images"]:
        raise ValueError("coverage prefix size changed")
    for record in records.values():
        record["zero_valid_cell_fraction"] = record["no_valid_energy_cell"] / max(
            1, record["instances"]
        )
    result = {
        "passed": True,
        "images": seen,
        "optimizer_steps": 0,
        "device": "cpu",
        "seed": cfg["seed"],
        "physical_batch": cfg["batch"],
        "sample_rule": "fixed augmented train prefix; no val",
        "categories": records,
        "trace_sha256": trace.hexdigest(),
        "seconds": time.time() - started,
        "limitation": "prefix is not full-dataset estimate; zero own-box cells does not mean zero native/overlap gradients",
        "h4": {
            "supported_for_review": records["sports_ball"]["instances"] >= 20
            and records["sports_ball"]["zero_valid_cell_fraction"] >= 0.05,
            "proposal": "若確認小球盲區，後續可比較僅零 cell 框的最近有效 cell 補點；需評估背景污染與梯度密度。",
            "implemented": False,
            "training_started": False,
        },
    }
    write_json(ARTIFACTS / "coverage.json", result)
    return result


def calibrate(folder, callback):
    data = review()
    cfg = data["protocol"]
    seed(cfg["seed"])
    ds, _, args = dataset("train", cfg["batch"], epochs=cfg["epochs"])
    model = checkpoint.build().to("cuda:0")
    model.args = args
    configure(model)
    criterion(model)
    before = {
        n: checkpoint.stage3.buffer_digest(v) for n, v in model.state_dict().items()
    }
    parameters = [
        model.model[16].cv2.conv.weight,
        model.model[23].one2one_cv2[0][0].conv.weight,
    ]
    sn = sh = dot = 0.0
    count = valid = 0
    records = []
    torch.cuda.reset_peak_memory_stats()
    for index, raw in enumerate(loader(ds, cfg["batch"], cfg["seed"])):
        if count >= cfg["calibration_images"]:
            break
        batch = transfer(raw, "cuda:0")
        if index == 0:
            with torch.autocast("cuda", dtype=torch.float16):
                native, _ = model.loss(batch)
                native = native.sum()
            gn0 = torch.autograd.grad(native * 1024, parameters)
            with torch.autocast("cuda", dtype=torch.float16):
                zero, _, aux, _, _ = parts(model, batch, 0.0)
            gz = torch.autograd.grad(zero * 1024, parameters)
            if (
                aux is not None
                or not torch.equal(native.detach(), zero.detach())
                or not all(torch.equal(a, b) for a, b in zip(gn0, gz))
            ):
                raise ValueError("mu zero changes native loss or gradients")
            del native, zero, gn0, gz
        with torch.autocast("cuda", dtype=torch.float16):
            _, native, aux, feature, _ = parts(model, batch, 0.0, force=True)
        gn = (
            torch.autograd.grad(native * 1024, feature, retain_graph=True)[0].float()
            / 1024
        )
        gh = (
            torch.autograd.grad(aux.loss_sum * 1024, feature, retain_graph=True)[
                0
            ].float()
            / 1024
        )
        producer, head = torch.autograd.grad(
            aux.loss_sum * 1024,
            [model.model[16].cv2.conv.weight, model.hog_aux.projection.weight],
        )
        a, b = float(gn.double().square().sum()), float(gh.double().square().sum())
        if (
            not math.isfinite(a + b)
            or min(a, b) <= 0
            or any(
                not torch.isfinite(v).all() or float(v.float().norm()) <= 0
                for v in (producer, head)
            )
        ):
            raise ValueError("invalid native/HOG gradients")
        sn += a
        sh += b
        dot += float((gn.double() * gh.double()).sum())
        records.append(
            {
                "raw_ratio": math.sqrt(b / a),
                "p3_producer_hog_grad": float(producer.float().norm()) / 1024,
                "aux_weight_grad": float(head.float().norm()) / 1024,
            }
        )
        count += len(batch["img"])
        valid += int(aux.targets.valid_mask.sum())
        callback({"kind": "calibration", "images": count})
        del native, aux, feature, gn, gh, producer, head
    mu = cfg["mu_target_gradient_ratio"] * math.sqrt(sn / sh)
    ratios = [mu * v["raw_ratio"] for v in records]
    median = statistics.median(ratios)
    if (
        not 0.02 <= median <= 0.10
        or count != cfg["calibration_images"]
        or not math.isfinite(mu)
    ):
        raise ValueError("HOG calibration ratio outside reviewed band")
    if before != {
        n: checkpoint.stage3.buffer_digest(v) for n, v in model.state_dict().items()
    }:
        raise ValueError("calibration modified parent/BN/aux weights")
    result = {
        "passed": True,
        "mu": mu,
        "parent_sha256": data["parent_sha256"],
        "images": count,
        "gradient_space": "raw P3, native and HOG batch sums",
        "target_ratio": 0.05,
        "median_ratio": median,
        "microbatch_ratios": ratios,
        "gradient_cosine": dot / math.sqrt(sn * sh),
        "optimizer_steps": 0,
        "valid_cells": valid,
        "validation_used": False,
        "mu_zero_loss_gradient_exact": True,
        "state_unchanged": True,
        "gradient_proofs": records,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(folder / "calibration.json", result)
    return result
