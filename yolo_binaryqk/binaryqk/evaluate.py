"""固定 E2E、FP32 的 native matching 與 COCO bbox 指標，禁止隱式 fusion。"""

import contextlib
import io
import time
from pathlib import Path

import numpy as np
import torch
from faster_coco_eval import COCO, COCOeval_faster
from torch.utils.data import DataLoader
from ultralytics.models.yolo.detect.val import DetectionValidator

from .calibration import COCO as COCO_ROOT
from .data import config
from .model import sha256


def official_metrics(predictions, image_ids):
    annotation = COCO_ROOT / "annotations/instances_val2017.json"
    if not predictions:
        raise ValueError("no detections; cannot silently skip official evaluation")
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotation))
        detected = ground_truth.loadRes(predictions)
        evaluator = COCOeval_faster(ground_truth, detected, "bbox", print_function=lambda *args: None)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]

    def valid_mean(values):
        values = values[values > -1]
        return float(values.mean()) if values.size else None

    classes = {}
    for name, category in (("person", 1), ("sports_ball", 37), ("baseball_bat", 39)):
        index = evaluator.params.catIds.index(category)
        values = precision[:, :, index, 0, -1]
        classes[name] = {"AP50": valid_mean(values[0]), "AP50_95": valid_mean(values)}
    return {"AP50_95": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
            "AP_small": float(evaluator.stats[3]), "AP_medium": float(evaluator.stats[4]),
            "AP_large": float(evaluator.stats[5]), "classes": classes,
            "annotation_sha256": sha256(annotation)}


@torch.inference_mode()
def evaluate(model, ds, data, output, batch=8, device="cuda:0", limit_batches=None, progress=None):
    cfg = config(batch)
    cfg.save_json = True
    output = Path(output)
    validator = DetectionValidator(args=vars(cfg), save_dir=output)
    validator.training = False
    validator.device, validator.data = torch.device(device), data
    validator.iouv = validator.iouv.to(device)
    model.eval()
    if model.end2end is not True:
        raise ValueError("evaluation requires inherited E2E head")
    validator.init_metrics(model)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2,
                        collate_fn=ds.collate_fn, pin_memory=device.startswith("cuda"))
    started = time.perf_counter()
    image_ids = []
    for index, raw in enumerate(loader):
        item = validator.preprocess(raw)
        preds = validator.postprocess(model(item["img"]))
        validator.update_metrics(preds, item)
        image_ids.extend(int(Path(p).stem) for p in raw["im_file"])
        if progress and (index + 1) % 50 == 0:
            progress({"kind": "evaluation", "images": len(image_ids), "seconds": time.perf_counter() - started})
        if limit_batches is not None and index + 1 >= limit_batches:
            break
    stats = {k: float(v) for k, v in validator.get_stats().items()}
    classes = {}
    box = validator.metrics.box
    indices = list(box.ap_class_index)
    for name, cls in (("person", 0), ("sports_ball", 32), ("baseball_bat", 34)):
        if cls in indices:
            i = indices.index(cls)
            classes[name] = {"AP50": float(box.ap50[i]), "AP50_95": float(box.ap[i]),
                             "precision": float(box.p[i]), "recall": float(box.r[i]),
                             "instances": int(validator.metrics.nt_per_class[cls])}
        else:
            classes[name] = {"AP50": None, "AP50_95": None, "reason": "no evaluated class targets"}
    if limit_batches is None and (len(image_ids) != 5000 or len(set(image_ids)) != 5000):
        raise ValueError("formal evaluation must cover all 5000 images exactly once")
    report = {"protocol": "E2E FP32 conf=.001 iou=.7 max_det=300; native matching and official COCO bbox",
              "images": len(image_ids), "formal": limit_batches is None,
              "internal": stats, "classes_internal": classes,
              "seconds": time.perf_counter() - started}
    if limit_batches is None:
        report["official"] = official_metrics(validator.jdict, image_ids)
    if not all(np.isfinite(v) for v in stats.values()):
        raise FloatingPointError("nonfinite validation metrics")
    return report
