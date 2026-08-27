"""將 ball／bat pose 標註轉成 COCO80 detection 驗證集並匯出指標。"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO

from .checkpoint import file_sha256

BALL_CLASS = 32
BAT_CLASS = 34
CLASS_REMAP = {0: BALL_CLASS, 1: BAT_CLASS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree_digest(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def convert_pose_label_line(
    line: str,
    class_remap: dict[int, int] | None = None,
) -> str:
    """移除兩個 keypoint，並依指定 mapping 轉換 detection 類別。"""

    fields = line.split()
    if len(fields) != 11:
        raise ValueError(f"ball／bat pose 標註應有 11 欄，實際為 {len(fields)} 欄")
    try:
        source_class = int(fields[0])
    except ValueError as exc:
        raise ValueError(f"類別欄不是整數：{fields[0]!r}") from exc
    class_remap = CLASS_REMAP if class_remap is None else class_remap
    if source_class not in class_remap:
        raise ValueError(f"不支援的 ball／bat 類別：{source_class}")
    for value in fields[1:5]:
        coordinate = float(value)
        if not 0.0 <= coordinate <= 1.0:
            raise ValueError(f"bbox 正規化座標超出 [0, 1]：{coordinate}")
    return " ".join((str(class_remap[source_class]), *fields[1:5]))


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _link_image(source_image: Path, image_link: Path) -> None:
    if image_link.is_symlink():
        if image_link.resolve() != source_image:
            raise RuntimeError(f"既有 image symlink 指向錯誤位置：{image_link}")
        return
    if image_link.exists():
        raise RuntimeError(f"預期 image 為 symlink，實際已有其他檔案：{image_link}")
    relative_target = os.path.relpath(source_image, start=image_link.parent)
    image_link.symlink_to(relative_target)


def _prepare_detection_split(
    *,
    source_root: Path,
    output_root: Path,
    split: str,
    class_remap: dict[int, int],
) -> dict[str, Any]:
    source_images = source_root / split / "images"
    source_labels = source_root / split / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError(f"來源缺少 {split}/images 或 {split}/labels")
    images = sorted(path for path in source_images.iterdir() if path.is_file())
    labels = sorted(source_labels.glob("*.txt"))
    if len(images) != len(labels):
        raise ValueError(f"{split} 影像／標註數不同：{len(images)}／{len(labels)}")
    if {path.stem for path in images} != {path.stem for path in labels}:
        raise ValueError(f"{split} 影像與標註檔名無法一一對應")

    output_images = output_root / split / "images"
    output_labels = output_root / split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    image_names = {path.name for path in images}
    label_names = {path.name for path in labels}
    for source_image in images:
        _link_image(source_image, output_images / source_image.name)

    class_instances = {target: 0 for target in class_remap.values()}
    empty_labels = 0
    for source_label in labels:
        converted: list[str] = []
        for line_number, raw_line in enumerate(
            source_label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                converted_line = convert_pose_label_line(raw_line, class_remap)
            except ValueError as exc:
                raise ValueError(f"{source_label}:{line_number}: {exc}") from exc
            converted.append(converted_line)
            class_instances[int(converted_line.split()[0])] += 1
        if not converted:
            empty_labels += 1
        _atomic_text(
            output_labels / source_label.name,
            "\n".join(converted) + ("\n" if converted else ""),
        )

    unexpected_images = sorted(
        path.name for path in output_images.iterdir() if path.name not in image_names
    )
    unexpected_labels = sorted(
        path.name
        for path in output_labels.glob("*.txt")
        if path.name not in label_names
    )
    if unexpected_images or unexpected_labels:
        raise RuntimeError(
            f"{split} 衍生資料殘留來源中不存在的檔案："
            f"images={unexpected_images[:3]} labels={unexpected_labels[:3]}"
        )
    derived_label_paths = sorted(output_labels.glob("*.txt"))
    return {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "instances_by_class": {str(key): value for key, value in class_instances.items()},
        "source_label_tree_sha256": _tree_digest(labels, source_labels),
        "derived_label_tree_sha256": _tree_digest(derived_label_paths, output_labels),
    }


def prepare_ball_bat_detect_dataset(
    *,
    source_root: Path,
    output_root: Path,
    coco_data: Path,
) -> tuple[Path, Path]:
    """建立完整 2-class detection dataset 與供既有 COCO80 detector 驗證的 view。"""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_yaml = source_root / "data.yaml"
    if not source_yaml.is_file() or not coco_data.resolve().is_file():
        raise FileNotFoundError(source_yaml if not source_yaml.is_file() else coco_data.resolve())
    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if source_config.get("names") != ["ball", "bat"] or source_config.get("kpt_shape") != [2, 3]:
        raise ValueError("來源資料契約不是 ball／bat、兩個 keypoint 的 pose dataset")

    coco_config = yaml.safe_load(coco_data.read_text(encoding="utf-8"))
    names_payload = coco_config.get("names")
    if isinstance(names_payload, dict) and set(names_payload) == set(range(80)):
        coco_names = [str(names_payload[index]) for index in range(80)]
    elif isinstance(names_payload, list) and len(names_payload) == 80:
        coco_names = names_payload
    else:
        raise ValueError("COCO data YAML 必須提供 80 個依序排列的類別名稱")
    if coco_names[BALL_CLASS] != "sports ball" or coco_names[BAT_CLASS] != "baseball bat":
        raise ValueError("COCO80 類別索引 32/34 與 sports ball/baseball bat 不符")

    two_class_splits = {
        split: _prepare_detection_split(
            source_root=source_root,
            output_root=output_root,
            split=split,
            class_remap={0: 0, 1: 1},
        )
        for split in ("train", "valid")
    }
    coco80_root = output_root / "coco80"
    coco80_splits = {
        split: _prepare_detection_split(
            source_root=source_root,
            output_root=coco80_root,
            split=split,
            class_remap=CLASS_REMAP,
        )
        for split in ("train", "valid")
    }
    two_class_data = output_root / "data.yaml"
    coco80_data = coco80_root / "data.yaml"
    common_paths = {"train": "train/images", "val": "valid/images"}
    _atomic_text(
        two_class_data,
        yaml.safe_dump(
            {"path": str(output_root), **common_paths, "names": ["ball", "bat"]},
            sort_keys=False,
            allow_unicode=True,
        ),
    )
    _atomic_text(
        coco80_data,
        yaml.safe_dump(
            {"path": str(coco80_root), **common_paths, "names": coco_names},
            sort_keys=False,
            allow_unicode=True,
        ),
    )

    coco_root = Path(str(coco_config["path"]))
    if not coco_root.is_absolute():
        coco_root = (coco_data.resolve().parent / coco_root).resolve()
    coco_train_names = {
        path.stem for path in (coco_root / "images/train2017").glob("*.jpg")
    }
    valid_images = sorted((source_root / "valid/images").iterdir())
    overlap_names = [
        image.name
        for image in valid_images
        if image.name[:12].isdigit() and image.name[:12] in coco_train_names
    ]
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "source": {"root": str(source_root), "data_yaml": str(source_yaml)},
        "views": {
            "two_class": {
                "root": str(output_root),
                "data_yaml": str(two_class_data),
                "class_mapping": {"0": "ball", "1": "bat"},
                "splits": two_class_splits,
            },
            "coco80": {
                "root": str(coco80_root),
                "data_yaml": str(coco80_data),
                "class_mapping": {"32": "sports ball", "34": "baseball bat"},
                "splits": coco80_splits,
            },
        },
        "storage": "影像為逐檔相對 symlink；detection labels 為獨立的 5 欄文字檔",
        "independence_warning": {
            "coco_train2017_id_overlap_images": len(overlap_names),
            "fraction_of_validation_images": len(overlap_names) / len(valid_images),
            "examples": overlap_names[:10],
        },
    }
    _atomic_text(
        output_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    _atomic_text(
        output_root / "README.md",
        """# Ball／Bat Detection Dataset

此目錄由相鄰的 `dataset/` pose 資料衍生，來源檔案不會被修改。

- `data.yaml`：2 類別 detection 契約，`ball=0`、`bat=1`。
- `coco80/data.yaml`：既有 COCO80 detector 的驗證契約，`sports ball=32`、`baseball bat=34`。
- 影像使用逐檔相對 symlink；標註已移除 keypoint，只保留 `class x y w h`。
- 來源沒有實際 `test/` 內容，因此只提供 train／valid。

驗證集有部分 COCO train2017 ID 重疊；精確數量與雜湊請見 `manifest.json`。
""",
    )
    return two_class_data, coco80_data


def prepare_ball_bat_detection_validation(
    *,
    source_root: Path,
    output_root: Path,
    coco_data: Path,
) -> Path:
    """建立不修改來源檔案的 detection-only 驗證映射與可稽核 manifest。"""

    source_root = source_root.resolve()
    output_root = output_root.resolve()
    source_images = source_root / "valid/images"
    source_labels = source_root / "valid/labels"
    source_yaml = source_root / "data.yaml"
    for required in (source_images, source_labels, source_yaml, coco_data.resolve()):
        if not required.exists():
            raise FileNotFoundError(required)

    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if source_config.get("names") != ["ball", "bat"] or source_config.get("kpt_shape") != [2, 3]:
        raise ValueError("來源資料契約不是 ball／bat、兩個 keypoint 的 pose dataset")
    coco_config = yaml.safe_load(coco_data.read_text(encoding="utf-8"))
    names_payload = coco_config.get("names")
    if isinstance(names_payload, dict) and set(names_payload) == set(range(80)):
        names = [str(names_payload[index]) for index in range(80)]
    elif isinstance(names_payload, list):
        names = names_payload
    else:
        names = []
    if len(names) != 80:
        raise ValueError("COCO data YAML 必須提供 80 個依序排列的類別名稱")
    if names[BALL_CLASS] != "sports ball" or names[BAT_CLASS] != "baseball bat":
        raise ValueError("COCO80 類別索引 32/34 與 sports ball/baseball bat 不符")

    images = sorted(path for path in source_images.iterdir() if path.is_file())
    labels = sorted(source_labels.glob("*.txt"))
    if len(images) != len(labels):
        raise ValueError(f"驗證影像／標註數不同：{len(images)}／{len(labels)}")
    if {path.stem for path in images} != {path.stem for path in labels}:
        raise ValueError("驗證影像與標註檔名無法一一對應")

    derived_valid = output_root / "valid"
    derived_labels = derived_valid / "labels"
    derived_labels.mkdir(parents=True, exist_ok=True)
    derived_images = derived_valid / "images"
    if derived_images.is_symlink():
        if derived_images.resolve() != source_images:
            raise RuntimeError(f"既有 images symlink 指向錯誤位置：{derived_images}")
        # 不能 symlink 整個目錄：Ultralytics 會 resolve 後回到來源 pose labels。
        derived_images.unlink()
    derived_images.mkdir(parents=True, exist_ok=True)
    image_names = {path.name for path in images}
    for source_image in images:
        image_link = derived_images / source_image.name
        if image_link.is_symlink():
            if image_link.resolve() != source_image:
                raise RuntimeError(f"既有 image symlink 指向錯誤位置：{image_link}")
        elif image_link.exists():
            raise RuntimeError(f"預期 image 為 symlink，實際已有其他檔案：{image_link}")
        else:
            image_link.symlink_to(source_image)
    unexpected_images = sorted(
        path.name for path in derived_images.iterdir() if path.name not in image_names
    )
    if unexpected_images:
        raise RuntimeError(f"衍生驗證集殘留來源中不存在的影像：{unexpected_images[:3]}")

    class_instances = {BALL_CLASS: 0, BAT_CLASS: 0}
    empty_labels = 0
    for source_label in labels:
        converted: list[str] = []
        for line_number, raw_line in enumerate(
            source_label.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not raw_line.strip():
                continue
            try:
                converted_line = convert_pose_label_line(raw_line)
            except ValueError as exc:
                raise ValueError(f"{source_label}:{line_number}: {exc}") from exc
            converted.append(converted_line)
            class_instances[int(converted_line.split()[0])] += 1
        if not converted:
            empty_labels += 1
        text = "\n".join(converted)
        if converted:
            text += "\n"
        _atomic_text(derived_labels / source_label.name, text)

    label_names = {path.name for path in labels}
    unexpected = sorted(
        path.name for path in derived_labels.glob("*.txt") if path.name not in label_names
    )
    if unexpected:
        raise RuntimeError(f"衍生驗證集殘留來源中不存在的標註：{unexpected[:3]}")

    data_path = output_root / "data.yaml"
    data_payload = {
        "path": str(output_root),
        # Ultralytics 即使只執行 split=val，仍要求 YAML 同時具有 train/val 鍵。
        "train": "valid/images",
        "val": "valid/images",
        "names": names,
    }
    _atomic_text(data_path, yaml.safe_dump(data_payload, sort_keys=False, allow_unicode=True))

    coco_root = Path(str(coco_config["path"]))
    if not coco_root.is_absolute():
        coco_root = (coco_data.resolve().parent / coco_root).resolve()
    coco_train_names = {
        path.stem for path in (coco_root / "images/train2017").glob("*.jpg")
    }
    overlap_names: list[str] = []
    for image in images:
        coco_id = image.name[:12]
        if coco_id.isdigit() and coco_id in coco_train_names:
            overlap_names.append(image.name)

    derived_label_paths = sorted(derived_labels.glob("*.txt"))
    manifest = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "purpose": "將 ball/bat pose bbox 以 COCO80 sports ball/baseball bat 類別進行 detection 驗證",
        "source": {
            "root": str(source_root),
            "data_yaml": str(source_yaml),
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_labels,
            "label_tree_sha256": _tree_digest(labels, source_labels),
        },
        "derived": {
            "root": str(output_root),
            "data_yaml": str(data_path),
            "label_tree_sha256": _tree_digest(derived_label_paths, derived_labels),
        },
        "class_mapping": {
            "0": {"source_name": "ball", "coco80_class": BALL_CLASS, "target_name": names[BALL_CLASS]},
            "1": {"source_name": "bat", "coco80_class": BAT_CLASS, "target_name": names[BAT_CLASS]},
        },
        "instances": {
            "sports_ball": class_instances[BALL_CLASS],
            "baseball_bat": class_instances[BAT_CLASS],
        },
        "independence_warning": {
            "coco_train2017_id_overlap_images": len(overlap_names),
            "fraction_of_validation_images": len(overlap_names) / len(images),
            "examples": overlap_names[:10],
        },
    }
    _atomic_text(
        output_root / "manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return data_path


def _per_class_metrics(results: Any) -> dict[str, dict[str, float | int | str]]:
    indices = [int(value) for value in results.box.ap_class_index]
    output: dict[str, dict[str, float | int | str]] = {}
    for class_id, key in ((BALL_CLASS, "sports_ball"), (BAT_CLASS, "baseball_bat")):
        if class_id not in indices:
            raise RuntimeError(f"驗證結果缺少 COCO80 class {class_id}")
        position = indices.index(class_id)
        output[key] = {
            "class_id": class_id,
            "name": results.names[class_id],
            "images": int(results.nt_per_image[class_id]),
            "instances": int(results.nt_per_class[class_id]),
            "precision": float(results.box.p[position]),
            "recall": float(results.box.r[position]),
            "f1": float(results.box.f1[position]),
            "ap50": float(results.box.ap50[position]),
            "ap75": float(results.box.all_ap[position, 5]),
            "ap50_95": float(results.box.ap[position]),
        }
    return output


def validate_ball_bat_checkpoint(
    *,
    checkpoint: Path,
    data: Path,
    run_dir: Path,
    imgsz: int = 640,
    batch: int = 8,
    device: str = "0",
    workers: int = 6,
) -> Path:
    """以固定的 32/34 類別過濾執行一個 checkpoint 的 ball／bat 驗證。"""

    checkpoint = checkpoint.resolve()
    data = data.resolve()
    run_dir = run_dir.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not data.is_file():
        raise FileNotFoundError(data)
    if run_dir.exists():
        raise FileExistsError(run_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(checkpoint))
    attention_backends = [
        module.config.normalization.value
        for module in model.model.modules()
        if module.__class__.__name__ == "HardwareFriendlyAttention"
    ]
    if attention_backends != ["bit_true_pwl", "bit_true_pwl"]:
        raise ValueError(f"checkpoint 不是雙站點 Bit-True PWL：{attention_backends}")
    results = model.val(
        data=str(data),
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        classes=[BALL_CLASS, BAT_CLASS],
        split="val",
        save_json=False,
        plots=False,
        verbose=True,
        project=str(run_dir),
        name="ultralytics",
        exist_ok=False,
    )
    per_class = _per_class_metrics(results)
    payload = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
        "data": str(data),
        "contract": {
            "task": "detect",
            "imgsz": imgsz,
            "batch": batch,
            "device": device,
            "workers": workers,
            "classes": [BALL_CLASS, BAT_CLASS],
            "attention_backends": attention_backends,
        },
        "overall": {
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
            "map50": float(results.box.map50),
            "map75": float(results.box.map75),
            "map50_95": float(results.box.map),
        },
        "per_class": per_class,
        "speed_ms": {name: float(value) for name, value in results.speed.items()},
        "cuda_peak_vram_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
        ),
    }
    destination = run_dir / "metrics.json"
    _atomic_text(
        destination,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )
    return destination
