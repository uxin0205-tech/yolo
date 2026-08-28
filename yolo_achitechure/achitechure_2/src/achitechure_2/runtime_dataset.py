"""把 Ultralytics label cache 限制在可重建 runtime View。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.data.dataset import (
    DATASET_CACHE_VERSION,
    YOLODataset,
    get_hash,
    load_dataset_cache_file,
)
from ultralytics.utils import colorstr


class RuntimeLabelCacheYOLODataset(YOLODataset):
    """保留原始 images/labels，只把可重建 label index 寫到指定位置。"""

    def __init__(
        self,
        *args: Any,
        label_cache_path: str | Path,
        **kwargs: Any,
    ) -> None:
        cache_path = Path(label_cache_path).expanduser().resolve()
        if cache_path.suffix != ".cache":
            raise ValueError("runtime label cache 必須使用 .cache 副檔名")
        self.runtime_label_cache_path = cache_path
        self.runtime_label_cache_hit = False
        super().__init__(*args, **kwargs)

    def cache_labels(self, path: Path = Path("labels.cache")) -> dict[str, Any]:
        """忽略 upstream source-adjacent path，改讀寫 runtime cache。"""

        del path
        target = self.runtime_label_cache_path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            cache = load_dataset_cache_file(target)
            assert cache["version"] == DATASET_CACHE_VERSION
            assert cache["hash"] == get_hash(self.label_files + self.im_files)
            self.runtime_label_cache_hit = True
            return cache
        except (FileNotFoundError, AssertionError, AttributeError, ModuleNotFoundError):
            self.runtime_label_cache_hit = False
            return super().cache_labels(target)


def build_runtime_yolo_dataset(
    cfg: Any,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    *,
    label_cache_path: str | Path,
    mode: str,
    rect: bool,
    stride: int,
    fraction: float | None = None,
) -> RuntimeLabelCacheYOLODataset:
    """等價於鎖定版 build_yolo_dataset，但不會建立 source-adjacent cache。"""

    if cfg.task not in {"detect", "pose"}:
        raise ValueError(f"runtime cache dataset 不支援 task={cfg.task}")
    if mode not in {"train", "val"}:
        raise ValueError("dataset mode 必須是 train 或 val")
    if fraction is None:
        fraction = cfg.fraction if mode == "train" else 1.0
    pad = 0.0 if mode == "train" else 0.5
    return RuntimeLabelCacheYOLODataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=stride,
        pad=pad,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
        label_cache_path=label_cache_path,
    )
