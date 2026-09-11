"""Ultralytics 8.4.90 的窄範圍資料唯讀防護。

必須在建立任何 dataset／worker 之前安裝。僅攔截 native dataset 的
JPEG 原地修復、相鄰 .npy 修復／建立，以及離開優化工作區的 label cache
寫入；不攔截全域 open，也不改動 site-packages。資料錯誤直接中止掃描，
不允許 native verifier 靜默略過檔案而改變正式 split。
"""

from __future__ import annotations

from functools import wraps
from importlib.metadata import version
import inspect
from pathlib import Path

from PIL import Image


WORKSPACE = Path(__file__).resolve().parents[2]
SUPPORTED_ULTRALYTICS = "8.4.90"
_installed = False


class ReadOnlyDatasetError(RuntimeError):
    """資料來源需要修復或寫入；此工作流程沒有該授權。"""


def install_readonly_data_guard() -> None:
    """Process-local、可重複呼叫；spawn worker 必須再次安裝。"""
    global _installed
    if _installed:
        return
    actual = version("ultralytics")
    if actual != SUPPORTED_ULTRALYTICS:
        raise ReadOnlyDatasetError(
            f"唯讀資料 guard 僅稽核過 Ultralytics {SUPPORTED_ULTRALYTICS}，目前為 {actual}"
        )
    from ultralytics.data import dataset as native_dataset
    from ultralytics.data import utils as native_utils
    from ultralytics.data.base import BaseDataset

    native_verify = native_utils.verify_image_label
    native_save_cache = native_utils.save_dataset_cache_file
    native_init = BaseDataset.__init__
    native_load_image = BaseDataset.load_image
    init_signature = inspect.signature(native_init)
    if "cache" not in init_signature.parameters:
        raise ReadOnlyDatasetError("BaseDataset.__init__ 的 cache 契約已改變")

    def readonly_check_image(im_file: str) -> tuple[str, tuple[int, int]]:
        # 與指定 native 版本相同的 PIL、尺寸與格式驗證，唯獨不執行修復。
        with Image.open(im_file) as image:
            image.verify()
            width, height = native_utils.exif_size(image)
            image_format = image.format.lower()
        if height <= 9 or width <= 9:
            raise ReadOnlyDatasetError(f"影像尺寸 {(height, width)} 不符：{im_file}")
        if image_format not in native_utils.IMG_FORMATS:
            raise ReadOnlyDatasetError(f"不支援的影像格式 {image_format}：{im_file}")
        if image_format in {"jpg", "jpeg"}:
            with open(im_file, "rb") as handle:
                handle.seek(-2, 2)
                if handle.read() != b"\xff\xd9":
                    raise ReadOnlyDatasetError(
                        f"JPEG 檔尾缺少 EOF，拒絕原地修復；來源未更動：{im_file}"
                    )
        return "", (height, width)

    @wraps(native_verify)
    def readonly_verify_image_label(args: tuple) -> list:
        result = native_verify(args)
        # 原生函式會捕捉 check_image／label 錯誤並返回 None；不能因此偷減資料。
        if result[0] is None:
            raise ReadOnlyDatasetError(f"資料驗證失敗，禁止略過或修復：{result[-1]}")
        return result

    @wraps(native_init)
    def readonly_init(self, *args, **kwargs):
        bound = init_signature.bind(self, *args, **kwargs)
        cache = bound.arguments.get("cache", False)
        if isinstance(cache, str) and cache.lower() == "disk":
            raise ReadOnlyDatasetError("唯讀 runtime View 禁止 cache='disk'，不得在來源建立 .npy")
        return native_init(self, *args, **kwargs)

    @wraps(native_load_image)
    def readonly_load_image(self, i: int, *args, **kwargs):
        # native 即使 cache=False，也會讀取並可能 unlink 同名 .npy。
        # 有既存檔時先停機，避免隱式使用舊 cache 或擅自修復來源。
        image_cache = Path(self.npy_files[i])
        if image_cache.exists() or image_cache.is_symlink():
            raise ReadOnlyDatasetError(
                f"發現相鄰 .npy，拒絕隱式載入／刪除；請先稽核 cache：{image_cache}"
            )
        return native_load_image(self, i, *args, **kwargs)

    def reject_disk_cache(self, i: int) -> None:
        raise ReadOnlyDatasetError("唯讀 runtime View 禁止建立／修復磁碟影像 cache")

    @wraps(native_save_cache)
    def readonly_save_dataset_cache_file(prefix: str, path: Path, payload: dict, cache_version: str):
        resolved = Path(path).expanduser().resolve()
        if resolved == WORKSPACE or not resolved.is_relative_to(WORKSPACE):
            raise ReadOnlyDatasetError(f"label cache 必須留在優化工作區，拒絕來源寫入：{resolved}")
        return native_save_cache(prefix, resolved, payload, cache_version)

    native_utils.check_image = readonly_check_image
    native_utils.verify_image_label = readonly_verify_image_label
    native_dataset.verify_image_label = readonly_verify_image_label
    native_utils.save_dataset_cache_file = readonly_save_dataset_cache_file
    native_dataset.save_dataset_cache_file = readonly_save_dataset_cache_file
    BaseDataset.__init__ = readonly_init
    BaseDataset.load_image = readonly_load_image
    BaseDataset.cache_images_to_disk = reject_disk_cache
    _installed = True
