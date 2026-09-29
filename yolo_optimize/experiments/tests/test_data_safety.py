"""僅以臨時 fixtures 驗證 native dataset 的不可變來源保護。"""

from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_optimize.data_safety import ReadOnlyDatasetError, install_readonly_data_guard


def jpeg_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 24), (30, 120, 80)).save(buffer, format="JPEG")
    return buffer.getvalue()


class ReadOnlyDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        install_readonly_data_guard()

    def _native_verify(self, image_path: Path, label_path: Path):
        # 呼叫 dataset.cache_labels 真正使用的已綁定 verifier，而非只測自訂函式。
        from ultralytics.data.dataset import verify_image_label
        return verify_image_label((str(image_path), str(label_path), "", False, 1, 0, 0, False))

    def test_jpeg_without_terminal_eof_is_rejected_without_source_mutation(self) -> None:
        for fixture_bytes in (jpeg_bytes()[:-2], jpeg_bytes() + b"trailing_bytes"):
            with self.subTest(trailing_bytes=fixture_bytes[-14:]):
                with TemporaryDirectory() as folder:
                    root = Path(folder)
                    source = root / "source.jpg"
                    source.write_bytes(fixture_bytes)
                    link = root / "runtime.jpg"
                    link.symlink_to(source)
                    labels = root / "runtime.txt"
                    labels.write_text("0 0.5 0.5 0.5 0.5\n")
                    before_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                    with self.assertRaisesRegex(ReadOnlyDatasetError, "EOF"):
                        self._native_verify(link, labels)
                    self.assertEqual(source.read_bytes(), fixture_bytes)
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before_sha)

    def test_valid_native_image_label_verification_is_preserved(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "valid.jpg"
            image.write_bytes(jpeg_bytes())
            label = root / "valid.txt"
            label.write_text("0 0.5 0.5 0.5 0.5\n")
            before = image.read_bytes()
            result = self._native_verify(image, label)
            self.assertEqual(result[0], str(image))
            self.assertEqual(result[2], (24, 24))
            self.assertEqual(result[1].shape, (1, 5))
            self.assertEqual(image.read_bytes(), before)

    def test_invalid_labels_stop_instead_of_silently_removing_image(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            image = root / "valid.jpg"
            image.write_bytes(jpeg_bytes())
            label = root / "invalid.txt"
            label.write_text("0 0.5 0.5\n")
            with self.assertRaisesRegex(ReadOnlyDatasetError, "禁止略過"):
                self._native_verify(image, label)

    def test_disk_cache_constructor_fails_before_dataset_access(self) -> None:
        from ultralytics.data.dataset import YOLODataset
        with self.assertRaisesRegex(ReadOnlyDatasetError, "cache='disk'"):
            YOLODataset(img_path="/nonexistent/data-guard-fixture", data={"names": {0: "fixture"}}, cache="disk")

    def test_direct_disk_cache_call_is_also_rejected(self) -> None:
        from ultralytics.data.base import BaseDataset
        with self.assertRaises(ReadOnlyDatasetError):
            BaseDataset.cache_images_to_disk(SimpleNamespace(), 0)

    def test_existing_corrupt_npy_is_neither_loaded_nor_deleted(self) -> None:
        from ultralytics.data.base import BaseDataset
        with TemporaryDirectory() as folder:
            image_cache = Path(folder) / "source.npy"
            image_cache.write_bytes(b"corrupt cache fixture")
            fake_dataset = SimpleNamespace(npy_files=[image_cache])
            with self.assertRaisesRegex(ReadOnlyDatasetError, "拒絕隱式載入／刪除"):
                BaseDataset.load_image(fake_dataset, 0)
            self.assertEqual(image_cache.read_bytes(), b"corrupt cache fixture")

    def test_native_label_cache_write_outside_workspace_is_rejected(self) -> None:
        from ultralytics.data.dataset import save_dataset_cache_file
        with TemporaryDirectory() as folder:
            cache = Path(folder) / "labels.cache"
            cache.write_bytes(b"preexisting source cache")
            with self.assertRaisesRegex(ReadOnlyDatasetError, "拒絕來源寫入"):
                save_dataset_cache_file("", cache, {}, "fixture-version")
            self.assertEqual(cache.read_bytes(), b"preexisting source cache")

    def test_install_is_idempotent(self) -> None:
        from ultralytics.data import utils
        first = utils.check_image
        install_readonly_data_guard()
        self.assertIs(utils.check_image, first)


if __name__ == "__main__":
    unittest.main()
