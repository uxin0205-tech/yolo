from __future__ import annotations

from pathlib import Path

import pytest
import torch

from yolo_combine.models import SharedDualHeadModel
from yolo_combine.source import BuiltTaskModels, SourceBundle

BUNDLE_ROOT = Path("/home/uxin/yolo/yolo_achitechure/achitechure_1/final")


@pytest.fixture(scope="session")
def source_bundle() -> SourceBundle:
    return SourceBundle(BUNDLE_ROOT)


@pytest.fixture(scope="session")
def full35_models(source_bundle: SourceBundle) -> BuiltTaskModels:
    return source_bundle.build_task_models()


@pytest.fixture(scope="session")
def shared_model(full35_models: BuiltTaskModels) -> SharedDualHeadModel:
    return SharedDualHeadModel.from_task_models(full35_models.detect, full35_models.pose)


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for this integration test")
    return torch.device("cuda:0")
