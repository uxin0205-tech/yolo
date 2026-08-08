from __future__ import annotations

from masf_yolo.models.builder import build_model
from masf_yolo.training.runner import RepositoryDetectionTrainer


def test_resume_reuses_serialized_custom_model_without_yaml_rebuild() -> None:
    model = build_model("M0")
    trainer = object.__new__(RepositoryDetectionTrainer)

    restored = trainer.get_model(weights=model, cfg=model.yaml, verbose=False)

    assert restored is model
    assert restored.masf_variant == "M0"
