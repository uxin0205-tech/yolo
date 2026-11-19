from __future__ import annotations

import json
import re
from pathlib import Path

from activation_lab.training import load_region_rules

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_training_json_schemas_are_valid_json() -> None:
    schemas = sorted((PROJECT_ROOT / "training/schemas").glob("*.json"))
    assert len(schemas) == 4
    for schema in schemas:
        payload = json.loads(schema.read_text(encoding="utf-8"))
        assert payload["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert payload["type"] == "object"


def test_example_region_rules_have_one_match_per_representative_path() -> None:
    rules = load_region_rules(
        PROJECT_ROOT / "training/configs/region-rules.example.yaml"
    )
    examples = {
        "model.0.block.act": "early",
        "model.7.block.act": "deep",
        "model.10.block.act": "attention_ffn",
        "model.15.block.act": "neck",
        "model.23.branch.cv2.0.act": "head_one2many",
        "model.23.branch.one2one_cv2.0.act": "head_one2one",
    }
    for path, expected_region in examples.items():
        matches = [rule.region for rule in rules if re.search(rule.pattern, path)]
        assert matches == [expected_region], (path, matches)
