import importlib
import json
from pathlib import Path


def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("inventory_project")


def test_reference_cycle_terminates_and_external_file_is_not_parsed(
    tmp_path, monkeypatch
):
    tool = module(monkeypatch)
    root = tmp_path / "project"
    root.mkdir()
    a, b = root / "a.json", root / "b.json"
    external = tmp_path / "external.json"
    external.write_text("not valid json")
    a.write_text(json.dumps({"next": str(b), "outside": str(external)}))
    b.write_text(json.dumps({"back": str(a)}))
    references, warnings = tool.reference_closure(root, [a])
    assert set(references) == {str(a), str(b), str(external)}
    assert not warnings
    assert external.read_text() == "not valid json"


def test_invalid_metadata_is_a_warning_not_a_deletion(tmp_path, monkeypatch):
    tool = module(monkeypatch)
    source = tmp_path / "partial.json"
    source.write_text("{")
    references, warnings = tool.reference_closure(tmp_path, [source])
    assert not references and len(warnings) == 1
    assert source.read_text() == "{"


def test_symlink_metadata_not_followed(tmp_path, monkeypatch):
    tool = module(monkeypatch)
    source = tmp_path / "real.json"
    source.write_text("{")
    link = tmp_path / "linked.json"
    link.symlink_to(source)
    references, warnings = tool.reference_closure(tmp_path, [link])
    assert not references and not warnings
    assert link.is_symlink()
