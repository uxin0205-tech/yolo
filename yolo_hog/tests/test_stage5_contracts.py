"""主線繼承、輔助頭拆除、固定範圍與 schedule。"""

import pytest
import torch
from binaryqk.data import config
from hog_yolo import checkpoint
from hog_yolo.common import review
from hog_yolo.models import configure, criterion, parts, strip, strip_proof
from hog_yolo.training import HOGStudy

from yolo_attention import stage3_checkpoint

torch.set_num_threads(2)


def test_parent_parity_strip_and_checkpoint(tmp_path):
    model = checkpoint.build()
    parent, _ = stage3_checkpoint.load(review()["parent"])
    for n, v in parent.state_dict().items():
        assert torch.equal(v, model.state_dict()[n])
    assert strip_proof(model)["outputs_exact"]
    for index, candidate in enumerate((model, strip(model))):
        path = tmp_path / f"{index}.pt"
        checkpoint.save(path, candidate, checkpoint.manifest_for(candidate))
        actual, _ = checkpoint.load(path)
        assert all(
            torch.equal(v, actual.state_dict()[n])
            for n, v in candidate.state_dict().items()
        )
        assert hasattr(actual, "hog_aux") == (index == 0)
    model.get_submodule(stage3_checkpoint.SITES[0]).normalizer = torch.nn.Softmax(
        dim=-1
    )
    with pytest.raises(ValueError):
        checkpoint.manifest_for(model)


def test_native_mu_zero_gradients_and_hog_connectivity():
    model = checkpoint.build()
    model.args = config(2)
    configure(model)
    criterion(model)
    assert all(
        not p.requires_grad for n, p in model.named_parameters() if ".attn." in n
    )
    assert all(
        not m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)
    )
    batch = {
        "img": torch.rand(2, 3, 160, 160),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.4, 0.4, 0.3, 0.3]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[32.0], [34.0]]),
    }
    param = model.model[16].cv2.conv.weight
    native, _ = model.loss(batch)
    g0 = torch.autograd.grad(native.sum(), param)[0]
    zero, _, aux, _, _ = parts(model, batch, 0.0)
    gz = torch.autograd.grad(zero, param)[0]
    assert aux is None and torch.equal(native.sum(), zero) and torch.equal(g0, gz)
    _, _, aux, _, _ = parts(model, batch, 0.0, force=True)
    grads = torch.autograd.grad(aux.loss_sum, [param, model.hog_aux.projection.weight])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    assert not model.model[16]._forward_hooks
    assert model.criterion.o2m == 0.1
    for _ in range(15):
        model.criterion.update()
        assert model.criterion.o2m == 0.1


def test_schedule_enables_first_epoch_and_disables_tail():
    policy = object.__new__(HOGStudy)
    policy.calibration = {"mu": 0.8}
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD(
        [{"params": [p], "role": "hog", "group_name": "hog"}], lr=1.0
    )
    recipe = {"epochs": 10, "steps_per_epoch": 925, "group_lrs": {"hog": 3e-4}}
    policy.set_lr(optimizer, recipe, 0, 0, 0, False)
    assert 0 < policy.mu < 0.8
    policy.set_lr(optimizer, recipe, 7, 0, 0, False)
    assert policy.mu == 0.8
    policy.set_lr(optimizer, recipe, 8, 0, 0, False)
    assert policy.mu == 0
    assert not HOGStudy.should_stop({"minimum_epochs": 5, "patience": 4}, 4, 0, [])
    assert HOGStudy.should_stop({"minimum_epochs": 5, "patience": 4}, 5, 0, [])


def test_queue_calibration_result_contract(tmp_path, monkeypatch):
    from hog_yolo import diagnostics, workflow
    from hog_yolo.common import write_json
    from yolo_attention.queue_executor import _validate_result
    from yolo_attention.queue_store import QueueStore

    monkeypatch.setattr(workflow, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(workflow, "source_guard", lambda *a, **k: None)
    monkeypatch.setattr(workflow, "require_idle_gpu", lambda: None)
    monkeypatch.setattr(workflow, "seed", lambda *_: None)
    monkeypatch.setattr(
        diagnostics,
        "calibrate",
        lambda folder, callback: write_json(
            folder / "calibration.json", {"passed": True}
        ),
    )
    output = tmp_path / "queue"
    workflow.initialize(output)
    state = QueueStore(output).load()
    assert [j.id for j in state.jobs] == [
        "calibration",
        "smoke",
        "training",
        "verification",
        "selection",
    ]
    result = workflow.Backend().execute(state.jobs[0], state)
    _validate_result(state.jobs[0], result, state)
