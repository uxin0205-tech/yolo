"""驗證反覆進入 train mode 仍凍結 Neck，而非只從 optimizer 排除。"""
from types import SimpleNamespace
import torch
from test_training_contract import ToyBase
from yolo_optimize.training import RunState, recovery_scope
from yolo_combine.stage_policy import build_joint_optimizer


def test_heads_scope_persists_after_training_mode():
    model = ToyBase()
    model.graph.model[16].detect_head = torch.nn.Linear(2,2)
    model.graph.model[16].pose_head = torch.nn.Linear(2,2)
    scope = recovery_scope('heads')
    optimizer, _ = build_joint_optimizer(model, scope)
    state = SimpleNamespace(model=SimpleNamespace(base=model,aux=None),
        metadata={'variant':'heads','base_lr_scale':1.0})
    for _ in range(2):
        model.train()
        RunState.training_mode(state)
        assert not model.graph.model[16].conv.weight.requires_grad
        assert all('.detect_head.' in name or '.pose_head.' in name
                   for name, p in model.named_parameters() if p.requires_grad)
    for group in optimizer.param_groups:
        if group['role'] not in ('detect_head','pose_head'):
            assert group['lr'] == 0
            assert all(not p.requires_grad for p in group['params'])
    before = model.graph.model[16].conv.weight.detach().clone()
    assert any(p.requires_grad for p in model.parameters())
    sum(p.square().sum() for p in model.parameters() if p.requires_grad).backward()
    optimizer.step()
    assert torch.equal(before, model.graph.model[16].conv.weight)
    assert scope.learning_rates['pose_head']==2.5e-5
    assert recovery_scope('native').learning_rates['neck']==1e-5
