"""固定 256 張 train-only 真實 loss 校準；不更新模型、不選 val 樣本。"""
import hashlib
import json
import random
import numpy as np
from common import ROOT, SOURCE, setup, prepare_coco, sha256, write_json
import torch
from train_masf_p3 import Harness, training_mode
from continue_masf_head import SOURCE_SHA


def main():
    output = ROOT / 'artifacts/masf-task-calibration-v1'
    output.mkdir(parents=True, exist_ok=False)
    setup()
    torch.set_num_threads(8)
    source = ROOT / 'artifacts/masf-p3-fork-v1/epoch-05-resume.pt'
    assert sha256(source) == SOURCE_SHA['fork']
    Harness.variant = 'fork'
    trainer = Harness(overrides={
        'model': str(SOURCE / 'weights/bittrue/a0.pt'), 'data': str(prepare_coco()),
        'epochs': 10, 'batch': 32, 'nbs': 128, 'imgsz': 640, 'device': '0',
        'workers': 4, 'amp': False, 'optimizer': 'AdamW', 'project': str(output),
        'name': 'setup', 'exist_ok': False, 'seed': 20260927, 'deterministic': True,
        'mosaic': 0.0, 'mixup': 0.0, 'cutmix': 0.0, 'copy_paste': 0.0,
        'fliplr': .5, 'cache': False, 'fraction': 1.0, 'plots': False,
        'save_json': False, 'warmup_epochs': 1.0, 'save': False,
        'patience': 4, 'cos_lr': True, 'close_mosaic': 0,
    })
    trainer._setup_train()
    model = trainer.model
    snapshot = torch.load(source, map_location='cpu', weights_only=False)
    model.load_state_dict(snapshot['model'].state_dict(), strict=True)
    del snapshot
    training_mode(model)
    criterion = model.init_criterion()
    for _ in range(5):
        criterion.update()
    head = model.model[23]
    params = tuple(head.p3_masf.parameters())
    before = {n: t.detach().cpu().clone() for n, t in model.state_dict().items()}
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    torch.cuda.manual_seed_all(5)
    trace = hashlib.sha256()
    rows = []
    for index, batch in enumerate(trainer.train_loader):
        if index >= 8:
            break
        if index < 4:
            trace.update(json.dumps(batch['im_file']).encode())
            for key in ('img', 'cls', 'bboxes', 'batch_idx'):
                trace.update(batch[key].contiguous().numpy().tobytes())
        batch = trainer.preprocess_batch(batch)
        raw = []
        handle = head.register_forward_pre_hook(lambda module, args: raw.extend(t.detach() for t in args[0]))
        try:
            with torch.no_grad():
                model(batch['img'])
        finally:
            handle.remove()
        assert len(raw) == 3
        raw = [x.requires_grad_() for x in raw]
        native = head(raw)
        linked = head.forward_head([head.p3_masf(raw[0].detach()), raw[1].detach(), raw[2].detach()], **head.one2one)
        assert all(torch.equal(native['one2one'][k], linked[k]) for k in ('boxes', 'scores'))
        many = criterion.one2many.loss(native['one2many'], batch)[0].sum() * criterion.o2m
        one_native = criterion.one2one.loss(native['one2one'], batch)[0].sum() * criterion.o2o
        one_linked = criterion.one2one.loss(linked, batch)[0].sum() * criterion.o2o
        assert torch.equal(one_native, one_linked)
        old = torch.autograd.grad(one_native, params, allow_unused=True)
        assert all(g is None for g in old)
        gm = torch.autograd.grad(many, params)
        new = torch.autograd.grad(one_linked, params + tuple(raw), allow_unused=True)
        assert all(g is None for g in new[len(params):])
        go = new[:len(params)]
        assert all(g is not None and torch.isfinite(g).all() for g in (*gm, *go))
        a = torch.cat([g.flatten() for g in gm]).double()
        b = torch.cat([g.flatten() for g in go]).double()
        an, bn = float(a.norm()), float(b.norm())
        assert an > 0 and bn > 0
        rows.append({'batch': index, 'images': len(raw[0]), 'many_loss': float(many.detach()),
                     'one_loss': float(one_linked.detach()), 'many_grad_norm': an, 'one_grad_norm': bn,
                     'cosine': float(torch.dot(a, b) / (a.norm() * b.norm()))})
        del native, linked, raw, gm, go, new, a, b, many, one_native, one_linked
    assert len(rows) == 8 and sum(r['images'] for r in rows) == 256
    native_report = json.loads((ROOT / 'artifacts/masf-head-fork-v1/summary.json').read_text())
    assert trace.hexdigest() == native_report['first_macro_trace_sha256']
    # 每個校準 microbatch 的新增梯度不超過原 MASF 梯度 25%；固定為一個訓練係數。
    coefficient = min(.25, *(0.25 * r['many_grad_norm'] / r['one_grad_norm'] for r in rows))
    assert 1e-4 <= coefficient <= .25
    assert all(torch.equal(v, model.state_dict()[n].detach().cpu()) for n, v in before.items())
    assert sha256(source) == SOURCE_SHA['fork']
    write_json(output / 'summary.json', {
        'status': 'passed', 'source_sha256': SOURCE_SHA['fork'], 'training_updates': 0,
        'images': 256, 'subset': 'first eight batches of paired full-train loader; no validation selection',
        'first_macro_trace_sha256': trace.hexdigest(), 'precision': 'FP32',
        'native_one2one_masf_gradient_absent': True, 'linked_one2one_raw_gradient_absent': True,
        'forward_and_native_loss_exact': True, 'all_model_state_unchanged': True,
        'one2one_masf_gradient_coefficient': coefficient,
        'rule': 'min(0.25, min_batch(0.25 * norm(weighted_many_grad) / norm(weighted_one_grad)))',
        'rows': rows, 'accuracy_gain_proven': False,
    })
    print('JOB_DONE: MASF 真實 loss 梯度校準通過', flush=True)


if __name__ == '__main__':
    main()
