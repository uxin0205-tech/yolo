"""根據最後 P2 結果選擇無MASF來源，驗證原factory可組裝且單次抽取特徵。"""
import copy
import dataclasses
import json
from pathlib import Path
from local_source import HERE, ROOT, SELECTION, LocalSource, initialize, sha256
import torch
from yolo_combine.fusion_model import assemble_graph_shared_model
from yolo_combine.graph_materialize import build_graph_validation_models
from verify_qk_challenger import tensors


def exact(a, b):
    left, right = tensors(a), tensors(b)
    assert len(left) == len(right) and all(torch.equal(x, y) for x, y in zip(left, right))


def main():
    initialize()
    torch.set_num_threads(4)
    artifacts = ROOT / 'studies/pre-fusion-full35-b100/artifacts'
    decision = json.loads((artifacts / 'masf-p2-queue-v1-state.json').read_text())
    assert decision['status'] == 'awaiting_analysis' and not decision['threshold_passed']
    report = json.loads((artifacts / 'masf-p2-control-v1/summary.json').read_text())
    assert report['status'] == 'complete'
    keys = ('coco/box/map50_95', 'coco/person/box/map50_95')
    eligible = [e for e in report['epochs'] if all(e['ema'][k] >= report['reference'][k] for k in keys)]
    assert eligible
    chosen = max(eligible, key=lambda e: e['ema'][keys[0]])
    checkpoint = artifacts / 'masf-p2-control-v1' / f"epoch-{chosen['epoch']:02d}-resume.pt"
    prior_audit = json.loads((HERE / 'artifacts/source-audit-v1.json').read_text())
    selection = {'status': 'selected_pending_full_validation', 'masf_abandoned': True,
        'detect_checkpoint': str(checkpoint), 'detect_sha256': sha256(checkpoint),
        'detect_epoch': chosen['epoch'], 'detect_expected_metrics': chosen['ema'],
        'selection_rule': 'max overall EMA among complete control epochs with overall/person >= verified prior parent',
        'pose_checkpoint': prior_audit['pose_source'], 'pose_sha256': prior_audit['pose_sha256'],
        'p2_failed': True, 'training_enabled': False, 'pwl_range': [-10, 0]}
    with SELECTION.open('x') as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
    source = LocalSource()
    source.verify_environment()
    source.verify_manifest()
    pair = source.build_task_models('float')
    image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(20261001))
    with torch.inference_mode():
        detect_expected, pose_expected = pair.detect(image), pair.pose(image)
    model, assembly = assemble_graph_shared_model(pair.detect, pair.pose)
    assert assembly.complete and assembly.audit.compatible
    counter = [0]
    hook = model.graph.model[0].register_forward_hook(lambda *args: counter.__setitem__(0, counter[0] + 1))
    try:
        with torch.inference_mode():
            actual = model(image, task='both')
        assert counter[0] == 1
    finally:
        hook.remove()
    exact(detect_expected, actual['detect'])
    exact(pose_expected, actual['pose'])
    backends = {}
    for kind in ('float', 'bittrue'):
        materialized = build_graph_validation_models(model, source, kind=kind)
        assert materialized.detect_report.complete and materialized.pose_report.complete
        assert not any('masf' in n for n, _ in materialized.detect.named_modules())
        assert not any('masf' in n for n, _ in materialized.pose.named_modules())
        backends[kind] = {'detect_complete': True, 'pose_complete': True}
    output = HERE / 'artifacts/no-masf-assembly-v1.json'
    with output.open('x') as handle:
        json.dump({'status': 'passed', 'device': 'cpu', 'training_started': False,
                   'selection': selection, 'assembly': assembly.as_dict(), 'contract': model.contract(),
                   'shared_trunk_forward_calls': counter[0], 'detect_initialization_exact': True,
                   'pose_matches_new_trunk_reference': True, 'pose_matches_old_standalone_not_claimed': True,
                   'materialization': backends}, handle, ensure_ascii=False, indent=2)
    print('PASSED: 無 MASF 融合組裝、Detect 初始化、單次 trunk、雙backend物化通過', flush=True)


if __name__ == '__main__':
    main()
