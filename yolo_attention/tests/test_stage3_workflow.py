import json

import pytest

from yolo_attention.cli import main
from yolo_attention.queue_backend import ResearchQueueBackend
from yolo_attention.queue_executor import QueueExecutor
from yolo_attention.queue_model import JobKind, QueueJob, QueueState
from yolo_attention.queue_store import QueueStore
from yolo_attention.stage3_report import plot, select


def test_stage3_dry_run_never_executes_backend(tmp_path, monkeypatch):
    job = QueueJob(id='s3-diagnostics', run_name='diagnostics', stage='integrated-stage3',
                   kind=JobKind.VALIDATE, order=0, requires_gpu=True)
    QueueStore(tmp_path).initialize(QueueState.initial((job,)))
    def fail(*args):
        raise AssertionError('dry run entered GPU/backend')
    monkeypatch.setattr(ResearchQueueBackend, 'execute', fail)
    assert main(['stage3', 'run', '--output', str(tmp_path)]) == 0
    assert QueueExecutor(QueueStore(tmp_path), backend=None).preview_next().job_id == job.id


def test_full_path_selection_refuses_accuracy_regression_and_requires_all_reciprocals():
    rows = [{'candidate': name, 'formal': True, 'AP50_95': ap, 'reciprocal_table_bytes': size}
            for name, ap, size in [('N2', .5143, 0), ('D1', .5142, 544), ('D2', .5142, 37), ('D3', .5140, 136)]]
    assert select(rows, .5141)['selected'] == 'D2'
    assert select(rows, .5141)['ready']
    assert not select(rows, .52)['ready']
    with pytest.raises(ValueError):
        select(rows[:-1], .5141)


def test_report_plot_needs_only_small_json(tmp_path):
    data = {'parent_AP50_95': .5141, 'rows': [
        {'candidate': 'D1', 'formal': True, 'AP50_95': .5142, 'reciprocal_table_bytes': 544}]}
    (tmp_path/'comparison.json').write_text(json.dumps(data))
    plot(tmp_path)
    assert (tmp_path/'comparison.svg').stat().st_size > 1000


def test_diagnostics_separates_probability_pv_and_exp_error():
    import torch

    from yolo_attention.stage3_diagnostics import Accumulator
    torch.manual_seed(11)
    scores = torch.randn(2, 4, 6, 6)
    reference = scores.softmax(-1)
    values = torch.randn(2, 4, 3, 6)
    collector = Accumulator()
    collector.update('site', 'same', reference, reference, values)
    collector.scores('site', scores)
    weights = (scores-scores.amax(-1, keepdim=True)).exp()
    collector.exp_error('site', 'same', weights, weights, weights)
    rows = collector.summary()
    assert len(rows) == 4
    assert all(r['probability_mae'] == r['pv_relative_l2'] == 0 for r in rows)
    assert all(r['row_sum_error_max'] < 1e-6 for r in rows)
    assert all(r['max_error'] == r['vs_float_max'] == 0 for r in collector.exp_data.values())
