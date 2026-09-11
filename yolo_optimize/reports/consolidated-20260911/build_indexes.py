"""由已完成封存與實測結果建立可讀索引，不載入或修改 checkpoint。"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT/'archives/research-through-20260911-v1'


def main():
    summary = json.loads((ARCHIVE/'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['anomalies'] == 0
    records = [json.loads(line) for line in (ARCHIVE/'manifest.jsonl').read_text().splitlines()]
    assert len(records) == summary['files']
    assert len({r['archive'] for r in records}) == len(records)
    assert all((ARCHIVE/r['archive']).is_file() and (ARCHIVE/r['archive']).stat().st_size == r['bytes'] for r in records)
    weights = json.loads((ARCHIVE/'checkpoints.json').read_text())
    assert len(weights) == summary['checkpoints']
    with (HERE/'checkpoints.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(weights[0]))
        writer.writeheader()
        writer.writerows(weights)
    runs = json.loads((ARCHIVE/'run-index.json').read_text())
    with (HERE/'runs.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['path', 'status', 'archive'])
        writer.writeheader()
        writer.writerows(runs)
    baseline = ROOT/'kd/dual_task_v1/artifacts/student-baseline-v1.json'
    head = ROOT/'kd/pose_focus_v1/artifacts/direct-kd-result-v1.json'
    mix = ROOT/'inference/pose_branch_v1/artifacts/branch-v2/metrics.json'
    route = ROOT/'inference/routing_v1/artifacts/one2many-v1/summary.json'
    cases = [
        ('qSiLU E2 one2one', baseline, json.loads(baseline.read_text())['metrics']),
        ('head KD E2 one2one', head, json.loads(head.read_text())['best_state']['best_pose']['metrics']),
        ('original boxes + KD keypoints', mix, json.loads(mix.read_text())['bittrue']),
        ('qSiLU E2 one2many Pose only', route, json.loads(route.read_text())['results']['bittrue']['metrics']),
    ]
    with (HERE/'current-model-comparison.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'backend', 'metric', 'value', 'source'])
        writer.writeheader()
        for name, path, data in cases:
            for k, v in data.items():
                if k.endswith('map50_95'):
                    writer.writerow(dict(model=name, backend='bittrue', metric=k, value=v, source=str(path)))
    with (HERE/'PRESERVATION.json').open('x') as f:
        json.dump({**summary, 'archive':str(ARCHIVE), 'path_size_checks':len(records),
                   'symlink_pt_files_in_optimize':0, 'inference_only_not_exact_resume':True,
                   'checkpoints_loaded_for_this_audit':0}, f, ensure_ascii=False, indent=2)
    print(f'PASS: {len(records)} archive paths/sizes; {len(weights)} checkpoint index rows; {len(runs)} summary rows')


if __name__ == '__main__':
    main()
