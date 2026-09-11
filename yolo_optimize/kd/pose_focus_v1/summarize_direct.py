"""KD 結束後自動整理既有驗證，不啟動額外模型或 GPU 工作。"""
import json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    run=HERE/'artifacts/runs/kd-e5-seed1-v1'
    summary=json.loads((run/'summary.json').read_text())
    assert summary['epochs_completed']==5
    baseline=json.loads((HERE.parent/'dual_task_v1/artifacts/student-baseline-v1.json').read_text())['metrics']
    rows=[]
    for path in sorted(run.glob('validation/epoch-*/bittrue/metrics.json')):
        m=json.loads(path.read_text())['metrics']
        epoch=int(path.parent.parent.name.split('-')[1])+1
        delta={k:m[k]-v for k,v in baseline.items()}
        rows.append({'epoch':epoch,'metrics':{k:m[k] for k in baseline},'delta':delta,
            'coco_unchanged':all(abs(v)<1e-8 for k,v in delta.items() if k.startswith('coco/')),
            'failed_metrics':[k for k,v in delta.items() if v<-.001]})
    assert len(rows)==5 and all(r['coco_unchanged'] for r in rows)
    output=HERE/'artifacts/direct-kd-result-v1.json'
    report={'status':'completed','epochs':rows,'best_state':summary['best_state'],
        'qualified_best_joint_present':'best_joint' in summary['best_state'],
        'native_control_cancelled_by_user':True,
        'limits':'僅與起始學生比較；沒有完整原生配對，不能歸因所有增益為KD。尚未另做匯出独立重驗。',
        'next':'依結果決定共享特徵小範圍解凍與驗證；未自動啟動共享層訓練。'}
    with output.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('JOB_DONE: KD 結果已整理至 '+str(output),flush=True)

if __name__=='__main__':main()
