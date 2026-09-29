"""交付前核對實際權重、數值、來源與暫停狀態；不使用 GPU。"""
import math
from pose_candidate import *

def main():
    initialize();torch.set_num_threads(4)
    summary=json.loads((HERE/'artifacts/summary-v1.json').read_text())
    bench=json.loads((HERE/'artifacts/benchmark-v1.json').read_text())
    diagnostic=json.loads((HERE/'artifacts/residual-diagnostic-v1.json').read_text())
    assert summary['status']=='completed' and bench['status']=='completed' and diagnostic['status']=='passed'
    rows=summary['cases'];base=rows['baseline']['metrics']
    for name,row in rows.items():
        assert row['status']=='passed' and row['coco_val_images']==5000 and row['bbat_val_images']==683
        assert row['parent_sha256']==sha256(CHECKPOINT)
        assert all(math.isfinite(float(v)) for v in row['metrics'].values())
        assert sha256(Path(row['checkpoint']))==row['sha256']
        if name=='pose_alpha_zero':assert all(abs(row['metrics'][k]-v)<1e-8 for k,v in base.items())
    assert all(abs(rows['pose_masf']['metrics'][k]-v)<1e-8 for k,v in base.items() if k.startswith('coco/'))
    candidate,_=build(True)
    exported=torch.load(rows['pose_masf']['checkpoint'],map_location='cpu',weights_only=True)
    assert exported['checkpoint_kind']=='inference_only'
    assert exported['metadata']['metrics']==rows['pose_masf']['metrics']
    before=candidate.state_dict();state=exported['state_dict']
    assert before.keys()==state.keys() and all(torch.equal(before[k],state[k]) for k in before)
    image=torch.linspace(0,1,3*160*160).reshape(1,3,160,160)
    with torch.inference_mode():expected=tensors(candidate(image,task='both'))
    candidate.load_state_dict(state,strict=True)
    with torch.inference_mode():actual=tensors(candidate(image,task='both'))
    assert len(expected)==len(actual)
    for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
    pwl=verify_pwl(candidate)
    pause=json.loads(PAUSE.read_text())
    assert pause['status']=='PAUSED' and sha256(Path(pause['checkpoint']))==pause['checkpoint_sha256']
    assert (ATTENTION/'artifacts/queue-v1/pause-request.json').exists()
    added_mac=bench['rows']['pose_masf']['operation_estimate']['mac']-bench['rows']['baseline']['operation_estimate']['mac']
    assert added_mac==80*80*(256*9+256*25+256*256)==475136000
    assert rows['pose_masf']['parameters']-rows['baseline']['parameters']==75777
    assert diagnostic['images']==683 and diagnostic['metric_reproduction_with_hook']
    for row in bench['rows'].values():
        assert row['cpu']['samples']==20 and row['gpu']['samples']==100
        assert all(math.isfinite(row[device][key]) and row[device][key]>0 for device in ('cpu','gpu') for key in ('median_ms','p95_ms','mean_ms'))
    save(HERE/'artifacts/final-audit-v1.json',{'status':'passed',
        'all_metrics_finite':True,'baseline_matches_pinned_parent':rows['baseline']['metrics']==payload()['metadata']['metrics'],
        'all_coco_metrics_unchanged':True,'all_alpha_zero_metrics_equal_baseline':True,
        'export_state_exact':True,'export_reload_outputs_exact':True,'pwl_sites':len(pwl),
        'all_checkpoint_sha256_verified':True,'full_resume_e2_unchanged':True,
        'extra_mac_formula_verified':True,'other_training_still_paused':True,
        'gpu_used':False,'no_new_training_performed':True})
    print('PASS 最終權重／指標／alpha-zero／MAC／原 E2 續訓檔核對。',flush=True)

if __name__=='__main__':main()
