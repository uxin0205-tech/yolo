#!/usr/bin/env python3
"""HOG 完成後 CPU 對照分析與實際 Full35 layer17 RepConv 前置檢查。"""
import copy
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from yolo_optimize import runtime
import torch

def main():
    torch.set_num_threads(2)
    root = ROOT / 'artifacts/direction1-20260908'
    output = root / 'hog-results-repconv-preflight-v2.json'
    if output.exists():
        raise FileExistsError(output)
    hog = json.loads((root/'hog-parent-ema-band-v1/summary.json').read_text())
    ctrl = json.loads((root/'native-parent-ema-control-adopted/summary.json').read_text())
    parent = hog['metadata']['criteria_continuation']['parent_metrics']
    rows = []
    for h,c in zip(hog['epochs'], ctrl['epochs']):
        assert h['epoch'] == c['epoch']
        m, n = h['metrics']['bittrue'], c['metrics']['bittrue']
        rows.append({'epoch':h['epoch'], 'hog_joint':h['scores']['best_joint'],
            'control_joint':c['scores']['best_joint'],
            'joint_delta_control':h['scores']['best_joint']-c['scores']['best_joint'],
            'delta_control':{k:m[k]-n[k] for k in parent},
            'delta_parent':{k:m[k]-v for k,v in parent.items()}})
    assert len(rows)==4
    checks={}
    for key in ('base_state_sha256','seed','physical_batch','detect_logical','detect_per_macro',
                'pose_per_macro','scope','optimizer_hyperparameters','scheduler','ema'):
        # EMA fixed-state names contain the additional aux branch only if applicable;
        # record differences rather than claiming all metadata are identical.
        checks[key]=hog['metadata'][key]==ctrl['metadata'][key]
    source, model, _, _ = runtime.load_model(runtime.load_config(root),
        runtime.FINAL_ROOT/'weights/combined/inference/best_joint.pt', torch.device('cpu'))
    from ultralytics.nn.modules.conv import RepConv
    from yolo_optimize.repconv import from_conv, to_eval_conv
    from yolo_combine.graph_materialize import build_graph_validation_models
    old = model.graph.model[17]
    conv = old.conv
    assert conv.kernel_size == (3,3) and conv.stride == (2,2)
    rep = from_conv(old).eval()
    torch.manual_seed(0)
    x = torch.randn(2,conv.in_channels,17,17)
    old.eval()
    with torch.no_grad():
        a,b = old(x),rep(x)
    print(json.dumps({'probe':'repconv_transfer', 'old_eps':old.bn.eps,
        'new_eps':rep.conv1.bn.eps, 'old_dtype':str(conv.weight.dtype),
        'new_dtype':str(rep.conv1.conv.weight.dtype),
        'max_abs':float((a-b).abs().max())}), flush=True)
    assert torch.equal(a,b)
    fused = copy.deepcopy(rep)
    fused.fuse_convs()
    with torch.no_grad():
        c = fused.forward_fuse(x)
    torch.testing.assert_close(c,a,rtol=1e-4,atol=1e-5)
    for name in ('i','f','type','np'):
        if hasattr(old,name): setattr(rep,name,getattr(old,name))
    model.graph.model[17] = rep
    # Exercise actual immutable materialization; do not weaken its strict state guard.
    try:
        result = build_graph_validation_models(model,source)
    except (ValueError,RuntimeError,TypeError) as error:
        materialization={'passed':False,'error_type':type(error).__name__, 'error':str(error)}
    else:
        materialization={'passed':result.detect_report.complete and result.pose_report.complete}
    # 下一個工程步驟：僅將評分副本折回官方 Conv 的 state 格式。
    folded = to_eval_conv(rep, old)
    with torch.no_grad():
        folded_output = folded(x)
    torch.testing.assert_close(folded_output,b,rtol=1e-4,atol=1e-5)
    model.graph.model[17] = folded
    converted = build_graph_validation_models(model,source)
    converted_ok = converted.detect_report.complete and converted.pose_report.complete
    assert converted_ok
    for task_model in (converted.detect,converted.pose):
        with torch.no_grad():
            torch.testing.assert_close(task_model.model[17](x),b,rtol=1e-4,atol=1e-5)
    optimizers=[]
    for summary in (hog,ctrl):
        opt=copy.deepcopy(summary['metadata']['optimizer_hyperparameters'])
        opt.pop('auxiliary_lr')
        opt['groups']=[g for g in opt['groups'] if g['role']!='aux']
        optimizers.append(opt)
    checks['base_optimizer_excluding_hog_aux']=optimizers[0]==optimizers[1]
    assert checks['base_optimizer_excluding_hog_aux']
    result={'hog_status':hog['status'], 'completed_epochs':4, 'paired_metadata_equal':checks,
            'epochs':rows, 'repconv':{'scope':'actual Full35 layer17 CPU-only',
            'in_channels':conv.in_channels,'out_channels':conv.out_channels,
            'zero_branch_exact':True,'fused_max_abs':float((c-a).abs().max()),
            'materialization':materialization,
            'eval_conv_materialization_passed':converted_ok,
            'eval_conv_max_abs':float((folded_output-b).abs().max()),
            'full_graph_forward_validated':False},
            'notes':['不重新評分，不讀或修改資料集。',
                     '單層等價不等於完整部署契約通過，更不表示 AP 提升。']}
    runtime.write_json(output,result)
    print(json.dumps({'epochs':rows,'paired_metadata_equal':checks,'repconv':result['repconv']},
                     ensure_ascii=False),flush=True)

if __name__=='__main__': main()
