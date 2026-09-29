"""用全部 BBAT5 val 的實際特徵量測 MASF 擾動；不抽樣或調參。"""
import math
import statistics
from pose_candidate import *
from yolo_combine.graph_materialize import build_graph_validation_models
from yolo_combine.data import prepare_bbt5_view
from yolo_combine.validation import extract_pose_metrics
from ultralytics.models.yolo.pose import PoseValidator

def main():
    initialize();torch.set_num_threads(4)
    target=HERE/'artifacts/residual-diagnostic-v1.json';assert not target.exists()
    original,source=build(True)
    pose=build_graph_validation_models(original,source,kind='bittrue').pose
    view=prepare_bbt5_view('/home/uxin/yolo/configs/datasets/bbat5-v1.yaml',HERE/'artifacts/datasets/bbat5-v1-runtime')
    state={'collect':False};ratios=[];energies=[]
    def capture(module,inputs,output):
        if not state['collect']:return
        x=inputs[0].detach().float();delta=output.detach().float()-x
        squared_x=x.flatten(1).square().sum(1);squared_delta=delta.flatten(1).square().sum(1)
        values=torch.stack((squared_x,squared_delta),1).cpu().tolist()
        for a,b in values:
            assert a>0 and math.isfinite(a) and math.isfinite(b)
            energies.append((a,b));ratios.append(math.sqrt(b/a))
    handle=pose.model[-1].p3_masf.register_forward_hook(capture)
    class ProbeValidator(PoseValidator):
        def preprocess(self,batch):
            result=super().preprocess(batch);state['collect']=True;return result
    validator=ProbeValidator(save_dir=HERE/'artifacts/residual-diagnostic-v1',args={
        'task':'pose','data':str(view.yaml),'imgsz':640,'batch':16,'workers':4,'device':'0',
        'plots':False,'save_json':False,'compile':False,'rect':True,'split':'val','mode':'val'})
    validator(model=pose.eval());handle.remove()
    assert len(validator.dataloader.dataset)==683 and len(ratios)==683
    metrics=extract_pose_metrics(validator.metrics,names=pose.names)
    expected=json.loads((HERE/'artifacts/comparison-v1/pose_masf/summary.json').read_text())['metrics']
    assert all(abs(metrics[k]-v)<1e-8 for k,v in expected.items() if k.startswith('bbat/'))
    ordered=sorted(ratios)
    alpha=float(pose.model[-1].p3_masf.alpha)
    result={'status':'passed','images':683,'sampled':False,'tuned_parameters':False,
        'metric_reproduction_with_hook':True,'warmup_excluded':True,'alpha':alpha,
        'relative_residual_l2_global':math.sqrt(sum(b for a,b in energies)/sum(a for a,b in energies)),
        'relative_residual_l2_median':statistics.median(ratios),
        'relative_residual_l2_p95':ordered[math.ceil(.95*len(ordered))-1],
        'relative_residual_l2_max':max(ratios),
        'definition':'||MASF(P3)-P3||_2 / ||P3||_2；per-image 與全資料聚合分開記錄',
        'note':'特徵擾動大小不是 AP 的因果證明；不據此調整 alpha。'}
    save(target,result);print('JOB_DONE 全量特徵殘差診斷',flush=True)

if __name__=='__main__':main()
