"""CPU 排程回歸：驗證失敗只補驗、RNG 隔離與大幅退化安全閘。"""
import ast
import json
from pathlib import Path
import random
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import training_b as b

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.alpha=torch.nn.Parameter(torch.tensor(0.02))
    def state_dict(self,*args,**kwargs):return {b.PREFIX+'p3_masf.alpha':self.alpha.detach().clone()}
    def load_state_dict(self,state,strict=True):
        with torch.no_grad():self.alpha.copy_(state[b.PREFIX+'p3_masf.alpha'])

class Guard:
    def check(self,model):assert not model.training

def main():
    assert not torch.cuda.is_initialized()
    cfg=json.loads(b.CONFIG.read_text());metrics=b.parent.payload()['metadata']['metrics']
    calls=[]
    with tempfile.TemporaryDirectory(prefix='pose-b-queue-test-') as folder:
        out=Path(folder);model=Tiny().eval()
        payload={'resolved_config':cfg,'progress':{'next_epoch':1},'ema_state':model.state_dict(),
                 'model_state':model.state_dict(),'loader_state':{'reports':[],'train_seconds':1}}
        snapshot=out/'checkpoints/epoch-01.pt';snapshot.parent.mkdir();torch.save(payload,snapshot)
        def export(path,**kwargs):
            calls.append('export');torch.save({'state_dict':kwargs['model'].state_dict()},path)
        def validate(*args,**kwargs):
            calls.append('validate')
            if calls.count('validate')==1:raise RuntimeError('synthetic validator failure')
            return metrics
        with patch.object(b,'save_inference_weights',export),patch.object(b,'validate',validate):
            try:b.finish_epoch(out,0,model,object(),Guard(),cfg)
            except RuntimeError as e:assert str(e)=='synthetic validator failure'
            else:raise AssertionError('測試必須捕獲驗證失敗')
            original=(out/'epochs/e1/ema.pt').read_bytes()
            assert not (out/'epochs/e1/metrics.json').exists()
            b.finish_epoch(out,0,model,object(),Guard(),cfg)
            b.finish_epoch(out,0,model,object(),Guard(),cfg)
            assert calls==['export','validate','validate']
            assert original==(out/'epochs/e1/ema.pt').read_bytes()
    @b.preserve_rng
    def consume():
        random.random();np.random.rand();torch.rand(1)
        raise ValueError('expected')
    random.seed(1);np.random.seed(1);torch.manual_seed(1)
    states=(random.getstate(),np.random.get_state(),torch.get_rng_state().clone())
    try:consume()
    except ValueError:pass
    assert random.getstate()==states[0]
    assert np.array_equal(np.random.get_state()[1],states[1][1])
    assert torch.equal(torch.get_rng_state(),states[2])
    bad=dict(metrics);bad['bbat/pose/map50_95']-=.06
    b.metric_guard(bad) # alpha-off 可記錄大降，不能把分析本身擋掉。
    try:b.metric_guard(bad,training_safety=True)
    except AssertionError:pass
    else:raise AssertionError('訓練大幅退化未攔截')
    # 靜態結構核對：已保存回合的補驗分支在新 epoch loop 之前。
    tree=ast.parse((b.HERE/'training_b.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    restored_if=next(n for n in run.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='snapshots')
    assert any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='finish_epoch' for n in ast.walk(restored_if))
    assert run.body.index(restored_if)<next(i for i,n in enumerate(run.body) if isinstance(n,ast.For))
    assert not torch.cuda.is_initialized()
    files=['training_b.py','queue_b.py','finalize_b.py','execution-config.json']
    result={'status':'passed','gpu_used':False,'tests':['驗證失敗只補驗，不重匯出／不改權重','完成回合不重驗',
       '驗證 RNG 在例外也還原','BBAT 大幅退化阻擋續訓，alpha-off 可保留退化結果','恢復分支先补驗後訓練'],
       'source_sha256':{name:b.sha256(b.HERE/name) for name in files}}
    target=b.HERE/'artifacts/queue-preflight-v1.json'
    b.save(target,result)
    print('PASS：5 項 B 組 CPU 排程與恢復回歸；無 GPU。')

if __name__=='__main__':main()
