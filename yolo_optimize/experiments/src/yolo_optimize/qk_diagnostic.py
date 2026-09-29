"""只供免訓練 site-isolation 的 FP dot 介入，保留 bias／PWL。

這不是 VariantConfig 的 P0 FP baseline，也不宣稱可直接部署或 QAT。
原 binary score 仍先執行，因此此診斷不能用來報 latency。
"""
import torch

SITES={10:'model.10.m.0.attn.score',22:'model.22.m.0.1.attn.score'}

def fp_dot(q,k):
    if q.ndim!=4 or q.shape!=k.shape:
        raise ValueError('Q/K 必須為相同 B,H,D,N')
    return (q * (q.shape[-2]**-0.5)).transpose(-2,-1) @ k

class ScoreDiagnosticSource:
    """只為正式 validator 新建的 task 圖掛入 eval-only score 替換。"""
    def __init__(self,source,fp_site):
        if fp_site not in SITES: raise ValueError('只接受 site10／22')
        self.source,self.fp_site=source,fp_site
        self.installations=[]

    def __getattr__(self,name): return getattr(self.source,name)

    def build_task_models(self,kind):
        models=self.source.build_task_models(kind)
        for task,model in (('detect',models.detect),('pose',models.pose)):
            modules=dict(model.named_modules())
            if any(path not in modules for path in SITES.values()):
                raise ValueError('完整兩 site mapping 不符')
            score=modules[SITES[self.fp_site]]
            if str(score.basis)!='hadamard' or not bool(score.fixed_coefficients_ready):
                # templates 尚未載入正式 state，ready 通常為 false；basis 必須吻合。
                if str(score.basis)!='hadamard': raise ValueError('原 basis 不是 Hadamard')
            entry={'task':task,'backend':kind,'fp_site':self.fp_site,'calls':0}
            self.installations.append(entry)
            def replace_score(module,inputs,output,record=entry):
                if module.training: raise RuntimeError('FP score diagnostic 禁止訓練')
                record['calls']+=1
                return fp_dot(*inputs)
            score.register_forward_hook(replace_score)
        return models

    def report(self):
        return {'kind':'eval_only_fp_dot_retained_bias_and_pwl',
            'fp_site':self.fp_site,'binary_site':22 if self.fp_site==10 else 10,
            'baseline_config_modified':False,'state_tensors_modified':False,
            'qk_ste_enabled':False,'deployable':False,'latency_valid':False,
            'installations':self.installations}
