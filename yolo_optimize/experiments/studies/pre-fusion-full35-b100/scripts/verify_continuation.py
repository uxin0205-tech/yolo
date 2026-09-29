"""檢查成對實際續訓的資料 trace、optimizer step 與 EMA age，非無中斷 replay。"""
import json
from common import ROOT,write_json
import torch
import qk_challenger


def main():
    reports={}
    for arm in ('control','qk'):
        path=ROOT/f'artifacts/a0-{arm}-continue-smoke-v2'
        report=json.loads((path/'summary.json').read_text())
        payload=torch.load(path/'epoch-02-resume.pt',map_location='cpu',weights_only=False)
        steps={int(s['step']) for s in payload['optimizer']['state'].values() if 'step' in s}
        assert steps=={926},steps
        assert payload['ema_updates']==926 and report['initial_ema_updates']==925
        assert payload['model'].criterion.updates==1
        assert report['optimizer_steps']==926 and report['smoke_images']==128
        reports[arm]={'optimizer_steps':sorted(steps),'ema_updates':payload['ema_updates'],
            'criterion_updates':payload['model'].criterion.updates,'trace':report['first_macro_trace_sha256']}
        del payload
    assert reports['control']['trace']==reports['qk']['trace']
    old=json.loads((ROOT/'artifacts/a0-control-smoke/summary.json').read_text())['first_macro_trace_sha256']
    assert reports['control']['trace']!=old,'續訓不應重播相同 E1 增強資料'
    write_json(ROOT/'artifacts/continuation-proof-v2.json',{'status':'passed','arms':reports,
        'exact_uninterrupted_dataloader_replay':False,'paired_new_stream':True})
    print('CONTINUATION_PASSED',reports)


if __name__=='__main__':main()
