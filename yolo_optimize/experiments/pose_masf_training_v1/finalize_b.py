"""全 B 組完成後 CPU 稽核、曲線、接受門檻與中文交付，不自動升版。"""
import csv
import io
import json
from pathlib import Path
import sys
import torch
from training_b import HERE,RUN,CONFIG,PREFIX,build,FrozenGuard,initialize,sha256,save,parent
sys.path.insert(0,str(HERE.parents[1]))
from tools.restructure_layout import edit

def main():
    initialize();torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    train=json.loads((RUN/'summary.json').read_text())
    analysis=json.loads((RUN/'analysis/summary.json').read_text())
    assert train['status']==analysis['status']=='completed'
    cfg=json.loads(CONFIG.read_text());model,source=build();guard=FrozenGuard(model)
    records=[]
    for epoch in range(1,6):
        p=RUN/f'checkpoints/epoch-{epoch:02d}.pt'
        ckpt=torch.load(p,map_location='cpu',weights_only=True)
        assert ckpt['checkpoint_kind']=='full_resume' and ckpt['resolved_config']==cfg
        assert ckpt['progress']['next_epoch']==epoch and ckpt['progress']['global_macro_step']==epoch*47
        assert ckpt['scheduler_state']['current_step']==epoch*47 and ckpt['ema_updates']==epoch*47
        reports=ckpt['loader_state']['reports']
        assert len(reports)==47 and sum(r['pose_images'] for r in reports)==5964
        assert reports[-1]['pose_images']==76 and all(r['detect_images']==0 for r in reports)
        for key in ('model_state','ema_state'):
            model.load_state_dict(ckpt[key],strict=True);model.eval();guard.check(model)
        export=RUN/f'epochs/e{epoch}/ema.pt'
        inference=torch.load(export,map_location='cpu',weights_only=True)
        assert inference['metadata']['parent_sha256']==cfg['parent_inference_sha256']
        assert set(inference['state_dict'])==set(ckpt['ema_state'])
        assert all(torch.equal(v,inference['state_dict'][n]) for n,v in ckpt['ema_state'].items())
        records.append({'epoch':epoch,'resume':str(p),'resume_sha256':sha256(p),'inference':str(export),
            'inference_sha256':sha256(export),'inference_bytes':export.stat().st_size,
            'alpha_live':float(ckpt['model_state'][PREFIX+'p3_masf.alpha']),
            'alpha_ema':float(ckpt['ema_state'][PREFIX+'p3_masf.alpha']),
            'pose_loss':sum(r['pose_mean_loss']*r['pose_images'] for r in reports)/5964})
    base=analysis['baseline'];e5=analysis['e5'];off=analysis['results']['e5_alpha_off']
    keys=[k for k in base if k.endswith('/map50_95')]
    assert all(abs(analysis['results']['e5_bittrue_recheck'][k]-v)<=1e-8 for k,v in e5.items())
    coco_ok=all(abs(e5[k]-v)<=1e-8 for k,v in base.items() if k.startswith('coco/'))
    pose_ok=e5['bbat/pose/map50_95']-base['bbat/pose/map50_95']>=.002
    other_ok=all(e5[k]-base[k]>=-.001 for k in keys if k.startswith('bbat/') and k!='bbat/pose/map50_95')
    gate={'coco_exact':coco_ok,'overall_pose_gain_at_least_0_2pp':pose_ok,
          'other_bbat_ap_drop_at_most_0_1pp':other_ok,'passed':coco_ok and pose_ok and other_ok}
    assert coco_ok
    assert sha256(parent.CHECKPOINT)==cfg['parent_inference_sha256']
    best=train['best_pose_epoch'];best_metrics=train['metrics_by_epoch'][str(best)]
    buf=io.StringIO();writer=csv.writer(buf);writer.writerow(['epoch','alpha_live','alpha_ema','pose_loss',*keys])
    for r in records:writer.writerow([r['epoch'],r['alpha_live'],r['alpha_ema'],r['pose_loss'],*[train['metrics_by_epoch'][str(r['epoch'])][k] for k in keys]])
    edit(HERE/'training-curves.csv',buf.getvalue())
    buf=io.StringIO();writer=csv.writer(buf);writer.writerow(['metric','parent_E2','B5','B5_alpha_off','B_best','B5_minus_E2_pp'])
    for k in keys:writer.writerow([k,base[k],e5[k],off[k],best_metrics[k],100*(e5[k]-base[k])])
    edit(HERE/'comparison.csv',buf.getvalue())
    # 原生圖表資產，不使用生成式圖片；資料來源可逐點核對。
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for task,ax in zip(('pose','box'),axes[:2]):
        for cls in ('ball','bat'):
            k=f'bbat/{cls}/{task}/map50_95'
            ax.plot(range(6),[100*base[k]]+[100*train['metrics_by_epoch'][str(i)][k] for i in range(1,6)],marker='o',label=cls)
        ax.set(title=f'BBAT {task} AP50-95 (%)',xlabel='E2 baseline (0) -> B epoch');ax.legend();ax.grid(alpha=.25)
    axes[2].plot(range(1,6),[r['alpha_ema'] for r in records],marker='o',label='EMA alpha')
    axes[2].set(title='Pose MASF gate',xlabel='B epoch');axes[2].grid(alpha=.25)
    fig.savefig(HERE/'figures/b-training-curves.svg');fig.savefig(HERE/'figures/b-training-curves.png',dpi=150);plt.close(fig)
    lines=['# B 組 Pose MASF：訓練、驗證與最終分析','',
        '5 epoch 與完整驗證已完成。A 組已取消；本次不自動替換正式模型、不再加訓。','',
        '## 結論','',
        ('B5 通過事前工程參考門檻，可列為待人工確認候選，仍不能宣稱 MASF 獨立增準。' if gate['passed'] else
         'B5 未同時通過全部工程參考門檻，先保留固定 E2；不因單一指標較好就自動採用。'),'',
        f'最佳 Pose 回合是 B{best}；它是驗證集選出的補充候選，主要比較仍固定 B5。', '',
        '沒有 A5，B5−E2 包含額外訓練與 MASF 的共同效果。B5 關閉 α 只代表已訓模型的分支依賴，不能替代無 MASF 重訓對照。', '',
        '## 完整精度比較','', '| AP50–95 (%) | 原 E2 | B5 | B5 α=0 | 最佳 Pose 回合 | B5−E2 (pp) |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for k in keys:lines.append(f'| {k} | {100*base[k]:.4f} | {100*e5[k]:.4f} | {100*off[k]:.4f} | {100*best_metrics[k]:.4f} | {100*(e5[k]-base[k]):+.4f} |')
    lines+=['','[完整 CSV](comparison.csv)；AP50、AP75、precision、recall 與 Float 數值保存在 analysis/summary.json 及逐回合 metrics.json。',
            '', '## 訓練與 MASF 曲線','', '![B 組曲線](figures/b-training-curves.png)', '',
            '[逐回合 CSV](training-curves.csv)包含 alpha live／EMA、Pose loss 及全部 AP50–95。0 是 native QK E2 起點，不是另一組新訓練。', '',
            '## 訓練契約與驗證','',
            '只訓 Pose head＋Pose P3 MASF，其他 shared／Detect 參數与全部 BN running 固定；β=1、α=0 啟動，原生 QK＋PWL [-10,0] 20 段與 qSiLU 不變。', '',
            'AdamW、5 epoch、warmup 1、physical batch16×累積8、每回合5964張／47更新／尾端76張。Pose head LR5e-6、context1e-5、alpha1e-4；詳細設定見 [execution-config.json](execution-config.json)。', '',
            '每回合完整 COCO val5000／BBAT5 v1 val683；沒有重切或抽樣。E5 另從匯出權重重建獨立驗證並核對全指標；best 不同於 E5 時另重验並關 α。Float 與 BitTrue 分開保存。', '',
            '最終 CPU 稽核逐回合核對續訓 state／EMA／匯出逐項相等、固定參數與 BN 不变、235 次總更新及完整來源 SHA。', '',
            '## 成本與推論使用','',
            f"完整雙 head Params：{sum(p.numel() for p in model.parameters()):,}；B5 inference 檔：{records[-1]['inference_bytes']/2**20:.3f} MiB。",
            '', '[相同新增 Pose MASF 結構的既有成本量測](../pose_masf_priority_v1/RESULTS.md)只作參考，本次未重測 MAC／FLOPs、peak memory、CPU／GPU latency、target latency 或 energy/frame，不混成新實測。', '',
            '以 training_b.build() 重建、對 epochs/e5/ema.pt 使用 weights_only=True 取 state_dict 並 strict=True 載入，再 materialize 成 bittrue；權重不是官方 YOLO 可直接猜結構的通用 pickle。alpha-off 僅關 Pose gate，Detect MASF 不變。', '',
            '## 來源與剩餘限制','',
            f"固定 parent SHA：`{cfg['parent_inference_sha256']}`。所有五回合 checkpoint 的 SHA 與路徑見 [最終稽核](artifacts/b-e5-seed1-v1/final-audit.json)。",
            '', '無獨立 test／多 seed 顯著性；沒有硬體實板或能耗驗證。原 Attention／scale_bias 仍暫停，A 不執行；不自動推 Git、不覆寫舊 best。']
    edit(HERE/'RESULTS.md','\n'.join(lines))
    result={'status':'completed','gpu_used':False,'records':records,'gate':gate,'best_pose_epoch':best,
        'default_model_replaced':False,'A_canceled':True,'parent_sha256':cfg['parent_inference_sha256'],
        'frozen_all_epochs_exact':True,'exports_equal_ema':True,'total_optimizer_steps':235,
        'report':str(HERE/'RESULTS.md'),'limitations':['無 A 加訓對照','無新硬體延遲／能耗量測','無多 seed 或 test']}
    save(RUN/'final-audit.json',result)
    note=HERE/'README.md';edit(note,note.read_text()+'\n## B 組佇列完成\n\n5 epoch／獨立重驗／alpha-off／CPU 稽核已完成，現況以 [最終分析](RESULTS.md)為準；上文未啟動文字是發布當下的歷史狀態。A 未跑，沒有自動升版。\n')
    work=HERE.parents[1]/'docs/worklogs/2026-09-13-pose-masf-b-queue.md'
    edit(work,work.read_text()+'\n## ALL_DONE\n\n5 epoch、独立驗證、alpha-off 與全部權重 CPU 稽核完成。結果：'+('通過' if gate['passed'] else '未通過')+'工程參考門檻，未自動升版。無執行錯誤；統計與硬體限制見最終報告。\n')
    print('ALL_DONE B 組結果與權重 CPU 稽核完成',flush=True)

if __name__=='__main__':main()
