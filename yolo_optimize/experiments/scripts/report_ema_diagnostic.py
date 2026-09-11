#!/usr/bin/env python3
"""只讀已完成 EMA age 診斷，產生中文結果與解讀邊界；不使用 GPU。"""

import argparse
import json
import math
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
LABELS = {
    'coco/box/map50_95': 'COCO Box', 'coco/person/box/map50_95': 'COCO Person Box',
    'bbat/box/map50_95': 'BBAT Box', 'bbat/pose/map50_95': 'BBAT Pose',
    'bbat/ball/box/map50_95': 'Ball Box', 'bbat/bat/box/map50_95': 'Bat Box',
    'bbat/ball/pose/map50_95': 'Ball Pose', 'bbat/bat/pose/map50_95': 'Bat Pose',
}


def render(root):
    result = json.loads((root / 'summary.json').read_text())
    if result['status'] != 'complete':
        raise ValueError('EMA 診斷尚未完成，不產生完成報告')
    parent = json.loads((root.parent / 'best-joint-revalidation/summary.json').read_text())
    variants = result['versions']
    count = result['ema_observers_at_end']['observations']
    age = result['metadata']['parent_ema_updates']
    retained = math.exp(sum(math.log(.9999 * -math.expm1(-(age + i) / 2000)) for i in range(1, count + 1)))
    lines = ['# EMA age 配對診斷：結果與後續', '',
        '## 結論', '',
        '延續 EMA age 把同一訓練軌跡的評分拉回原 BEST 附近；fresh EMA 與 live 仍退步。這確認 EMA age 對「同一軌跡的評分結果」有實測影響，不表示 live 學習已改善，也不代表新的架構帶來增益。原始 BEST 不自動替換。', '',
        '## 同口徑結果', '',
        '| BitTrue internal AP | 原 BEST | Live | Fresh EMA | Continued EMA |',
        '|---|---:|---:|---:|---:|']
    for key, label in LABELS.items():
        values = [parent['metrics']['bittrue'][key]] + [variants[name]['metrics']['bittrue'][key]
            for name in ('live', 'fresh', 'continued')]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:.8f}' for v in values) + ' |')
    values = [parent['joint_scores']['bittrue']] + [variants[name]['joint_scores']['bittrue']
        for name in ('live', 'fresh', 'continued')]
    lines += ['| Joint | ' + ' | '.join(f'{v:.8f}' for v in values) + ' |', '',
        '| Joint backend | 原 BEST | Live | Fresh EMA | Continued EMA |', '|---|---:|---:|---:|---:|']
    for backend in ('float', 'bittrue'):
        values = [parent['joint_scores'][backend]] + [variants[name]['joint_scores'][backend]
            for name in ('live', 'fresh', 'continued')]
        lines.append('| ' + backend + ' | ' + ' | '.join(f'{v:.8f}' for v in values) + ' |')
    delta = result['comparison']['bittrue']
    lines += ['',
        f'Continued 相對 fresh joint：{delta["continued_minus_fresh_joint"]:+.8f}；相對原 BEST：{delta["continued_minus_parent_joint"]:+.8f}。後者太小，不宣稱有意義或顯著改善。Continued 仍有部分 Box AP 低於 parent，不能只報 joint 上升。', '',
        '## 為何會這樣', '',
        'EMA 更新為 Eₜ = dₜEₜ₋₁ + (1−dₜ)Wₜ，其中 Wₜ 是同一份 live 權重，d(u)=0.9999×(1−exp(−u/2000))。Fresh 的 u 從 0 開始，continued 從 paired snapshot 的頂層 ema_updates 開始；兩份模型起點完全相同。', '',
        f'本次成功 {count} 個 macros；ages 由 0／{age} 變成 {count}／{age+count}。Continued 的初始 parent 係數乘積為 {retained:.8f}，亦即對可平滑 state 約保留 {100*retained:.2f}% 起點權重，所有新 live 觀察合計約 {100*(1-retained):.2f}%。固定 state 則使用 exact-copy。', '',
        '所以 continued 接近原 BEST 有合理機制：它對新、且此刻表現較差的 live 更新反應很慢。EMA 改善評分不等於 live 優化方向正確；後續仍需觀察 live，並用原 parent 作非劣比較。', '',
        '## 公平性與驗證', '',
        '這是一次 native 訓練、同步兩個 EMA observer，不是兩次獨立重複實驗。已由 4 項 CPU 測試確認：不回饋 loss／optimizer、相同 live 軌跡的各 EMA 結果與分開計算一致、AMP overflow 不多算更新、固定 state 不變。GPU 結束後的更新計數、固定 state 與驗證前後 state digest 也通過檢查。', '',
        '原 source、原 BEST、COCO80 與 canonical BBAT5 未變；HOG、MASF 移位、BinaryQK、RepConv 均未加入本次診斷。warmup=1 epoch，AdamW／LR／BN／criterion 皆沿用既定對照。完整 snapshots 在驗證之前保存，兩份共用同一 epoch-end RNG 邊界；三份 inference 權重明確區分 live／ema。', '',
        f'GPU child 執行約 {result["elapsed_seconds"] / 60:.2f} 分鐘。600 秒監測及提前完成偵測由 supervisor 執行；本次只有一個正式訓練 epoch，不能列成兩個獨立 epoch 的統計重複。', '',
        '## 後續決策', '',
        '後續既定 native5／HOG10 對照共同採 parent EMA age，兩者仍從同一原始 BEST 重建 fresh optimizer，不從本次診斷候選偷偷續跑。每 epoch 另記 live BitTrue AP 作健康診斷；模型評分與原來的安全暫停仍以 EMA 八項 AP 對 parent 為準。這是共同續訓規則，不是 HOG 特有收益。', '',
        '先做原生 5-epoch 對照並依安全 gate 決定是否繼續；HOG 的 train-only μ 尚未校準，不能宣稱 HOG 已具備精度收益。部署仍保留原 BEST。', '',
        '## 產物', '', f'結果目錄：`{root}`。', '',
        '- `summary.json`：三版本 Float／BitTrue 完整結果。',
        '- `checkpoints/fresh-epoch-0001.pt`、`continued-epoch-0001.pt`：相同 live 邊界的兩份 snapshot。',
        '- `inference/live-epoch-0001.pt`、`fresh-epoch-0001.pt`、`continued-epoch-0001.pt`：僅供診斷，未升格 BEST。', '',
        '工作紀錄見 [EMA age 診斷](../../docs/worklogs/2026-09-08-ema-age-diagnostic.md)。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=WORKSPACE / 'artifacts/direction1-20260908/ema-age-paired')
    parser.add_argument('--write', type=Path)
    args = parser.parse_args()
    report = render(args.root.resolve())
    print(report)
    if args.write:
        target = args.write.resolve()
        if not target.is_relative_to(WORKSPACE):
            raise ValueError('報告只允許寫入優化專案')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding='utf-8')


if __name__ == '__main__':
    main()
