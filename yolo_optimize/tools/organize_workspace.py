"""一次性目錄整理：保留歷史、搬移文件與工具，不動訓練程式／權重／raw metrics。"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'docs/history/organization-20260912'
LINK = re.compile(r'(?<!!)\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)')
TOOLS = ('archive_research.py', 'archive_addendum.py', 'prepare_report_publication.py')
PDFS = {
    'Design and Implementation of a Multi-Precision Deep Learning Accelerator for YOLOX-Based Object Detection in Synthetic Aperture Radar Images.pdf':
        'docs/references/papers/sar-yolox-multiprecision.pdf',
    'experiments/combine/碩一_陳宥炘_0831.pdf': 'docs/references/user-reports/combine-experiment4-0831.pdf',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def edit(path, text):
    """文件編輯統一經 apply_patch；不以 shell 重導向覆蓋。"""
    path = ROOT/path
    text = text.rstrip()+'\n'
    if path.exists():
        old=path.read_text()
        patch='*** Begin Patch\n*** Update File: '+str(path)+'\n@@\n'
        patch+=''.join('-'+line+'\n' for line in old.splitlines())
        patch+=''.join('+'+line+'\n' for line in text.splitlines())
    else:
        patch='*** Begin Patch\n*** Add File: '+str(path)+'\n'
        patch+=''.join('+'+line+'\n' for line in text.splitlines())
    subprocess.run(['apply_patch'],input=patch+'*** End Patch\n',text=True,check=True,stdout=subprocess.DEVNULL)


def inventory():
    protected={}
    for folder,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=[d for d in dirs if d not in {'archives','datasets','__pycache__','.git','.venv'}
                 and not (Path(folder)/d).is_symlink()]
        for name in names:
            p=Path(folder)/name
            rel=p.relative_to(ROOT)
            if p.is_symlink(): continue
            training_code=p.suffix=='.py' and 'maintenance' not in rel.parts and rel.as_posix() not in {'experiments/scripts/'+n for n in TOOLS}
            evidence=p.suffix in {'.pt','.pth','.onnx','.engine'} or ('artifacts' in rel.parts and p.suffix in {'.json','.csv','.jsonl'})
            if training_code or evidence:
                s=p.stat()
                protected[str(rel)]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
    return protected


def save_history(rel):
    src=ROOT/rel
    if not src.exists(): return None
    raw=src.read_bytes()
    stem=rel.replace('/','__').removesuffix('.md')
    backup=HISTORY/(stem+'.txt')
    shutil.copyfile(src, backup)
    dst=HISTORY/(stem+'.md')
    def rewrite(m):
        label,target=m.groups();target=target.strip('<>')
        if '://' in target or target.startswith('#'):return m.group(0)
        path,sep,anchor=target.partition('#')
        local=Path(os.path.normpath(src.parent/unquote(path)))
        if local.is_relative_to(ROOT):
            local=ROOT/PDFS.get(str(local.relative_to(ROOT)),str(local.relative_to(ROOT)))
        target=os.path.relpath(local,dst.parent)+(('#'+anchor) if sep else '')
        return '['+label+'](<'+target+'>)'
    text='# 歷史入口快照（非目前狀態）\n\n原路徑：`'+rel+'`。原始位元組另存同名 `.txt`；整理日期 2026-09-12。\n\n'+LINK.sub(rewrite,raw.decode())
    edit(dst.relative_to(ROOT),text)
    return {'original':rel,'backup':str(backup.relative_to(ROOT)),'readable':str(dst.relative_to(ROOT)),
            'sha256':hashlib.sha256(raw).hexdigest()}


PAGES = {
'README.md': '''# YOLO Optimize

從 BinaryQK、MASF 到 Detect＋Pose 融合、activation、KD 與推論的研究工作區。

## 先看這三個入口

- **看完整成果**：[詳細總報告](reports/consolidated-20260911/README.md)／[從 BinaryQK 開始的閱讀順序](reports/5090-done-0912/README.md)。
- **找模型與 checkpoint**：[權重導航](reports/checkpoints/README.md)。原始權重不搬移，研究候選不等於合格最佳模型。
- **找程式與資料夾**：[目錄地圖](docs/WORKSPACE.md)／[各階段結果總覽](reports/README.md)。

## 目前狀態

新主線預設仍為 qSiLU P3 bridge E2；並未全面追回舊 combine 的 BBAT 表現。KD、分支重組與 one2many 推論實验都有結果，但未產生可全面替換預設的新模型。沒有本研究的待執行 queue；資料夾整理不啟動 GPU。

| 找什麼 | 位置 |
| --- | --- |
| 早期融合後研究 | [artifacts](artifacts/README.md)／[結果稽核](optimizations/integrated-roadmap/round1-audit.md) |
| 融合前方向 1：BinaryQK／HOG／RepConv／MASF | [studies](studies/README.md) |
| 新 Detect＋Pose 融合與恢復 | [combine](combine/README.md) |
| qSiLU／SiLU 等 activation | [activation](activation/README.md) |
| 雙教師與 Pose-head KD | [kd](kd/README.md) |
| 推論分支比較 | [inference](inference/README.md) |
| 原始假說與未執行提案 | [optimizations](optimizations/README.md) |
| 論文、工作紀錄與歷史 | [docs](docs/README.md) |
| 共用程式／操作工具／測試 | [src](src/README.md)／[scripts](scripts/README.md)／[tests](tests/README.md) |
| 本機保存副本 | [archives](archives/README.md)，不是日常實驗入口 |

## 資料與安全邊界

BBAT 固定使用 `/home/uxin/yolo/original/pose/derived/bbat5-v1/`，registry 為 `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`；COCO80 使用 `/home/uxin/yolo/coco2017.yaml`。不重切、不改 labels；詳見[資料集規範](../docs/agents/bbat5-datasets.md)。

歷史 checkpoint、原始指標及兩個封存包保留原位。新工作不覆寫既有 run；不要直接執行舊 queue。全部修改記於[中文工作紀錄](docs/worklogs/README.md)。

GitHub 已發布 `5090 Done 0912`：`927c42bd`，[發布確認](reports/5090-done-0912/PUBLISHED.md)。本次目錄整理僅在本機完成，尚未另行提交／上傳。舊首頁所有歷史進度已移至[歷史入口](docs/history/README.md)，不再堆在首頁。
''',
'docs/WORKSPACE.md': '''# 目錄地圖與存放規則

## 閱讀與工作分開

```text
yolo_optimize/
├── README.md             目前狀態與總入口
├── reports/              正式結果、GitHub 發布、checkpoint 導航
├── docs/
│   ├── research/         論文分析與方法推導
│   ├── references/       使用者 PDF（papers／user-reports）
│   ├── history/          整理前的入口快照；不是目前狀態
│   └── worklogs/         每次操作與驗證紀錄
├── optimizations/        原始優化提案；與實測結果分開
├── studies/              融合前方向 1
├── combine/              融合與 Pose 恢復
├── activation/           Activation 實驗
├── kd/                   雙教師／Pose-head KD
├── inference/            推論實驗
├── artifacts/            早期融合後原始產物
├── src/                  共用模組
├── scripts/
│   └── maintenance/      封存、報告發布與整理工具
├── tests/                CPU 契約測試
└── archives/             固定保存副本，不作 active run
```

實驗目錄仍分開保留，是因程式使用 `Path(__file__).resolve().parents[...]`、跨模組 import 與固定 checkpoint 路徑。未重新設計來源解析前，不把它們硬搬進新的父目錄；本次没有製造第二套大型資料或訓練根。

## 各種檔案放哪裡

| 類型 | 存放位置／規則 |
| --- | --- |
| 結論與跨階段比較 | reports 下新增具日期／用途的報告，不覆蓋舊報告 |
| 尚未測試的想法 | optimizations；明記 proposed，不寫成已提升 |
| 文獻解讀 | docs/research；原 PDF 放 docs/references |
| 操作過程 | docs/worklogs；同步索引，不再把過程貼滿各層 README |
| 訓練／驗證程式 | 所屬階段目錄；共用模組才放 src |
| checkpoint／原始 metrics | 所屬 run 的 artifacts，維持 inference／checkpoints 區分 |
| 模型入口 | reports/checkpoints，只保存指向原檔的索引，不複製權重 |
| 保存副本 | archives；已完成的版本不修改 |

## 這次實際搬移

两份 PDF 已移至 docs/references；三個封存／發布工具移至 scripts/maintenance。舊的長 README 完整保存為歷史 Markdown 與原始 txt，現行入口改為短版。精確舊／新路徑、hash 與保護檢查見[整理紀錄](history/organization-20260912/manifest.json)。

## 不清除的內容

checkpoint、raw metrics、失敗 log、queue 事件、cache 及封存全部 Keep。沒有刪除候選或刪除操作；本次整理不是回收磁碟容量。所有已訓練／候選模型維持原路徑，不把 `best_pose.pt`、`best_joint.pt` 的檔名當成跨階段合格標章。
''',
'docs/README.md': '''# 文件中心

| 目的 | 入口 |
| --- | --- |
| 看資料夾怎麼使用 | [目錄地圖](WORKSPACE.md) |
| 看實際成果 | [正式報告](../reports/README.md) |
| 看論文研究與推導 | [research](research/README.md) |
| 找原始 PDF | [references](references/README.md) |
| 看執行、修正與驗證 | [worklogs](worklogs/README.md) |
| 找整理前的歷史入口 | [history](history/README.md) |

工作紀錄是過程證據，原始提案在 optimizations，最終結果在 reports；三者用途不同。資料集政策仍以[全域 BBAT5 規範](../../docs/agents/bbat5-datasets.md)為準。
''',
'reports/README.md': '''# 正式結果與交付

| 入口 | 用途 |
| --- | --- |
| [5090 Done 0912](5090-done-0912/README.md) | 從 BinaryQK 開始的完整閱讀順序與 GitHub 發布 |
| [全階段詳細總報告](consolidated-20260911/README.md) | 架構、所有階段、AP、超參數、失敗原因與限制 |
| [融合前方向 1](direction1-20260910/README.md) | FP／A0／B100 與 HOG／RepConv／MASF 對照 |
| [權重導航](checkpoints/README.md) | 哪份模型現在使用、哪些只供比較、完整 checkpoint 清單 |

階段詳細結果：[MASF](../combine/pose-masf/RESULTS.md)、[融合恢復](../combine/bridge_v1/BBAT_RECOVERY_RESULTS.md)、[Activation](../activation/bridge_v1/RESULTS.md)、[Pose-head KD](../kd/pose_focus_v1/README.md)、[推論](../inference/README.md)。

本地保存清單不是模型已上 GitHub 的證明；大型權重與封存不隨報告發布。歷史報告內的「尚未開始」只代表當時，最新入口不沿用那些狀態。
''',
'experiments/studies/README.md': '''# 階段 1：融合前研究

本區是獨立 Detect 的 Full35-B100 方向 1，不是早期融合後研究。

- [執行目錄與模型來源](pre-fusion-full35-b100/README.md)
- [正式結果報告](../reports/direction1-20260910/README.md)
- [全部階段總覽](../reports/README.md)

程式放研究目錄 scripts，checkpoint／metrics 保留 artifacts。沒有待執行 queue；現有路徑不搬移。
''',
'experiments/studies/pre-fusion-full35-b100/README.md': '''# 融合前 Full35-B100：方向 1

狀態：BinaryQK 梯度恢復、narrow／late 適應、HOG、RepConv17、P3 shared／fork／bridge、P2 最後對照均已完成或依門檻停止。使用者選 P3 bridge E8 接續後續融合；這不代表它已明顯勝過無 MASF 配對。

| 找什麼 | 位置 |
| --- | --- |
| 結果／架構／成本 | [方向 1 報告](../../reports/direction1-20260910/README.md) |
| P3／P2 BBAT 比較 | [MASF 報告](../../combine/pose-masf/RESULTS.md) |
| 程式 | scripts/；保留固定目錄深度 |
| 每組訓練與診斷 | artifacts/；含 checkpoint、summary、失敗紀錄 |
| 已驗證候選 | artifacts/direction1-candidate-verification-v1/ |
| 模型選擇與完整清單 | [權重導航](../../reports/checkpoints/README.md) |

P3 bridge COCO AP 0.508212／person 0.627664；仍低於 FP，額外模組收益未過方法 gate。PWL 保持 [-10,0]／20 段。資料是完整 COCO80；BBAT 觀察使用 canonical v1。沒有新增 P2 Detect head，也沒有本研究 P2 Pose 訓練成果。

原始分段計畫、當時的 queue 狀態與全部文字保留於[歷史入口](../../docs/history/README.md)。新工作應另開 run，不直接重啟舊 queue。
''',
'experiments/combine/README.md': '''# 階段 2：Detect＋Pose 融合

| 區域 | 用途／狀態 |
| --- | --- |
| [bridge_v1](bridge_v1/README.md) | 使用者選定 P3 bridge 的新融合主線，已完成融合與恢复試驗 |
| [pose-masf](pose-masf/RESULTS.md) | 現有 MASF 權重的 COCO／BBAT 比較及架構說明 |
| artifacts/、full35/、configs/ | 初期無 MASF J0 與準備流程的原始產物，歷史保留 |
| monitor.py | 共用 600 秒子工作監測工具；本次沒有新 queue |

詳細[新舊 combine 與恢復結果](bridge_v1/BBAT_RECOVERY_RESULTS.md)。目前新模型较保護 COCO，但 bat 等 BBAT 指標仍有缺口；沒有原嚴格 gate 全過的新模型。

使用者實驗 4 PDF 已移至[文件參考區](../docs/references/README.md)。其他研究程式、設定、checkpoint 與 artifacts 路徑不改，避免破壞來源載入。舊入口與初期排程見[歷史快照](../docs/history/README.md)，不要直接啟動舊 run_pose_pair.py。
''',
'experiments/combine/bridge_v1/README.md': '''# P3 bridge 融合主線

完整 Pose、J0、balanced J1/J2/J3、Pose head 恢復及 BN 校準已有結果；之後 activation 與 KD 在各自獨立目錄。沒有本研究待執行 queue。

- [融合／恢復詳細結果](BBAT_RECOVERY_RESULTS.md)
- [目前計畫狀態](plan.json)
- [全階段總報告](../../reports/consolidated-20260911/README.md)
- [權重與續訓檔導航](../../reports/checkpoints/README.md)

`artifacts/fusion/` 保存每個完整 run，`inference/` 與 `checkpoints/` 用途不同；前者不含完整 optimizer 狀態。原 J3 恢復 E5 作為 activation 的父模型，非所有融合 gate 通過的 best_joint。

MASF 只接 Detect P3；Pose 使用 raw P3/P4/P5，因此推論只切 MASF α 不會直接改 Pose。原始超參數、分段理由與舊執行中訊息已移入[歷史入口](../../docs/history/README.md)。本區程式位置不改，舊來源與所有失敗紀錄保留。
''',
'experiments/activation/README.md': '''# 階段 3：Activation

現行研究：[P3 bridge activation](bridge_v1/README.md)。四臂 zero-shot 與 SiLU／qSiLU 各 10 輪已完成，沒有新 queue。

[詳細結果](bridge_v1/RESULTS.md)／[qSiLU 權重入口](../reports/checkpoints/README.md)。Hardswish、PolyShift 未追加訓練；不把固定多項式結構稱為已測板端加速。
''',
'experiments/activation/bridge_v1/README.md': '''# P3 bridge Activation 配對

狀態：已完成。相同父模型下 SiLU／qSiLU 各 10 輪，qSiLU E2 為新主線起點；不是每項都優於 SiLU，也沒有通過原始所有融合 gate。

- [完整表格、超參數與限制](RESULTS.md)
- [權重導航](../../reports/checkpoints/README.md)
- [原始指標與 runs](artifacts/)

重建入口為 `verify_selected.py::SelectedSource`，必須恢復 qSiLU 類型、PWL [-10,0]／20 段與 BinaryQK，不可直接以原 SiLU 架構載入 state dict。舊配對 queue 已完成，不應重跑；當時規劃保留於[歷史入口](../../docs/history/README.md)。
''',
'experiments/kd/README.md': '''# 階段 4：知識蒸餾 KD

| 研究 | 實際結果 |
| --- | --- |
| [dual_task_v1](dual_task_v1/README.md) | COCO 用 YOLO26L 教師、BBAT 用獨立 Pose 教師；K0／KD 各 5 輪，無合格新 best_joint |
| [pose_focus_v1](pose_focus_v1/README.md) | 固定共享／Detect，只更新完整 Pose head；5 輪完成，E2 keypoints 小升但框退化 |

兩者都由 qSiLU E2 開始，不串接退化末輪。教師不進部署模型，資料不混 class IDs。共享 Conv 小範圍適應與創新區域 KD 仍是未執行方向；沒有待執行 queue。
''',
'experiments/kd/dual_task_v1/README.md': '''# 雙教師 KD：已完成

qSiLU E2 為起點，K0／空間 KD 各 5 輪已完成。KD E4 有 bat Pose 收益，但 ball 框下降，沒有新的合格 best_joint；原起點不被替換。

- [方法、teacher、μ、超參數與 gate](PLAN.md)
- [後续 Pose-only 的推導與原計畫](POSE_ONLY_NEXT_PLAN.md)
- [實際結果與全階段比較](../../reports/consolidated-20260911/README.md)
- 原始資料：artifacts/runs/、artifacts/teacher-validation-v1/、artifacts/training-queue-v1/。

MuSGD 在本起點校準未過，主線使用 AdamW；不外推為 MuSGD 普遍無效。`run_training_pair.py` 是已完成實驗入口，不是現在應直接啟動的 queue。舊說明見[歷史快照](../../docs/history/README.md)。
''',
'experiments/kd/pose_focus_v1/README.md': '''# Pose-head KD：已完成

5 輪／1,865 macro 已完成；正式 native 對照依使用者指示取消，失敗／取消產物保留。固定全部非 Pose state，COCO／person 精確不變；最佳 E2 的 Pose AP 小升但 ball／bat 框下降，沒有合格 best_joint。

| BitTrue AP | qSiLU 起點 | head KD E2 |
| --- | ---: | ---: |
| BBAT 框 | 0.618008 | 0.613915 |
| BBAT Pose | 0.891329 | 0.892962 |
| ball 框／Pose | 0.505192／0.859649 | 0.501879／0.861716 |
| bat 框／Pose | 0.730823／0.923008 | 0.725951／0.924207 |

[完整逐輪結果](artifacts/direct-kd-result-v1.json)／[權重角色](../../reports/checkpoints/README.md)／[全階段報告](../../reports/consolidated-20260911/README.md)。原框＋KD keypoints 的後續[推論重組](../../inference/pose_branch_v1/README.md)也沒有收益，不能拼接最高指標。

實際設定：AdamW、Pose head LR1e-5、warmup1、batch16、5 epochs、patience0，固定 μ10.063553373151326 對應 feature 梯度5%。沒有本研究待執行 queue，共享 Conv 方案尚未實施。原提案與取消前配對安排保留於[歷史快照](../../docs/history/README.md)。
''',
'experiments/inference/README.md': '''# 階段 5：推論處理

| 實驗 | 結論 |
| --- | --- |
| [pose_branch_v1](pose_branch_v1/README.md) | 原框分類＋KD E2 keypoints：框精確保留，但 Pose 無增益，不升版 |
| [routing_v1](routing_v1/README.md) | one2many＋NMS：bat 明顯提升、ball 下降，不全面採用 |

這兩項都是既有權重的推論驗證，不是新訓練。預設仍是原 qSiLU one2one；ball／bat 分流 routing 尚未實作、未量測雙分支成本。每個子目錄包含程式、參數、原始 metrics 與 README，未合併覆寫。
''',
'proposals/README.md': '''# 優化假說與原始計畫

本區保存各方向最初的推導、對照設計與停止條件，不作現在的 queue 狀態。實測結果以[總報告](../reports/consolidated-20260911/README.md)及各實驗目錄為準；子計畫的 proposed／未開始可能是歷史文字。

| 方向 | 原始計畫 | 現行判斷 |
| --- | --- | --- |
| BinaryQK | [accuracy-recovery](binaryqk-accuracy-recovery/README.md) | 已做正式梯度修復與分段訓練，未完全追回 FP |
| 固定 scale／codebook | [scale-codebook](binaryqk-scale-codebook/README.md) | 固定尺度等價改善有證據；新8選1 selector未實施 |
| HOG | [HOG 計畫](p3-hog-companion-training/README.md) | 兩條研究線都有試驗，目前版本未採用 |
| P3 MASF | [Detect entry](p3-masf-detect-entry/README.md) | P3／P2／bridge有結果；使用者選bridge，不代表通過額外增準gate |
| 訓練衝突 | [conflict-safe](training-conflict-safe/README.md) | 投影啟動條件沒有足夠支持，未實施 |
| 已訓模型恢復 | [recovery](trained-model-recovery/README.md) | 原始設計保留；後續新combine與KD結果另列 |
| 第二輪創新 | [round2](round2-innovation/README.md) | 普通KD已測，區域／排序創新未完成 |
| person-only | [專用head](coco-person-specialized-head/README.md) | 依使用者要求延期，不建立新資料 |
| 整合計畫 | [integrated-roadmap](integrated-roadmap/README.md) | 早期融合後計畫與實測稽核，不冒充現在主線 |

新想法另建資料夾並清楚標記 proposed；程式與大型產物放實驗所屬階段。不要直接修改舊方法／parent來掩蓋失敗。整理前完整索引見[歷史入口](../docs/history/README.md)。
''',
'experiments/artifacts/README.md': '''# 早期融合後原始產物

本根層 artifacts 是舊融合後方向 1 的研究資料，不是所有新階段的共同輸出目錄。

- `direction1-20260908/`：BEST 重驗、EMA、native／HOG／RepConv／MASF、heads 恢復等原始 run。
- `direction1-20260909/`：逐圖定位與排序診斷。
- [該階段實際結果](../optimizations/integrated-roadmap/round1-audit.md)。

全部保留原路徑，不搬移、不覆寫、不刪除。新的研究使用各自階段 artifacts。此處不是待執行 queue；歷史 epoch 與現在模型不能混為同一 parent。
''',
'experiments/scripts/README.md': '''# 操作工具

## 維護／報告（不訓練）

[maintenance](maintenance/README.md)：封存、GitHub 報告準備、目錄整理。三個工具已由 scripts 根層移入此處；舊命令需更新路徑。

## 早期融合後研究工具（位置保留）

| 類型 | 主要檔案 |
| --- | --- |
| 訓練與監測 | run_recovery.py、supervise_recovery.py、start_quiet_recovery.py、quiet_recovery_supervisor.py、blocking_job_monitor.py、monitor_existing_recovery.py |
| EMA／BN／LR | run_ema_diagnostic.py、report_ema_diagnostic.py、diagnose_native_recovery.py、analyze_native_control.py |
| 模型與安全稽核 | audit_recovery_states.py、audit_head_updates.py、verify_recovery_safety.py、audit_accuracy_regressions.py |
| Pose／視覺診斷 | audit_pose_errors.py、analyze_pose_ranking.py、analyze_pose_localization.py、probe_pose_classifier.py、render_pose_error_report.py |
| HOG／RepConv／scale | analyze_hog_and_repconv.py、audit_repconv_seams.py、verify_rep17_integration.py、verify_fixed_scale.py、validate_fixed_scale.py |
| 優化器與結果表 | probe_musgd_updates.py、build_round1_evidence.py、report_recovery.py |

這些不是目前應直接執行的 queue。後來融合前／combine／activation／KD程式留在各階段，避免跨研究誤用入口。沒有為整理而重跑訓練。
''',
'tools/README.md': '''# 維護工具

| 工具 | 用途與安全邊界 |
| --- | --- |
| archive_research.py | 使用全新 --name 建立封存並驗證 SHA；不覆蓋既有包 |
| archive_addendum.py | 2026-09-11 補充封存的一次性工具；固定目的地已存在，不應再次執行 |
| prepare_report_publication.py | 在乾淨隔離 worktree 建立報告發布樹；不 commit／push，不發布權重 |
| organize_workspace.py | 本次一次性整理；已有 manifest 時拒絕再次執行，不是每次開案都要跑 |

從專案根層使用 `scripts/maintenance/` 新路徑。工具根目錄解析已隨搬移修正；保留checkpoint／raw metrics／來源資料，不載入模型。封存與發布工具的存在不等於授權再次封存或上傳。
''',
'experiments/src/README.md': '''# 共用研究模組

`yolo_optimize/` 保存早期融合後研究使用的 training、安全保護、HOG、RepConv、MASF bridge、QK、EMA與固定scale模組。實際階段入口在 scripts 或各研究資料夾。

本次只補導航，沒有搬移或修改這些模組。相應 CPU 測試在 [tests](../tests/README.md)，報告見 [reports](../reports/README.md)。
''',
'experiments/tests/README.md': '''# CPU 契約測試

本區測試共用模組的凍結／資料保護、EMA、HOG、RepConv、QK／scale與訓練邊界。部分測試需要本機原始source bundle及Python環境，不是只clone報告就能執行的完整部署驗收。

目錄整理只驗證搬移工具的root解析、CLI help、文件連結與產物保護，不重跑無關的訓練測試或GPU。本次未改src與原有tests；不把語法檢查稱為精度或硬體效能測試。
''',
'docs/references/README.md': '''# 使用者提供的參考文件

| 文件 | 新位置 | 研究報告 |
| --- | --- | --- |
| Multi-Precision YOLOX／SAR 論文 | [papers/sar-yolox-multiprecision.pdf](papers/sar-yolox-multiprecision.pdf) | [第3.1節訓練方法適配](../research/2026-09-04-paper31-to-yolo26m-training-adaptation.md) |
| 碩一＿陳宥炘＿0831 報告 | [user-reports/combine-experiment4-0831.pdf](user-reports/combine-experiment4-0831.pdf) | [實驗4分析](../research/2026-09-10-combine-report-experiment4.md) |

兩份文件只改存放位置與方便使用的檔名，bytes與SHA256不變；原檔名、舊位置見[整理manifest](../history/organization-20260912/manifest.json)。不擅自上傳使用者／第三方PDF；原始文字內容沒有更改。
''',
}


def main():
    assert not HISTORY.exists(), '已有整理紀錄，拒絕覆蓋或重跑'
    for src,dst in PDFS.items():
        assert (ROOT/src).is_file() and not (ROOT/dst).exists()
    for name in TOOLS:
        assert (ROOT/'scripts'/name).is_file() and not (ROOT/'experiments/scripts/maintenance'/name).exists()
    protected=inventory()
    HISTORY.mkdir(parents=True)
    edit(HISTORY/'protected-before.json', json.dumps(protected,indent=2))
    histories=[]
    for rel in PAGES:
        row=save_history(rel)
        if row:histories.append(row)
    moves=[]
    for old,new in PDFS.items():
        src,dst=ROOT/old,ROOT/new
        sha=digest(src);dst.parent.mkdir(parents=True,exist_ok=True)
        src.rename(dst)
        assert digest(dst)==sha
        moves.append(dict(old=old,new=new,sha256=sha,kind='unchanged_pdf'))
    # 只編輯並搬移三個維護工具，訓練入口不動。
    for name in TOOLS:
        src=ROOT/'scripts'/name;dst=ROOT/'experiments/scripts/maintenance'/name
        old=src.read_text();new=old.replace('ROOT = Path(__file__).resolve().parents[1]','ROOT = Path(__file__).resolve().parents[1]')
        if name=='archive_addendum.py':
            new=new.replace("'experiments/scripts/archive_research.py'","'tools/archive_research.py'").replace("'experiments/scripts/archive_addendum.py'","'tools/archive_addendum.py'")
        if name=='prepare_report_publication.py':
            new=new.replace("if src.is_symlink() or src.suffix not in ALLOWED:","if src.name in {'publication-manifest.json', 'publication-check.json', 'PUBLISHED.md'}:\n                continue\n            if src.is_symlink() or src.suffix not in ALLOWED:")
        patch='*** Begin Patch\n*** Update File: '+str(src)+'\n*** Move to: '+str(dst)+'\n@@\n'
        patch+=''.join('-'+x+'\n' for x in old.splitlines())+''.join('+'+x+'\n' for x in new.splitlines())+'*** End Patch\n'
        subprocess.run(['apply_patch'],input=patch,text=True,check=True,stdout=subprocess.DEVNULL)
        moves.append(dict(old=str(src.relative_to(ROOT)),new=str(dst.relative_to(ROOT)),kind='maintenance_tool',
                          original_sha256=hashlib.sha256(old.encode()).hexdigest(),sha256=digest(dst)))
    for rel,text in PAGES.items():edit(rel,text)
    # 更新三份研究報告的 PDF 連結；歷史紀錄中的舊路徑文字維持原樣。
    for p in (ROOT/'docs/research').glob('*.md'):
        old=p.read_text()
        def replace_pdf(m):
            label,target=m.groups();target=target.strip('<>')
            if '://' in target:return m.group(0)
            local=Path(os.path.normpath(p.parent/unquote(target)))
            if local.is_relative_to(ROOT) and str(local.relative_to(ROOT)) in PDFS:
                dst=ROOT/PDFS[str(local.relative_to(ROOT))]
                return '['+label+'](<'+os.path.relpath(dst,p.parent)+'>)'
            return m.group(0)
        new=LINK.sub(replace_pdf,old)
        if new!=old:edit(p.relative_to(ROOT),new)
    history='# 歷史入口\n\n下列為整理前的完整入口文字，僅保留當時狀態；現行導航見[首頁](../../README.md)。每份 `.txt` 與原檔bytes完全相同，Markdown只調整相對連結以便閱讀。\n\n'
    for row in histories:
        history+='- ['+row['original']+']('+os.path.relpath(ROOT/row['readable'],ROOT/'docs/history')+')\n'
    history+='\n[搬移與保護清單](organization-20260912/manifest.json)。checkpoint、訓練程式與raw metrics沒有搬移；沒有刪除任何原始內容。\n'
    edit('docs/history/README.md',history)
    # 集中六個重要模型的角色，不複製或以symlink冒充一般YOLO權重。
    choices=[
      ('default_joint','experiments/activation/bridge_v1/artifacts/runs/qsilu_pq-short-e10-seed1-v1/inference/best_joint.pt','目前新研究預設；activation-relative，非原嚴格gate全過'),
      ('prefusion_masf','experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/masf-e8-bittrue.pt','使用者選定融合前P3 bridge E8'),
      ('prefusion_control','experiments/studies/pre-fusion-full35-b100/artifacts/direction1-candidate-verification-v1/control-e8-bittrue.pt','融合前無MASF配對'),
      ('pose_recovery','experiments/combine/bridge_v1/artifacts/fusion/j3-pose-head-recovery-v1/inference/best_pose.pt','activation父模型'),
      ('pose_kd_candidate','experiments/kd/pose_focus_v1/artifacts/runs/kd-e5-seed1-v1/inference/best_pose.pt','headKD E2；框退化，不升版'),
      ('mixed_keypoint_candidate','experiments/inference/pose_branch_v1/artifacts/branch-v2/candidate.pt','推論重組負結果，不升版'),
    ]
    registry=[]
    catalog='# Checkpoint 與模型導航\n\n索引不複製權重，所有原路徑不變。不要直接把自訂state-dict交給原生YOLO載入；需使用對應source重建qSiLU、MASF及PWL，並維持weights_only安全載入。\n\n| 角色 | 原始模型 | 判斷 |\n| --- | --- | --- |\n'
    for key,rel,note in choices:
        p=ROOT/rel;assert p.is_file()
        row=dict(id=key,path=rel,sha256=digest(p),bytes=p.stat().st_size,role=note)
        if '/inference/' in rel:
            resume=ROOT/rel.replace('/inference/','/checkpoints/')
            if resume.is_file():row['training_snapshot']=str(resume.relative_to(ROOT))
        registry.append(row)
        catalog+='| '+key+' | [權重]('+os.path.relpath(p,ROOT/'reports/checkpoints')+') | '+note+' |\n'
    catalog+='\n[六個模型的來源／SHA／續訓檔索引](registry.json)；[全部406個封存模型產物CSV](../consolidated-20260911/checkpoints.csv)。406包含外部來源、不同bank與研究候選，不是406個全部合格模型。\n\n有training snapshot不等於任意中斷exact resume已驗證。原始舊combine仍在 `/home/uxin/yolo/yolo_combine/final/full35/`，不與本輪qSiLU主線混名；完整比較見[總報告](../consolidated-20260911/README.md)。\n'
    edit('reports/checkpoints/README.md',catalog)
    edit('reports/checkpoints/registry.json', json.dumps(registry,ensure_ascii=False,indent=2))
    for path,expected in protected.items():
        s=(ROOT/path).stat()
        assert dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino)==expected,path
    for row in histories:assert digest(ROOT/row['backup'])==row['sha256']
    manifest=dict(status='organized',moves=moves,histories=histories,readme_pages=len(PAGES)+3,
                  protected_files=len(protected),protected_stat_checks_passed=True,
                  checkpoints_rehashed=len(registry),model_registry='reports/checkpoints/registry.json',
                  no_training=True,no_deletions=True,no_git_publication=True,
                  limitation='training/artifact roots kept for path compatibility; archived packages untouched')
    edit(HISTORY/'manifest.json', json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(dict(status='organized',moves=len(moves),historical_readmes=len(histories),
                         protected_files=len(protected),model_entries=len(registry)),ensure_ascii=False))


if __name__=='__main__':main()
