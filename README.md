# YOLO 研究工作區

本目錄是 YOLO 與棒球視覺研究的共用工作區。所有子專案均遵守根層代理規則、固定資料集政策與中文工作紀錄要求。

## 全域文件入口

- [代理全域規則](AGENTS.md)
- [BBAT5／BBT5 棒球資料集使用規範](docs/agents/bbat5-datasets.md)
- [工作紀錄索引與格式](docs/worklogs/README.md)
- [領域文件規範](docs/agents/domain.md)
- [GitHub Issue 使用規範](docs/agents/issue-tracker.md)

## BBAT5 v1 正式資料版本：同時包含 Pose 與 Detect

`bbat5-v1` 不是「只給 Pose」或「只給 Detect」的單一 YAML，而是一個成對版本容器。
Pose 與 ball/bat 二類 Detect 共用 6,647 張影像及完全相同的 train/val assignment，只使用
不同格式的 labels。根目錄本身不能直接交給 Ultralytics，請依任務選擇入口：

| 要執行的任務 | 正式入口 | 說明 |
| --- | --- | --- |
| ball/bat Pose | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml` | 2 classes、2 個 keypoints |
| ball/bat 二類 Detect | `/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/detect.yaml` | 2 classes，只使用 bbox |
| COCO80／person Detect | `/home/uxin/yolo/coco2017.yaml` | 不屬於 BBAT5，不可改用二類 YAML |
| Detect–Pose 融合 | [`configs/datasets/bbat5-v1.yaml`](configs/datasets/bbat5-v1.yaml) 加上任務所需的 COCO 設定 | 由程式解析各 Task View |

全專案 BBAT5 registry 是 [`configs/datasets/bbat5-v1.yaml`](configs/datasets/bbat5-v1.yaml)。
`original/pose/dataset` 與 `detect_dataset` 是唯讀來源／歷史資料；新訓練不得直接使用。
各專案可以建立隔離 cache 的 runtime View，但 split、labels 與 lineage 必須完全來自
`bbat5-v1`，不得形成第二套資料版本。architecture_2 另保留使用者已核准的 Portable GitHub
Snapshot 作 clone／稽核用途，不是正式訓練入口。

資料目錄角色見 [`original/pose/README.md`](original/pose/README.md)，選型理由見
[ADR 0001](docs/adr/0001-use-bbat5-v1-as-canonical-dataset.md)。`original/pose/` 是唯一正式
BBAT5 資料資產庫；portable snapshot 只改變發布形式，不改變 canonical assignment 或 lineage。

### GitHub 的 original 發布範圍

依 2026-08-23 授權，GitHub 保存 `original/pose/dataset/`、`detect_dataset/` 與
`derived/bbat5-v1/` 的可用資料內容，供稽核與重建。這不改變正式訓練入口：新 run 仍只讀取
`bbat5-v1` registry。發布永久排除 `.pt`／checkpoint、Ultralytics cache、run，以及超過
GitHub 單檔限制且與已解壓目錄重複的 `detect_dataset.zip`；詳細範圍見
[`original/README.md`](original/README.md) 與 [ADR 0002](docs/adr/0002-publish-original-data-without-weights.md)。

2026-08-22 核准的可攜副本固定在
`yolo_achitechure/achitechure_2/artifacts/datasets/bbat5-v1/github-dataset/`；完整影像與 labels 可發布，
但不得作為本機 canonical 訓練來源，也不得包含 weight、checkpoint、cache 或 run。

## 報告與變更

Full35 activation 数学、完整 COCO2017 + Canonical BBAT5 v1 实验、可重建 SiLU／qSiLU 权重与
量化交接见 [`yolo_activation/`](yolo_activation/README.md)；最终逐项分析见
[2026-08-29 收尾报告](yolo_activation/reports/full35-activation-final-analysis.md)。

Full35 activation-output A3至A8量化預選、老師版圖表、30格source evidence與SD4公平實驗設計見
[`yolo_quantize/`](yolo_quantize/README.md)；主要論證見
[2026-08-29 activation預選報告](yolo_quantize/docs/reports/2026-08-29-activation-preselection-report.md)。
這是無訓練proxy先行發布，不是完整mAP、QAT或硬體結果。

YOLO26m Binary Q/K＋Bit-True PWL 的完整可攜 workspace、程式碼、權重與文件見
[`yolo_attention_final/final/`](yolo_attention_final/final/README.md)；本次 GitHub 發布過程見
[2026-08-27 工作紀錄](docs/worklogs/2026-08-27-yolo-attention-final-github-publication.md)。

architecture_2 的 C1～C3 已因 Float20 精度下降過大而不採用；C2／C3 full、PTQ 與 QAT-lite
已永久關閉。完整負結果、清理與發布範圍見
[2026-08-28 architecture_2 封存工作紀錄](docs/worklogs/2026-08-28-architecture2-archive-and-publication.md)。

所有進度、報告、實驗結果、變更紀錄，以及困難與解法都使用中文。每份新工作紀錄都必須加入[工作紀錄索引](docs/worklogs/README.md)，讓 README 保持可追溯的入口。
