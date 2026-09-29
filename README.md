# YOLO 研究工作區

## Cleanup-v1：程式與研究報告基線

此分支從 GitHub `main` 的 `1d409c4fe21d3287d6f19be48f4457407a6bb50e` 建立獨立起始提交，供後續更新使用。保留既有程式、測試、設定、研究報告、統計圖表及資料血緣；模型／checkpoint、分片權重、資料集本體、逐筆預測、batch 預覽和冗長日誌不納入本分支。

- [完整清理結果、容量口徑與資產需求](docs/cleanup-v1/README.md)
- [逐檔排除清單](docs/cleanup-v1/excluded-files.csv)及[盤點摘要](docs/cleanup-v1/inventory.json)
- [中文工作紀錄](docs/worklogs/2026-09-29-cleanup-v1.md)
- [提交前檔案檢查](tools/README.md)：`python3 tools/check_repository_hygiene.py`

只下載這個基線：

```bash
git clone --single-branch --branch Cleanup-v1 https://github.com/uxin0205-tech/yolo.git
```

此版本沒有附帶可直接推論或續訓的權重；歷史報告中的 checkpoint 路徑與結果不代表資產仍隨分支發布。BBAT5 v1 的本機資料及 split／labels 不變，重跑前須提供原資料及對應權重。舊 `main` 或其他分支的 Git／LFS 儲存量不會因建立此分支立即歸零。

## 5090 Done 0912：BinaryQK 至推論優化報告

[完整研究閱讀入口](yolo_optimize/reports/publication/README.md)整合 BinaryQK／固定 scale、HOG、RepConv、MASF P3／P2、重新 combine、activation、KD 與推論。詳細結果、超參數、架構圖與 checkpoint 索引見[全階段總報告](yolo_optimize/reports/final/README.md)。本次不發布大型權重、封存副本或新增資料集，保留所有失敗與未達門檻結論。

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
Snapshot 的設定與血緣紀錄作稽核用途；本分支不含影像與 labels。

資料目錄角色見 [`original/pose/README.md`](original/pose/README.md)，選型理由見
[ADR 0001](docs/adr/0001-use-bbat5-v1-as-canonical-dataset.md)。`original/pose/` 是唯一正式
BBAT5 資料資產庫；portable snapshot 只改變發布形式，不改變 canonical assignment 或 lineage。

### 歷史 GitHub 資料發布範圍（Cleanup-v1 已改為僅保留 metadata）

依 2026-08-23 授權，GitHub 保存 `original/pose/dataset/`、`detect_dataset/` 與
`derived/bbat5-v1/` 的可用資料內容，供稽核與重建。這不改變正式訓練入口：新 run 仍只讀取
`bbat5-v1` registry。發布永久排除 `.pt`／checkpoint、Ultralytics cache、run，以及超過
GitHub 單檔限制且與已解壓目錄重複的 `detect_dataset.zip`；詳細範圍見
[`original/README.md`](original/README.md) 與 [ADR 0002](docs/adr/0002-publish-original-data-without-weights.md)。

2026-08-22 核准的可攜副本固定在
`yolo_achitechure/achitechure_2/artifacts/datasets/bbat5-v1/github-dataset/`；完整影像與 labels 可發布，
但不得作為本機 canonical 訓練來源，也不得包含 weight、checkpoint、cache 或 run。

## 報告與變更

Full35 activation 數學、完整 COCO2017 + Canonical BBAT5 v1 實驗、可重建 SiLU／qSiLU 權重與
量化交接見 [`yolo_activation/`](yolo_activation/README.md)。2026-08-30 已完成繁體中文與文件結構整理；
閱讀順序見[文件導覽](yolo_activation/docs/README.md)，目前結論見
[Activation 權威整合報告](yolo_activation/reports/completed-activation-integrated-report.md)。

Full35 activation-output A3至A8量化預選、老師版圖表、30格source evidence與SD4公平實驗設計見
[`yolo_quantize/`](yolo_quantize/README.md)；主要論證見
[2026-08-29 activation預選報告](yolo_quantize/docs/reports/2026-08-29-activation-preselection-report.md)。
這是無訓練proxy先行發布，不是完整mAP、QAT或硬體結果。

YOLO26m Binary Q/K＋Bit-True PWL 的歷史可攜 workspace 的程式碼與文件見
[`yolo_attention_final/final/`](yolo_attention_final/final/README.md)；本次 GitHub 發布過程見
[2026-08-27 工作紀錄](docs/worklogs/2026-08-27-yolo-attention-final-github-publication.md)。

architecture_2 的 C1～C3 已因 Float20 精度下降過大而不採用；C2／C3 full、PTQ 與 QAT-lite
已永久關閉。完整負結果、清理與發布範圍見
[2026-08-28 architecture_2 封存工作紀錄](docs/worklogs/2026-08-28-architecture2-archive-and-publication.md)。

所有進度、報告、實驗結果、變更紀錄，以及困難與解法都使用中文。每份新工作紀錄都必須加入[工作紀錄索引](docs/worklogs/README.md)，讓 README 保持可追溯的入口。
