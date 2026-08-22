# YOLO 研究工作區

本目錄是 YOLO 與棒球視覺研究的共用工作區。所有子專案均遵守根層代理規則、固定資料集政策與中文工作紀錄要求。

## 全域文件入口

- [代理全域規則](AGENTS.md)
- [BBAT5／BBT5 棒球資料集使用規範](docs/agents/bbat5-datasets.md)
- [工作紀錄索引與格式](docs/worklogs/README.md)
- [領域文件規範](docs/agents/domain.md)
- [GitHub Issue 使用規範](docs/agents/issue-tracker.md)

## BBAT5 固定資料來源

| 任務 | 唯一預設資料集 |
| --- | --- |
| Detection | `/home/uxin/yolo/original/pose/detect_dataset/` |
| Pose | `/home/uxin/yolo/original/pose/dataset/` |

為控制實驗變因，後續不得自行重切、抽樣、複製、替換或重建這兩套資料集。需要例外時，必須先取得使用者明確授權，並在[工作紀錄](docs/worklogs/README.md)寫明原因、範圍與影響。

使用者已核准 architecture_2 將既有 `bbat5-v1` 以不改 split／patch 的方式物化為完整 GitHub
snapshot；固定位置與禁止權重規則見[BBAT5 資料集規範](docs/agents/bbat5-datasets.md)。

## 報告與變更

所有進度、報告、實驗結果、變更紀錄，以及困難與解法都使用中文。每份新工作紀錄都必須加入[工作紀錄索引](docs/worklogs/README.md)，讓 README 保持可追溯的入口。
