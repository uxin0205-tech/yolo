# BBT5 資料與固定切分

## 來源

- 唯一來源：`bbt5-detect-baseline/dataset/`。
- 類別：`ball=0`、`bat=1`。
- Dataset 不納入 GitHub 發布包；本 repo 只保存 manifest、固定清單、COCO JSON 與統計。
- Dataset SHA-256 contract：`6e16c975941dbae2af174a2eb4b5424bffd4736c74aad56d424805da019b8fbc`。

## 數量

| Split | Unique frames | Exported records/images | Ball objects | Bat objects |
|---|---:|---:|---:|---:|
| train | 1,987 | 5,093 | 2,696 | 4,178 |
| validation | 300 | 781 | 414 | 685 |
| test | 291 | 773 | 503 | 590 |
| 合計 | 2,578 | 6,647 | 3,613 | 5,453 |

`unique frames` 是 group/hash 去重後的原始 frame 數；`exported records/images` 包含來源匯出的多筆記錄，兩者不是同一口徑。

## Leakage audit

- Split ratio：80/10/10，seed=42。
- Groups：1,128。
- Group overlap：0。
- Image hash overlap：0。
- Audit 狀態：`ok=true`。

## 證據

- [`artifacts/static-phase1/dataset/audit.json`](../../artifacts/static-phase1/dataset/audit.json)
- [`artifacts/static-phase1/dataset/manifest.json`](../../artifacts/static-phase1/dataset/manifest.json)
- [`artifacts/static-phase1/dataset/dataset_profile.json`](../../artifacts/static-phase1/dataset/dataset_profile.json)
- [`artifacts/static-phase1/dataset/data.yaml`](../../artifacts/static-phase1/dataset/data.yaml)
- 固定 train/val/test 清單與 COCO JSON：[`artifacts/static-phase1/dataset/`](../../artifacts/static-phase1/dataset/)

## 資料暴露限制

`yolo11m_bat_detect_init.pt` 由 BBT5 pose-derived 權重轉成 detection initializer，已接觸同源資料。因此目前結果適合比較相同資料、初始化與 split 下的架構差異，不適合宣稱 unseen-data generalization。
