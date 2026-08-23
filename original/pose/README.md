# BBAT5 共用資料資產庫

此目錄保存 BBAT5 的原始來源、不可變正式版本與歷史權重。未來新實驗只使用
`derived/bbat5-v1`。這個根目錄是同時包含 Pose 與二類 Detect 的版本容器，不是只能做
其中一項；其他目錄不再直接作新訓練入口。
這是 `/home/uxin/yolo` 內唯一 BBAT5 資料資產位置；`architecture_2`、`yolo_combine` 與後續專案
只引用這裡，不再維護資料副本。


| 目錄 | 角色 | 新實驗可直接使用 |
| --- | --- | --- |
| `dataset/` | 原始 Pose 來源與舊 Roboflow split，唯讀 | 否 |
| `detect_dataset/` | 從原始 Pose 衍生的歷史 Detect／COCO80 View，唯讀 | 否 |
| `derived/bbat5-v1/` | 唯一正式 BBAT5 v1，Detect/Pose 共用 grouped split | 是 |
| `weight/` | 歷史模型權重，不是資料集 | 否 |

如何選擇正式入口：

| 工作 | 使用檔案 |
| --- | --- |
| ball/bat Pose | `derived/bbat5-v1/configs/pose.yaml` |
| ball/bat 二類 Detect | `derived/bbat5-v1/configs/detect.yaml` |
| 同時使用兩個 Task View／融合 | `/home/uxin/yolo/configs/datasets/bbat5-v1.yaml` |
| COCO80／person Detect | `/home/uxin/yolo/coco2017.yaml`，不在本資料版本內 |

`bbat5-v1` 有 6,647 張影像，formal train/val 為 5,964/683；依 `.rf.` 前來源名稱
分組，跨 split leakage 為 0。Pose 與 Detect 僅是同一版本的兩種任務 View；不得各自重切。

不要刪除或改寫 `dataset/`、`detect_dataset/`：舊 checkpoint 與報告仍需靠它們解釋
歷史血緣。若資料內容或 split 必須改變，建立 `bbat5-v2`，不得覆寫 v1。

## GitHub 發布規則

GitHub 會保存 `dataset/`、`detect_dataset/` 與 `derived/bbat5-v1/` 的影像、labels、設定與
lineage，讓來源可以被稽核及重建。GitHub 上看得到 raw/basic split 不代表可拿它直接做新
訓練；正式入口仍是上方 `bbat5-v1` YAML／registry。

不提交 `weight/`、`.pt`、checkpoint、cache、run，以及 726,351,400-byte 且與完整
`detect_dataset/` 重複的 `detect_dataset.zip`。本機權重與 zip 保留不動；只有 Git 發布樹排除。
