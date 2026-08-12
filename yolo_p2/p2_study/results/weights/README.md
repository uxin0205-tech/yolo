# 正式權重

本目錄只保留四個有效 checkpoint。權重檔由 Git ignore，SHA-256 與大小記錄在 `../metadata/archive_manifest.json`。

| 檔案 | 用途 | 最佳 epoch | Detect stride | SHA-256 |
| --- | --- | ---: | --- | --- |
| `A0_yolo11m.pt` | 官方 YOLO11m 基線與重新訓練起點 | N/A | 8/16/32 | `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95` |
| `A1_best.pt` | P2 Direct 正式最佳權重 | 89 | 4/8/16/32 | `1f3b96464b1867f0e7c5ba9cb956049f2faaa9aa7102969c20152a62608d1005` |
| `A2_stage1_best.pt` | P2-only 20 epochs 階段最佳權重 | 18 | 4/8/16/32 | `d2b51e1790a6c7747809ef7dd9e17773b86730fbcc6cd56a29d6fbde8101a658` |
| `A2_best.pt` | A2 Stage 2 最終正式評估權重 | 39 | 4/8/16/32 | `2e7067454a42564b54e7520f4fb049bc6ec1dac62496aa6c5d7c8a417df3a767` |

`last.pt`、gate/health、取消與 invalidated 權重不屬於正式成果，整理時已移除。
