# Full35 / Partial75 CPU validation audit

- 日期：2026-08-22
- 裝置：CPU；所有命令皆以 `CUDA_VISIBLE_DEVICES=""` 執行
- 正式 GPU training：未啟動
- 結論：兩個架構的 folder isolation、Float/Bit-True 推論等價與 joint backward 通過

## Workspace isolation

| Architecture | folder-locked entrypoint | mutable root |
|---|---|---|
| Full35 | `variants/full35/run.py` | `variants/full35/artifacts/` |
| Partial75 | `variants/partial75/run.py` | `variants/partial75/artifacts/` |

入口不接受 `--architecture`。設定、prepared dataset、Ultralytics cache、CPU report、
future checkpoint 與 run output 都從所在 workspace 推導。共享的只有唯讀 source、
原始 dataset 與無狀態的 `src/yolo_combine/` implementation。

## CPU acceptance results

| Gate | Full35 | Partial75 |
|---|---:|---:|
| Source/checkpoint SHA256 | pass | pass |
| BBT5 Pose→Detect derivation | 0 mismatch | 0 mismatch |
| Float trunk transfer | 587/587 | 587/587 |
| Float independent→shared outputs | bit-exact | bit-exact |
| Float shared→official task graphs | bit-exact | bit-exact |
| Bit-True trunk transfer | 585/585 | 585/585 |
| Bit-True independent→shared outputs | bit-exact | bit-exact |
| Synthetic 2:1 joint backward | pass | pass |
| Real COCO/BBT5 2:1 joint step | pass | pass |
| Inherited Attention/MASF drift | none | none |

參數結果：

| Architecture | independent pair | shared dual-head | removed | reduction |
|---|---:|---:|---:|---:|
| Full35 | 45,580,762 | 26,529,701 | 19,051,061 | 41.796% |
| Partial75 | 45,442,522 | 26,460,581 | 18,981,941 | 41.771% |

真實資料 smoke 使用每個 workspace 自己的：

- COCO Detect smoke view：128 images，1 background，cache 位於該 workspace。
- BBT5 Pose view：6,080 train / 567 valid，4 個微小負 keypoint coordinate 在 derived
  labels 夾到 0，source labels 不變。
- ratio：2 Detect microbatches : 1 Pose microbatch。

機器可讀結果：

- `variants/full35/artifacts/cpu-validation.json`
- `variants/full35/artifacts/joint-smoke.json`
- `variants/partial75/artifacts/cpu-validation.json`
- `variants/partial75/artifacts/joint-smoke.json`

這些是工程 smoke，不是 mAP 或正式精度證據。

## COCO source-cache incident

第一版 real-data smoke 直接把 full COCO YAML 與 `fraction=0.001` 交給 Ultralytics。
Ultralytics 因 image list 改成 118 張而重建來源路徑的：

```text
/home/uxin/yolo/coco2017/labels/train2017.cache
```

結果該 cache 從先前稽核的完整 118,287-image cache（232,587,697 bytes）變成
238,616 bytes；事故版本 SHA256：

```text
96f2d15dcbd75767f8399f6408ee2f767255dfe489266ce4fc9954aa2450bef3
```

COCO images、labels 與 train/val text manifests 沒有被該命令修改；受影響的只有
可重建 cache。直接復原會再次寫入 workspace 外的來源路徑，因此先等待使用者明確
授權，並在 2026-08-22 取得「可以重建 cache」的確認後執行。

已完成的防再發措施：

1. `joint-smoke` 先在各 workspace 建立固定 128-image symlink/label view。
2. Ultralytics 只會在 `variants/<architecture>/artifacts/datasets/` 建 cache。
3. Full35 隔離 smoke 前後來源 cache 的 SHA256、size、mtime 完全不變。
4. Partial75 使用同一隔離介面並成功完成。

### Recovery result

重建前先把事故版本逐位元備份到：

```text
artifacts/cache-recovery/2026-08-22/train2017.cache.partial-118.backup
SHA256 96f2d15dcbd75767f8399f6408ee2f767255dfe489266ce4fc9954aa2450bef3
```

`scripts/rebuild_coco_train_cache.py` 直接讀取完整 `train2017.txt`、image 與 YOLO
label；不載入模型或 checkpoint。命令以 `CUDA_VISIBLE_DEVICES=""`、4 個 CPU worker
執行，先在 workspace 產生候選檔，通過全部內容 gate 後才原子替換來源 cache。

正式來源 cache 已恢復：

| Field | Value |
|---|---|
| Path | `/home/uxin/yolo/coco2017/labels/train2017.cache` |
| Bytes | 232,587,697 |
| SHA256 | `881dc0031a50a721b166e9d97cb9014d7cbf680eac6c1df8402b55ebc679c6ad` |
| Cache version | `1.0.3` |
| Dataset hash | `54f27e05f48d28d9b11459fd995dd19edc64bfa855dafa5d66d6bfd234055ded` |
| Results `(found, missing, empty, corrupt, total)` | `(117266, 1021, 0, 0, 118287)` |
| Cached label entries | 118,287 |
| CUDA available during recovery | false |

正式路徑重新載入後再次得到相同 results、entry count 與空 warning list；正式檔案
SHA256 亦與安裝前已驗證的候選檔一致。機器可讀證據位於
`artifacts/cache-recovery/2026-08-22/rebuild-report.json`。
