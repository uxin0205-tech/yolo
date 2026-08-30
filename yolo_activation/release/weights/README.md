# Full35 Activation 可重建權重

本目錄發布量化交接所需的兩個最小充分 inference 權重。每個原始 `.pt` 為 106,825,541 bytes，超過
GitHub 一般 100 MB 單 blob 限制，因此切成一個 90,000,000-byte 分片與一個 16,825,541-byte 分片；
這不是重新序列化、壓縮或量化，按順序接回後與原檔逐位元相同。

| Weight ID | 用途 | 原檔 SHA-256 |
| --- | --- | --- |
| `accepted-silu-best-joint` | accepted J3 SiLU baseline／量化父權重 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| `qsilu-pq-short-recovery-best-joint` | 已完成並通過八項 gate 的 10-epoch qSiLU 候選 | `7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e` |

## 重建

在本目錄執行：

```bash
python reconstruct.py accepted-silu-best-joint
python reconstruct.py qsilu-pq-short-recovery-best-joint
```

程式會先逐片檢查大小與 SHA-256，再串接並檢查完整檔；若最終校驗失敗，會移除無效輸出。重建檔案
分別位於各自目錄的 `best_joint.pt`，受倉庫 `*.pt` 規則排除，不會意外再次提交。

也可先獨立檢查分片：

```bash
sha256sum -c SHA256SUMS
```

`weights.json` 是 machine-readable lineage 與 part manifest。

## 2026-08-30 重建驗證

四個分片皆通過 `sha256sum -c SHA256SUMS`。兩組權重在 `/tmp` 重新串接後，檔案大小皆為
106,825,541 bytes，完整 SHA-256 分別回到上表的 `d67fb45c…ec74c` 與 `76791866…6190e`。

## 明確未發布的權重

- qSiLU 20-epoch finalist：依使用者要求中止，沒有 completion marker，也沒有正式 `best_joint.pt`。
- 425 MB full-resume／optimizer checkpoints：量化 inference 不需要，且會使 Git 歷史膨脹。
- Hardswish、`poly_shift`、`poly_quality`：10-epoch 八項 gate 未過，不作為量化 finalists。

本目錄不改變 `original/` 永久禁止發布任何 weights/checkpoints 的規則；這些分片屬於
`yolo_activation/release/weights/` 的明確實驗交付。
