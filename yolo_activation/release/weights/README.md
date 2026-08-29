# Full35 Activation 可重建权重

本目录发布量化交接所需的两个最小充分 inference 权重。每个原始 `.pt` 为 106,825,541 bytes，超过
GitHub 一般 100 MB 单 blob 限制，因此切成一个 90,000,000-byte 分片与一个 16,825,541-byte 分片；
这不是重新序列化、压缩或量化，按顺序接回后与原档逐位元相同。

| Weight ID | 用途 | 原档 SHA-256 |
| --- | --- | --- |
| `accepted-silu-best-joint` | accepted J3 SiLU baseline／量化父权重 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| `qsilu-pq-short-recovery-best-joint` | 已完成并通过八项 gate 的 10-epoch qSiLU 候选 | `7679186695317e431cd7deb17289f426f4b39b7a4993e4548e74f5ba2766190e` |

## 重建

在本目录执行：

```bash
python reconstruct.py accepted-silu-best-joint
python reconstruct.py qsilu-pq-short-recovery-best-joint
```

程式会先逐片检查大小与 SHA-256，再串接并检查完整档；若最终校验失败，会移除无效输出。重建档案
分别位于各自目录的 `best_joint.pt`，受仓库 `*.pt` 规则排除，不会意外再次提交。

也可先独立检查分片：

```bash
sha256sum -c SHA256SUMS
```

`weights.json` 是 machine-readable lineage 与 part manifest。

## 明确未发布的权重

- qSiLU 20-epoch finalist：依使用者要求中止，没有 completion marker，也没有正式 `best_joint.pt`。
- 425 MB full-resume／optimizer checkpoints：量化 inference 不需要，且会使 Git 历史膨胀。
- Hardswish、`poly_shift`、`poly_quality`：10-epoch 八项 gate 未过，不作为量化 finalists。

本目录不改变 `original/` 永久禁止发布任何 weights/checkpoints 的规则；这些分片属于
`yolo_activation/release/weights/` 的明确实验交付。
