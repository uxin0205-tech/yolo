# 2026-08-29：Full35 Activation 收尾、权重发布与量化交接

## 任务与范围

依使用者要求，把本子专案从数学设计、资料规范、实验顺序、训练配置、正式／失败／中止结果到权重
lineage 全部统整，准备以 commit subject `5090 Finish 0829` 发布 GitHub。此次不恢复 qSiLU finalist
训练；将 activation-only 阶段冻结为可供下一条 quantization 工作列读取的交付。

正式资料入口保持不变：

- COCO2017 Detect：`/home/uxin/yolo/coco2017.yaml`，118,287 train／5,000 val。
- BBAT5 Pose／box branch：`/home/uxin/yolo/original/pose/derived/bbat5-v1/configs/pose.yaml`，
  5,964 formal train／683 formal val。
- BBAT5 registry：`/home/uxin/yolo/configs/datasets/bbat5-v1.yaml`。
- `fraction=1.0`、`resampling=false`；没有改动样本、labels、split，也没有建立 30% 资料版本。

## 变更内容与原因

1. 新增 `docs/research/full35-activation-mathematical-derivations.md`，完整推导 registry 六种 activation。
   控制组包含 SiLU 偶残差、Hardswish 分段式与 ReLU；proposed 部分包含 SIPA 一般四约束线性解、
   `poly_quality`／`poly_shift` 的逐系数展开、qSiLU 截断平方基底与 exact-tail 三条件、各区段展开，
   以及 fixed-point `A(n)-A(-n)=n` 的整数证明。
2. 新增最终中文分析报告、JSON 与 CSV。JSON 保留 full Float／Bit-True zero-shot 数值、八项 selector、
   11-region sensitivity、queue、OOM probes、profiling 与权重 lineage；CSV 提供 12 个关键节点的八项
   mAP50-95／delta 宽表。
3. 新增 `scripts/export_full35_results.py`，从本机原始 artifacts 重建 committed JSON／CSV，避免人工
   抄写误差；用 `/tmp` 重生成并以 `cmp` 证明两份报告逐位元可复现。
4. 发布两个 inference 权重：accepted SiLU 与完成 10 epochs、通过 gate 的 qSiLU。每个原档
   106,825,541 bytes，因超过 GitHub 100 MB 单 blob 上限，切成 90,000,000 + 16,825,541-byte 两片；
   新增 `weights.json`、`SHA256SUMS` 与安全重建程式。没有发布中止 qSiLU finalist、425 MB resume
   checkpoints 或已被 gate 淘汰候选。
5. 新增 release contract tests，防止后续把 provisional qSiLU 标成 final、改变关键 pass/fail，或
   产生超过 GitHub 单档上限的分片。
6. 同步子专案 README、研究索引、Full35 配置 README、报告索引与工作纪录索引，使最终结论、数学、
   权重及量化入口可从根层一路追踪。
7. 在根层 `.gitignore` 仅放行 `yolo_activation/release/weights/`，让已核准且有 checksum 的分片可提交；
   重建出的 `*.pt` 仍由全域规则排除。
8. 机械格式化 `scripts/full35_queue.py` 的既有长路径表达式；没有改变 selector 或 queue 语意。

## 最终实验结论

- 静态 queue：`5 completed / 0 pending / 14 blocked`。14 个 blocked 是 `poly_shift` uniform
  prerequisite 失败后的预注册 gate 传播，不是执行错误。
- 10-epoch recovery：qSiLU worst delta `-0.008635`，八项通过；Hardswish `-0.016884`、
  `poly_quality -0.020970`、`poly_shift -0.030138`，均未过。
- SiLU finalist control：第 6 个已完成 epoch early-stop，gate-passing best worst delta `-0.012872`。
- qSiLU finalist：只完成 epoch 1，并在 epoch 2 macro 106 后依使用者要求中止；epoch 1 provisional
  worst delta `-0.015966`，没有 completion marker，不作最终胜负或量化父权重。
- qSiLU physical batch 128／64／32 均在第一个 forward OOM；16 可稳定完成。logical Detect batch
  仍为 128，靠 gradient accumulation 保持 macro 语意。
- 后续先跑 activation × quantization PTQ 分析，只有超出相同八项预算才做 matched QAT；目前不继续
  activation-only 调参或补跑被 gate 封锁的 policy。

## 验证方式与结果

- `/home/uxin/yolo/.venv/bin/python -m pytest`：`61 passed`；4 个既有 legacy ONNX exporter
  deprecation warnings，无功能失败。
- `python -m ruff check .`：`All checks passed!`。
- `python -m ruff format --check .`：`64 files already formatted`。
- `scripts/validate_phase0.py`：六种函数报告完成；qSiLU／`poly_shift` 的 symmetry、exact tails、Q16.10
  与 ONNX gates 通过。
- `scripts/activation_training.py toy-dry-run`：manifest、policy replacement 与 finite output 通过。
- `scripts/full35_activation.py preflight`：`ready=true`、`blockers=[]`；资料比例 1.0、正式路径与父权重
  hash 正确。TensorBoard 未安装只影响额外 UI，不影响 JSONL／CSV／PNG；base config 的 VRAM warning
  已由本轮真实 qSiLU probes 与 per-job physical microbatch override 明确处理。
- 报告重生成：`reports/*.json/csv` 与 `/tmp` 重建档 `cmp` 一致；JSON parser 通过。
- 权重：四个分片 `sha256sum -c` 全部 OK；两个 `/tmp` 重建 `.pt` 分别回到原 SHA
  `d67fb45...ec74c` 与 `767918...190e`。
- Markdown：26 份文件的相对链接检查，`missing=[]`；字面 `\\n` 残留检查为空。
- 训练程序：唯读 process 查核没有本专案 queue/train；未重启 GPU 作业。

## 困难与解法

1. 修改既有 Markdown 时，`apply_patch` 再次因环境 `bwrap: loopback: Failed RTM_NEWADDR` 失败。
   依既有工作流程改用旧内容必须唯一命中的精确 Perl replacement。
2. 第一次 Perl fallback 使用不插值的 `q{}`，在研究／训练 README 写入字面 `\\n`；立即唯读检查发现，
   精确转换为真实换行并再次检查全专案无残留。README 多段 replacement 第一次命中数为 0，未写入；
   改用 `qq{}` 后三段必须全部唯一命中才完成。
3. 初次完整 Ruff 找到新增 exporter 的 executable bit、import order 与三个格式问题；设定 CLI 可执行、
   自动整理 imports 与机械格式化后重跑完整检查，全数通过。
4. 权重单档超过 GitHub blob 限制且仓库没有可验证的 Git LFS remote；采用无损分片、逐片 hash 与最终
   full-file hash，既能一般 Git push，也保留逐位元可追溯性。

## 清理与保留

- 没有删除本机 29 GB 原始 experiments、OOM probes、quarantine 或 checkpoints；它们受 `.gitignore`
  排除并保留作稽核证据。
- 发布只包含 source/config/tests/docs、约 51 KB 正规化报告及约 214 MB 的两个必要权重分片。
- `/tmp` 的报告与重建权重只用于验证，发布前移除，不影响本机正式 artifacts。

## 未解事项或风险

- qSiLU seed-1 finalist 未完成，seed 2／claim ablation 未执行；不得宣称多 seed 最终优越性。
- 尚未完成 PTQ、QAT、全输入码 Q8/Q12/Q16 枚举与 compiler/HLS/RTL bit-exact 比对。
- 没有指定 target FPGA／ASIC、clock、precision 与 synthesis flow，尚无 latency、power、LUT/DSP/BRAM
  板上结论。
- qSiLU eager PyTorch 的 OOM 不代表 fused kernel 的极限；未来若优化 kernel 必须重新实测。
