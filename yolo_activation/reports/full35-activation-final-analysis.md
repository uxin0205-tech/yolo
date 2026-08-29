# Full35 Activation 最终整合分析与量化交接

## 结论先行

本阶段可以收尾并交给量化工作列，但不能宣称 activation 最终胜负已经由 20-epoch finalist 完整证明。
可以冻结的事实如下：

1. 所有实验从 `yolo_combine/final/full35` accepted J3 `best_joint` 接手；不是从随机初始化训练。
2. 使用完整 COCO2017 与不可变 Canonical BBAT5 v1，`fraction=1.0`、不重切、不抽样、没有 30% 资料夹。
3. 完成 baseline、全 train profiling、五种 uniform zero-shot、11-region `poly_shift` sensitivity、四种
   10-epoch recovery、SiLU finalist control，以及 qSiLU finalist 的一个 provisional epoch。
4. 10-epoch hard gate 中，唯一八项全过的非 SiLU 候选是 `qsilu_pq`；Hardswish、`poly_shift`、
   `poly_quality` 均失败。故「Hardswish 比 qSiLU 好」不是本次正式结果。
5. `poly_shift` prerequisite 失败后，8 个 region recovery 与 6 个 mixed policy 共 14 jobs 按预注册
   dependency 正确封锁，不应为了填满表格继续浪费训练。
6. qSiLU 20-epoch finalist 依使用者要求在 epoch 1 完成、epoch 2 macro 106 后停止，不能把其
   `last.pt` 称为 finalist 或量化父权重。
7. 量化应先比较 accepted SiLU 与已完成 10-epoch qSiLU 两个权重。activation 与 quantization 确实强耦合：
   dynamic range、负谷、分段门槛、导数、rounding 与 saturation 会共同决定 PTQ／QAT 误差。

## 资料与模型血缘

| 项目 | 冻结值 |
| --- | --- |
| 架构 | YOLO26m Full35 shared trunk + COCO80 Detect + BBAT5 Pose26 |
| 父权重 | `yolo_combine/final/full35/weights/combined/inference/best_joint.pt` |
| 父权重 SHA-256 | `d67fb45c576035e1b9c607914c62fa2c46bad84a5f53dea2c95ea7d4155ec74c` |
| 父权重来源 | J3 global epoch 58 best joint；训练在 global epoch 63 early-stop |
| COCO2017 | 118,287 train／5,000 val；COCO80 Detect |
| BBAT5 | `/home/uxin/yolo/original/pose/derived/bbat5-v1/`；5,964 formal train／683 formal val |
| 资料比例 | `1.0`；`resampling=false` |
| 影像尺寸 | 640 |
| Activation sites | 190 个 references，分 11 regions |
| Selector | Bit-True；八项 mAP50-95 各自相对 accepted baseline 比较 |
| Hard gate | 每一项 delta 都必须 `>= -0.015`；不平均 COCO 与 BBAT raw AP |

COCO 与 BBAT5 是同一个 shared-trunk run 的不同 task head。BBAT5 box 来自 Pose26 head 的 box branch，
COCO 指标来自 COCO80 Detect head；没有把 BBAT5 二类 Detect YAML 错接到 COCO80 head。

## 原始 Full35 与 activation recovery 配置

原父权重的 stage lineage：

| Stage | Epoch 上限 | Patience | Warm-up | 主要可训练范围／LR |
| --- | ---: | ---: | ---: | --- |
| J0 | 8 | 0 | 1 | Pose head `2e-4` |
| J1 | 20 | 8 | 1 | heads `2e-4`、neck `7.5e-5` |
| J2 | 80 | 17 | 1 | heads `2e-4`、MASF `1.5e-4`、neck `7.5e-5`、backbone `1.5e-5` |
| J3 | 20 | 5 | 3 | heads `5e-5`、MASF `3.8e-5`、neck `1.9e-5`、backbone `3.8e-6`、attention `5e-7` |

Activation 实验不是把每个候选重新跑 128 epochs，而是从同一个 accepted inference EMA state
乾净载入，重新建立 optimizer、scheduler、EMA 与 RNG：

| Phase | Epoch | Seed | Patience／warm-up | LR scale | 实际状态 |
| --- | ---: | ---: | --- | ---: | --- |
| Baseline reproduction | 0 | 0 | — | — | 完成 |
| Full train profiling | 0 | 1 | — | — | 完成，COCO 118,287／BBAT5 5,964 全看完 |
| Uniform zero-shot | 0 | 1 | — | — | 5 activations 完成 |
| `poly_shift` region zero-shot | 0 | 1 | — | — | 11/11 完成 |
| Uniform short recovery | 10 | 1 | 0／1 | J3 `0.1×` | 4 候选完成 |
| Region recovery／BCSP | 10 | 1 | 0／1 | J3 `0.1×` | 14 jobs 因 prerequisite 失败封锁 |
| SiLU finalist control | 最多 20 | 1 | 5／3 | J3 `1.0×` | epoch 6 early-stop，完成并过 gate |
| qSiLU finalist | 最多 20 | 1 | 5／3 | J3 `1.0×` | 人工停止；只有 epoch 1 provisional |
| Seed 2／ablation | 20 | 2／1 | 5／3 | J3 `1.0×` | 未启动 |
| PTQ | 0 | 1 | calibration | — | 交给量化工作列 |
| QAT if needed | 10 | 1 | 0／1 | J3 `0.5×` | 只在 PTQ 超预算时启动 |

所有 recovery 共用 AdamW、`weight_decay=2.7e-4`、betas `(0.948,0.999)`、AMP、gradient clip 10、
cosine final factor 0.5。10-epoch LR 为 backbone `3.8e-7`、neck `1.9e-6`、MASF `3.8e-6`、
attention `5e-8`、两个 heads `5e-6`；finalist 使用上表原 J3 LR。

## Batch 与记忆体事实

- Detect logical batch 固定 128，每 macro 两个 logical batches，共 256 Detect images；这才是 optimizer
  看到的有效 batch 语义。
- qSiLU／多项式 recovery 的 physical microbatch 为 16，以 16 次 physical forward 组成 macro；
  Pose physical batch 16、每 macro 一批、loss weight 0.25。
- SiLU finalist physical microbatch 32；validation 固定 Detect 32／Pose 16。
- qSiLU physical batch 128、64、32 都在第一个 forward OOM，尚未进入 backward；稳定完成训练的是 16。
  因此目前 31.35 GiB GPU 上不能在「不做梯度累积」的前提下使用 32／64／128。
- 这是 PyTorch eager reference 的显存结论，不等于未来 fused qSiLU kernel 的上限。

## Accepted baseline

| 指标 | mAP50-95 |
| --- | ---: |
| COCO box | 0.498022 |
| COCO person box | 0.620381 |
| BBAT box | 0.630036 |
| BBAT pose | 0.903717 |
| BBAT ball box | 0.507437 |
| BBAT bat box | 0.752634 |
| BBAT ball pose | 0.859909 |
| BBAT bat pose | 0.947526 |

## Uniform zero-shot 结果

| Activation | Worst delta | Gate | 解释 |
| --- | ---: | --- | --- |
| `qsilu_pq` | -0.013091 | 通过 | 未训练替换已在八项预算内 |
| `poly_quality` | -0.005501 | 通过 | zero-shot 最贴近，但 recovery 不稳定 |
| `poly_shift` | -0.022064 | 失败 | BBAT overall／ball box 较敏感 |
| Hardswish | -0.176230 | 失败 | 直接替换造成大幅 feature drift |
| ReLU | -0.947526 | 失败 | 八项均为 0，仅作为诊断下界 |

Hardswish 起初看起来「某些数值较高」的混淆来自不同阶段、不同指标或修复前异常 run；在同一 accepted
父权重、同一完整资料、同一 Bit-True 八项口径下，zero-shot 并不高。它经 10 epochs 可大幅恢复，
但 bat pose 仍超过 gate。

## 10-epoch short recovery：正式 selector

| Activation | COCO | BBAT box | BBAT pose | Worst delta | Gate／失败项 |
| --- | ---: | ---: | ---: | ---: | --- |
| `qsilu_pq` | 0.495273 | 0.625751 | 0.901986 | -0.008635 | 通过；八项全过 |
| Hardswish | 0.483861 | 0.618224 | 0.898519 | -0.016884 | 失败；BBAT bat pose |
| `poly_quality` | 0.496755 | 0.617772 | 0.898036 | -0.020970 | 失败；BBAT ball box |
| `poly_shift` | 0.492545 | 0.613396 | 0.889203 | -0.030138 | 失败；BBAT box、ball box、ball pose |

这张表说明：只看 COCO 会误选 `poly_quality`，只看 overall 指标也会漏掉 ball／bat class-level 风险。
八项逐项 gate 正是为了保护 COCO + BBAT5 的共同父模型。

## `poly_shift` 11-region sensitivity

| Region | Sites | COCO delta | BBAT box delta | BBAT pose delta |
| --- | ---: | ---: | ---: | ---: |
| backbone attention | 3 | -0.000228 | -0.000775 | +0.000571 |
| backbone deep | 21 | -0.001191 | -0.001247 | -0.001273 |
| backbone early | 21 | +0.000255 | +0.000465 | -0.002520 |
| detect one-to-many | 18 | 0 | 0 | 0 |
| detect one-to-one | 18 | -0.001220 | 0 | 0 |
| MASF | 3 | -0.000006 | +0.000202 | -0.000332 |
| neck | 33 | -0.000934 | -0.003098 | -0.007104 |
| neck attention | 1 | -0.000075 | -0.000045 | -0.000001 |
| pose flow | 24 | 0 | 0 | 0 |
| pose one-to-many | 24 | 0 | 0 | 0 |
| pose one-to-one | 24 | 0 | -0.012352 | -0.000141 |

zero-shot 显示 attention／MASF 较低风险、neck 与 pose one-to-one 较敏感；但预注册规则要求
`poly_shift` uniform 10-epoch prerequisite 先通过。它没有通过，所以这些观察不能被包装成已训练的
mixed policy 成果。

## Finalist 状态

SiLU seed-1 control 使用 20-epoch 上限、patience 5，在第 6 个已完成 epoch early-stop；选中的
gate-passing epoch 八项最差 delta 为 `-0.012872`，正式完成。

qSiLU 使用相同 seed／epoch／LR budget，但 physical microbatch 16。停止前只完成 epoch 1：COCO
`0.494248`、BBAT box `0.620522`、BBAT pose `0.897755`；唯一未过的是 BBAT ball box，delta
`-0.015966`，只超门槛 `0.000966`。它随后进入 epoch 2 macro 106，才依使用者要求中止。
没有 completion marker、没有 `best_joint.pt`，所以此数值只能写 provisional，不能拿来宣称 qSiLU
finalist 失败或获胜。

## 数学与硬体定位

六个函数的逐式推导见[完整数学文件](../docs/research/full35-activation-mathematical-derivations.md)。
简要定位：

- `poly_quality`／`poly_shift` 属于 SIPA 受约束积分多项式族，精确保持
  `A(x)-A(-x)=x`、zero anchor、负谷、`C²` 与 exact ReLU tails。
- `qsilu_pq` 使用 `|x|=1,2,4,8`、dyadic 系数的 `C¹` 分段二次偶残差；一个共享平方，尾端精确 ReLU。
- qSiLU 是 hardware-friendly 设计候选，但目前只有函数、Q16.10、ONNX 与 Full35 accuracy 证据；没有
  指定 FPGA／ASIC 的 synthesis、latency、power、LUT/DSP/BRAM 量测，不能说已经实现硬体加速。
- qSiLU 名称与分段二次 SiLU 近似都已有相关 prior art；可主张本专案冻结候选与证据，不能主张广义方法新颖。

## 权重发行为何只放两个

GitHub 一般单 blob 上限为 100 MB，而每个 inference 权重是 106,825,541 bytes。本次以 90,000,000-byte
以下分片发布，并附重建与 SHA-256 检查：

1. accepted SiLU `best_joint`：量化 baseline／正式父权重。
2. 完成 10-epoch 且过 gate 的 qSiLU `best_joint`：唯一可交付的 non-SiLU quantization candidate。

没有上传 425 MB optimizer resume checkpoint、29 GB runs、OOM probe payload、未完成 qSiLU finalist
`last.pt`，也没有上传已被 gate 淘汰候选的权重。这样保留量化所需的最小充分对照，不把 Git 历史变成
不可维护的 artifact 仓库。

## 给量化工作列的建议顺序

1. 重建并校验两个发布权重；禁止使用中止的 qSiLU finalist `last.pt` 当正式父权重。
2. 同一 calibration split、同一 observer、同一 per-channel／per-tensor 规则分别跑 SiLU 与 qSiLU PTQ。
3. 除网络 AP 外，逐 activation region 记录 scale、zero-point、clip ratio、saturation、负谷 code、
   门槛 1/2/4/8 附近误差，以及式 `A(n)-A(-n)=n` 是否 bit-exact。
4. Q8／Q12／Q16 枚举全输入码，比较 floor／round-to-nearest／symmetric rounding、overflow 与 saturation。
5. 只有 PTQ 任一八项超出 `-0.015` activation budget，才对 SiLU 与 qSiLU 使用相同 10-epoch QAT budget；
   不应只替失败者补训。
6. 完成 PTQ/QAT 后再决定是否恢复 qSiLU 20-epoch finalist；目前继续做 activation-only 最佳化的收益
   小于先厘清 activation × quantization interaction。

## 可追溯产物与限制

- 完整 selector 与 full Float／Bit-True zero-shot 数值：
  [`full35-activation-results.json`](full35-activation-results.json)。
- 八项宽表：[`full35-activation-results.csv`](full35-activation-results.csv)。
- 冻结 recipe：[`training/full35/activation-recipe.yaml`](../training/full35/activation-recipe.yaml)。
- 原始 29 GB artifacts 留在本机、受 `.gitignore` 排除且未删除。
- 无指定 target board／clock／precision，板上速度与功耗仍是未解事项。
- qSiLU finalist 未完成、seed 2 未执行；任何多 seed 最终优越性主张都必须等待量化决策后补齐。
