# PWL Attention final-training rationale

Access date for every external source in this note: **2026-08-17**.

## Scope and status of the claims

This note records the evidence behind the final YOLO26m PWL Attention training recipe. It deliberately distinguishes:

- **source facts**, which are statements made by an original paper or by official PyTorch/Ultralytics documentation or source; and
- **project inferences**, which are engineering choices for this fixed checkpoint, fixed hardware table, short recovery budget, and audit requirements.

The sources do not prove that this particular recipe will improve COCO mAP. That is determined only by the planned full COCO2017 validation runs and the predeclared gates.

## Direct Float-PWL surrogate and Bit-True reference

### Why a differentiable surrogate is necessary

The deployment path contains Q8.8 rounding, saturation, integer conversion, and discrete segment lookup. PyTorch autograd supports gradients only for floating-point and complex tensors, not integer tensors ([PyTorch `torch.autograd`](https://docs.pytorch.org/docs/stable/autograd)). PyTorch's own autograd definitions assign a zero derivative to `floor`, while fake-quantization has an explicit custom backward rule ([PyTorch `derivatives.yaml`](https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml)). Therefore, merely executing the Bit-True integer path does not provide a useful ordinary gradient to upstream Q/K parameters.

The straight-through estimator (STE) is not the mathematical derivative of the discrete forward operation. Bengio, Léonard, and Courville introduced the name for a heuristic that copies the gradient through a hard stochastic/non-smooth unit ([original STE paper](https://arxiv.org/abs/1308.3432)). Likewise, PyTorch describes quantization-aware training as simulating quantization in floating point and typically using an STE because rounding is non-differentiable ([official PyTorch QAT article](https://pytorch.org/blog/quantization-aware-training/)). These sources support treating an STE as an explicit optimization approximation, not as an invisible property of bit-accurate inference.

**Project inference.** Training therefore uses a direct **Float-PWL surrogate**: the fixed 20-segment, range `[-10, 0]`, `delta=0.5` piecewise-linear function is evaluated in floating point, with the same fixed endpoint values and clamping semantics but without Q8.8 rounding, integer casts, or integer indexing in the gradient path. Linear interpolation supplies an ordinary finite derivative inside each segment (with conventional subgradient behavior at clamps/knots), so gradients can reach both relative-bias tables and Q/K. The exact denominator remains the same float reference specified by the project.

This surrogate is intentionally narrower than generic QAT: it does not claim to reproduce every integer rounding effect during backpropagation. Its adequacy must be checked by finite/non-zero gradient tests during training and by actual Bit-True evaluation after every candidate checkpoint.

### Why the Bit-True path must not secretly use STE

An STE changes only backward semantics; it makes the backward pass intentionally differ from the derivative of the executed hard operation ([Bengio et al.](https://arxiv.org/abs/1308.3432)). Hiding such a rule inside the Bit-True implementation would combine two different responsibilities:

1. a reference oracle for Q8.8 rounding, saturation, table lookup, and row normalization; and
2. a biased gradient estimator used for optimization.

**Project inference.** The Bit-True implementation remains reference-only and contains no detach trick or custom STE. This makes save/reload equivalence, endpoint packing, rounding boundaries, and Float-to-Bit-True reconfiguration independently testable. Model selection always runs the Bit-True path on all 5,000 COCO2017 validation images; Float-PWL metrics and live weights are diagnostic only.

## EMA and non-floating state synchronization

Ultralytics `ModelEMA.update()` iterates through the EMA `state_dict` but appends a tensor for interpolation only when `v.dtype.is_floating_point`; its actual update is `lerp`/multiply-add over that filtered list ([Ultralytics official `ModelEMA` reference and source](https://docs.ultralytics.com/reference/utils/torch_utils/#ultralytics.utils.torch_utils.ModelEMA.update)). Consequently, an integer buffer such as `current_epoch` is copied at EMA construction but is not subsequently updated by the normal EMA update. This is a source-level fact even though the class-level prose broadly says it keeps a moving average of the state dict.

**Project inference.** Any discrete control state that affects forward behavior must be copied explicitly from the live model to the EMA model at the relevant lifecycle boundary, and a regression test must assert live/EMA agreement. Otherwise, validation of EMA `best.pt` can execute a different normalization mode from the live training model. The final recipe does not use progressive normalization, but the synchronization fix remains necessary to prevent recurrence and to make older progressive checkpoints auditable.

EMA synchronization is separate from freezing. `requires_grad=False` prevents optimizer gradients for parameters, but BatchNorm training behavior maintains running estimates in buffers by default ([official `BatchNorm2d` documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)). **Project inference:** non-target modules must therefore be kept in evaluation mode or have their buffer state otherwise locked, and before/after state snapshots must prove that non-Attention running mean, variance, and batch counters are byte-identical.

## Multiple seeds and reporting

Reimers and Gurevych showed that changing only the random seed could materially change neural-network scores and argued for distributions from multiple executions instead of a single score ([original score-distributions study](https://arxiv.org/abs/1707.09861)). The NeurIPS reproducibility program likewise treats robust experimental workflow and transparent reporting as part of reliable ML research ([Pineau et al., JMLR](https://jmlr.org/papers/v22/20-303.html)). PyTorch also cautions that completely reproducible results are not guaranteed across releases, commits, platforms, or CPU/GPU execution even with identical seeds ([official reproducibility note](https://docs.pytorch.org/docs/stable/notes/randomness.html)).

**Project inference.** The winning pilot recipe is repeated with seeds 0, 1, and 2, and the report includes every result plus mean, sample standard deviation, minimum, and maximum. The highest single run is not by itself evidence of a stable recipe: the trained recipe must improve the zero-train parent mean by at least `0.001` mAP50-95 before replacing it. Three runs are a compute-bounded stability check, not a claim of statistical significance; the `0.001` rule is a predeclared engineering tolerance for evaluator/run noise, not a p-value.

The immutable manifests therefore record the seed and the exact Git, Ultralytics, Python, PyTorch, CUDA, GPU, dataset, and parent-checkpoint provenance needed to interpret remaining variation.

## Progressive unfreezing and simultaneous sites

Howard and Ruder proposed gradual unfreezing as a fine-tuning technique intended to retain pretrained knowledge and avoid catastrophic forgetting, and their ablations compare last-layer, full, and gradual-unfreezing strategies ([original ULMFiT paper](https://proceedings.mlr.press/v80/howard18a.html)). That evidence comes from language modeling rather than object detection, so it establishes the design pattern, not expected YOLO effect size.

**Project inference.** Recovery expands capacity only after the narrower state has been evaluated:

1. Phase A adapts both sites' decomposed relative-bias tables.
2. Phase B adds both sites' Q/K convolutions and Q/K BatchNorm affine parameters.
3. Phase C adds the remainder of both `HardwareFriendlyAttention` modules at half the pilot LR.

Both Attention sites are unfrozen together in every phase. This preserves a single symmetric deployment state and avoids attributing an intermediate architecture mismatch to the PWL method. Each phase is an immutable run with a new optimizer; only a checkpoint accepted by Bit-True validation becomes the next parent. If a child is worse than its parent by more than `0.001` mAP50-95, the next phase starts from the better parent rather than inheriting the regression.

## Why the recipe excludes adjacent techniques

These exclusions are scope and identifiability decisions, not claims that the methods are generally ineffective.

### No progressive Exact-to-PWL blend

PyTorch's QAT example notes an empirical case where delaying fake quantization for initial steps helped a particular LLM setup ([official PyTorch QAT article](https://pytorch.org/blog/quantization-aware-training/)); it does not establish that a progressive schedule is universally required. Here the target approximation and parent are already fixed, the phases are only two to four epochs, and the earlier integer epoch buffer exposed an EMA/live mismatch.

**Project inference.** Float-PWL is active from the first batch. This directly optimizes under one normalization surrogate, removes a schedule hyperparameter and discrete control state from formal training, and keeps every phase checkpoint semantically comparable. Progressive-state synchronization remains tested only as a regression safeguard.

### No learnable PWL endpoints

Learnable quantization parameters can be effective: PACT explicitly optimizes its clipping parameter to determine quantization scale ([original PACT paper](https://arxiv.org/abs/1805.06085)). That result shows that learnable quantizer degrees of freedom are a real alternative; it does not make them compatible with a fixed table contract.

**Project inference.** The 21 UQ1.15 endpoints, 336-bit table, range, and segment width are the hardware specification and are held fixed. Learning endpoints would change the deployed function being evaluated, require constraints for monotonicity/range/packing, and confound whether recovery came from Attention weights or from redesigning the approximation. Endpoint learning belongs in a separately specified hardware-table search, not this recovery run.

### No SWA

The original SWA method averages multiple points along an SGD trajectory generated with a cyclical or constant learning rate ([Izmailov et al.](https://arxiv.org/abs/1803.05407)). PyTorch's implementation also calls out special buffer handling and the need to update BatchNorm statistics after averaging ([official `AveragedModel` documentation](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html)).

**Project inference.** These short, independently gated AdamW phases deliberately do not create an SWA trajectory or add a second model-averaging policy beside Ultralytics EMA. Adding SWA would expand scheduler and BatchNorm-buffer choices and weaken the phase checkpoint/state-scope audit. It can be evaluated later as a separately predeclared recipe if the direct recovery is stable.

### No knowledge distillation

Knowledge distillation trains a student to transfer behavior from a teacher or ensemble, including information carried in softened output distributions ([Hinton, Vinyals, and Dean](https://arxiv.org/abs/1503.02531)). It therefore adds a teacher, temperature/loss weighting, and a second target beyond the COCO detection objective.

**Project inference.** This experiment is intended to measure recovery caused by the fixed PWL surrogate and constrained Attention updates. KD would change that question and make gains harder to attribute, so it is outside scope rather than a rejected claim about KD quality.

### No full-model recovery

Gradual-unfreezing work motivates controlling how much pretrained state is disturbed during adaptation and discusses catastrophic forgetting under aggressive fine-tuning ([Howard and Ruder](https://proceedings.mlr.press/v80/howard18a.html)). It does not prove that full YOLO fine-tuning would fail here.

**Project inference.** The observed degradation was introduced only at two normalization sites. Updating backbone, neck, Detect, FFN, or outer blocks would vastly enlarge the mutable state, allow ordinary detector retraining to mask PWL error, and make the non-regression audit less informative. The project therefore stops at complete updates of the two Attention modules and retains the zero-train Bit-True parent whenever constrained recovery is not stable across seeds.

## Decision summary

- Optimize through fixed Float-PWL interpolation; never optimize through a disguised Bit-True STE.
- Evaluate and select only reconfigured Bit-True EMA checkpoints on full COCO2017 val.
- Explicitly synchronize non-floating forward-control state between live and EMA models; freeze both out-of-scope parameters and buffers.
- Expand trainable scope bias -> Q/K -> full Attention at both sites together, with immutable phase boundaries and rollback gates.
- Report all three seeds and distribution summaries; treat `0.001` as a predeclared practical gate, not statistical significance.
- Keep endpoints, PoT coefficients, denominator reference, and all non-Attention model state fixed so the experiment answers one narrow question.

## Primary-source index

All entries below were accessed 2026-08-17.

- Bengio, Léonard, Courville, *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation*: <https://arxiv.org/abs/1308.3432>
- Choi et al., *PACT: Parameterized Clipping Activation for Quantized Neural Networks*: <https://arxiv.org/abs/1805.06085>
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network*: <https://arxiv.org/abs/1503.02531>
- Howard and Ruder, *Universal Language Model Fine-tuning for Text Classification*: <https://proceedings.mlr.press/v80/howard18a.html>
- Izmailov et al., *Averaging Weights Leads to Wider Optima and Better Generalization*: <https://arxiv.org/abs/1803.05407>
- Pineau et al., *Improving Reproducibility in Machine Learning Research*: <https://jmlr.org/papers/v22/20-303.html>
- Reimers and Gurevych, *Reporting Score Distributions Makes a Difference*: <https://arxiv.org/abs/1707.09861>
- PyTorch official autograd documentation: <https://docs.pytorch.org/docs/stable/autograd>
- PyTorch official autograd derivative source: <https://github.com/pytorch/pytorch/blob/main/tools/autograd/derivatives.yaml>
- PyTorch official BatchNorm2d documentation: <https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>
- PyTorch official QAT article: <https://pytorch.org/blog/quantization-aware-training/>
- PyTorch official reproducibility note: <https://docs.pytorch.org/docs/stable/notes/randomness.html>
- PyTorch official `AveragedModel` documentation: <https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html>
- Ultralytics official `ModelEMA` reference/source: <https://docs.ultralytics.com/reference/utils/torch_utils/#ultralytics.utils.torch_utils.ModelEMA.update>
