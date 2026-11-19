"""Stage 2 score composition 與嚴格 checkpoint；保留 Stage 1 B2 血緣。"""

import copy
import json
from pathlib import Path

import torch
from ultralytics.nn.tasks import DetectionModel
from yolo_attention.config import FixedScaleFormat, ScoreAffineConfig
from yolo_attention.score_affine import QuantizedScoreAffine

from .basis import BASES, BranchScore, branch_dots
from .checkpoint import CONTRACT as STAGE1_CONTRACT
from .checkpoint import activations
from .model import SITES, BranchAttention, ProbabilityValue, convert, sha256

CANDIDATES = {"S1": FixedScaleFormat.Q10, "S2": FixedScaleFormat.POWER_OF_TWO}
CONTRACT = {**copy.deepcopy(STAGE1_CONTRACT), "schema": "binaryqk-stage2-v1",
            "scale_format": "learned-fixed-Q10-or-PoT",
            "bias_format": "decomposed-signed16/1024-query-minus-key",
            "score_format": "signed18/1024-int32-container-before-native-softmax",
            "rounding": "nearest-ties-to-even", "saturation": "scale-and-bias-before-rounding",
            "shape_policy": "1<=H,W<=32; N=H*W<=1024",
            "scale_backward": "bounded-amplitude-STE; exact-integer-score-forward"}


class AffineBranchScore(torch.nn.Module):
    def __init__(self, candidate, coefficients):
        super().__init__()
        self.candidate, self.variant, self.bases = candidate, "B2", BASES["B2"]
        self.affine = QuantizedScoreAffine(ScoreAffineConfig(format=CANDIDATES[candidate]), coefficients)

    def forward(self, q, k, *, height=None, width=None):
        if q.shape != k.shape or q.ndim != 4 or q.shape[1:3] != (4, 32):
            raise ValueError("Stage 2 requires matching B2 Q/K shapes")
        if height is None or width is None or height * width != q.shape[-1]:
            raise ValueError("explicit H/W required; never infer square tokens")
        dots = list(branch_dots(q, k, self.bases, training=self.training))
        return self.affine(dots, height=height, width=width)


def install(model, candidate):
    replacements, initial = {}, {}
    for site in SITES:
        module = model.get_submodule(site)
        if type(module) is not BranchAttention or type(module.score) is not BranchScore or module.score.variant != "B2":
            raise ValueError("Stage 2 requires the Stage 1 B2 parent at both sites")
        initial[site] = module.score.coefficients.tolist()
        replacements[site] = AffineBranchScore(candidate, module.score.coefficients)
    for site, score in replacements.items():
        model.get_submodule(site).score = score
    model.stage2_initial_coefficients = initial
    return model


def manifest_for(model, calibration_sha):
    candidates, constants = set(), {}
    for site in SITES:
        m = model.get_submodule(site)
        if (type(m) is not BranchAttention or type(m.score) is not AffineBranchScore
                or type(m.normalizer) is not torch.nn.Softmax or m.normalizer.dim != -1
                or type(m.aggregate) is not ProbabilityValue):
            raise ValueError("Stage 2 attention implementation changed")
        if m.score.affine.config != ScoreAffineConfig(format=CANDIDATES[m.score.candidate]):
            raise ValueError("Stage 2 affine config changed")
        candidates.add(m.score.candidate)
        constants[site] = m.score.affine.export_constants()
    if len(candidates) != 1:
        raise ValueError("mixed scale candidates")
    result = {**copy.deepcopy(CONTRACT), "variant": "B2", "bases": list(BASES["B2"]),
              "scale_candidate": candidates.pop(), "scale_slots": 16, "bias_entries": 1008,
              "constants": constants, "initial_coefficients": copy.deepcopy(model.stage2_initial_coefficients),
              "calibration_sha256": calibration_sha, "activation_calls": activations(model),
              "architecture": copy.deepcopy(model.yaml), "names": copy.deepcopy(model.names),
              "lineage": copy.deepcopy(model.research_lineage)}
    validate_manifest(result)
    return result


def validate_manifest(manifest):
    for key, value in CONTRACT.items():
        if manifest.get(key) != value:
            raise ValueError(f"missing or changed Stage 2 contract: {key}")
    if (manifest.get("scale_candidate") not in CANDIDATES or manifest.get("variant") != "B2"
            or manifest.get("bases") != list(BASES["B2"]) or manifest.get("scale_slots") != 16
            or manifest.get("bias_entries") != 1008):
        raise ValueError("invalid Stage 2 candidate")
    for field in ("constants", "initial_coefficients"):
        if set(manifest.get(field, {})) != set(SITES):
            raise ValueError(f"missing Stage 2 {field}")
    kind = CANDIDATES[manifest['scale_candidate']]
    expected = {'format': kind.value, 'fraction_bits': 10, 'bias_signed_bits': 16,
                'accumulator_signed_bits': 18, 'rounding': 'nearest-ties-to-even',
                'bias_orientation': 'query-minus-key', 'max_height': 32, 'max_width': 32,
                'scale_code_meaning': 'm/1024' if kind is FixedScaleFormat.Q10 else '2^k'}
    for site in SITES:
        constants = manifest['constants'][site]
        if any(constants.get(k) != v for k, v in expected.items()):
            raise ValueError('invalid deployment constants contract')
        low, high = (1, 1024) if kind is FixedScaleFormat.Q10 else (-10, 0)
        for field, shape, lower, upper in (
            ('scale_codes', (4, 2), low, high), ('bias_x_codes', (4, 63), -32768, 32767),
            ('bias_y_codes', (4, 63), -32768, 32767),
        ):
            tensor = torch.as_tensor(constants.get(field, []))
            if (tuple(tensor.shape) != shape or tensor.is_floating_point()
                    or tensor.dtype == torch.bool or tensor.min() < lower or tensor.max() > upper):
                raise ValueError(f'invalid deployment {field}')
        initial = torch.as_tensor(manifest['initial_coefficients'][site])
        if initial.shape != (4, 2) or not torch.isfinite(initial).all() or (initial <= 0).any():
            raise ValueError('invalid parent coefficients')
    lineage = manifest.get("lineage", {})
    digest = lineage.get("parent_sha256", "")
    if (lineage.get("phase") != "scale_bias" or not lineage.get("parent")
            or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest)):
        raise ValueError("missing Stage 2 lineage")
    for field in ("architecture", "names", "activation_calls", "calibration_sha256"):
        if not manifest.get(field):
            raise ValueError(f"missing {field}")
    digest = manifest['calibration_sha256']
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('invalid calibration SHA')


def save(path, model, manifest):
    path = Path(path)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(path)
    validate_manifest(manifest)
    current = manifest_for(model, manifest["calibration_sha256"])
    if any(current[k] != v for k, v in manifest.items() if k != "constants"):
        raise ValueError("Stage 2 immutable model contract changed")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in state.values()):
        raise ValueError("Stage 2 requires finite FP32 checkpoint state")
    torch.save({"manifest": current, "state_dict": state}, path)
    path.with_suffix('.json').write_text(json.dumps({"checkpoint_sha256": sha256(path)}, indent=2) + '\n')


def load(path):
    path = Path(path)
    if sha256(path) != json.loads(path.with_suffix('.json').read_text())["checkpoint_sha256"]:
        raise ValueError("Stage 2 checkpoint SHA mismatch")
    payload = torch.load(path, map_location='cpu', weights_only=True)
    manifest = payload['manifest']
    validate_manifest(manifest)
    if any(v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())
           for v in payload['state_dict'].values()):
        raise ValueError('nonfinite or non-FP32 Stage 2 state')
    model = DetectionModel(copy.deepcopy(manifest['architecture']), ch=3, nc=80, verbose=False)
    convert(model, 'B2', manifest['initial_coefficients'])
    install(model, manifest['scale_candidate'])
    model.names = copy.deepcopy(manifest['names'])
    model.research_lineage = copy.deepcopy(manifest['lineage'])
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    if manifest_for(model, manifest['calibration_sha256']) != manifest:
        raise ValueError("reloaded Stage 2 constants/activation/normalization mismatch")
    return model, manifest
