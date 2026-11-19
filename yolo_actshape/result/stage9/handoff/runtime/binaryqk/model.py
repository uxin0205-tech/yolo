"""由同一個經 SHA 驗證的 Detect parent 建立分支，嚴格保存非 Q/K 權重。"""

import copy
import hashlib
from pathlib import Path

import torch
from torch import nn
from ultralytics.nn.modules.block import Attention
from ultralytics.nn.modules.conv import Conv

from .basis import BranchScore

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / "yolo_attention/weights/yolo26m.pt"
PARENT_SHA = "401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7"
SITES = ("model.10.m.0.attn", "model.22.m.0.1.attn")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_parent():
    if sha256(PARENT) != PARENT_SHA:
        raise ValueError("parent checkpoint SHA mismatch")
    # 唯一允許 pickle 的既有可信 parent；新 checkpoint 使用 weights_only=True。
    payload = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = payload["model"].float().eval()
    verify_native(model)
    return model


def verify_native(model):
    actual = tuple(n for n, m in model.named_modules() if isinstance(m, Attention))
    if actual != SITES or model.yaml.get("scale") != "m" or model.yaml.get("nc") != 80:
        raise ValueError(f"unexpected parent architecture/sites: {actual}")
    for site in SITES:
        attn = model.get_submodule(site)
        if (attn.num_heads, attn.key_dim, attn.head_dim) != (4, 32, 64):
            raise ValueError("unexpected head dimensions")


def slice_projection(source, indices):
    conv = source.conv
    if conv.groups != 1 or conv.bias is not None or not isinstance(source.act, nn.Identity):
        raise ValueError("requires unfused native QKV Conv/BN with identity activation")
    target = Conv(conv.in_channels, len(indices), k=conv.kernel_size, s=conv.stride,
                  p=conv.padding, d=conv.dilation[0], act=False).to(conv.weight)
    target.bn.eps, target.bn.momentum = source.bn.eps, source.bn.momentum
    indices = torch.tensor(indices, device=conv.weight.device)
    with torch.no_grad():
        target.conv.weight.copy_(conv.weight[indices])
        for name in ("weight", "bias", "running_mean", "running_var"):
            getattr(target.bn, name).copy_(getattr(source.bn, name)[indices])
        target.bn.num_batches_tracked.copy_(source.bn.num_batches_tracked)
    return target.train(source.training)


class SplitQKV(nn.Module):
    def __init__(self, source):
        super().__init__()
        h, d, v = source.num_heads, source.key_dim, source.head_dim
        indices = [[], [], []]
        for head in range(h):
            offset = head * (2 * d + v)
            for group, start, length in ((0, 0, d), (1, d, d), (2, 2 * d, v)):
                indices[group].extend(range(offset + start, offset + start + length))
        self.q, self.k, self.v = (slice_projection(source.qkv, idx) for idx in indices)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


class ProbabilityValue(nn.Module):
    def forward(self, probabilities, value):
        return value @ probabilities.transpose(-2, -1)


class BranchAttention(nn.Module):
    def __init__(self, source, variant, coefficients):
        super().__init__()
        self.num_heads, self.key_dim, self.head_dim = source.num_heads, source.key_dim, source.head_dim
        self.qkv = SplitQKV(source)
        self.score = BranchScore(variant, coefficients)
        if self.score.coefficients.shape[0] != self.num_heads:
            raise ValueError("coefficient head count differs from source")
        self.normalizer = nn.Softmax(dim=-1)
        self.aggregate = ProbabilityValue()
        self.pe, self.proj = copy.deepcopy(source.pe), copy.deepcopy(source.proj)
        self.train(source.training)

    def forward(self, x):
        b, c, height, width = x.shape
        maps = self.qkv(x)
        q, k, v = (t.reshape(b, self.num_heads, -1, height * width) for t in maps)
        # AMP 訓練仍以 FP32 計算 binary score／Softmax；eval 保持同一路徑。
        with torch.autocast(device_type=x.device.type, enabled=False):
            probabilities = self.normalizer(self.score(q.float(), k.float(), height=height, width=width))
        value = self.aggregate(probabilities, v).reshape(b, c, height, width)
        return self.proj(value + self.pe(maps[2]))


def convert(model, variant, coefficients):
    verify_native(model)
    if set(coefficients) != set(SITES):
        raise ValueError("coefficients must explicitly cover both sites")
    replacements = {site: BranchAttention(model.get_submodule(site), variant, coefficients[site])
                    for site in SITES}
    # 全部先建立成功，避免失敗留下只替換一處的模型。
    for site, module in replacements.items():
        parent, name = site.rsplit(".", 1)
        setattr(model.get_submodule(parent), name, module)
    return model


def train_qk_only(model):
    model.train()
    prefixes = tuple(f"{site}.qkv.{part}." for site in SITES for part in ("q", "k"))
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith(prefixes))
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
    return [param for param in model.parameters() if param.requires_grad]


def train_scope(model, scope):
    if scope not in {"attention", "full"}:
        raise ValueError(f"unsupported scope: {scope}")
    model.train()
    for name, parameter in model.named_parameters():
        selected = scope == "full" or name.startswith(tuple(f"{s}." for s in SITES))
        parameter.requires_grad_(selected and ".dfl." not in name)
    return [p for p in model.parameters() if p.requires_grad]


def bn_buffers(model):
    return {f"{name}.{key}" if name else key
            for name, module in model.named_modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
            for key, _ in module.named_buffers(recurse=False)}


def frozen_hashes(model, mutable_buffers=()):
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for n, t in model.state_dict().items() if n not in trained and n not in mutable_buffers}
