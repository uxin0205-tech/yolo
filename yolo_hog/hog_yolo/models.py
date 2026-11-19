"""raw P3 的短生命週期 hook；輔助頭不進推論 forward。"""

import copy

import torch
from torch import nn

from .hog import HOGAuxiliary

HOG_SPEC = {
    "raw_site": "model.16",
    "channels": 256,
    "cell_size": 8,
    "bins": 9,
    "energy_epsilon": 1e-6,
    "mask": "all80 GT center hard union",
    "target": "augmented RGB unsigned cell-L1 HOG",
    "loss": "valid-cell image mean then batch sum",
    "deployment_auxiliary": False,
    "aux_seed": 20260909,
}


def attach(model):
    if hasattr(model, "hog_aux") or model.model[-1].f != [16, 19, 22]:
        raise ValueError("unexpected HOG parent graph")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(HOG_SPEC["aux_seed"])
        model.hog_aux = HOGAuxiliary(256)
    return model


def role(name):
    if name.startswith("hog_aux."):
        return "hog"
    if name.startswith("model.23.") and ".dfl." not in name:
        return "head"
    if (
        any(name.startswith(f"model.{i}.") for i in (13, 16, 17, 19, 20, 22))
        and ".attn." not in name
    ):
        return "neck"
    return "fixed"


def configure(model, scope="neck-head-hog-fixed-attention"):
    if scope != "neck-head-hog-fixed-attention":
        raise ValueError(scope)
    model.train()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(role(name) != "fixed")
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    return [p for p in model.parameters() if p.requires_grad]


def criterion(model, start_epoch=0):
    model.criterion = model.init_criterion()
    c = model.criterion
    c.o2m = c.o2m_copy = c.final_o2m = 0.1
    c.o2o = 0.9
    c.updates = 5 + start_epoch


def parts(model, batch, mu, force=False):
    captured = []
    hook = model.model[16].register_forward_hook(
        lambda _m, _i, value: captured.append(value)
    )
    try:
        native, items = model.loss(batch)
        native = native.sum()
    finally:
        hook.remove()
    if len(captured) != 1 or not captured[0].requires_grad:
        raise ValueError("raw P3 must be captured exactly once with gradients")
    raw = captured.pop()
    aux = (
        model.hog_aux(raw, batch["img"], batch["bboxes"], batch["batch_idx"])
        if mu > 0 or force
        else None
    )
    total = native if aux is None else native + mu * aux.loss_sum
    return total, native, aux, raw, items


def strip(model):
    result = copy.deepcopy(model).eval()
    if not hasattr(result, "hog_aux"):
        raise ValueError("cannot strip missing HOG head")
    del result.hog_aux
    if hasattr(result, "criterion"):
        del result.criterion
    result.stage5_manifest = copy.deepcopy(result.stage5_manifest)
    result.stage5_manifest["auxiliary_present"] = False
    return result


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [x for item in value.values() for x in tensors(item)]
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in tensors(item)]
    return []


def strip_proof(model):
    original = copy.deepcopy(model).float().cpu().eval()
    deployment = strip(original)
    removed = set(original.state_dict()) - set(deployment.state_dict())
    if removed != {"hog_aux.projection.weight", "hog_aux.projection.bias"}:
        raise ValueError("strip changed native model keys")
    image = torch.rand(1, 3, 160, 160, generator=torch.Generator().manual_seed(7))
    with torch.inference_mode():
        expected, actual = tensors(original(image)), tensors(deployment(image))
    if len(expected) != len(actual) or not all(
        torch.equal(a, b) for a, b in zip(expected, actual)
    ):
        raise ValueError("HOG strip changed predictions")
    if any(
        not torch.equal(v, original.state_dict()[n])
        for n, v in deployment.state_dict().items()
    ):
        raise ValueError("HOG strip changed native weights")
    return {
        "passed": True,
        "dtype": "FP32",
        "input": [1, 3, 160, 160],
        "outputs_exact": True,
        "native_state_exact": True,
        "removed_keys": sorted(removed),
    }
