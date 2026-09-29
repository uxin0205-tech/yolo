"""依 Fusion Winner handoff 路徑凍結不可變模組。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


def _validated_paths(model: nn.Module, paths: Iterable[str]) -> tuple[str, ...]:
    result = tuple(paths)
    if not result:
        raise ValueError("frozen module paths 不得為空")
    if len(set(result)) != len(result):
        raise ValueError("frozen module paths 不得重複")
    ordered = sorted(result)
    for index, path in enumerate(ordered):
        if not path or path.startswith(".") or path.endswith("."):
            raise ValueError(f"無效 frozen module path：{path!r}")
        try:
            model.get_submodule(path)
        except AttributeError as error:
            raise ValueError(f"frozen module path 不存在：{path}") from error
        for other in ordered[index + 1 :]:
            if other.startswith(path + "."):
                raise ValueError(f"frozen module paths 有父子重疊：{path} / {other}")
    return result


def apply_frozen_paths(
    model: nn.Module,
    paths: Iterable[str],
    *,
    reset_trainable: bool = True,
) -> tuple[str, ...]:
    """凍結 handoff 宣告路徑，並讓其 BN/dropout 維持 eval。"""

    selected = _validated_paths(model, paths)
    if reset_trainable:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    for path in selected:
        module = model.get_submodule(path)
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return selected


def enforce_frozen_eval(model: nn.Module, paths: Iterable[str]) -> None:
    """在外部呼叫 model.train() 後重新鎖住 frozen module mode。"""

    for path in _validated_paths(model, paths):
        model.get_submodule(path).eval()


@dataclass
class FrozenStateGuard:
    """拒絕 frozen parameters、buffers 或 train/eval mode 漂移。"""

    paths: tuple[str, ...]
    state: dict[str, torch.Tensor]

    @classmethod
    def capture(
        cls,
        model: nn.Module,
        paths: Iterable[str],
        *,
        reset_trainable: bool = True,
    ) -> FrozenStateGuard:
        selected = apply_frozen_paths(
            model,
            paths,
            reset_trainable=reset_trainable,
        )
        state: dict[str, torch.Tensor] = {}
        for path in selected:
            module = model.get_submodule(path)
            for name, value in module.state_dict().items():
                state[f"{path}.{name}"] = value.detach().cpu().clone()
        return cls(selected, state)

    def assert_unchanged(self, model: nn.Module) -> None:
        current: dict[str, torch.Tensor] = {}
        for path in _validated_paths(model, self.paths):
            module = model.get_submodule(path)
            if module.training:
                raise AssertionError(f"frozen module 進入 train mode：{path}")
            for name, value in module.state_dict().items():
                current[f"{path}.{name}"] = value.detach().cpu()
        if current.keys() != self.state.keys():
            raise AssertionError("frozen state keys 改變")
        changed = [
            name
            for name, value in current.items()
            if not torch.equal(value, self.state[name])
        ]
        if changed:
            raise AssertionError(f"frozen parameters 或 buffers 改變：{changed[:10]}")
