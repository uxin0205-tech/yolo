"""不改動 native 訓練規則的 BN/RNG retry 與固定 state EMA 修補。

這兩項是正確性保護，不表示已證明它們造成任何特定 AP 下降。
不更改 optimizer/scaler 的 step 順序、EMA 初始 age、decay 或 tau。
"""

from __future__ import annotations

from dataclasses import dataclass

from . import runtime as _runtime  # 只建立唯讀 final import path
import torch
from torch import Tensor, nn
from ultralytics.utils.torch_utils import ModelEMA, unwrap_model
from yolo_combine.hardware_contract import _IMMUTABLE_BUFFER_PARTS
from yolo_combine.joint_loss import MacroStepEngine


_BN_STATE_NAMES = ("running_mean", "running_var", "num_batches_tracked")


@dataclass
class _MacroSnapshot:
    bn_state: tuple[tuple[Tensor, Tensor], ...]
    cpu_rng: Tensor
    cuda_rng: tuple[Tensor, ...] | None

    @classmethod
    def capture(cls, model: nn.Module) -> "_MacroSnapshot":
        state = []
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for name in _BN_STATE_NAMES:
                    value = getattr(module, name, None)
                    if isinstance(value, Tensor):
                        state.append((value, value.detach().clone()))
        # 不呼叫 is_available/device_count，不因 CPU 工作而初始化 CUDA。
        cuda_rng = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else None
        return cls(tuple(state), torch.get_rng_state().clone(), cuda_rng)

    @torch.no_grad()
    def restore(self) -> None:
        for live, saved in self.bn_state:
            live.copy_(saved)
        torch.set_rng_state(self.cpu_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(list(self.cuda_rng))


class RetrySafeMacroStepEngine(MacroStepEngine):
    """一個成功 macro 只留下成功嘗試的 BN/RNG 變化。

沿用原生遞迴重試，由最外層保存唯一快照；中間失敗嘗試的 BN/RNG 不會
成為下一次起點。若最終失敗，也恢復 macro 前 BN/RNG。這不是 optimizer
transaction：若 native 在實際 optimizer step 之後才拋錯，不回滾已更新權重。
同一 engine 不支援並行／重入呼叫。
"""

    def run(
        self,
        *,
        detect_batches,
        pose_batches,
        record_gradient_statistics: bool = False,
        _amp_overflow_retries: int = 0,
    ):
        if _amp_overflow_retries < 0:
            raise ValueError("_amp_overflow_retries cannot be negative")
        outermost = _amp_overflow_retries == 0
        if outermost:
            if hasattr(self, "_macro_safety_snapshot"):
                raise RuntimeError("同一 MacroStepEngine 不可重入或並行執行")
            self._macro_safety_snapshot = _MacroSnapshot.capture(self.model)
        else:
            if not hasattr(self, "_macro_safety_snapshot"):
                raise RuntimeError("AMP retry 沒有對應的最外層 macro 快照")
            self._macro_safety_snapshot.restore()
        try:
            reset_statistics = getattr(self.losses, "reset_attempt_statistics", None)
            if callable(reset_statistics):
                reset_statistics()
            return super().run(
                detect_batches=detect_batches,
                pose_batches=pose_batches,
                record_gradient_statistics=record_gradient_statistics,
                _amp_overflow_retries=_amp_overflow_retries,
            )
        except BaseException:
            if outermost:
                self._macro_safety_snapshot.restore()
            raise
        finally:
            if outermost:
                del self._macro_safety_snapshot


class FixedStateEMA(ModelEMA):
    """保留 native EMA 平滑，對 constructor 判定的固定 state exact-copy。

請在設定 requires_grad 與 BN train/eval scope 後建立；固定清單不會隨
驗證階段的暫時 eval() 改變。若後續真的變更訓練 scope，須建立對應新 EMA，
不能把本實例當作動態 freeze/unfreeze 政策。
"""

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        live = unwrap_model(model)
        names = {
            name for name, parameter in live.named_parameters(remove_duplicate=False)
            if not parameter.requires_grad
        }
        for name, _buffer in live.named_buffers(remove_duplicate=False):
            if any(pattern in f".{name}" for pattern in _IMMUTABLE_BUFFER_PARTS):
                names.add(name)
        for module_name, module in live.named_modules(remove_duplicate=False):
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not module.training:
                for state_name in _BN_STATE_NAMES:
                    if isinstance(getattr(module, state_name, None), Tensor):
                        names.add(f"{module_name}.{state_name}" if module_name else state_name)
        # EMA 只處理 state_dict；非 persistent buffers 不在 native EMA 更新範圍。
        names.intersection_update(live.state_dict())
        super().__init__(model, decay=decay, tau=tau, updates=updates)
        names.intersection_update(self.ema.state_dict())  # 沿用 native 的 teacher 排除規則
        self.fixed_state_names = tuple(sorted(names))

    @torch.no_grad()
    def update(self, model):
        super().update(model)
        if not self.enabled:
            return
        live = unwrap_model(model).state_dict()
        averaged = self.ema.state_dict()
        for name in self.fixed_state_names:
            averaged[name].copy_(live[name].detach())
