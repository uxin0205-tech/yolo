"""同一條 live 訓練軌跡的 EMA age 對照；本模組不啟動訓練或 GPU。

呼叫端先載入選定 parent EMA 權重、設定相同訓練 scope，再建立 observer，
並將它傳入 RetrySafeMacroStepEngine 的 ``ema`` 參數。成功 optimizer step
才會更新兩份 EMA；observer 本身不是 nn.Module，不掛入 trainable model。
"""

from __future__ import annotations

from collections.abc import Mapping

from .training_safety import FixedStateEMA


_DECAY = 0.9999
_TAU = 2000


def _validated_age(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("parent ema_updates 必須是非負 Python 整數，不能是 bool 或浮點數")
    return value


def parent_ema_updates(snapshot: Mapping) -> int:
    """只接受 paired snapshot 頂層 ema_updates；不猜測、不由 epoch 推算。"""
    if not isinstance(snapshot, Mapping):
        raise TypeError("paired snapshot 必須是 mapping")
    if "ema_updates" not in snapshot:
        raise ValueError("paired snapshot 缺少頂層 ema_updates，不能以 0 代替")
    return _validated_age(snapshot["ema_updates"])


class PairedEMAObserver:
    """以同一個 live model 同步觀察 fresh 與 continued-age EMA。

兩個 FixedStateEMA 都直接複製 constructor 收到的同一份模型；僅初始
updates 不同。不要把 observer 當成單一 EMA 存檔：分別存 fresh／continued。
observer 不擁有 optimizer、scaler、criterion 或任何訓練參數。
"""

    def __init__(self, model, parent_updates: int):
        self.parent_updates = _validated_age(parent_updates)
        self.fresh = FixedStateEMA(model, decay=_DECAY, tau=_TAU, updates=0)
        self.continued = FixedStateEMA(model, decay=_DECAY, tau=_TAU, updates=self.parent_updates)
        if self.fresh.fixed_state_names != self.continued.fixed_state_names:
            raise RuntimeError("兩份 EMA 的固定 state 清單不一致")
        self.observations = 0

    def update(self, model) -> None:
        """由 macro engine 在成功 optimizer step 後呼叫一次。"""
        if not self.fresh.enabled or not self.continued.enabled:
            raise RuntimeError("EMA age 對照不能停用其中一份 observer")
        if (self.fresh.updates != self.observations
                or self.continued.updates != self.parent_updates + self.observations):
            raise RuntimeError("EMA update age 與同軌跡 observer 計數不一致，拒絕繼續")
        self.fresh.update(model)
        self.continued.update(model)
        self.observations += 1

    def metadata(self) -> dict:
        """JSON-ready 診斷設定與目前更新次數；不含模型 tensor。"""
        return {
            "schema_version": 1,
            "kind": "paired_ema_age_same_live_trajectory",
            "decay": _DECAY,
            "tau": _TAU,
            "parent_ema_updates": self.parent_updates,
            "initial_updates": {"fresh": 0, "continued": self.parent_updates},
            "current_updates": {"fresh": self.fresh.updates, "continued": self.continued.updates},
            "observations": self.observations,
            "fixed_state_names": list(self.fresh.fixed_state_names),
        }
