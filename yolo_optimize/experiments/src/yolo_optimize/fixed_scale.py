"""固定 scale 的本地等價早退；不修改封存來源或係數。"""
from types import MethodType


def install_fixed_early_return(score):
    """僅省略固定、非校準路徑的無用 reduction，其他路徑委回原方法。

    保存未綁定函式而不是 bound method，避免 patched score 被 deepcopy 後，
    dynamic／calibration fallback 仍回頭呼叫原始 score。
    """
    if getattr(score, '_fixed_early_return_installed', False):
        raise ValueError('不可重複安裝 fixed early-return')
    original = score._coefficient
    if not isinstance(original, MethodType) or original.__self__ is not score:
        raise TypeError('score._coefficient 必須是綁定目前 score 的 instance method')
    original_function = original.__func__

    def coefficient(self, index, q, k, attention_scale):
        if str(self.scale_mode) != 'dynamic' and not bool(self.calibration_enabled):
            return self._fixed(index)
        return original_function(self, index, q, k, attention_scale)

    score._coefficient = MethodType(coefficient, score)
    score._fixed_early_return_installed = True
