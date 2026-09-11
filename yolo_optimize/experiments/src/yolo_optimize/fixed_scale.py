"""固定 scale 的本地等價早退；不修改封存來源或係數。"""
from types import MethodType


def install_fixed_early_return(score):
    """僅省略固定、非校準路徑的無用 reduction，其他路徑委回原方法。"""
    if getattr(score, '_fixed_early_return_installed', False):
        raise ValueError('不可重複安裝 fixed early-return')
    original = score._coefficient

    def coefficient(self, index, q, k, attention_scale):
        if str(self.scale_mode) != 'dynamic' and not bool(self.calibration_enabled):
            return self._fixed(index)
        return original(index, q, k, attention_scale)

    score._coefficient = MethodType(coefficient, score)
    score._fixed_early_return_installed = True
