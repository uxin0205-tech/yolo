from __future__ import annotations

import os

import pytest


@pytest.mark.integration
def test_formal_handoff_cpu_validation_is_explicitly_opt_in() -> None:
    """正式 winner 尚未交付時，整合測試要明確跳過而不是猜測模型。"""

    if not os.environ.get("ARCHITECHURE_2_HANDOFF"):
        pytest.skip("設定 ARCHITECHURE_2_HANDOFF 後才驗證正式 Fusion Winner")
    pytest.skip("正式 handoff loader 由 yolo_combine builder 契約提供後啟用")
