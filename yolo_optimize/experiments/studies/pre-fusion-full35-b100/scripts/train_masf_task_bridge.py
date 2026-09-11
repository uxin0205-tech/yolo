"""重用已驗證的 E5→E10 續訓；只更換訓練中的 MASF 梯度路徑。"""
import json
import sys
from common import ROOT, write_json, sha256
import continue_masf_head as continuation
from masf_task_bridge import TaskAlignedMASFDetect


class BridgeHarness(continuation.Harness):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        assert self.variant == 'fork'
        calibration = json.loads((ROOT / 'artifacts/masf-task-calibration-v1/summary.json').read_text())
        assert calibration['status'] == 'passed'
        assert calibration['source_sha256'] == continuation.SOURCE_SHA['fork']
        model.model[23].__class__ = TaskAlignedMASFDetect
        model.model[23].bridge_coefficient = calibration['one2one_masf_gradient_coefficient']
        return model


def main():
    assert '--variant' in sys.argv and sys.argv[sys.argv.index('--variant') + 1] == 'fork'
    name = sys.argv[sys.argv.index('--name') + 1]
    assert name.startswith('masf-task-') and name.replace('-', '').isalnum()
    # 只替換本程序使用的 trainer factory，不改原始續訓檔案或已完成結果。
    continuation.Harness = BridgeHarness
    continuation.main()
    output = ROOT / 'artifacts' / name
    report = json.loads((output / 'summary.json').read_text())
    calibration = json.loads((ROOT / 'artifacts/masf-task-calibration-v1/summary.json').read_text())
    report.update(method='train-only one2one MASF gradient bridge',
                  change='same native fork E5 continuation; only add calibrated one2one gradient to MASF',
                  bridge_coefficient=calibration['one2one_masf_gradient_coefficient'],
                  bridge_source_sha256=sha256(__file__.replace('train_masf_task_bridge.py', 'masf_task_bridge.py')),
                  wrapper_source_sha256=sha256(__file__),
                  paired_control='masf-head-fork-v1', inference_extra_masf_passes=0)
    write_json(output / 'summary.json', report)


if __name__ == '__main__':
    main()
