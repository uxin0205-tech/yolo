"""第三組 P2 的真實 Pose 梯度與固定既有權重驗證。"""
import smoke_j0 as smoke
from pose_p2_masf import install, P2Source, P2EMA


def configure():
    config, stages = install()
    smoke.JOINT_STAGES = stages
    return config


if __name__ == '__main__':
    from j0_runtime import require_training_enabled
    require_training_enabled()
    smoke.install = configure
    smoke.PoseOnlyEMA = P2EMA
    smoke.main(run_name='j0-p2-smoke-v1', source_class=P2Source, extra_trainable_part='.p2_masf.')
