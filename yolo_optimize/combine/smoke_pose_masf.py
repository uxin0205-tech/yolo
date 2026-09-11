"""沿用 J0 真實更新 smoke，另驗證 MASF alpha／context 更新。"""
from pose_masf import PoseMASFSource
from smoke_j0 import main

if __name__ == '__main__':
    main(run_name='j0-masf-smoke-v1', source_class=PoseMASFSource)
