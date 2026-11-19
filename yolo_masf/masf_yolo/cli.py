"""MASF 單一入口；舊 YOLO11 實驗已結束並移除執行程式。"""

import sys


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] != 'stage4':
        raise SystemExit('用法：python -m masf_yolo.cli stage4 {diagnostics,init,status,run,worker,plot}')
    from .stage4.cli import main as stage4_main
    return stage4_main(args[1:])


if __name__ == '__main__':
    raise SystemExit(main())
