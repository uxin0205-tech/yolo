"""序列評估；正常不讀 log、不查 GPU，600 秒監測邊界；完成立即處理。"""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    out = HERE/'artifacts'
    for job in ('binary_masf_off','fp_masf_off'):
        if (HERE/'STOP_AFTER_CURRENT').exists():
            print('PAUSED 使用者改為移除後重訓，不啟動下一組',flush=True)
            return 0
        result = out/(job+'.json')
        if result.exists():
            assert json.loads(result.read_text())['status'] == 'passed'
            continue
        logs = out/'logs'; logs.mkdir(exist_ok=True)
        attempt = 1
        while (logs/f'{job}-{attempt}.log').exists(): attempt += 1
        log = logs/f'{job}-{attempt}.log'
        with log.open('x') as stream:
            process = subprocess.Popen([sys.executable,str(HERE/'evaluate_switches.py'),job],
                stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            while True:
                try:
                    code = process.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    # 十段組成 600 秒週期；無額外 log、GPU 或檔案探查。
                    continue
        if code:
            print('ERROR '+job+' '+str(log),flush=True)
            return code
        assert json.loads(result.read_text())['status'] == 'passed'
        print('JOB_DONE '+job,flush=True)
    print('ALL_DONE 四格消融的兩組新增完整驗證已完成',flush=True)
    return 0


if __name__ == '__main__': sys.exit(main())
