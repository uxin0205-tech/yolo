"""Pose-only 正式入口；需先通過兩臂真實更新及 head KD 前置。"""
import argparse
from dataclasses import asdict
import json
import experiment as exp
from state_checks import digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=('native','kd'),required=True)
    exp.ARM=parser.parse_args().arm
    config,stage=exp.configure()
    for arm in ('native','kd','calibrate'):
        assert json.loads((config.run_root/(arm+'-probe-v1/summary.json')).read_text())['status']=='passed'
    session=exp.Session(config,device='0',run_name=exp.ARM+'-e5-seed1-v1')
    json.dumps(session._resolved_config())
    result=session.run()
    for router in exp.ROUTERS:
        assert digest(router.teacher)==router.initial_teacher_hash
        assert all(p.grad is None for p in router.teacher.parameters())
    with (session.run_dir/'summary.json').open('x') as f:json.dump(asdict(result),f,indent=2,default=str)
    print('JOB_DONE: Pose-only '+exp.ARM+' five epochs completed',flush=True)

if __name__=='__main__':main()
