"""依序執行一次性成本比較；每 600 秒等待檢查，正常不讀 log、不查 GPU。"""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
OUT=HERE/'artifacts/comparison-v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases=json.loads((HERE/'manifest.json').read_text())['cases']
    completed=[]
    for row in cases:
        dst=OUT/(row['id']+'.json')
        if dst.exists():
            saved=json.loads(dst.read_text())
            assert saved['status']=='passed' and saved['case']['sha256']==row['sha256'] and not saved['quick']
            completed.append(row['id']);continue
        logs=OUT/'logs';logs.mkdir(exist_ok=True)
        attempt=1
        while (logs/f'{row["id"]}-{attempt}.log').exists():attempt+=1
        log=logs/f'{row["id"]}-{attempt}.log'
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,str(HERE/'measure.py'),'--case',row['id'],'--output',str(dst)],
                stdout=stream,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            while True:
                try:code=p.wait(timeout=600);break
                except subprocess.TimeoutExpired:
                    # 正常 process 仍在執行；不讀 log、不重查 GPU。
                    continue
        if code:
            print('ERROR '+row['id']+' '+str(log),flush=True);return code
        completed.append(row['id']);print('JOB_DONE '+row['id'],flush=True)
    final=OUT/'summary.json'
    if not final.exists():
        final.write_text(json.dumps({'status':'completed','cases':completed,'count':len(completed),
            'model_sha_preserved':True,'no_training':True,'monitor_interval_seconds':600,
            'measurement_script_sha256':hashlib.sha256((HERE/'measure.py').read_bytes()).hexdigest()},indent=2)+'\n')
    print('ALL_DONE cost comparison',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
