"""Wait for a specifically live cache process, then execute the gated pipeline once."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import psutil

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('cache_pid',type=int);args=p.parse_args()
process=psutil.Process(args.cache_pid)
if process.cmdline()[-3:]!=['-u','-m','alignment.build_cache']:
    raise RuntimeError('PID is not the expected live teacher cache process')
started=process.create_time()
print('WAITING_VERIFIED_CACHE_PROCESS',args.cache_pid,started,flush=True)
while True:
    try:
        if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE or process.create_time()!=started:break
    except psutil.NoSuchProcess:break
    time.sleep(5)
info=json.loads((ROOT/'logs/cache_build.json').read_text())
if info['samples']!=11745 or info['entries']!=11745*4:raise ValueError('Full cache did not complete')
print('CACHE_COMPLETE_START_PIPELINE',flush=True)
started=time.time()
result=subprocess.run(['bash','scripts/pipeline.sh'],cwd=ROOT)
(ROOT/'logs/pipeline_runtime.json').write_text(json.dumps(dict(exit_code=result.returncode,seconds=time.time()-started),indent=2))
raise SystemExit(result.returncode)
