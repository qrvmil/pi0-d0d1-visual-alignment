"""Execute an immutable in-memory snapshot of the pipeline; record every attempt."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
script=(ROOT/'scripts/pipeline.sh').read_text()
started=time.time()
signature=hashlib.sha256(script.encode()).hexdigest()
print('PIPELINE_SNAPSHOT',signature,flush=True)
result=subprocess.run(['bash','-c',script],cwd=ROOT)
record=dict(started_unix=started,seconds=time.time()-started,exit_code=result.returncode,script_sha256=signature)
with (ROOT/'logs/pipeline_attempts.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
print('PIPELINE_EXIT',record,flush=True)
raise SystemExit(result.returncode)
