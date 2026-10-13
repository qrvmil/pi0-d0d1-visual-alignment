"""Attach to a verified local worker and its NVML host PID without touching training."""
import argparse
import json
from pathlib import Path
import time
import psutil
import pynvml

p=argparse.ArgumentParser();p.add_argument('--local-pid',type=int,required=True);p.add_argument('--nvml-pid',type=int,required=True)
p.add_argument('--output',required=True);p.add_argument('--attached-after-update',type=int,required=True);args=p.parse_args()
process=psutil.Process(args.local_pid);created=process.create_time()
if process.cmdline()[-3:]!=['-m','alignment.train','B-align']:raise ValueError('Unexpected worker command')
pynvml.nvmlInit();handle=pynvml.nvmlDeviceGetHandleByIndex(0)
initial=pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
if len(initial)!=1 or initial[0].pid!=args.nvml_pid:raise ValueError('Expected a single observed GPU worker host PID')
started=time.time();rows=[]
while True:
    try:
        if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE or process.create_time()!=created:break
    except psutil.NoSuchProcess:break
    memory=pynvml.nvmlDeviceGetMemoryInfo(handle)
    candidates=[r for r in pynvml.nvmlDeviceGetComputeRunningProcesses(handle) if r.pid==args.nvml_pid]
    rows.append(dict(seconds=time.time()-started,device_used_bytes=memory.used,
                     process_used_bytes=None if not candidates else candidates[0].usedGpuMemory))
    time.sleep(.25)
valid=[r['process_used_bytes'] for r in rows if r['process_used_bytes'] is not None]
result=dict(scope='observed sole GPU compute worker host PID; attached after initialization/first training updates',
    local_pid=args.local_pid,nvml_host_pid=args.nvml_pid,attached_after_update=args.attached_after_update,
    sample_interval_seconds=.25,peak_process_vram_bytes=max(valid) if valid else None,
    peak_device_vram_bytes=max(r['device_used_bytes'] for r in rows),samples=rows)
Path(args.output).write_text(json.dumps(result,indent=2));print('ATTACHED_VRAM_MONITOR_COMPLETE',result['peak_process_vram_bytes'])
