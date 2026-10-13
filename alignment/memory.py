"""Per-process NVML VRAM sampling, distinct from JAX allocator accounting."""
import json
import os
import threading
import time


def process_bytes(rows,pid,total):
    value=sum(memory for process,memory in rows if process==pid)
    if not 0<=value<=total:raise ValueError('NVML process memory unavailable')
    return value


def matched_process_bytes(rows,pid,total):
    if not any(process==pid for process,_ in rows):return None
    return process_bytes(rows,pid,total)


class VramMonitor:
    def __init__(self):
        import pynvml
        self.nvml=pynvml;pynvml.nvmlInit();self.handle=pynvml.nvmlDeviceGetHandleByIndex(0)
        self.total=pynvml.nvmlDeviceGetMemoryInfo(self.handle).total
        self.pid=os.getpid();self.stop_event=threading.Event();self.samples=[];self.errors=[]
        self.started=time.time();self.thread=threading.Thread(target=self.run,daemon=True)
        self.sample();self.thread.start()

    def sample(self):
        processes=self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        memory=matched_process_bytes([(p.pid,p.usedGpuMemory) for p in processes],self.pid,self.total)
        device_memory=self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
        self.samples.append(dict(seconds=time.time()-self.started,process_vram_bytes=memory,device_vram_bytes=device_memory))

    def run(self):
        while not self.stop_event.wait(.25):
            try:self.sample()
            except Exception as e:self.errors.append(str(e))

    def finish(self,path):
        self.stop_event.set();self.thread.join();self.sample()
        previous=json.loads(path.read_text()) if path.exists() else None
        offset=0 if previous is None else previous['wall_seconds']
        samples=([dict(r,seconds=r['seconds']+offset) for r in self.samples])
        if previous is not None:samples=previous['samples']+samples
        valid=[r['process_vram_bytes'] for r in samples if r['process_vram_bytes'] is not None]
        device=[r['device_vram_bytes'] for r in samples if 'device_vram_bytes' in r]
        result=dict(pid=self.pid,sample_interval_seconds=.25,peak_process_vram_bytes=max(valid) if valid else None,
                    peak_device_vram_bytes=max(device) if device else None,process_pid_visible=bool(valid),
                    wall_seconds=offset+time.time()-self.started,total_vram_bytes=self.total,errors=self.errors,samples=samples)
        path.write_text(json.dumps(result,indent=2))
        if self.errors:raise RuntimeError('VRAM measurement had sampling errors')
        return result
