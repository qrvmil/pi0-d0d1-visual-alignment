"""Evaluation-only paired rollout runner; never imported by a training dataloader."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
from .stats import paired_effect,d0_decision

ROOT=Path(__file__).resolve().parents[1]


def seed_plan(extended=False):
    return list(range(10000,10500 if extended else 10050))


def validate_pairs(control,align):
    if [r['seed'] for r in control]!=[r['seed'] for r in align]:raise ValueError('Paired seed mismatch')
    for c,a in zip(control,align):
        if c['reset_hash']!=a['reset_hash']:raise ValueError('Paired reset mismatch')
        n=min(len(c['noise_hashes']),len(a['noise_hashes']))
        if n==0 or c['noise_hashes'][:n]!=a['noise_hashes'][:n]:raise ValueError('Paired flow-noise mismatch')


def validate_cohort(rows,task):
    expected=seed_plan(len(rows)==500)
    if task!='D0' and len(rows)!=50:raise ValueError('Only D0 may extend to500')
    if [r['seed'] for r in rows]!=expected:raise ValueError('Exact unique preregistered seed cohort required')


def require_current_binding(saved,current):
    if any(saved.get(k)!=v for k,v in current.items()):raise ValueError('Current evaluation binding differs from saved artifacts')


def require_evaluation_resume_binding(saved,current):
    serialized=json.loads(json.dumps(current))
    if saved!=serialized:raise ValueError('Evaluation checkpoint/normalization/runtime binding drift')


def needs_extended_cohort(cohorts):
    return any(r['seed']>=10050 for rows in cohorts for r in rows)


class NoiseLoggedPolicy:
    def __init__(self,policy):self.policy=policy;self.hashes=[]
    def infer(self,obs,**kwargs):
        noise=np.asarray(kwargs['noise'])
        self.hashes.append(hashlib.sha256(noise.tobytes()).hexdigest())
        return self.policy.infer(obs,**kwargs)


def run(arm,task,extended=False):
    if extended and task!='D0':raise ValueError('Only D0 extension permitted')
    for name in ('B-control','B-align'):
        complete=json.loads((ROOT/'logs'/f'{name}_complete.json').read_text())
        if complete['steps']!=3001:raise RuntimeError('Both main runs must finish before D2 access')
    os.environ['MUJOCO_GL']='egl'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ['JAX_PLATFORMS']='cuda'
    os.environ['XLA_FLAGS']='--xla_gpu_autotune_level=0'
    os.environ['MULTITASK_WORK']=str(ROOT/'private/eval_runtime')
    sys.path[:0]=['/root/maniskill_myws/src','/root/maniskill_myws/third_party/openpi/third_party/libero']
    from maniskill_myws.pld.multitask_eval import EvalConfig,EvaluationSession
    from maniskill_myws.pld.multitask_eval import save_video
    from maniskill_myws.pld.libero_runtime import configure_base_inference
    from maniskill_myws.pld.multitask_data import V4
    configure_base_inference(V4)
    # Matching architecture for newly created LoRA32 params.
    from .training import config
    config(arm)
    if task=='D0':
        cohorts=[]
        for other_arm in ('B-control','B-align'):
            existing_path=ROOT/'evaluation'/other_arm/task/'episodes.json'
            if existing_path.exists():cohorts.append(json.loads(existing_path.read_text()))
        extended=extended or needs_extended_cohort(cohorts)
    checkpoint=ROOT/'private/checkpoints'/arm/'3001'
    cfg=EvalConfig(str(checkpoint),'external',task=task,seeds=tuple(seed_plan(extended)),videos=0,
                   normalization=str(ROOT/'configs/norm_stats.json'),seen_tasks=('D0','D1'),
                   reset_dir=str(ROOT/'evaluation/paired_resets'))
    session=EvaluationSession(cfg)
    logger=NoiseLoggedPolicy(session.model.policy);session.model.policy=logger
    session.set_task(task)
    output=ROOT/'evaluation'/arm/task;output.mkdir(parents=True,exist_ok=True)
    path=output/'episodes.json'
    rows=json.loads(path.read_text()) if path.exists() else []
    binding_path=output/'binding.json'
    binding=dict(session.binding,arm=arm,task=task,noise_stream='numpy.default_rng(reset_seed), fixed shape50x32, consumed on replanning')
    if binding_path.exists():
        require_evaluation_resume_binding(json.loads(binding_path.read_text()),binding)
    elif rows:raise ValueError('Unbound stale evaluation rows')
    else:binding_path.write_text(json.dumps(binding,indent=2))
    if len({r['seed'] for r in rows})!=len(rows) or any(r['seed'] not in seed_plan(extended) for r in rows):
        raise ValueError('Invalid partial evaluation cohort')
    existing={r['seed'] for r in rows}
    for seed in seed_plan(extended):
        if seed in existing:continue
        logger.hashes=[]
        row,transitions=session.run_one(seed)
        row.update(arm=arm,noise_hashes=logger.hashes.copy())
        rows.append(row);rows.sort(key=lambda r:r['seed']);path.write_text(json.dumps(rows,indent=2))
        if seed in (10000,10001):
            metadata=save_video(output/'videos','external',task,row,transitions)
            metadata['arm']=arm
            # Original filename contains external; parent folder identifies arm.
        print('PAIRED_EVAL',arm,task,seed,row['success'],row['length'],flush=True)
    session.env.close()
    return rows


def summarize():
    results={}
    from .data import sha256
    sys.path.insert(0,'/root/maniskill_myws/src')
    from maniskill_myws.pld.libero_protocol import directory_manifest
    for task in ('D0','D1','D2'):
        c=json.loads((ROOT/'evaluation/B-control'/task/'episodes.json').read_text())
        a=json.loads((ROOT/'evaluation/B-align'/task/'episodes.json').read_text())
        for arm in ('B-control','B-align'):
            binding=json.loads((ROOT/'evaluation'/arm/task/'binding.json').read_text())
            checkpoint=ROOT/'private/checkpoints'/arm/'3001'
            current=dict(checkpoint=str(checkpoint.resolve()),checkpoint_params=directory_manifest(checkpoint/'params'),
                normalization_path=str((ROOT/'configs/norm_stats.json').resolve()),normalization_sha256=sha256(ROOT/'configs/norm_stats.json'),
                arm=arm,task=task,seen_tasks=['D0','D1'],action_horizon=50,replan_steps=5)
            require_current_binding(binding,current)
        validate_cohort(c,task);validate_cohort(a,task)
        validate_pairs(c,a)
        if len(c) not in ((50,500) if task=='D0' else (50,)):
            raise ValueError('Preregistered seed count incomplete')
        stats=(d0_decision if task=='D0' else paired_effect)([int(r['success']) for r in c],[int(r['success']) for r in a])
        if task!='D0':stats['status']='ПРОЙДЕН' if stats['ci95'][0]>0 else 'НЕ ПРОЙДЕН' if stats['ci95'][1]<=0 else 'НЕОПРЕДЕЛЁН'
        results[task]=stats
    (ROOT/'evaluation/results.json').write_text(json.dumps(results,indent=2))
    final_path=ROOT/'logs/evaluation_complete.json'
    if not results['D0']['extend_to500']:
        result_sha256=sha256(ROOT/'evaluation/results.json')
        previous=json.loads(final_path.read_text()) if final_path.exists() else None
        if previous is None or previous['results_sha256']!=result_sha256:
            import time
            final_path.write_text(json.dumps(dict(finished_unix=time.time(),results_sha256=result_sha256),indent=2))
    import csv
    with (ROOT/'evaluation/summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['task','n','control_sr','align_sr','delta','lower95','upper95','status'])
        writer.writeheader()
        for t,r in results.items():writer.writerow(dict(task=t,n=r['n'],control_sr=r['control_sr'],align_sr=r['align_sr'],
            delta=r['delta'],lower95=r['ci95'][0],upper95=r['ci95'][1],status=r['status']))
    print(json.dumps(results,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['B-control','B-align']);p.add_argument('--task',choices=['D0','D1','D2'])
    p.add_argument('--extended',action='store_true');p.add_argument('--summarize',action='store_true');args=p.parse_args()
    if args.summarize:summarize()
    else:
        import shutil
        missing=[name for name in ('ffmpeg','ffprobe') if shutil.which(name) is None]
        if missing:raise RuntimeError('Video validation requires system executables: '+', '.join(missing)+'; install ffmpeg before evaluation')
        for arm in ('B-control','B-align'):
            if json.loads((ROOT/'logs'/f'{arm}_complete.json').read_text())['steps']!=3001:
                raise RuntimeError('Both main runs must finish before evaluator preflight')
        resume=ROOT/'logs/fresh_process_resume.json'
        if not resume.exists():
            import subprocess
            env=dict(os.environ);env.pop('XLA_FLAGS',None);env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
            with (ROOT/'logs/fresh_process_resume.log').open('w') as output:
                subprocess.check_call(['/root/maniskill_myws/third_party/openpi/.venv/bin/python','-u','-m','alignment.fresh_resume'],cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
        if json.loads(resume.read_text())['status']!='passed':raise RuntimeError('Fresh-process resume preflight failed')
        run(args.arm,args.task,args.extended)

if __name__=='__main__':main()
