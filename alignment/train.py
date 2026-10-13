"""CLI: real-data smoke, frozen projector calibration, pilot, matched training."""
import argparse
import json
from pathlib import Path
import time
import gc
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.training import checkpoints
from .cache import CacheReader
from .corpus import Corpus
from .data import sha256
from .model import middle_state,projector_init,project,cosine_loss
from .stats import choose_lambda
from .training import (ROOT,config,initial_state,parameter_digest,training_recipe,Batches,microbatch,
                       gradient_function,updater,accumulated_grad,loader_assets,require_resume_binding,reconcile_rows)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))


def frozen_features(state,obs):
    m=nnx.merge(state.model_def,state.params)
    return middle_state(m,obs)


def calibrate(state,batches,steps,output):
    extract=jax.jit(frozen_features)
    initial=parameter_digest(state.params)
    batch,_=batches.get(0)
    hidden,mask=extract(state,microbatch(batch,0,1)[0]);jax.block_until_ready(hidden)
    params=projector_init(jax.random.key(101),hidden.shape[-1],batch[2].shape[-1])
    tx=optax.adam(1e-3);opt=tx.init(params)
    def step(p,o,h,t,mask):
        loss,g=jax.value_and_grad(lambda p:cosine_loss(project(p,h,freeze=False),t,mask))(p)
        updates,o=tx.update(g,o,p);return optax.apply_updates(p,updates),o,loss
    step=jax.jit(step)
    rows=[]
    for i in range(steps):
        batch,records=batches.get(i)
        # Calibrate one balanced effective batch, frozen policy; cache middle features per update.
        hs=[];ms=[]
        for start in range(8):
            h,m=extract(state,microbatch(batch,start,1)[0]);hs.append(h);ms.append(m)
        h=jnp.concatenate(hs);mask=jnp.concatenate(ms)
        params,opt,loss=step(params,opt,jax.lax.stop_gradient(h),batch[2],mask)
        jax.block_until_ready((params,opt,loss))
        if not np.isfinite(float(loss)):raise ValueError('Nonfinite projector loss')
        rows.append(dict(step=i+1,loss=float(loss)))
        if i%10==0:print('PROJECTOR',i+1,float(loss),flush=True)
    assert parameter_digest(state.params)==initial
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez(output,**{k:np.asarray(v) for k,v in params.items()})
    write(ROOT/'logs'/('projector_smoke.json' if steps<200 else 'projector_calibration.json'),
          dict(updates=steps,lr=1e-3,optimizer='Adam',policy_parameters_unchanged=True,
               initial_parameter_sha256=initial,projector_sha256=sha256(output),losses=rows,tasks=['D0','D1']))
    return params


def load_projector(path):
    with np.load(path) as f:return {k:jnp.asarray(f[k]) for k in f.files}


def finite(info):
    if not all(np.isfinite(float(v)) for v in info.values()):raise ValueError('Nonfinite loss/gradient')


def smoke(state,batches,projector,cfg):
    grad=gradient_function(cfg,alignment=True)
    batch,_=batches.get(0);events=[]
    chosen=None
    for size in (8,4,2,1):
        try:
            g,info=accumulated_grad(state,batch,projector,.2,grad,size);finite(info);chosen=size
            print('MICROBATCH_PASSED',size,{k:float(v) for k,v in info.items()},flush=True);break
        except Exception as e:
            if 'RESOURCE_EXHAUSTED' not in str(e) and 'out of memory' not in str(e).lower():raise
            events.append(dict(microbatch=size,error=str(e)));jax.clear_caches();gc.collect()
    if chosen is None:raise RuntimeError('Microbatch1 OOM')
    update=updater(cfg)
    projector_hash=parameter_digest(projector)
    manager,_=checkpoints.initialize_checkpoint_dir(ROOT/'private/smoke_checkpoints',keep_period=1,overwrite=False,resume=False)
    loader=loader_assets(batches.norm);rows=[]
    for i in range(5):
        batch,records=batches.get(i)
        g,info=accumulated_grad(state,batch,projector,.2,grad,chosen);finite(info)
        state=update(state,g);jax.block_until_ready(state)
        rows.append(dict(step=int(state.step),**{k:float(v) for k,v in info.items()}))
        print('DATA_SMOKE_UPDATE',rows[-1],flush=True)
        if i==1:
            signature=parameter_digest((state.params,state.opt_state))
            checkpoints.save_state(manager,state,loader,2);manager.wait_until_finished()
            state=checkpoints.restore_state(manager,state,loader,step=2)
            jax.block_until_ready(state)
            assert int(state.step)==2 and parameter_digest((state.params,state.opt_state))==signature
            assert batches.order[16:24]==batches.order[int(state.step)*8:(int(state.step)+1)*8]
            print('RESUME_VERIFIED',flush=True)
    assert parameter_digest(projector)==projector_hash
    manager.close()
    write(ROOT/'configs/memory_recipe.json',dict(microbatch=chosen,accumulation=8//chosen,effective_batch=8,oom_events=events))
    write(ROOT/'logs/data_training_smoke.json',dict(status='passed',updates=rows,projector_frozen=True,
        resume_params_and_optimizer_exact=True,camera_masks={'base':True,'left_wrist':True,'right_wrist':False},
        memory_stats=jax.devices()[0].memory_stats(),tasks=['D0','D1'],microbatch=chosen))


def pilot(state,batches,projector,cfg,size):
    from .integration import validate
    flow_grad,align_grad=validate(state,batches,projector,cfg,size)
    update=updater(cfg);rows=[];ratios=[]
    for i in range(20):
        batch,records=batches.get(i)
        gf,fi=accumulated_grad(state,batch,projector,.2,flow_grad,size)
        ga,ai=accumulated_grad(state,batch,projector,.2,align_grad,size)
        fn=float(optax.global_norm(gf));an=float(optax.global_norm(ga));r=.2*an/max(fn,1e-30)
        finite(fi);finite(ai)
        if not np.isfinite(r) or r<=0:raise ValueError('Alignment gradient absent or nonfinite')
        total=jax.tree.map(lambda f,a:f+.2*a,gf,ga)
        state=update(state,total);jax.block_until_ready(state)
        row=dict(step=i+1,L_flow=float(fi['flow']),L_align=float(ai['align']),flow_grad_norm=fn,align_grad_norm=an,weighted_gradient_ratio=r)
        rows.append(row);ratios.append(r);print('PILOT',row,flush=True)
    lam=choose_lambda(ratios)
    write(ROOT/'configs/lambda_decision.json',dict(initial=.2,selected=lam,
        rule='median last10 r>1: .2/r; r<.01: min(2,.02/r); otherwise .2; single adjustment',
        median_last10=float(np.median(ratios[-10:])),tasks=['D0','D1'],pilot_updates=20))
    write(ROOT/'logs/pilot.json',rows)


def train_arm(state,batches,projector,cfg,arm,size,lam):
    directory=ROOT/'private/checkpoints'/arm
    manager,_=checkpoints.initialize_checkpoint_dir(directory,keep_period=1,overwrite=False,resume=directory.exists())
    loader=loader_assets(batches.norm)
    recipe=training_recipe(arm);binding=dict(recipe,microbatch=size,lambda_align=lam if recipe['alignment'] else 0,
        openpi_config=repr(cfg),base_manifest_sha256=sha256(ROOT/'configs/base_manifest.json'),
        cache_manifest_sha256=sha256(ROOT/'private/cache/manifest.json'),normalization_sha256=sha256(ROOT/'configs/norm_stats.json'),
        projector_sha256=sha256(ROOT/'private/projector.npz'))
    binding_path=ROOT/'configs'/f'{arm}.json'
    if binding_path.exists():require_resume_binding(json.loads(binding_path.read_text()),binding)
    else:write(binding_path,binding)
    if manager.all_steps():
        saved_step=max(manager.all_steps())
        state=checkpoints.restore_state(manager,state,loader,step=saved_step);jax.block_until_ready(state)
        if int(state.step)!=saved_step:raise ValueError('Resume optimizer counter differs from checkpoint step')
        saved_metadata=directory/str(int(state.step))/'experiment_metadata.json'
        require_resume_binding(json.loads(saved_metadata.read_text())['binding'],binding)
        print('MAIN_RESUMED',arm,int(state.step),flush=True)
    else:
        digest=parameter_digest(state.params)
        signature=ROOT/'configs/initialization_sha256.json'
        if signature.exists():
            if json.loads(signature.read_text())['sha256']!=digest:raise ValueError('Control/align initial parameters differ')
        else:write(signature,dict(sha256=digest,seed=0,base_manifest_sha256=sha256(ROOT/'configs/base_manifest.json')))
        write(ROOT/'configs'/f'{arm}_initialization.json',dict(sha256=digest,independent_load=True))
        checkpoints.save_state(manager,state,loader,0);manager.wait_until_finished()
        write(directory/'0/experiment_metadata.json',dict(arm=arm,step=0,binding=binding))
    grad=gradient_function(cfg,alignment=recipe['alignment']);update=updater(cfg)
    projector_hash=parameter_digest(projector);started=time.time()
    log=ROOT/'logs'/f'{arm}.jsonl'
    existing_rows=[json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    durable,previous_time=reconcile_rows(existing_rows,int(state.step))
    timing_path=ROOT/'logs'/f'{arm}_timing.json'
    if timing_path.exists():previous_time=max(previous_time,json.loads(timing_path.read_text())['seconds'])
    discarded=[r for r in existing_rows if r['step']>int(state.step)]
    if discarded:
        with (ROOT/'logs'/f'{arm}_discarded_attempts.jsonl').open('a') as f:
            for row in discarded:f.write(json.dumps(row)+'\n')
    log.write_text(''.join(json.dumps(r)+'\n' for r in durable))
    for i in range(int(state.step),recipe['steps']):
        batch,records=batches.get(i)
        g,info=accumulated_grad(state,batch,projector,lam,grad,size);finite(info)
        state=update(state,g);jax.block_until_ready(state)
        row=dict(step=int(state.step),**{k:float(v) for k,v in info.items()},seconds=previous_time+time.time()-started,
                 batch_sha256=hashlib_records(records),samples=records,align_metric_computed=recipe['alignment'])
        with log.open('a') as f:f.write(json.dumps(row)+'\n')
        write(timing_path,dict(seconds=row['seconds']))
        if int(state.step)%10==0:print('TRAIN',arm,int(state.step),row['flow'],row['align'],round(row['seconds'],1),flush=True)
        if int(state.step) in recipe['save_steps']:
            checkpoints.save_state(manager,state,loader,int(state.step));manager.wait_until_finished()
            write(directory/str(int(state.step))/'experiment_metadata.json',dict(arm=arm,step=int(state.step),recipe=recipe,binding=binding,
                normalization_sha256=sha256(ROOT/'configs/norm_stats.json'),projector_sha256=sha256(ROOT/'private/projector.npz'),
                lambda_align=lam if recipe['alignment'] else 0,selection='fixed final3001; D2 unseen until final evaluation'))
    assert parameter_digest(projector)==projector_hash
    write(ROOT/'logs'/f'{arm}_complete.json',dict(status='complete',steps=int(state.step),seconds=previous_time+time.time()-started,
        checkpoints=list(manager.all_steps()),memory_stats=jax.devices()[0].memory_stats(),projector_frozen=True))
    manager.close()


def hashlib_records(records):
    import hashlib
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','calibrate','pilot','B-control','B-align']);args=p.parse_args()
    from jax._src.lib import cuda_versions
    from .smoke import validate_runtime
    validate_runtime(jax.devices()[0].platform,cuda_versions.cuda_runtime_get_version())
    if json.loads((ROOT/'logs/compatibility_smoke.json').read_text())['status']!='passed':raise RuntimeError('Smoke gate closed')
    if args.mode!='smoke' and json.loads((ROOT/'logs/data_training_smoke.json').read_text())['status']!='passed':raise RuntimeError('Data smoke gate closed')
    corpus=Corpus(ROOT)
    cache=CacheReader(ROOT/'private'/('cache_smoke' if args.mode=='smoke' else 'cache'))
    batches=Batches(corpus,cache,smoke=args.mode=='smoke')
    cfg=config(args.mode);print('INITIALIZE_PRETRAINED',args.mode,flush=True)
    state=initial_state(cfg);print('PRETRAINED_READY',flush=True)
    if args.mode=='smoke':
        projector=calibrate(state,batches,5,ROOT/'private/projector_smoke.npz')
        smoke(state,batches,projector,cfg)
    elif args.mode=='calibrate':calibrate(state,batches,200,ROOT/'private/projector.npz')
    else:
        calibration=json.loads((ROOT/'logs/projector_calibration.json').read_text())
        if calibration['updates']!=200 or not calibration['policy_parameters_unchanged']:
            raise ValueError('Projector calibration incomplete')
        if sha256(ROOT/'private/projector.npz')!=calibration['projector_sha256']:
            raise ValueError('Frozen projector differs from completed calibration')
        if parameter_digest(state.params)!=calibration['initial_parameter_sha256']:
            raise ValueError('Policy initialization differs from frozen-policy calibration')
        projector=load_projector(ROOT/'private/projector.npz');size=json.loads((ROOT/'configs/memory_recipe.json').read_text())['microbatch']
        if args.mode=='pilot':pilot(state,batches,projector,cfg,size)
        else:
            lam=json.loads((ROOT/'configs/lambda_decision.json').read_text())['selected']
            train_arm(state,batches,projector,cfg,args.mode,size,lam)

if __name__=='__main__':
    from .memory import VramMonitor
    import sys
    monitor=VramMonitor()
    try:main()
    finally:monitor.finish(ROOT/'logs'/f'{sys.argv[1]}_vram.json')
