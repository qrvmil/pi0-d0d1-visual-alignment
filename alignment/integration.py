"""Real π0 preflight: selected state, dummy-view invariance and gradient decomposition."""
import dataclasses
import json
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.models import gemma
from openpi.models.pi0 import make_attn_mask
from .model import middle_state
from .training import ROOT,microbatch,gradient_function,accumulated_grad,parameter_digest


def require_finite(tree):
    leaves=jax.tree.leaves(tree)
    if not leaves or not all(bool(jnp.all(jnp.isfinite(x))) for x in leaves):
        raise ValueError('Nonfinite integration features, gradients or metrics')


def raw_reference(model,obs,actions):
    prefix,pm,pa=model.embed_prefix(obs)
    # Deterministic arbitrary suffix; prefix cannot attend to it.
    suffix,sm,sa,cond=model.embed_suffix(obs,actions,jnp.ones((actions.shape[0],))*.5)
    mask=make_attn_mask(jnp.concatenate([pm,sm],axis=1),jnp.concatenate([pa,sa]))[:,None]
    positions=jnp.cumsum(jnp.concatenate([pm,sm],axis=1),axis=1)-1
    params=nnx.state(model.PaliGemma.llm).to_pure_dict()['layers']
    configs=tuple(model.PaliGemma.llm.module.configs)
    xs=jax.tree.map(lambda x:x.astype(model.PaliGemma.llm.module.embed_dtype),[prefix,suffix])
    # Test oracle manually applies exactly9 blocks, retaining only the current state.
    for i in range(9):
        perlayer=jax.tree.map(lambda x:x[i],params)
        xs,_=gemma.Block(configs=configs).apply({'params':perlayer},xs,None,positions,mask,[None,cond])
    return xs[0][:,:2*256].reshape(actions.shape[0],2,16,16,-1)


def validate(state,batches,projector,cfg,size):
    batch,_=batches.get(0);small=microbatch(batch,0,1);obs,actions,target=small
    def features(params,obs):return middle_state(nnx.merge(state.model_def,params),obs)[0]
    extract=jax.jit(features)
    got=extract(state.params,obs);jax.block_until_ready(got)
    oracle=jax.jit(lambda params,obs,actions:raw_reference(nnx.merge(state.model_def,params),obs,actions))
    ref=oracle(state.params,obs,actions);jax.block_until_ready(ref)
    require_finite((got,ref))
    rel=float(jnp.linalg.norm(got.astype(jnp.float32)-ref.astype(jnp.float32))/jnp.maximum(jnp.linalg.norm(ref.astype(jnp.float32)),1e-8))
    if rel>.005:raise ValueError(f'Real π0 middle-state correspondence failed: {rel}')
    altered=dataclasses.replace(obs,images=dict(obs.images,right_wrist_0_rgb=jax.random.uniform(jax.random.key(99),obs.images['right_wrist_0_rgb'].shape,minval=-1,maxval=1)))
    dummy=extract(state.params,altered);jax.block_until_ready(dummy)
    if not np.array_equal(np.asarray(got),np.asarray(dummy)):raise ValueError('Masked dummy camera affects valid visual tokens')
    flow_fn=gradient_function(cfg,alignment=False,component='flow')
    align_fn=gradient_function(cfg,alignment=True,component='align')
    total_fn=gradient_function(cfg,alignment=True)
    gf,fi=accumulated_grad(state,batch,projector,.2,flow_fn,size)
    ga,ai=accumulated_grad(state,batch,projector,.2,align_fn,size)
    gt,ti=accumulated_grad(state,batch,projector,.2,total_fn,size)
    require_finite((gf,ga,gt,fi,ai,ti))
    delta=jax.tree.map(lambda t,f,a:t-f-.2*a,gt,gf,ga)
    residual=float(optax.global_norm(delta)/jnp.maximum(optax.global_norm(gt),1e-8))
    require_finite((rel,residual))
    if residual>.02:raise ValueError(f'Gradient decomposition exceeded bf16 tolerance: {residual}')
    if abs(float(fi['flow'])-float(ti['flow']))>1e-5:raise ValueError('Initial paired flow losses differ')
    groups={};late_max=0.
    for path,value in ga.flat_state().items():
        name='/'.join(map(str,path));v=value.value
        group='vision' if '/img/' in name else 'paligemma_lora' if '/llm/' in name and '_1/' not in name else 'other'
        groups[group]=groups.get(group,0.)+float(jnp.sum(jnp.square(v.astype(jnp.float32))))
        if '/llm/layers/' in name and v.ndim>0 and v.shape[0]==18:
            late_max=max(late_max,float(jnp.max(jnp.abs(v[9:]))))
    groups={k:float(np.sqrt(v)) for k,v in groups.items()}
    if groups.get('vision',0)<=0 or groups.get('paligemma_lora',0)<=0 or late_max!=0:
        raise ValueError('Alignment gradients do not reach intended visual/middle groups')
    record=dict(status='passed',middle_relative_rmse=rel,tolerance=.005,masked_dummy_invariance_exact=True,
        gradient_decomposition_relative_residual=residual,gradient_tolerance=.02,
        initial_control_flow=float(fi['flow']),initial_align_flow=float(ti['flow']),
        align_group_gradient_norms=groups,alignment_grad_after_layer9_max=late_max,
        projector_sha256=parameter_digest(projector),tasks=['D0','D1'])
    (ROOT/'logs/model_integration.json').write_text(json.dumps(record,indent=2))
    print('REAL_MODEL_INTEGRATION',record,flush=True)
    return flow_fn,align_fn
