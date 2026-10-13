"""Compatibility gate; real π0 architecture, synthetic inputs, no dataset access."""
import json
import os
from pathlib import Path
import subprocess
import time


def validate_runtime(platform, version):
    if platform != 'gpu':
        raise RuntimeError('GPU backend required')
    if version < 12080:
        raise RuntimeError(f'CUDA runtime >=12.8 required; got {version}')


def main():
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    import jax
    import jax.numpy as jnp
    import numpy as np
    import flax.nnx as nnx
    import optax
    from jax._src.lib import cuda_versions
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared import nnx_utils
    started=time.time()
    inventory=subprocess.check_output(['nvidia-smi'],text=True)
    version=cuda_versions.cuda_runtime_get_version()
    print(inventory,flush=True)
    print('JAX',jax.__version__, 'DEVICES',jax.devices(),'CUDA_RUNTIME',version,flush=True)
    validate_runtime(jax.devices()[0].platform,version)
    x=jnp.ones((1024,1024),dtype=jnp.float32)
    assert float((x@x).block_until_ready()[0,0]) == 1024
    cfg=Pi0Config(paligemma_variant='gemma_2b_lora',action_expert_variant='gemma_300m_lora')
    print('INITIALIZING_ACTUAL_PI0',flush=True)
    model=cfg.create(jax.random.key(0))
    obs=cfg.fake_obs()
    obs.image_masks['right_wrist_0_rgb']=jnp.zeros((1,),dtype=bool)
    actions=cfg.fake_act()
    graph,state=nnx.split(model)
    # Same frozen bf16 policy as pinned trainer.
    state=nnx_utils.state_map(state,cfg.get_freeze_filter(),lambda p:p.replace(p.value.astype(jnp.bfloat16)))
    trainable=nnx.All(nnx.Param,nnx.Not(cfg.get_freeze_filter()))
    tx=optax.adamw(2.5e-5)
    optimizer=tx.init(state.filter(trainable))
    def step(state,optimizer):
        m=nnx.merge(graph,state)
        def loss(m):
            # false disables all extra image augmentations.
            return m.compute_loss(jax.random.key(1),obs,actions,train=False).mean()
        value,grads=nnx.value_and_grad(loss,argnums=nnx.DiffState(0,trainable))(m)
        updates,optimizer=tx.update(grads,optimizer,state.filter(trainable))
        params=optax.apply_updates(state.filter(trainable),updates)
        nnx.update(m,params)
        return nnx.state(m),optimizer,value,optax.global_norm(grads)
    print('COMPILING_FORWARD_BACKWARD',flush=True)
    state,optimizer,loss,grad=jax.jit(step)(state,optimizer)
    jax.block_until_ready((state,optimizer,loss,grad))
    assert np.isfinite(float(loss)) and np.isfinite(float(grad)) and float(grad)>0
    stats=jax.devices()[0].memory_stats()
    result=dict(status='passed',jax=jax.__version__,cuda_runtime=version,devices=[str(d) for d in jax.devices()],
                architecture='pi0 gemma_2b_lora/gemma_300m_lora + SigLIP So400m/14',batch_size=1,
                loss=float(loss),gradient_norm=float(grad),memory_stats=stats,seconds=time.time()-started,
                hardware=inventory,blackwell_verified=False)
    Path('logs/compatibility_smoke.json').write_text(json.dumps(result,indent=2))
    print('SMOKE_PASSED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
