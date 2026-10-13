"""Matched SFT recipe, deterministic batch assembly and functional gradient steps."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from openpi.models import gemma
from openpi.models.model import Observation
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training import config as oc, weight_loaders, sharding
from openpi.shared.normalize import deserialize_json
from .data import sample_order
from .model import middle_state,project,cosine_loss

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path('/root/maniskill_myws/third_party/openpi')


def training_recipe(arm):
    if arm not in ('B-control','B-align'):raise ValueError('Unknown training arm')
    return dict(alignment=arm=='B-align',seed=0,effective_batch=8,steps=3001,save_steps=[0,500,1000,2000,3001],
                lora_rank=32,lora_alpha=32,teacher_frozen=True,projector_frozen=True,action_horizon=50,
                data_tasks=['D0','D1'],optimizer='upstream AdamW',schedule='upstream CosineDecaySchedule',
                selection='fixed final3001 before OOD evaluation',extra_augmentations=False)


def batch_slice(order,step,batch_size):
    result=order[step*batch_size:(step+1)*batch_size]
    if len(result)!=batch_size:raise ValueError('Sampler exhausted')
    return result


def parameter_digest(params):
    digest=hashlib.sha256()
    for path,value in jax.tree_util.tree_flatten_with_path(params)[0]:
        arr=np.asarray(value);digest.update(str(path).encode());digest.update(str(arr.shape).encode())
        digest.update(str(arr.dtype).encode());digest.update(arr.tobytes())
    return digest.hexdigest()


def config(arm):
    if not getattr(gemma.get_config,'alignment_rank32',False):
        original=gemma.get_config
        def get(variant):
            c=original(variant)
            if variant in ('gemma_2b_lora','gemma_300m_lora'):
                c=dataclasses.replace(c,lora_configs={k:dataclasses.replace(v,rank=32,alpha=32.) for k,v in c.lora_configs.items()})
            return c
        get.alignment_rank32=True;gemma.get_config=get
    cfg=oc.get_config('pi0_libero_low_mem_finetune')
    return dataclasses.replace(cfg,name='pi0_visual_alignment',exp_name=arm,batch_size=8,num_workers=0,
        seed=0,num_train_steps=3001,ema_decay=None,wandb_enabled=False,
        weight_loader=weight_loaders.CheckpointWeightLoader(str(ROOT/'private/base/params')),
        checkpoint_base_dir=str(ROOT/'private/checkpoints'),assets_base_dir=str(ROOT/'private/assets'),
        data=dataclasses.replace(cfg.data,repo_id='local/d0d1',extra_delta_transform=False),keep_period=1)


def initial_state(cfg):
    spec=importlib.util.spec_from_file_location('pinned_openpi_trainer',UPSTREAM/'scripts/train.py')
    trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
    _,init_rng=jax.random.split(jax.random.key(0))
    mesh=sharding.make_mesh(1)
    state,_=trainer.init_train_state(cfg,init_rng,mesh,resume=False)
    jax.block_until_ready(state)
    return state


class Batches:
    def __init__(self,corpus,cache,smoke=False):
        self.corpus=corpus;self.cache=cache;self.smoke=smoke
        norm_path=ROOT/'configs/norm_stats.json'
        if not norm_path.exists():corpus.make_normalization()
        self.norm=deserialize_json(norm_path.read_text())
        tok=PaligemmaTokenizer(48)
        self.tokens={t:tok.tokenize(corpus.files[t]['data'].attrs.get('language_instruction',
            ' '.join(__import__('alignment.data',fromlist=['TASKS']).TASKS[t][1].split('_')))) for t in corpus.tasks}
        self.order=sample_order([1,1] if smoke else corpus.lengths,(3001+1)*8,seed=0)

    def get(self,step):
        items=[];records=[];features=[];tokens=[];masks=[]
        for task,index,aug in batch_slice(self.order,step,8):
            row=self.corpus.samples[self.corpus.tasks[task]][index]
            item=self.corpus.augmented(row,aug);target=self.cache.lookup(row['sample_id'],aug)
            if item['params']!=target['params'] or item['image_sha256']!=target['image_sha256']:
                raise ValueError('Teacher/student pixels or augmentation parameters differ')
            items.append(item);records.append(dict(sample_id=row['sample_id'],augmentation=aug,image_sha256=item['image_sha256']))
            features.append(target['features']);t,m=self.tokens[row['task']];tokens.append(t);masks.append(m)
        state=np.stack([x['state'] for x in items]);actions=np.stack([x['actions'] for x in items])
        state=(state-np.asarray(self.norm['state'].mean))/(np.asarray(self.norm['state'].std)+1e-6)
        actions=(actions-np.asarray(self.norm['actions'].mean))/(np.asarray(self.norm['actions'].std)+1e-6)
        views=np.stack([x['images'] for x in items]).astype(np.float32)/255*2-1
        obs=Observation(images={'base_0_rgb':jnp.asarray(views[:,0]),'left_wrist_0_rgb':jnp.asarray(views[:,1]),
            'right_wrist_0_rgb':-jnp.ones_like(jnp.asarray(views[:,0]))},
            image_masks={'base_0_rgb':jnp.ones((8,),bool),'left_wrist_0_rgb':jnp.ones((8,),bool),
                'right_wrist_0_rgb':jnp.zeros((8,),bool)},
            state=jnp.asarray(np.pad(state,((0,0),(0,24))),jnp.float32),
            tokenized_prompt=jnp.asarray(np.stack(tokens),jnp.int32),tokenized_prompt_mask=jnp.asarray(np.stack(masks),bool))
        return (obs,jnp.asarray(np.pad(actions,((0,0),(0,0),(0,25))),jnp.float32),
                jnp.asarray(np.stack(features),jnp.float32)),records


def microbatch(batch,start,size):
    return jax.tree.map(lambda x:x[start:start+size],batch)


def gradient_function(cfg,alignment=True,component=None):
    def gradients(model_def,params,batch,projector,rng,lam):
        m=nnx.merge(model_def,params)
        def objective(m):
            obs,actions,target=batch
            flow=m.compute_loss(rng,obs,actions,train=False).mean() if component!='align' else jnp.array(0.)
            if alignment or component=='align':
                hidden,mask=middle_state(m,obs)
                align=cosine_loss(project(projector,hidden,freeze=True),target,mask)
            else:align=jnp.array(0.)
            total=align if component=='align' else flow if component=='flow' else flow+lam*align
            return total,(flow,align)
        (loss,aux),grads=nnx.value_and_grad(objective,argnums=nnx.DiffState(0,cfg.trainable_filter),has_aux=True)(m)
        return grads,dict(loss=loss,flow=aux[0],align=aux[1],grad_norm=optax.global_norm(grads))
    return jax.jit(gradients)


def updater(cfg):
    def update(state,grads):
        params=state.params.filter(cfg.trainable_filter)
        updates,opt_state=state.tx.update(grads,state.opt_state,params)
        model=nnx.merge(state.model_def,state.params)
        nnx.update(model,optax.apply_updates(params,updates))
        return dataclasses.replace(state,step=state.step+1,params=nnx.state(model),opt_state=opt_state)
    return jax.jit(update,donate_argnums=(0,))


def accumulated_grad(state,batch,projector,lam,grad_fn,micro_size):
    grads=None;infos=[]
    for start in range(0,8,micro_size):
        # Stream stable across paired arms and microbatch configuration fixed for both.
        rng=jax.random.fold_in(jax.random.fold_in(jax.random.key(0),state.step),start)
        g,info=grad_fn(state.model_def,state.params,microbatch(batch,start,micro_size),projector,rng,jnp.asarray(lam))
        jax.block_until_ready((g,info))
        grads=g if grads is None else jax.tree.map(lambda a,b:a+b,grads,g)
        infos.append(info)
    count=8//micro_size
    grads=jax.tree.map(lambda x:x/count,grads)
    info=jax.tree.map(lambda *v:sum(v)/count,*infos)
    return grads,info


def loader_assets(norm):
    return SimpleNamespace(data_config=lambda:SimpleNamespace(norm_stats=norm,asset_id='local/d0d1'))


def require_resume_binding(saved,current):
    if saved!=current:raise ValueError('Checkpoint resume binding differs from immutable recipe')


def reconcile_rows(rows,durable_step):
    seen=set();last=0
    for row in rows:
        if row['step'] in seen or row['step']<=last:raise ValueError('Invalid training log order or duplicate step')
        seen.add(row['step']);last=row['step']
    return [r for r in rows if r['step']<=durable_step],max([float(r['seconds']) for r in rows],default=0.)
