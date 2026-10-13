"""Middle-only PaliGemma extraction and a separately frozen LN+MLP projector."""
import dataclasses
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import gemma
from openpi.models.pi0 import make_attn_mask


class PrefixMiddle(gemma.Module):
    """Same scanned parameter layout, only first depth blocks; no layer stack output."""
    def __call__(self,embedded,positions,mask):
        embedded=jax.tree.map(lambda e:e.astype(self.embed_dtype),embedded)
        xs,_=self.layers(embedded,None,positions,mask[:,None,:,:],[None]*len(self.configs),True)
        return xs[0]


def sliced_params(params,depth):
    result=dict(params)
    result['layers']=jax.tree.map(lambda x:x[:depth],params['layers'])
    return result


def middle_state(model,observation,depth=9):
    prefix,mask,ar=model.embed_prefix(observation)
    llm=model.PaliGemma.llm
    configs=[dataclasses.replace(c,depth=depth) for c in llm.module.configs]
    module=PrefixMiddle(configs=configs,embed_dtype=llm.module.embed_dtype)
    params=sliced_params(nnx.state(llm).to_pure_dict(),depth)
    middle=module.apply({'params':params},[prefix,None],jnp.cumsum(mask,axis=1)-1,make_attn_mask(mask,ar))
    # Image tokens are concatenated camera-major, row-major patches within each view.
    names=list(observation.images)
    if names!=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb']:
        raise ValueError('Unexpected view order; patch/view correspondence cannot be inferred')
    middle=middle[:,:3*256].reshape(middle.shape[0],3,16,16,-1)
    return middle[:,:2],jnp.stack([observation.image_masks[n] for n in names[:2]],axis=1)


def projector_init(rng,input_dim,output_dim,hidden_dim=1024):
    r1,r2=jax.random.split(rng)
    return dict(scale=jnp.ones((input_dim,)),bias=jnp.zeros((input_dim,)),
                w1=jax.random.normal(r1,(input_dim,hidden_dim))*jnp.sqrt(1/input_dim),b1=jnp.zeros((hidden_dim,)),
                w2=jax.random.normal(r2,(hidden_dim,output_dim))*jnp.sqrt(1/hidden_dim),b2=jnp.zeros((output_dim,)))


def project(params,x,freeze=True):
    if freeze:params=jax.tree.map(jax.lax.stop_gradient,params)
    x=x.astype(jnp.float32)
    mean=jnp.mean(x,axis=-1,keepdims=True)
    variance=jnp.mean(jnp.square(x-mean),axis=-1,keepdims=True)
    x=(x-mean)*jax.lax.rsqrt(variance+1e-5)*params['scale']+params['bias']
    return jax.nn.gelu(x@params['w1']+params['b1'])@params['w2']+params['b2']


def cosine_loss(student,teacher,mask):
    if student.shape!=teacher.shape or student.ndim!=5 or student.shape[1:4]!=(2,16,16):
        raise ValueError('Student/teacher spatial/view grids must match exactly')
    if mask.shape!=student.shape[:2]:raise ValueError('View mask shape mismatch')
    t=jax.lax.stop_gradient(teacher.astype(jnp.float32));s=student.astype(jnp.float32)
    t=t/jnp.maximum(jnp.linalg.norm(t,axis=-1,keepdims=True),1e-8)
    s=s/jnp.maximum(jnp.linalg.norm(s,axis=-1,keepdims=True),1e-8)
    losses=1-jnp.sum(t*s,axis=-1)
    weights=mask[:,:,None,None].astype(jnp.float32)
    denominator=mask.sum()*256
    # An all-dummy sample contributes nothing; an all-dummy batch fails finite gate.
    return jnp.where(denominator>0,jnp.sum(losses*weights)/denominator,jnp.nan)
