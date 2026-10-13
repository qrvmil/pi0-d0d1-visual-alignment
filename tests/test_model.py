import dataclasses
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from openpi.models import gemma
from alignment.model import PrefixMiddle, sliced_params, cosine_loss, projector_init, project


def test_middle_prefix_matches_full_suffix_manual_blocks():
    configs=[dataclasses.replace(gemma.get_config('dummy'),depth=4),dataclasses.replace(gemma.get_config('dummy'),depth=4)]
    full=gemma.Module(configs=configs,embed_dtype='float32')
    prefix=jax.random.normal(jax.random.key(4),(1,8,configs[0].width))
    suffix=jax.random.normal(jax.random.key(5),(1,3,configs[1].width))
    pos=jnp.arange(11)[None,:]
    mask=jnp.ones((1,11,11),bool).at[:,:8,8:].set(False)
    variables=nn.Module.init(full,jax.random.key(0),[prefix,suffix],pos,mask)
    params=variables['params']
    xs=[prefix,suffix]
    for i in range(2):
        perlayer=jax.tree.map(lambda x:x[i],params['layers'])
        xs,_=gemma.Block(configs=tuple(configs)).apply({'params':perlayer},xs,None,pos,mask[:,None], [None,None])
    middle=PrefixMiddle(configs=[dataclasses.replace(c,depth=2) for c in configs],embed_dtype='float32')
    got=middle.apply({'params':sliced_params(params,2)},[prefix,None],pos[:,:8],mask[:,:8,:8])
    np.testing.assert_allclose(got,xs[0],rtol=1e-5,atol=1e-5)


def test_masks_patch_view_and_frozen_target_gradient():
    student=jnp.ones((2,2,16,16,3))
    target=student.at[:,1].set(-student[:,1])
    mask=jnp.array([[True,False],[True,False]])
    assert float(cosine_loss(student,target,mask))<1e-6
    assert float(cosine_loss(student,target,jnp.ones((2,2),bool)))>.99
    assert np.all(np.asarray(jax.grad(lambda z:cosine_loss(student,z,mask))(target))==0)
    with pytest.raises(ValueError):cosine_loss(student,target[:,:,:8],mask)


def test_projector_gradient_is_frozen_but_student_gradient_nonzero():
    params=projector_init(jax.random.key(3),4,3)
    x=jax.random.normal(jax.random.key(5),(1,2,16,16,4))
    target=jax.random.normal(jax.random.key(6),(1,2,16,16,3))
    def loss(p,x):return cosine_loss(project(p,x,freeze=True),target,jnp.ones((1,2),bool))
    gp,gx=jax.grad(loss,argnums=(0,1))(params,x)
    assert all(np.all(np.asarray(v)==0) for v in jax.tree.leaves(gp))
    assert float(jnp.linalg.norm(gx))>0


def test_integration_finite_guard_rejects_nan_and_inf():
    from alignment.integration import require_finite
    require_finite({'a':jnp.array([1.,2.]),'b':jnp.array(0.)})
    with pytest.raises(ValueError,match='Nonfinite'):require_finite({'a':jnp.array([jnp.nan])})
    with pytest.raises(ValueError,match='Nonfinite'):require_finite({'a':jnp.array([jnp.inf])})
