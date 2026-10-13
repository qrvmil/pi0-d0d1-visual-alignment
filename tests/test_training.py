import numpy as np
import pytest
from alignment.training import batch_slice, parameter_digest, training_recipe


def test_same_recipe_for_both_arms():
    c=training_recipe('B-control');a=training_recipe('B-align')
    assert c['seed']==a['seed']==0
    assert c['effective_batch']==a['effective_batch']==8
    assert c['steps']==a['steps']==3001
    assert c['save_steps']==a['save_steps']==[0,500,1000,2000,3001]
    assert {k:v for k,v in c.items() if k!='alignment'}=={k:v for k,v in a.items() if k!='alignment'}
    with pytest.raises(ValueError):training_recipe('B-original')


def test_slice_resumes_by_consumed_updates():
    order=list(range(200))
    assert batch_slice(order,4,8)==list(range(32,40))
    assert batch_slice(order,5,8)==list(range(40,48))


def test_parameter_digest_detects_weight_change():
    p={'a':np.ones((2,3))}
    assert parameter_digest(p)==parameter_digest({'a':p['a'].copy()})
    assert parameter_digest(p)!=parameter_digest({'a':np.zeros((2,3))})


def test_resume_binding_rejects_recipe_or_projector_drift():
    from alignment.training import require_resume_binding
    require_resume_binding({'projector':'a','lambda':.2},{'projector':'a','lambda':.2})
    with pytest.raises(ValueError,match='resume'):
        require_resume_binding({'projector':'a','lambda':.2},{'projector':'b','lambda':.2})


def test_resume_log_prunes_replayed_tail_but_accounts_time():
    from alignment.training import reconcile_rows
    rows=[{'step':i,'seconds':float(i)} for i in range(1,11)]
    durable,previous_time=reconcile_rows(rows,5)
    assert [r['step'] for r in durable]==[1,2,3,4,5]
    assert previous_time==10.
    with pytest.raises(ValueError):reconcile_rows([rows[0],rows[0]],5)
