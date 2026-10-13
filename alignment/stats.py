"""Conservative exact paired-binomial bounds, valid even for zero discordance."""
import numpy as np
from scipy.stats import beta


def cp(k,n,tail):
    lower=0. if k==0 else float(beta.ppf(tail,k,n-k+1))
    upper=1. if k==n else float(beta.ppf(1-tail,k+1,n-k))
    return lower,upper


def paired_effect(control,align):
    c=np.asarray(control);a=np.asarray(align)
    if c.ndim!=1 or a.shape!=c.shape or c.size==0 or not np.isin(c,[0,1]).all() or not np.isin(a,[0,1]).all():
        raise ValueError('Matched nonempty binary paired outcomes required')
    gains=int(((a==1)&(c==0)).sum());losses=int(((a==0)&(c==1)).sum());n=len(a)
    gl,gu=cp(gains,n,.0125);ll,lu=cp(losses,n,.0125)
    return dict(n=n,control_sr=float(c.mean()),align_sr=float(a.mean()),delta=(gains-losses)/n,
                gains=gains,losses=losses,ci95=[gl-lu,gu-ll],
                method='exact conservative marginal CP Bonferroni paired difference')


def d0_decision(control,align):
    result=paired_effect(control,align)
    # Each of two prespecified looks: total one-sided error .025; two marginals .0125.
    gl,gu=cp(result['gains'],result['n'],.0125)
    ll,lu=cp(result['losses'],result['n'],.0125)
    lower=gl-lu;upper=gu-ll
    result.update(one_sided95_lower_simultaneous=lower,noninferiority_margin=-.02,
                  status='ПРОЙДЕН' if lower>-.02 else 'НЕ ПРОЙДЕН' if upper<=-.02 else 'НЕОПРЕДЕЛЁН',
                  extend_to500=result['n']==50 and lower<=-.02)
    return result


def choose_lambda(ratios):
    values=np.asarray(ratios,float)
    if len(values)<10 or not np.isfinite(values).all() or np.any(values<=0):
        raise ValueError('Pilot needs finite positive gradient ratios')
    r=float(np.median(values[-10:]))
    return .2/r if r>1 else min(2.,.02/r) if r<.01 else .2
