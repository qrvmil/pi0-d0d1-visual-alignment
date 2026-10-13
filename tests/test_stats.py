import pytest
from alignment.stats import paired_effect, d0_decision, choose_lambda


def test_no_discordance_has_uncertainty():
    r=paired_effect([1]*50,[1]*50)
    assert r['delta']==0
    assert r['ci95'][0]<0<r['ci95'][1]
    assert d0_decision([1]*50,[1]*50)['status']=='НЕОПРЕДЕЛЁН'
    assert d0_decision([1]*500,[1]*500)['status']=='ПРОЙДЕН'


def test_pairs_not_independent_proportions():
    r=paired_effect([0]*25+[1]*25,[1]*25+[1]*25)
    assert r['delta']==0.5 and r['gains']==25 and r['losses']==0
    assert r['ci95'][0]>0


def test_loss_clear_failure():
    r=d0_decision([1]*500,[0]*500)
    assert r['status']=='НЕ ПРОЙДЕН'


def test_mismatched_pairs_rejected():
    with pytest.raises(ValueError): paired_effect([0],[0,1])
    with pytest.raises(ValueError): paired_effect([],[])
    with pytest.raises(ValueError): paired_effect([0.5],[1])


def test_preregistered_lambda():
    assert choose_lambda([.1]*20)==.2
    assert choose_lambda([2]*20)==.1
    assert choose_lambda([.001]*20)==2
    with pytest.raises(ValueError): choose_lambda([0]*20)
