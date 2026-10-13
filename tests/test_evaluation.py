import pytest
from alignment.evaluation import validate_pairs, seed_plan


def test_reset_and_noise_prefix_must_match():
    c=[dict(seed=10000,reset_hash='a',noise_hashes=['x','y'],success=True)]
    a=[dict(seed=10000,reset_hash='a',noise_hashes=['x'],success=False)]
    validate_pairs(c,a)
    with pytest.raises(ValueError,match='reset'):validate_pairs(c,[dict(a[0],reset_hash='b')])
    with pytest.raises(ValueError,match='noise'):validate_pairs(c,[dict(a[0],noise_hashes=['z'])])
    with pytest.raises(ValueError,match='seed'):validate_pairs(c,[dict(a[0],seed=10001)])


def test_prespecified_extension_no_cherry_picking():
    assert seed_plan(False)==list(range(10000,10050))
    assert seed_plan(True)==list(range(10000,10500))


def test_cohort_rejects_duplicates_and_unregistered_seeds():
    from alignment.evaluation import validate_cohort
    valid=[dict(seed=s) for s in seed_plan(False)]
    validate_cohort(valid,'D0')
    with pytest.raises(ValueError):validate_cohort(valid[:-1]+[valid[0]],'D0')
    with pytest.raises(ValueError):validate_cohort([dict(seed=s) for s in range(50)],'D0')
    with pytest.raises(ValueError):validate_cohort([dict(seed=s) for s in seed_plan(True)],'D2')
