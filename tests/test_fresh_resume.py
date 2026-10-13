import pytest
from alignment.fresh_resume import compare_metrics


def test_fresh_resume_metrics_match_saved_continuation():
    compare_metrics({'loss':.2,'flow':.1,'align':.5,'grad_norm':2.},
                    {'loss':.2,'flow':.1,'align':.5,'grad_norm':2.})
    with pytest.raises(ValueError):compare_metrics({'loss':.3,'flow':.1,'align':.5,'grad_norm':2.},
                                                  {'loss':.2,'flow':.1,'align':.5,'grad_norm':2.})
    with pytest.raises(ValueError):compare_metrics({'loss':float('nan'),'flow':.1,'align':.5,'grad_norm':2.},
                                                  {'loss':.2,'flow':.1,'align':.5,'grad_norm':2.})
