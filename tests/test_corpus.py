import h5py
import numpy as np
import pytest
from alignment.corpus import Corpus, chunk_indices, normalization


def test_native_postaction_shift_chunk_boundary():
    assert chunk_indices(0,4,5).tolist()==[1,2,3,3,3]
    assert chunk_indices(2,4,5).tolist()==[3,3,3,3,3]


def test_heldout_corpus_fail_before_open(tmp_path):
    with pytest.raises(ValueError,match='D0'):
        Corpus(tmp_path,tasks=('D0','D2'),verify=False)


def test_normalization_source_dimensions():
    state=np.array([[1.,2.],[3.,4.]],np.float32)
    actions=np.array([[[1.],[3.]],[[5.],[7.]]],np.float32)
    n=normalization(state,actions)
    np.testing.assert_allclose(n['state']['mean'],[2.,3.])
    np.testing.assert_allclose(n['actions']['mean'],[4.])
