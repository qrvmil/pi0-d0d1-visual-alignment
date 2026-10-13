import pytest
from alignment.smoke import validate_runtime


def test_old_cuda_rejected():
    with pytest.raises(RuntimeError, match='12.8'):
        validate_runtime('gpu',12060)


def test_cpu_rejected():
    with pytest.raises(RuntimeError, match='GPU'):
        validate_runtime('cpu',12080)


def test_runtime_floor():
    validate_runtime('gpu',12080)
