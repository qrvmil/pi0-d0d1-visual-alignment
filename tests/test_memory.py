import pytest
from alignment.memory import process_bytes


def test_process_memory_separates_concurrent_teacher():
    rows=[(10,1024),(20,2048)]
    assert process_bytes(rows,20,8192)==2048
    assert process_bytes(rows,30,8192)==0
    with pytest.raises(ValueError):process_bytes([(20,2**64-1)],20,8192)
