import numpy as np
import pytest
from alignment.data import augmentation_params, replay, require_seen, sample_order
from alignment.cache import CacheWriter, CacheReader


def test_replay_and_view_seeds():
    image=np.arange(224*224*3,dtype=np.uint32).reshape(224,224,3).astype(np.uint8)
    p=augmentation_params('D0/demo_0/7',2,'base_0_rgb')
    assert p==augmentation_params('D0/demo_0/7',2,'base_0_rgb')
    assert p!=augmentation_params('D0/demo_0/7',2,'left_wrist_0_rgb')
    assert np.array_equal(replay(image,p),replay(image,p))
    assert replay(image,p).shape==image.shape


def test_heldout_rejected():
    for tasks in [('D2',),('D0','D2'),('D0','D1','D2')]:
        with pytest.raises(ValueError): require_seen(tasks)
    require_seen(('D0','D1'))


def test_order_resumes_and_balanced():
    a=sample_order([7,9],50,seed=0)
    b=sample_order([7,9],100,seed=0)
    assert a==b[:50]
    assert sum(x[0]==0 for x in a)==25
    assert all(0<=x[2]<4 for x in a)


def test_cache_lookup_and_checksums(tmp_path):
    w=CacheWriter(tmp_path,shard_size=2,metadata={'tasks':['D0','D1']})
    features=np.ones((2,16,16,3),np.float16)
    w.add('D0/demo_0/0',0,features,{'seed':3},['hash1','hash2'])
    w.add('D1/demo_0/0',1,features*2,{'seed':4},['hash3','hash4'])
    w.finish()
    r=CacheReader(tmp_path)
    got=r.lookup('D1/demo_0/0',1)
    assert np.array_equal(got['features'],features*2)
    assert got['image_sha256']==['hash3','hash4']
    with pytest.raises(KeyError): r.lookup('D0/demo_0/0',3)
    path=next(tmp_path.glob('shard*.npz')); path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'): CacheReader(tmp_path)


def test_duplicate_cache_keys_rejected(tmp_path):
    w=CacheWriter(tmp_path,2,{})
    features=np.ones((2,16,16,3),np.float16)
    w.add('D0/0',0,features,{},['a','b'])
    with pytest.raises(ValueError,match='duplicate'): w.add('D0/0',0,features,{},['a','b'])
