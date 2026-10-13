"""Sharded fp16 spatial cache with exact IDs and fail-closed integrity checks."""
import json
from pathlib import Path
import numpy as np
from .data import sha256


def key(sample_id,augmentation):
    return f'{sample_id}@{augmentation}'


class CacheWriter:
    def __init__(self,root,shard_size=32,metadata=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        if (self.root/'manifest.json').exists():raise FileExistsError('Completed cache already exists')
        self.size=shard_size;self.metadata=metadata or {};self.rows=[];self.index={};self.shards=[]

    def add(self,sample_id,augmentation,features,params,image_sha256):
        k=key(sample_id,augmentation)
        if k in self.index:raise ValueError('duplicate cache key')
        features=np.asarray(features,np.float16)
        if features.ndim!=4 or features.shape[:3]!=(2,16,16) or not np.isfinite(features).all():
            raise ValueError('Cache expects two finite16x16 view grids')
        record=dict(sample_id=sample_id,augmentation=augmentation,params=params,image_sha256=image_sha256,
                    shard=len(self.shards),row=len(self.rows))
        self.index[k]=record;self.rows.append(features)
        if len(self.rows)==self.size:self.flush()

    def flush(self):
        if not self.rows:return
        name=f'shard_{len(self.shards):05d}.npz'
        path=self.root/name
        # Uncompressed memory-friendly arrays; checksum protects all rows.
        with path.open('wb') as f:np.savez(f,features=np.stack(self.rows))
        self.shards.append(dict(file=name,sha256=sha256(path),bytes=path.stat().st_size,rows=len(self.rows)))
        self.rows=[]

    def finish(self):
        self.flush()
        result=dict(version=1,dtype='float16',views=['base_0_rgb','left_wrist_0_rgb'],grid=[16,16],
                    metadata=self.metadata,index=self.index,shards=self.shards)
        (self.root/'manifest.json').write_text(json.dumps(result,indent=2))
        return result


class CacheReader:
    def __init__(self,root,verify=True):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text())
        self.last_shard=None;self.last_data=None
        if verify:
            for s in self.manifest['shards']:
                if sha256(self.root/s['file'])!=s['sha256']:raise ValueError('Cache checksum mismatch')

    def lookup(self,sample_id,augmentation):
        row=self.manifest['index'][key(sample_id,augmentation)]
        n=row['shard']
        if self.last_shard!=n:
            self.last_data=np.load(self.root/self.manifest['shards'][n]['file'])['features']
            self.last_shard=n
        return dict(row,features=self.last_data[row['row']].copy())
