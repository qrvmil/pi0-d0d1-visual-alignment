"""Native LIBERO seen-only corpus, shared chunk-weighted normalization."""
import json
from pathlib import Path
import h5py
import numpy as np
from openpi_client.image_tools import resize_with_pad
from .data import TASKS,VIEWS,require_seen,sha256,augmentation_params,replay,image_hash


def chunk_indices(frame,length,horizon=50):
    if not 0<=frame<length-1:raise ValueError('Invalid native frame')
    return np.minimum(np.arange(frame+1,frame+1+horizon),length-1)


def normalization(states,actions):
    result={}
    for name,x in [('state',states),('actions',actions.reshape(-1,actions.shape[-1]))]:
        x=np.asarray(x,np.float64)
        result[name]=dict(mean=x.mean(axis=0).tolist(),std=x.std(axis=0).tolist(),
                          q01=np.quantile(x,.01,axis=0).tolist(),q99=np.quantile(x,.99,axis=0).tolist())
    return result


class Corpus:
    def __init__(self,root,tasks=('D0','D1'),verify=True):
        require_seen(tasks)
        self.root=Path(root);self.tasks=tasks;self.files={};self.samples={};self.records=[]
        manifest=json.loads((self.root/'configs/dataset_manifest.json').read_text())
        for task in tasks:
            path=self.root/'private/native'/manifest[task]['path']
            if verify and sha256(path)!=manifest[task]['sha256']:raise ValueError('Dataset checksum mismatch')
            f=h5py.File(path,'r');self.files[task]=f
            expected='/'.join(TASKS[task])+'.bddl'
            if not str(f['data'].attrs.get('bddl_file_name','')).endswith(expected):
                raise ValueError('Source task identity mismatch')
            demos=sorted(f['data'],key=lambda s:int(s.split('_')[-1]))
            if len(demos)!=50:raise ValueError('Exactly50 expert demos required')
            rows=[]
            for demo in demos:
                group=f['data'][demo];length=len(group['actions'])
                if group['actions'].shape[1]!=7:raise ValueError('Seven action dimensions required')
                for frame in range(length-1):
                    row=dict(task=task,demo=demo,frame=frame,length=length,sample_id=f'{task}/{demo}/{frame}')
                    rows.append(row);self.records.append(row)
            self.samples[task]=rows
        self.lengths=[len(self.samples[t]) for t in tasks]

    def raw(self,row):
        g=self.files[row['task']]['data'][row['demo']];o=g['obs'];i=row['frame']
        images=[resize_with_pad(np.ascontiguousarray(o[k][i][::-1,::-1]),224,224)
                for k in ('agentview_rgb','eye_in_hand_rgb')]
        state=np.concatenate([o['ee_pos'][i],o['ee_ori'][i],o['gripper_states'][i]]).astype(np.float32)
        # h5py fancy indices reject repeated terminal indices.
        indices=chunk_indices(i,row['length'])
        unique,inverse=np.unique(indices,return_inverse=True)
        action=g['actions'][unique][inverse].astype(np.float32)
        if state.shape!=(8,) or not np.isfinite(state).all() or not np.isfinite(action).all():
            raise ValueError('Invalid source states/actions')
        return images,state,action

    def augmented(self,row,aug):
        images,state,actions=self.raw(row)
        params={v:augmentation_params(row['sample_id'],aug,v) for v in VIEWS}
        images=[replay(img,params[v]) for v,img in zip(VIEWS,images)]
        return dict(images=images,state=state,actions=actions,params=params,image_sha256=[image_hash(x) for x in images])

    def make_normalization(self):
        states=[];actions=[]
        for row in self.records:
            g=self.files[row['task']]['data'][row['demo']];i=row['frame'];o=g['obs']
            states.append(np.concatenate([o['ee_pos'][i],o['ee_ori'][i],o['gripper_states'][i]]))
            indices=chunk_indices(i,row['length']);u,inv=np.unique(indices,return_inverse=True)
            actions.append(g['actions'][u][inv])
        result=normalization(np.asarray(states),np.asarray(actions))
        dest=self.root/'configs/norm_stats.json'
        dest.write_text(json.dumps({'norm_stats':result},indent=2))
        (self.root/'configs/normalization_provenance.json').write_text(json.dumps(dict(
            tasks=list(self.tasks),demos_per_task=50,frames=sum(self.lengths),action_horizon=50,
            method='all seen frames, native next-action shift, terminal-clamped chunk-weighted mean/std',
            source_manifest_sha256=sha256(self.root/'configs/dataset_manifest.json'),
            normalization_sha256=sha256(dest)),indent=2))
        return result
