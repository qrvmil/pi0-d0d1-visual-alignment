"""Frozen C-RADIO cache from exactly the RGB pixels replayed by π0."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
import sys
from .cache import CacheWriter
from .corpus import Corpus
from .data import sha256

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples',type=int,default=None)
    args=parser.parse_args()
    torch.set_num_threads(4)
    corpus=Corpus(ROOT)
    if args.samples is None:corpus.make_normalization()
    sys.path.insert(0,str(ROOT/'private'))
    from teacher.hf_model import RADIOModel,RADIOConfig
    from safetensors.torch import load_file
    teacher=RADIOModel(RADIOConfig.from_json_file(ROOT/'private/teacher/config.json'))
    teacher.load_state_dict(load_file(ROOT/'private/teacher/model.safetensors'),strict=True)
    teacher=teacher.eval().cuda()
    teacher.requires_grad_(False)
    assert not any(p.requires_grad for p in teacher.parameters())
    before=[(n,p.detach().clone()) for n,p in teacher.named_parameters()] if args.samples else None
    pins=json.loads((ROOT/'configs/pins.json').read_text())
    dest=ROOT/'private'/('cache_smoke' if args.samples else 'cache')
    if (dest/'manifest.json').exists():
        print('CACHE_ALREADY_COMPLETE',dest,flush=True);return
    writer=CacheWriter(dest,shard_size=32,metadata=dict(tasks=['D0','D1'],K=4,
        teacher_revision=pins['teacher'],dataset_manifest_sha256=sha256(ROOT/'configs/dataset_manifest.json'),
        teacher_input='uint8 RGB224/255 with internal RADIO input_conditioner; no external CLIP normalization',
        grid_resize='bilinear align_corners=False14x14→16x16',teacher_frozen=True))
    rows=corpus.records if args.samples is None else [corpus.samples[t][0] for t in corpus.tasks][:args.samples]
    started=time.time()
    for i,row in enumerate(rows):
        for aug in range(4):
            item=corpus.augmented(row,aug)
            pixels=torch.from_numpy(np.stack(item['images'])).permute(0,3,1,2).float().cuda()/255.
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                _,features=teacher(pixels)
            assert features.shape[1]==14*14
            grid=features.float().reshape(2,14,14,-1).permute(0,3,1,2)
            grid=F.interpolate(grid,size=(16,16),mode='bilinear',align_corners=False).permute(0,2,3,1)
            writer.add(row['sample_id'],aug,grid.cpu().numpy().astype(np.float16),item['params'],item['image_sha256'])
        if i%100==0:print('CACHE_PROGRESS',i,len(rows),round(time.time()-started,1),flush=True)
    manifest=writer.finish()
    unchanged=True if before is None else all(torch.equal(p,dict(teacher.named_parameters())[n]) for n,p in before)
    assert unchanged
    info=dict(samples=len(rows),entries=len(manifest['index']),seconds=time.time()-started,
              peak_vram_bytes=torch.cuda.max_memory_allocated(),teacher_parameters_frozen=True,
              teacher_state_unchanged=unchanged,manifest_sha256=sha256(dest/'manifest.json'))
    output=ROOT/'logs'/('cache_smoke.json' if args.samples else 'cache_build.json')
    output.write_text(json.dumps(info,indent=2))
    # Publish manifest without publishing the cache arrays.
    (ROOT/'configs'/('cache_smoke_manifest.json' if args.samples else 'cache_manifest.json')).write_text((dest/'manifest.json').read_text())
    print('CACHE_COMPLETE',info,flush=True)

if __name__=='__main__':main()
