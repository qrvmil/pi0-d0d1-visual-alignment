"""Download exact seen-only demonstrations and immutable teacher/base inventories."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import requests
from huggingface_hub import HfApi,hf_hub_download,snapshot_download

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.data import TASKS,require_seen,sha256

ROOT=Path(__file__).resolve().parents[1]
PRIVATE=ROOT/'private'


def main():
    gate=json.loads((ROOT/'logs/compatibility_smoke.json').read_text())
    if gate['status']!='passed' or gate['cuda_runtime']<12080:raise RuntimeError('Smoke gate closed')
    require_seen(tuple(TASKS))
    api=HfApi()
    repo='yifengzhu-hf/LIBERO-datasets'
    existing_pins=json.loads((ROOT/'configs/pins.json').read_text()) if (ROOT/'configs/pins.json').exists() else {}
    rev=existing_pins.get('dataset_revision') or api.dataset_info(repo).sha
    pins=dict(source='c6b77f51e3e5bd8873b812d258744c43380fdfeb',
              openpi='981483dca0fd9acba698fea00aa6e52d56a66c58',libero='8f1084e3132a39270c3a13ebe37270a43ece2a01',
              teacher='9d0413465e8a91e67bbf2c1ad342815478d1b906',dataset_revision=rev,
              reference_revision='0e27d62f16cc276c33d5e3b2107cb1f24dce7792')
    (ROOT/'configs/pins.json').write_text(json.dumps(pins,indent=2))
    sources={}
    for t,(suite,name) in TASKS.items():
        filename=f'{suite}/{name}_demo.hdf5'
        path=hf_hub_download(repo,filename,repo_type='dataset',revision=rev,local_dir=PRIVATE/'native')
        sources[t]=dict(path=filename,sha256=sha256(path),revision=rev)
        print('FETCHED_SEEN',t,sources[t],flush=True)
    (ROOT/'configs/dataset_manifest.json').write_text(json.dumps(sources,indent=2))
    print('TEACHER_FETCH',flush=True)
    snapshot_download('nvidia/C-RADIOv3-L',revision=pins['teacher'],local_dir=PRIVATE/'teacher',
                      ignore_patterns=['*.pth.tar'])
    prefix='checkpoints/pi0_base/'
    objects=[];page=None
    saved_inventory=ROOT/'configs/base_inventory.json'
    if saved_inventory.exists():objects=json.loads(saved_inventory.read_text())
    while not objects or page:
        params={'prefix':prefix}
        if page:params['pageToken']=page
        r=requests.get('https://storage.googleapis.com/storage/v1/b/openpi-assets/o',params=params,timeout=60)
        r.raise_for_status();body=r.json();objects.extend(body.get('items',[]));page=body.get('nextPageToken')
        if not page:break
    if not objects:raise RuntimeError('Missing base inventory')
    (ROOT/'configs/base_inventory.json').write_text(json.dumps(objects,indent=2))
    def fetch(obj):
        relative=obj['name'][len(prefix):]
        if not relative or obj['name'].endswith('/'):return
        dest=PRIVATE/'base'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists() or dest.stat().st_size!=int(obj['size']):
            tmp=dest.with_name(dest.name+'.download')
            with requests.get('https://storage.googleapis.com/openpi-assets/'+obj['name'],params={'generation':obj['generation']},stream=True,timeout=120) as response:
                response.raise_for_status()
                with tmp.open('wb') as f:
                    for block in response.iter_content(8*1024*1024):f.write(block)
            if tmp.stat().st_size!=int(obj['size']):raise ValueError('Base size mismatch')
            tmp.replace(dest)
        digest=hashlib.md5()
        with dest.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
        if base64.b64encode(digest.digest()).decode()!=obj['md5Hash']:raise ValueError('Base checksum mismatch')
        print('BASE_VERIFIED',relative,flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(fetch,objects))
    manifest={str(p.relative_to(PRIVATE/'base')):dict(sha256=sha256(p),bytes=p.stat().st_size)
              for p in (PRIVATE/'base').rglob('*') if p.is_file()}
    (ROOT/'configs/base_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('FETCH_COMPLETE',flush=True)

if __name__=='__main__':main()
