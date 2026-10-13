"""Publish only finalized artifacts and newly created checkpoints, then verify remote downloads."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re
import uuid
import requests
from huggingface_hub import HfApi,hf_hub_download

ROOT=Path(__file__).resolve().parents[1]
REPO='qrvmil/pi0-d0d1-visual-alignment'
SCHEDULE=[0,500,1000,2000,3001]


def command(args):
    return subprocess.check_output(args,cwd=ROOT,text=True).strip()


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def validate_staged_entries(entries):
    roots={'alignment','scripts','tests','docs','configs','logs','evaluation','figures'}
    topfiles={'README.md','REPORT_RU.md','environment.lock.txt','.gitignore'}
    for mode,name in entries:
        p=Path(name)
        if mode=='120000' or (len(p.parts)==1 and name not in topfiles) or (len(p.parts)>1 and p.parts[0] not in roots):
            raise ValueError(f'Unapproved staged artifact: {name}')
        if p.name.startswith('.env') or p.suffix in ('.env','.hdf5','.h5','.key','.pem','.pth','.safetensors'):
            raise ValueError(f'Forbidden staged artifact: {name}')


def validate_origin(url):
    if url.rstrip('/').removesuffix('.git') not in (f'https://github.com/{REPO}',f'git@github.com:{REPO}'):
        raise ValueError('Existing origin does not target the requested repository')


def audit_staged():
    entries=[]
    for row in command(['git','ls-files','--stage']).splitlines():
        head,path=row.split('\t',1);entries.append((head.split()[0],path))
    validate_staged_entries(entries)
    pattern=re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{20,}|-----BEGIN [^-]*PRIVATE KEY-----')
    for _,name in entries:
        p=ROOT/name
        if p.suffix not in ('.png','.pdf','.mp4','.npz'):
            if pattern.search(p.read_text(errors='ignore')):raise ValueError(f'Secret-like content in {name}')


def validate_local():
    sys.path.insert(0,str(ROOT))
    from alignment.evaluation import summarize
    from alignment.data import sha256
    subprocess.check_call(['/root/maniskill_myws/third_party/openpi/.venv/bin/python','-m','pytest','-q'],cwd=ROOT)
    stored=json.loads((ROOT/'evaluation/results.json').read_text())
    summarize()
    if stored!=json.loads((ROOT/'evaluation/results.json').read_text()):raise ValueError('Stored results differ from current cohorts')
    if stored['D0'].get('extend_to500'):raise ValueError('Prespecified D0 extension incomplete')
    old_report=(ROOT/'REPORT_RU.md').read_bytes()
    subprocess.check_call(['/root/maniskill_myws/third_party/openpi/.venv/bin/python','scripts/report.py'],cwd=ROOT)
    if (ROOT/'REPORT_RU.md').read_bytes()!=old_report:raise ValueError('Report was stale; regenerated, review before publication')
    if not (ROOT/'figures/success_rates.png').is_file():raise ValueError('Missing success-rate figure')
    for arm in ('B-control','B-align'):
        binding=json.loads((ROOT/'configs'/f'{arm}.json').read_text())
        for field,path in [('projector_sha256',ROOT/'private/projector.npz'),('normalization_sha256',ROOT/'configs/norm_stats.json'),('cache_manifest_sha256',ROOT/'private/cache/manifest.json')]:
            if binding[field]!=sha256(path):raise ValueError('Current training assets drift from saved binding')
    results=json.loads((ROOT/'evaluation/results.json').read_text())
    if set(results)!=set(['D0','D1','D2']):raise ValueError('Missing task results')
    if not (ROOT/'REPORT_RU.md').is_file() or not (ROOT/'figures/losses.png').is_file():raise ValueError('Missing report/figures')
    if 'passed' not in (ROOT/'logs/final_tests.log').read_text() or 'FAILED' in (ROOT/'logs/final_tests.log').read_text():
        raise ValueError('Final tests not verified')
    exported=[]
    for arm in ('B-control','B-align'):
        complete=json.loads((ROOT/'logs'/f'{arm}_complete.json').read_text())
        if complete['steps']!=3001:raise ValueError('Training incomplete')
        folder=ROOT/'private/checkpoints'/arm
        steps=sorted(int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdecimal())
        if not set(SCHEDULE)<=set(steps):raise ValueError('Missing scheduled checkpoint')
        for step in steps:
            checkpoint=folder/str(step)
            if not (checkpoint/'params/_METADATA').is_file():raise ValueError('Checkpoint params missing')
            if not (checkpoint/'assets/local/d0d1/norm_stats.json').is_file():raise ValueError('Checkpoint normalization missing')
            for p in checkpoint.rglob('*'):
                if p.is_file() and (p.name.startswith('.env') or p.suffix in ('.hdf5','.h5') or p.is_symlink()):
                    raise ValueError(f'Forbidden publication file: {p.name}')
            metadata=json.loads((checkpoint/'experiment_metadata.json').read_text())
            binding=json.loads((ROOT/'configs'/f'{arm}.json').read_text())
            if metadata['binding']!=binding:raise ValueError('Checkpoint binding mismatch')
            if sha(checkpoint/'assets/local/d0d1/norm_stats.json')!=sha(ROOT/'configs/norm_stats.json'):
                # Orbax serializes normalization with stable formatting; compare numeric content if formatting differs.
                if json.loads((checkpoint/'assets/local/d0d1/norm_stats.json').read_text())!=json.loads((ROOT/'configs/norm_stats.json').read_text()):raise ValueError('Checkpoint normalization drift')
            exported.append(f'{arm}/{step}')
    return exported


def main():
    exported=validate_local()
    try:
        command(['gh','auth','status'])
        api=HfApi();identity=api.whoami()
        if identity['name']!='qrvmil':raise RuntimeError('HF identity differs')
    except Exception as e:
        print('Publication authorization required: gh auth login; hf auth login',file=sys.stderr)
        raise RuntimeError('Training stays saved; rerun publish only after authorization') from e
    # GitHub must contain neither private artifacts nor credentials.
    tracked=command(['git','ls-files']).splitlines()
    if any(p.startswith(('private/','datasets/','cache/','checkpoints/')) or Path(p).name.startswith('.env') for p in tracked):
        raise ValueError('Forbidden tracked publication path')
    if 'origin' in command(['git','remote']).splitlines():validate_origin(command(['git','remote','get-url','origin']))
    command(['git','add','.']);audit_staged()
    export=ROOT/'private'/('hf_export_'+uuid.uuid4().hex);export.mkdir(parents=True)
    for arm in ('B-control','B-align'):
        dest=export/'checkpoints'/arm
        for checkpoint in exported:
            checkpoint_arm,step=checkpoint.split('/')
            if checkpoint_arm==arm:
                shutil.copytree(ROOT/'private/checkpoints'/arm/step,dest/step,copy_function=os.link)
    shutil.copy2(ROOT/'private/projector.npz',export/'projector.npz')
    configdest=export/'configs';configdest.mkdir(exist_ok=True)
    for p in (ROOT/'configs').glob('*.json'):
        if 'cache_manifest' in p.name or 'cache_smoke_manifest' in p.name or p.name=='published_hf_inventory.json':continue
        shutil.copy2(p,configdest/p.name)
    shutil.copy2(ROOT/'REPORT_RU.md',export/'REPORT_RU.md')
    for folder in ('figures','evaluation','logs','docs'):
        shutil.copytree(ROOT/folder,export/folder)
    card=f'''---
language:
- ru
library_name: openpi
tags:
- robotics
- pi0
- libero
- visual-alignment
---
# π0: visual alignment на D0+D1

Код и воспроизведение: https://github.com/{REPO}
Статья: https://arxiv.org/html/2510.25616
Результаты: [REPORT_RU.md](REPORT_RU.md).

Новые B-control и B-align из одной исходной π0, одинаковые seed0, data order, K=4 deterministic augmentations, optimizer/schedule и effectivebatch8. Отличие B-align — cosine middle PaliGemma visual features к frozen C-RADIOv3-L через предварительно обученный, затем frozen LN+MLP.

D0: libero_spatial / pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.
D1: libero_spatial / pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.
D2: libero_goal / put_the_bowl_on_the_stove — held-out OOD.

**D2 не использовалась для training, calibration, normalization, настройки λ, debugging или выбора модели.** Финальный checkpoint3001 выбран заранее. Все scheduled checkpoints сохранены: {', '.join(exported)}. Доказанный успех заявляется только при выполнении зарегистрированных статистических критериев; см. численные результаты в отчёте.

Веса в checkpoints/<arm>/<step>/params — Orbax/OpenPI format; train_state содержит optimizer state для resume; assets/local/d0d1/norm_stats.json — D0+D1-only normalization. Projector — projector.npz (LN scale/bias, w1/b1, w2/b2); не нужен при inference. Конфиги/metadata доступны рядом. Используйте pinned OpenPI и LoRA32 из GitHub scripts/setup.sh и alignment/training.py. Исходная π0: gs://openpi-assets/checkpoints/pi0_base; teacher revision и source hashes в configs/pins.json.

Данные, teacher feature cache, исходные upstream checkpoints и credentials в этот репозиторий не загружены. Фактическая машина A10080GB; Blackwell compatibility не проверена. Это один seed обучения; внешние задачи за пределами D2 не оценивались.
'''
    (export/'README.md').write_text(card)
    api.create_repo(REPO,repo_type='model',exist_ok=True,private=False)
    api.upload_large_folder(repo_id=REPO,repo_type='model',folder_path=export,num_workers=4)
    info=api.model_info(REPO,files_metadata=True);revision=info.sha
    inventory=[]
    for p in export.rglob('*'):
        if p.is_file() and '.cache' not in p.parts:
            inventory.append(dict(path=str(p.relative_to(export)),bytes=p.stat().st_size,sha256=sha(p)))
    remote={s.rfilename:s for s in info.siblings}
    for item in inventory:
        sibling=remote.get(item['path'])
        if sibling is None or sibling.size!=item['bytes']:raise ValueError('Remote HF inventory mismatch')
        if sibling.lfs is not None and sibling.lfs.sha256!=item['sha256']:raise ValueError('Remote HF LFS checksum mismatch')
    checks=[]
    for checkpoint in exported:
        for relative in ('params/_METADATA','assets/local/d0d1/norm_stats.json'):
            filename=f'checkpoints/{checkpoint}/{relative}'
            fetched=Path(hf_hub_download(REPO,filename,revision=revision))
            if sha(fetched)!=sha(export/filename):raise ValueError('Remote downloaded artifact differs')
            checks.append(filename)
    for filename in ('README.md','REPORT_RU.md','projector.npz'):
        fetched=Path(hf_hub_download(REPO,filename,revision=revision))
        if sha(fetched)!=sha(export/filename):raise ValueError('Remote download mismatch')
        checks.append(filename)
    (ROOT/'configs/published_hf_inventory.json').write_text(json.dumps(dict(revision=revision,files=inventory,download_verified=checks),indent=2))
    if subprocess.run(['gh','repo','view',REPO],cwd=ROOT,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:
        command(['gh','repo','create',REPO,'--public','--description','Reproducible pi0 visual alignment D0+D1 / held-out D2'])
    remotes=command(['git','remote']).splitlines()
    if 'origin' not in remotes:command(['git','remote','add','origin',f'https://github.com/{REPO}.git'])
    command(['git','add','.']);audit_staged()
    validate_origin(command(['git','remote','get-url','origin']))
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        command(['git','commit','-m','results: publish paired visual alignment experiment and complete artifacts'])
    command(['git','push','-u','origin','main'])
    commit=command(['git','rev-parse','HEAD'])
    tag='v1.0.0-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
    notes=ROOT/'private/release_notes.md'
    notes.write_text(f'Полный отчёт: REPORT_RU.md\n\nHF checkpoints: https://huggingface.co/{REPO}/tree/{revision}\n\nGit commit: {commit}\n\nCheckpoints: '+', '.join(exported)+'\n')
    command(['gh','release','create',tag,'REPORT_RU.md','evaluation/summary.csv','figures/losses.png','figures/success_rates.png',
             '--target',commit,'--title','π0 D0+D1 visual alignment — paired results','--notes-file',str(notes)])
    report_url=f'https://raw.githubusercontent.com/{REPO}/{commit}/REPORT_RU.md'
    response=requests.get(report_url,timeout=60);response.raise_for_status()
    if response.content!=(ROOT/'REPORT_RU.md').read_bytes():raise ValueError('Published GitHub report differs')
    release=json.loads(command(['gh','release','view',tag,'--json','url,assets']))
    for asset in release['assets']:
        response=requests.get(asset['url'],stream=True,timeout=60);response.raise_for_status()
        if not next(response.iter_content(1024),b''):raise ValueError('Release asset not downloadable')
        response.close()
    result=dict(github=f'https://github.com/{REPO}',hf=f'https://huggingface.co/{REPO}',commit=commit,
                release_tag=tag,release_url=release['url'],hf_revision=revision,checkpoints=exported)
    (ROOT/'logs/publication.json').write_text(json.dumps(result,indent=2))
    command(['git','add','logs/publication.json']);audit_staged()
    command(['git','commit','-m','docs: record verified publication URLs and release inventory'])
    command(['git','push','origin','main'])
    result['publication_log_commit']=command(['git','rev-parse','HEAD'])
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
