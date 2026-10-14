"""Separate published evidence from a fresh local run, preserving every artifact."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def prepare(root:Path):
    root=Path(root).resolve()
    private=root/'private'
    if private.is_symlink() or (private.exists() and any(private.iterdir())):
        raise RuntimeError('Refusing to prepare an existing local experiment; use a fresh clone before setup/download/smoke. Existing private data and checkpoints are preserved.')
    keep={'configs':{'pins.json','upstream-uv.lock','base_inventory.json'},'evaluation':{'B-original_historical.json'}}
    planned=[]
    for folder in ('logs','configs','evaluation','figures'):
        directory=root/folder
        if directory.is_symlink():raise RuntimeError('Refusing runtime directory symlink: '+folder)
        if directory.exists():
            planned.extend(p for p in sorted(directory.iterdir()) if p.name not in keep.get(folder,set()))
    if (root/'REPORT_RU.md').exists():planned.append(root/'REPORT_RU.md')
    archive=private/'published_results'
    archive.mkdir(parents=True,exist_ok=False)
    for source in planned:
        destination=archive/source.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(source),str(destination))
    for folder in ('logs','configs','evaluation','figures'):(root/folder).mkdir(exist_ok=True)
    result=dict(status='fresh_run_prepared',archive=str(archive),
                archived=[str(p.relative_to(root)) for p in planned],
                note='Published evidence archived; no prior completion, lambda, normalization, initialization, cache manifests or evaluation rows reused. Static pins/lock, immutable GCS base object generations and historical B-original reference retained.')
    (root/'logs/reproduction_preparation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    print(json.dumps(prepare(ROOT),indent=2))
