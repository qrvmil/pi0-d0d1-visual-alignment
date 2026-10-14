from pathlib import Path
import pytest


def published_clone(root):
    files={'configs/pins.json':'pins','configs/upstream-uv.lock':'lock',
           'configs/lambda_decision.json':'published lambda','configs/B-align.json':'published binding',
           'configs/norm_stats.json':'published norm','configs/memory_recipe.json':'published memory',
           'configs/initialization_sha256.json':'published initialization','configs/cache_manifest.json':'published cache manifest',
           'logs/B-control_complete.json':'published completion',
           'evaluation/B-align/D0/episodes.json':'published episodes',
           'evaluation/B-original_historical.json':'historical reference',
           'figures/losses.png':'published figure','REPORT_RU.md':'published report'}
    for name,value in files.items():
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
    return files


def test_fresh_clone_archives_evidence_and_removes_all_runtime_reuse(tmp_path):
    from scripts.prepare_reproduction import prepare
    before=published_clone(tmp_path)
    result=prepare(tmp_path)
    assert (tmp_path/'configs/pins.json').read_text()=='pins'
    assert (tmp_path/'configs/upstream-uv.lock').read_text()=='lock'
    assert (tmp_path/'evaluation/B-original_historical.json').read_text()=='historical reference'
    retained={'configs/pins.json','configs/upstream-uv.lock','evaluation/B-original_historical.json'}
    for name in set(before)-retained:
        assert not (tmp_path/name).exists()
        assert (Path(result['archive'])/name).read_text()==before[name]
    assert all((tmp_path/name).is_dir() for name in ('logs','configs','evaluation','figures'))


def test_preparation_refuses_existing_local_experiment_without_mutation(tmp_path):
    from scripts.prepare_reproduction import prepare
    before=published_clone(tmp_path)
    checkpoint=tmp_path/'private/checkpoints/B-align/3001/params/weights'
    checkpoint.parent.mkdir(parents=True);checkpoint.write_text('valuable weights')
    with pytest.raises(RuntimeError,match='existing local'):
        prepare(tmp_path)
    assert checkpoint.read_text()=='valuable weights'
    for name,value in before.items():assert (tmp_path/name).read_text()==value


def test_second_preparation_cannot_overwrite_archived_evidence(tmp_path):
    from scripts.prepare_reproduction import prepare
    published_clone(tmp_path);result=prepare(tmp_path)
    with pytest.raises(RuntimeError):prepare(tmp_path)
    assert (Path(result['archive'])/'REPORT_RU.md').read_text()=='published report'


def test_documented_preparation_cli_runs_before_openpi_setup(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    published_clone(tmp_path)
    script=tmp_path/'scripts/prepare_reproduction.py'
    script.parent.mkdir();shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/prepare_reproduction.py',script)
    result=subprocess.run([sys.executable,str(script)],cwd=tmp_path,text=True,capture_output=True,check=True)
    assert json.loads(result.stdout)['status']=='fresh_run_prepared'
    assert not (tmp_path/'configs/lambda_decision.json').exists()
    assert not (tmp_path/'evaluation/B-align').exists()
