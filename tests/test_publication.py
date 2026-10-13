import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('publication',Path(__file__).parents[1]/'scripts/publish.py')
pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)


def test_staged_allowlist_rejects_untracked_dataset_and_symlink():
    pub.validate_staged_entries([('100644','README.md'),('100644','evaluation/D0.json')])
    with pytest.raises(ValueError):pub.validate_staged_entries([('100644','random_dataset.hdf5')])
    with pytest.raises(ValueError):pub.validate_staged_entries([('100644','logs/token.env')])
    with pytest.raises(ValueError):pub.validate_staged_entries([('120000','logs/link')])


def test_wrong_origin_rejected():
    pub.validate_origin('https://github.com/qrvmil/pi0-d0d1-visual-alignment.git')
    with pytest.raises(ValueError):pub.validate_origin('https://github.com/qrvmil/maniskill_myws.git')
