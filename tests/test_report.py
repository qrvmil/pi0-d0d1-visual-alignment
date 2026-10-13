import importlib.util
from pathlib import Path
import pytest
import matplotlib.pyplot as plt

spec=importlib.util.spec_from_file_location('report',Path(__file__).parents[1]/'scripts/report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def test_zero_rate_interval_roundoff_remains_renderable(tmp_path):
    lower,upper=report.interval_errors(0.,6.938893903907228e-18,.07134759913335872)
    fig,ax=plt.subplots()
    ax.errorbar([0],[0],yerr=[[lower],[upper]],fmt='o')
    fig.savefig(tmp_path/'rates.png');plt.close(fig)
    assert (tmp_path/'rates.png').stat().st_size>1000


def test_invalid_interval_does_not_get_silently_clamped():
    with pytest.raises(ValueError):report.interval_errors(0.,.1,.2)
