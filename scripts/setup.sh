#!/usr/bin/env bash
set -euo pipefail
python -m pip install uv==0.13.0
if [[ ! -d /root/maniskill_myws/.git ]]; then
  git clone https://github.com/qrvmil/maniskill_myws.git /root/maniskill_myws
fi
git -C /root/maniskill_myws checkout c6b77f51e3e5bd8873b812d258744c43380fdfeb
GIT_LFS_SKIP_SMUDGE=1 git -C /root/maniskill_myws submodule update --init --recursive
cd /root/maniskill_myws/third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 --frozen
uv pip install --python .venv/bin/python 'timm==1.0.15' 'robosuite==1.4.1' 'mujoco==3.2.7' 'bddl==1.0.1' 'gym==0.26.2' future easydict
uv pip install --python .venv/bin/python --no-deps 'nvidia-cuda-runtime-cu12==12.8.90' 'nvidia-cuda-nvcc-cu12==12.8.93'
git -C third_party/libero checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
apt-get update -qq
apt-get install -y -qq libgl1 libegl1 libglvnd0 libgles2
.venv/bin/python - <<'PY'
from pathlib import Path
import yaml
root=Path('/root/maniskill_myws/third_party/openpi/third_party/libero/libero/libero')
p=Path('/root/.libero/config.yaml');p.parent.mkdir(exist_ok=True)
data={k:str(root/v) for k,v in [('benchmark_root',''),('bddl_files','bddl_files'),('init_states','init_files'),('assets','assets')]}
data['datasets']=str(root.parent/'datasets')
if p.exists() and yaml.safe_load(p.read_text())!=data:raise ValueError('Existing LIBERO paths differ')
p.write_text(yaml.safe_dump(data))
PY
