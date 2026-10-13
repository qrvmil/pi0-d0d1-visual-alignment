export ALIGN_ROOT=/root/pi0-d0d1-visual-alignment
export ALIGN_PY=/root/maniskill_myws/third_party/openpi/.venv/bin/python
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=/root/maniskill_myws/src:/root/maniskill_myws/third_party/openpi/third_party/libero${PYTHONPATH:+:$PYTHONPATH}
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
cd "$ALIGN_ROOT"
