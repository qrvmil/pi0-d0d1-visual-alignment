# Visual Alignment Implementation Plan

**Goal:** полностью выполнить docs/REQUEST_RU.txt.
**Architecture:** pinned OpenPI JAX + scoped deterministic D0/D1 cache; prefix middle-state extractor; frozen JAX projector; одинаковые paired SFT runs; LIBERO evaluator и exact paired statistics.
**Tech Stack:** JAX/Flax NNX, PyTorch teacher, NumPy, HDF5, LIBERO, Orbax, HF CLI, gh.
**Spec:** docs/DESIGN_RU.md.

## Global Constraints
D2 training/calibration/normalization/debugging запрещены. Training gated smoke, upstream pins неизменны. No random extra augmentations. No dataset/cache/upstream weights/secrets publication. All scheduled steps retained.

## Review Focus
Cache collision/corruption; no-valid-camera reduction; full-pass/middle correspondence; zero discordant paired outcomes; exact data-order resume.

## Tasks
1. Environment/smoke: scripts/smoke.py и environment lock; synthetic actual pi0 step, GPU matmul, CUDA runtime>=12.8, VRAM. No dataset download before pass.
2. Cache: alignment/data.py, alignment/cache.py, tests/test_data.py. Replay equality, view masks, corrupt manifest and lookup failure. Download only exact D0+D1 pinned dataset files. Shared normalization.
3. Model: alignment/model.py, patch upstream scan slice for middle only, tests/test_model.py. Compare prefix-only selected state against full forward, gradient reaches selected student layers, projector/teacher frozen.
4. Training: alignment/train.py. Calibrate projector200, pilot20, preregistered λ rule, smoke5, resume equality, train control and align3001 with same init/order/effectivebatch8. Save all scheduled checkpoints and hashes.
5. Eval/stats: alignment/evaluate.py, alignment/stats.py, tests/test_stats.py. 50 paired seeds D0/D1/D2, deterministic D0 extension500, exact reset/noise checks. Exact conservative paired CI including no-discordance case.
6. Report/publication: REPORT_RU.md, figures, metrics, README, Russian model card; upload created checkpoint steps/projector/assets only; release and remote download checks.

Each code task: write behavioral tests, observe failure, implement, run project suite, record outcome. User requested autonomous end-to-end execution; approval handoffs are superseded by that instruction. Execute inline. Review and audit against every REQUEST requirement before completion.
