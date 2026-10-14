# Проверка полного запроса

Локальные обучение, оценка и отчёт проверены. Публикация и remote-download proof ещё требуют исполнения; весь запрос до них не считается завершённым. Первоначальный запрос сохранён в REQUEST_RU.txt. Ниже указан scope каждого доказательства.

| Требование | Статус | Авторитетное доказательство |
|---|---|---|
| Прочитать статью и описать перенос/адаптации | Проверено | DESIGN_RU.md: frozen teacher, middle spatial cosine, LN+MLP; адаптации для flow objective, двух камер, grids и PaliGemma block9 |
| GPU/CUDA≥12.8/JAX matmul/OpenPI forward-backward до datasets | Проверено на фактическом A100 | logs/compatibility_smoke.json/.log: CUDA12080, JAX0.5.3, finite real π0 loss/gradients. Blackwell отсутствует и не объявлен проверенным |
| Все source/OpenPI/LIBERO/HF reference/teacher pins | Проверено | configs/pins.json/base_inventory.json/base_manifest.json; fresh git HEAD: c6b77f51 / 981483dc / 8f1084e3; teacher9d041346 и HF reference0e27d62f зафиксированы |
| Только D0+D1 в train/calibration/pilot/norm; D2 без tuning/debug/model selection | Проверено | Corpus.require_seen; dataset_manifest.json и normalization_provenance.json; pilot/calibration tasks=D0+D1; заранее fixed final3001; все train sample IDs проверены |
| K4/2valid cameras/fp16 sharded cache/IDs/seeds/params/checksums | Проверено | logs/cache_build.json:11745samples/46980entries; configs/cache_manifest.json; private/cache с features исключён из git/HF export |
| Точные augmented RGB и отсутствие extra random aug | Проверено для всех consumed batches | CacheReader/Batches сравнивают params/imageSHA; compute_loss train=False; все3001 пар batch/sample/image hashes equal, logs/B-align_audit.json |
| Dummy masks/middle only/spatial correspondence | Проверено | replay/cache/model tests; logs/model_integration.json: actual relativeRMSE0.00395654 < prewritten0.005, dummy invariance exact, no hidden-state layer stack, late alignment grads0 |
| Teacher frozen | Проверено с ограничением измерения | requires_gradFalse/eval/inference_mode, optimizer отсутствует; exact state equality в2-sample smoke. Full-cache exact state equality не измерялась, поле null |
| Projector обучен при frozen π0, затем frozen | Проверено | projector_calibration.json:200updates, policy SHA unchanged; projector SHA неизменён после обоих3001-step runs и совпадает с каждым binding |
| Pilot λ0.2/flow+align/gradient norms/заранее правило | Проверено | pilot.json20updates; lambda_decision.json median last10 ratio0.0154101, selected0.2; правило записано до outcomes |
| 2sample cache +1–5training steps до main | Проверено | cache_smoke.json8entries; data_training_smoke.json5finite updates; actual GPU integration beforepilot |
| Equal base/init/optimizer/schedule/batch/order/aug/seeds | Проверено | independent initialization SHA43f7c62a... одинаков; configs equality except arm/alignment/λ; all3001 paired inputs equal; initial flow rounding difference2.235e-7 within prewritten1e-5 |
| Finite loss/gradients/effectivebatch/OOM handling | Проверено | обе3001 строки finite; micro8/accum1/effective8, OOM0; configs/memory_recipe.json; accumulation regression tests |
| Resume после остановки | Проверено | exact smoke params+optimizer restore step2; fresh GPU process restored2/continued3 with losses exact and gradnorm within tolerance, fresh_process_resume.json; main500 train_state separately restored onCPU |
| PeakVRAM и длительность | Проверено в указанном scope | complete training JAX allocations20.39/25.30GiB и pool32GiB; process wall3483.77/4736.22s. Original NVML host/container PID zeros invalid, null; align attached NVML32.48GiB — partial window, не full-process peak |
| Все scheduled checkpoints | Проверено локально | B-control и B-align0/500/1000/2000/3001; params/train_state/assets/metadata и norm/binding/projector digests повторно проверены |
| Paired resets и flow-noise | Проверено все600пар | logs/final_evaluation_audit.json: exact seed cohorts, reset hashes и common noise prefixes equal; first50 snapshots unchanged |
| First50 всех задач и prespecified D0→500 без cherry-picking | Проверено | evaluation/first50 archive; D0 original lower−.208213983 triggeredextension; final D0 both500 unique seeds10000..10499, D1/D2 each50 seeds10000..10049;1200 new episodes |
| Δ/CI/strict D0 lower>−.02/D2 primary/no unsupported success | Проверено | evaluation/results.json и summary.csv: D0 Δ−.014, lower−.075148328, CI[−.081893146,.054171274]; D2 both0/50, CI±.083909878; все статусы НЕОПРЕДЕЛЁН |
| REPORT_RU.md/таблица/reference/status/directanswer/config/n/time/VRAM/cost/plots/commands | Проверено локально | generated REPORT_RU.md, losses.png/success_rates.png просмотрены; цена1.1854347$/h ×6.03h≈7.15$, scope firstsmoke→finaleval, excludes initialsetup/publication; all claims source-linked |
| Fresh-clone reproduction | Проверено изолированными fixtures/CLI | prepare_reproduction.py архивирует опубликованные runtime outputs в ignored private/published_results, сохраняет pins/lock/immutablebaseinventory/reference и отказывается менять existing private run; tests/test_prepare_reproduction.py; README commands require prep before setup/smoke |
| Все текущие tests | Проверено, final publication gate повторит | original actualGPU fullsuite39passed/44warnings; новые fresh-run/CLI/report tests6passed; новый fullsuite после всех фиксов43passed/44warnings наCPU, logs/final_tests.log |
| Все12 небольших videos | Проверено | полный ffmpeg decode, frame counts, SHA, row metadata и exact seeds10000/10001 perarm/task, final_evaluation_audit.json |
| GitHub code/config/tests/manifests/lock/report/results/videos/logs | Ожидает публикации | local artifacts готовы; scripts/publish.py staged allowlist/path/secret/symlink checks; repo qrvmil/pi0-d0d1-visual-alignment |
| HF все10checkpoints/projector/norm/metadata/config/Russian card | Ожидает публикации | publisher explicit checkpoint inventory/fresh export/hashes, запрещает data/cache/upstreamweights/secrets; remote proofs ещё отсутствуют |
| Cross-links/article/final release/download accessibility | Ожидает публикации | README crosslinks готовы; publisher проверит public HF inventory/LFS hashes/full small downloads/ranges forweights, GitHub report bytes/release assets |
| Финальные URLs/commit/tag/checkpoint list | Ожидает публикации | logs/publication.json будет создан только после remote checks; goal complete только после него |

Независимая read-only review подтвердила числа, all3001 input hashes и paired cohorts, указала fresh-clone reuse bug. Он исправлен архивированием published runtime metadata перед новым запуском; существующие actual checkpoints не затрагивались. Remote publication ещё не выполнена.
