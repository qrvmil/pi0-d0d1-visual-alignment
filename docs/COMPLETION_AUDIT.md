# Проверка полного запроса — до завершения

Это промежуточный audit, не утверждение о завершении. Авторитетные результаты — перечисленные файлы и живые процессы; поле «ожидается» требует повторной проверки после соответствующей стадии.

| Требование | Текущий статус | Доказательство / оставшаяся проверка |
|---|---|---|
| Прочитать исходный запрос и статью, зафиксировать перенос/адаптации | Проверено | REQUEST_RU.txt, DESIGN_RU.md; статья §6 и §8.1–8.5; LN+MLP/cosine/middle teacher transfer, flow objective/two-view grids адаптированы |
| Hardware/CUDA/JAX/GPU matmul/OpenPI forward-backward до datasets | Проверено на фактическом A100 | logs/compatibility_smoke.json и .log; CUDA runtime12080. Blackwell отсутствует и не объявлен проверенным |
| Source/OpenPI/LIBERO/HF reference/teacher revisions | Проверено при setup | configs/pins.json, base_inventory.json, base_manifest.json, environment.lock.txt; повторить git HEAD checks перед публикацией |
| Только D0+D1 в training/calibration/normalization, D2 без debugging/model selection | Проверено для текущих стадий | Corpus require_seen, source HDF5 BDDL identities, configs/dataset_manifest.json и normalization_provenance.json; tests/test_data.py, test_corpus.py |
| Две камеры, K=4, fp16 sharded features, IDs/seed/params/manifest/checksums | Проверено | logs/cache_build.json:11745 samples/46980 entries; configs/cache_manifest.json, private/cache shards (не публиковать) |
| Exact RGB replay и отсутствие extra random augmentation | Проверено для consumed batches | Batches.get сравнивает params и image SHA каждого view; compute_loss train=False; replay/cache tests |
| Dummy/masked cameras исключены; middle only, spatial/token correspondence | Проверено | tests/test_model.py; logs/model_integration.json: exact dummy invariance, actual π0 middle relativeRMSE0.00395654 в заранее заданном bf16 tolerance0.005; no hidden-state layer stack |
| Teacher frozen | Проверено с ограниченным scope state equality | requires_gradFalse/eval/inference_mode, отсутствует optimizer; exact state comparison в 2-sample smoke. Full-cache state equality не измерялась и помечена null |
| Projector сначала обучен с frozen π0, потом frozen | Проверено для calibration/pilot/current control | logs/projector_calibration.json:200 updates, policy SHA unchanged; configs/projector.json, projector file SHA; main end-digest checks прошли для обоих arms: B-control_audit.json и B-align_audit.json |
| Pilot λ0.2 / flow+align losses / gradient ratio / preregistered change rule | Проверено | logs/pilot.json:20 updates; configs/lambda_decision.json: median0.0154101, selected0.2; rule записан до результатов |
| 2-sample cache и 1–5 real training steps до main runs | Проверено | logs/cache_smoke.json, data_training_smoke.json, model_integration.json |
| Finite loss/gradients и effective batch при OOM | Оба main runs завершены и проверены | configs/memory_recipe.json: micro8/accum1/effective8, no OOM; finite gates каждой итерации. Control JAX peak allocation/pool измерены; NVML host/container PID mismatch исправлен как unavailable, не ноль; align partial-window host-PID monitor live |
| Одинаковая инициализация B-control/B-align | Проверено | logs/paired_training_prefix_audit.json: оба independent-load SHA43f7c62a совпадают с calibration source; initial flow difference2.235e-7 в prewritten tolerance1e-5 |
| Одинаковые optimizer/schedule/seeds/order/augmentation/effectivebatch | Все3001 paired batches проверены | training recipes/tests; checkpoint500_audit: exact4D0+4D1 per batch. Все3001 batch/sample/image hashes equal в logs/B-align_audit.json; independent initialization и recipes проверены |
| Resume после остановки | Проверено | logs/fresh_process_resume.json: fresh GPU process restores real smoke step2 and continues step3; losses exact/reference gradnorm within prewritten tolerance. Main500 train_state separately restored onCPU |
| Сохранить все scheduled checkpoint steps | Проверено локально; публикация ожидается | Оба arms0/500/1000/2000/3001 durable, metadata/norm/bindings audited в logs/B-control_audit.json и B-align_audit.json |
| Paired reset seeds и common flow-noise stream | Ожидается | evaluation binding/cohort/reset/noise validation реализованы; новых rollouts нет |
| D0/D1/D2 first50, D0 prespecified extension500 без cherry-picking | Ожидается | seeds10000–10049; extension10000–10499 по original one-sided bound. D0 displayed two-sided CI поправлен на2looks до любых новых rollouts |
| Δ/95% CI/D0 strict>-0.02 criterion/D2 OOD outcome/не объявлять неподтверждённый успех | Ожидается | stats exact CP+Bonferroni tests; реальные evaluation/results.json ещё отсутствуют |
| Короткий REPORT_RU.md с таблицей/reference/status/direct answer/config/n/time/VRAM/cost/plots/commands | Ожидается после eval | scripts/report.py; B-original_historical.json из pinned source; стоимость по readonly rate1.1854347USD/h, timeframe фиксируется на финальном evaluation |
| GitHub весь созданный код/configs/tests/manifests/locks/results/videos/logs | Ожидается | Новый local repo; staged allowlist, forbidden-path/secret checks реализованы; remote repo создать при публикации |
| HF все созданные checkpoints/projector/norm/metadata/configs/Russian card | Ожидается | scripts/publish.py: fresh export из explicit checkpoint inventory + vetted index paths; original base/teacher/data/cache не экспортировать |
| README/HF cross-links, release и доступные скачиваемые файлы | Ожидается | Publisher checks remote inventory/LFS SHA, small artifact downloads и anonymous range downloads больших weights; release/download proof ещё отсутствует |
| Финальные GitHub/HF URLs, commit SHA, release tag, checkpoint list | Ожидается | Только после проверенной публикации; logs/publication.json |
