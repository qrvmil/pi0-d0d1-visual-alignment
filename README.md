# π0 D0+D1 Visual Alignment

Воспроизводимый causal experiment: новый B-control vs B-align из одинаковых π0 weights, frozen C-RADIO spatial teacher и LN+MLP projector. D0/D1 — training; D2 — held-out OOD.

[Статья](https://arxiv.org/html/2510.25616) · [HF checkpoints](https://huggingface.co/qrvmil/pi0-d0d1-visual-alignment) · [Запрос](docs/REQUEST_RU.txt) · [Заранее заданный протокол](docs/DESIGN_RU.md)

Обучение и paired evaluation завершены: [REPORT_RU.md](REPORT_RU.md), [результаты](evaluation/results.json), [CSV](evaluation/summary.csv). D2: обе модели 0/50; D0: 81,6% vs 80,2%, сохранение в пределах 2 п.п. статистически не подтверждено. Статусы всех критериев — «НЕОПРЕДЕЛЁН». Числа подтверждены JSON и логами в `logs/`. Текущая машина — **A100 80GB**, исходное описание RTX6000D Blackwell не соответствует `nvidia-smi`. CUDA12.8 + JAX0.5.3 smoke пройден на A100; Blackwell не проверен.

Исходники: `qrvmil/maniskill_myws@c6b77f51e3e5bd8873b812d258744c43380fdfeb`; OpenPI и LIBERO pins, teacher и dataset revisions в [pins](configs/pins.json). Python311 и полный [environment lock](environment.lock.txt); upstream uv lock приложен. Репозиторий эксперимента предполагает путь `/root/pi0-d0d1-visual-alignment`.

```bash
git clone https://github.com/qrvmil/pi0-d0d1-visual-alignment.git /root/pi0-d0d1-visual-alignment
cd /root/pi0-d0d1-visual-alignment
python scripts/prepare_reproduction.py
bash scripts/setup.sh
source scripts/env.sh
$ALIGN_PY -m pytest -q
$ALIGN_PY -u -m alignment.smoke > logs/compatibility_smoke.log 2>&1
$ALIGN_PY -u scripts/fetch.py > logs/fetch.log 2>&1
$ALIGN_PY -u -m alignment.build_cache --samples 2 > logs/cache_smoke.log 2>&1
$ALIGN_PY -u -m alignment.train smoke > logs/data_training_smoke.log 2>&1
$ALIGN_PY -u scripts/run_pipeline.py
```

Для **нового запуска из fresh clone** команда `prepare_reproduction.py` переносит опубликованные результаты и runtime-конфиги в `private/published_results/`, сохраняя pins/lock, immutable GCS generations в `base_inventory.json` и B-original reference. Она отказывается работать при наличии локальных private-данных/checkpoints: существующий эксперимент не очищается. Выполняйте её до setup/download/smoke; для resume существующего запуска эта команда не нужна. Smoke-команды выполняются один раз в новом рабочем каталоге; повторный smoke поверх существующего checkpoint не является resume.

В `private/` находятся datasets, исходные π0/C-RADIO weights, teacher cache, frozen projector и созданные checkpoints; эта директория исключена из git. Код публикации загружает в HF **только созданные** B-control/B-align checkpoints и projector, normalization assets/configs/model card. Dataset, teacher cache и исходные upstream checkpoints не публикуются.

Восстановление основного training run: повторить `$ALIGN_PY -u -m alignment.train B-control` или `B-align`. Загружается последний сохранённый optimizer checkpoint, data order восстанавливается по счётчику update, RNG по seed/update/microbatch. Scheduled steps0/500/1000/2000/3001 сохраняются все. После остановки между scheduled checkpoints повторяется только хвост после последнего durable checkpoint.

D0/D1/D2 evaluation запускается только после завершения обоих runs. Первые50 seeds10000–10049; заранее заданное расширение D0 до500 seeds10000–10499. Проверяются hashes reset states и совпадение common prefix flow-noise stream. D0: simultaneous односторонняя95% нижняя граница>−0.02. D2: paired95% CI, superiority только если lower>0. Консервативные exact CP bounds не дают ложную нулевую ширину при отсутствии discordant pairs.

B-original — исторический reference step3001, n50 из pinned upstream таблицы, отдельно помечен как не-causal control. Нормализация новых runs рассчитана только по D0+D1, K=4 image augmentation replay общий для обеих моделей и teacher. Masked right-wrist slot исключён из alignment.

Публикация после финального аудита: `$ALIGN_PY scripts/publish.py`. Авторизация при необходимости: `gh auth login` и `hf auth login`; обучение повторять не требуется.
