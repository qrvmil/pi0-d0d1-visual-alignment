"""Produce the Russian result report only from complete paired evaluation artifacts."""
import json
import datetime
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from alignment.stats import cp

ROOT=Path(__file__).resolve().parents[1]


def interval_errors(rate,lower,upper):
    eps=1e-12
    if not np.isfinite([rate,lower,upper]).all() or not 0<=rate<=1 or lower < -eps or upper > 1+eps or lower>upper or lower>rate+eps or upper<rate-eps:
        raise ValueError('Success-rate interval invalid')
    return max(0.,rate-lower)*100,max(0.,upper-rate)*100


def cost_text():
    path=ROOT/'logs/cost_rate.json'
    if not path.exists():return 'Цена аренды недоступна: стоимость=walltimeGPU×фактическая ставка$/h.'
    rate=json.loads(path.read_text())['hourly_usd']
    smoke=json.loads((ROOT/'logs/compatibility_smoke.json').read_text())
    started=datetime.datetime.strptime(smoke['hardware'].splitlines()[0].strip(),'%a %b %d %H:%M:%S %Y').replace(tzinfo=datetime.timezone.utc).timestamp()
    finished=json.loads((ROOT/'logs/evaluation_complete.json').read_text())['finished_unix']
    hours=(finished-started)/3600
    value=hours*rate
    record=dict(hourly_usd=rate,wall_hours=hours,approx_usd=value,scope='first compatibility smoke to final paired evaluation; instance walltime, excludes initial setup and publication')
    (ROOT/'logs/run_cost.json').write_text(json.dumps(record,indent=2))
    return f'Фактическая ставка аренды: ${rate:.3f}/час ([billing log](logs/cost_rate.json)); от первого compatibility smoke до финального evaluation: {hours:.2f} ч, примерно ${value:.2f}. Это оплачиваемое walltime инстанса, включая ожидание/подготовку между стадиями; начальный setup и публикация исключены.'


def main():
    r=json.loads((ROOT/'evaluation/results.json').read_text())
    original=json.loads((ROOT/'evaluation/B-original_historical.json').read_text())
    original={x['task']:x for x in original['rows']}
    memory_recipe=json.loads((ROOT/'configs/memory_recipe.json').read_text())
    metrics={}
    for arm in ('B-control','B-align'):
        metrics[arm]=json.loads((ROOT/'logs'/f'{arm}_complete.json').read_text())
        if metrics[arm]['steps']!=3001:raise ValueError('Training incomplete')
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    colors={'B-original':'#777777','B-control':'#1f77b4','B-align':'#ff7f0e'}
    for arm in metrics:
        rows=[json.loads(s) for s in (ROOT/'logs'/f'{arm}.jsonl').read_text().splitlines()]
        steps=np.array([x['step'] for x in rows]);flow=np.array([x['flow'] for x in rows]);align=np.array([x['align'] for x in rows])
        # Show all measured points, plus an explicit rolling mean.
        series=[(axes[0],flow,'L_flow')]
        if arm=='B-align':series.append((axes[1],align,'L_align'))
        for ax,y,label in series:
            ax.plot(steps,y,alpha=.2,lw=.4,color=colors[arm])
            width=min(50,len(y));avg=np.convolve(y,np.ones(width)/width,mode='valid')
            ax.plot(steps[width-1:],avg,label=arm,color=colors[arm])
            ax.set_xlabel('Optimizer update');ax.set_ylabel(label);ax.legend()
    fig.tight_layout();fig.savefig(figures/'losses.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));xx=np.arange(3)
    for j,arm in enumerate(('B-original','B-control','B-align')):
        means=[];lower=[];upper=[]
        for task in ('D0','D1','D2'):
            if arm=='B-original':
                row=original[task];sr=float(row['success_rate']);lo=float(row['ci_low']);hi=float(row['ci_high'])
            else:
                rows=json.loads((ROOT/'evaluation'/arm/task/'episodes.json').read_text());k=sum(bool(x['success']) for x in rows)
                sr=k/len(rows);lo,hi=cp(k,len(rows),.0125 if task=='D0' else .025)
            error_low,error_high=interval_errors(sr,lo,hi)
            means.append(sr*100);lower.append(error_low);upper.append(error_high)
        ax.errorbar(xx+(j-1)*.15,means,yerr=[lower,upper],fmt='o',capsize=4,label=arm,color=colors[arm])
    ax.set_xticks(xx,['D0','D1','D2 OOD']);ax.set_ylabel('Success rate, % (95% CI)');ax.set_ylim(-4,104)
    ax.legend();fig.tight_layout();fig.savefig(figures/'success_rates.png',dpi=160);plt.close(fig)
    def pct(x):return f'{100*x:.1f}'
    d0=r['D0'];d2=r['D2']
    helped='доказанное улучшение' if d2['status']=='ПРОЙДЕН' else 'улучшение не подтверждено (критерий не пройден)' if d2['status']=='НЕ ПРОЙДЕН' else 'результат статистически неопределён; улучшение не доказано'
    preserved='сохранение в пределах 2 п.п. подтверждено' if d0['status']=='ПРОЙДЕН' else 'сохранение в пределах 2 п.п. не подтверждено'
    lines=[f'# Visual alignment π0 / LIBERO', '',
        f'Alignment на D2: {helped}; Δ={pct(d2["delta"])} п.п., 95% CI [{pct(d2["ci95"][0])}; {pct(d2["ci95"][1])}]. '
        f'D0: {preserved}; Δ={pct(d0["delta"])} п.п., односторонняя нижняя 95% с поправкой на два просмотра граница {pct(d0["one_sided95_lower_simultaneous"])} п.п. '
        'Обе новые модели обучены на одинаковых D0+D1 из одной исходной π0; выбран заранее последний update 3001. '
        'D2 не участвовала в обучении, calibration, λ, normalization, debugging или выборе checkpoint.', '',
        '| Task | B-original, reference | B-control | B-align | Δ, п.п. | paired 95% CI, п.п. | Статус |',
        '|---|---:|---:|---:|---:|---|---|']
    for t,v in r.items():lines.append(f'|{t}, n={v["n"]}|{pct(float(original[t]["success_rate"]))}% (50)|{pct(v["control_sr"])}%|{pct(v["align_sr"])}%|{pct(v["delta"])}|[{pct(v["ci95"][0])}; {pct(v["ci95"][1])}]|{v["status"]}|')
    lines.extend(['',f'**Прямой ответ:** D2 — {helped}. D0 — {preserved}. Статистически неопределённые результаты не объявляем успехом.',
        'B-original — исторический reference из pinned source, а не causal control. D0 критерий: нижняя граница строго>−2 п.п.; D2: нижняя paired 95% CI строго>0. D1 показана как exploratory superiority.',
        '', 'Обучены новые B-control и B-align: LoRA32 в PaliGemma/action expert плюс параметры, которые upstream low-memory recipe оставляет trainable (включая SigLIP). Отличие arms — frozen-projector cosine regularizer. '
        'Перенесены frozen C-RADIO, middle backbone patch alignment и предварительно обученный LN+MLP с последующим замораживанием. Адаптации: flow matching вместо CE, девятый блок PaliGemma, 2 камеры, bilinear 14×14 → 16×16. '
        'Middle state вычисляется prefix-only первыми 9 блоками; Интеграционная проверка π0 измерила relative RMSE 0,00396 при заранее заданном BF16 допуске 0,005 ([лог](logs/model_integration.json)); все слои не материализуются.',
        '',f'Конфигурация: seed обучения 0, 3001 обновление, effective batch 8; microbatch {memory_recipe["microbatch"]}, accumulation {memory_recipe["accumulation"]}, OOM событий {len(memory_recipe["oom_events"])}. AdamW: β₁=0,9, β₂=0,95, clip=1; cosine schedule, warmup 1000, peak LR 2,5·10⁻⁵, decay 30000, min LR 2,5·10⁻⁶; [полный конфиг](configs/B-align.json). '
        f'λ={json.loads((ROOT/"configs/lambda_decision.json").read_text())["selected"]}. Projector: 200 updates; pilot: 20 updates; K=4. '
        f'Демонстрации: 50 D0 + 50 D1; {json.loads((ROOT/"configs/normalization_provenance.json").read_text())["frames"]} кадров. '
        'Нормализация all-seen, chunk-weighted, next-action shift; отличается от исторического pipeline обработкой последнего неполного batch, это зафиксированная адаптация.',
        '', f'Evaluation: {2*sum(v["n"] for v in r.values())} новых эпизодов. D0 — {d0["n"]} пар, seeds 10000–{10000+d0["n"]-1}; D1/D2 — по 50 пар, seeds 10000–10049. Reset states и общий префикс flow-noise совпадают внутри каждой пары.',
        '', '| Run | Длительность процесса, ч | NVML peak, GiB | JAX allocations peak, GiB | JAX pool peak, GiB |', '|---|---:|---:|---:|---:|'])
    for arm,v in metrics.items():
        memory=json.loads((ROOT/'logs'/f'{arm}_vram.json').read_text())
        nvml='недоступен (PID namespace)'
        attached=ROOT/'logs'/f'{arm}_attached_vram.json'
        if attached.exists():
            observed=json.loads(attached.read_text())
            if observed['peak_process_vram_bytes'] is not None:nvml=f"{observed['peak_process_vram_bytes']/2**30:.2f} (частичное окно)"
        elif memory['peak_process_vram_bytes']:
            nvml=f"{memory['peak_process_vram_bytes']/2**30:.2f}"
        lines.append(f'|{arm}|{memory["wall_seconds"]/3600:.2f}|{nvml}|{v["memory_stats"]["peak_bytes_in_use"]/2**30:.2f}|{v["memory_stats"]["peak_pool_bytes"]/2**30:.2f}|')
    lines.extend(['', 'GPU: фактически A100-SXM4-80GB. Blackwell smoke не проверен на этой машине. Driver 580.159.03; JAX 0.5.3, CUDA runtime 12.8. '+cost_text(),
        '', 'L_align показана для B-align; в B-control этот loss не вычисляется. Интервалы SR на графике: exact CP; D0 с поправкой на два предусмотренных просмотра, D1/D2 fixed-cohort95%; исторический reference использует исходные Wilson95%.', '![Losses](figures/losses.png)', '![Success rates](figures/success_rates.png)', '',
        'Проблемы: NVML host PID отличается от PID контейнера; первоначальная per-process серия возвращала ложные нули и признана недоступной. JAX allocations/pool peaks измерены за весь процесс; B-align NVML probe подключён после первых updates и помечен как частичное окно, не полный peak всего run. '
        'CUDA 12.6 обновлена до 12.8; локальный HF dynamic loader и per-tensor loading RADIO несовместимы с выбранным Transformers, использована прямая загрузка полного pinned safetensors state_dict с strict=True. '
        'При первом D0 rollout отсутствовал system ffmpeg: видео валидировано после установки, эпизод сохранён; JSON tuple/list сравнение resume исправлено с сохранением строгой проверки binding ([лог](logs/evaluation_dependency_recovery.json)). '
        'Доверительные границы — консервативные exact CP с marginal Bonferroni. D0: односторонний критерий и двухсторонний CI учитывают 2 заранее заданных просмотра; D1/D2: fixed50. '
        f'При n=50 интервал может быть широким. D0: фактически {d0["n"]} пар; расширение до 500 '+('выполнено по заранее записанному правилу.' if d0['n']==500 else 'не потребовалось по заранее записанному правилу.')+' Один seed обучения ограничивает перенос вывода на другие seeds.',
        '', 'Первые 50 пар по всем задачам: [архив результатов](evaluation/first50/results.json), [CSV](evaluation/first50/summary.csv). '
        'Все числа: [paired results](evaluation/results.json), [CSV](evaluation/summary.csv), [logs](logs/), [pins](configs/pins.json), [дизайн](docs/DESIGN_RU.md). '
        'Исторический reference: [pinned table](https://github.com/qrvmil/maniskill_myws/blob/c6b77f51e3e5bd8873b812d258744c43380fdfeb/docs/multitask_sft/success_rates.csv). '
        '[Статья](https://arxiv.org/html/2510.25616).',
        '', 'Воспроизведение (полные команды и environment lock в README):', '```bash',
        'python scripts/prepare_reproduction.py', 'bash scripts/setup.sh', 'source scripts/env.sh', '$ALIGN_PY -m pytest -q', '$ALIGN_PY -m alignment.smoke',
        '$ALIGN_PY scripts/fetch.py', '$ALIGN_PY -m alignment.build_cache --samples 2',
        '$ALIGN_PY -m alignment.train smoke', '$ALIGN_PY -u scripts/run_pipeline.py', '```'])
    (ROOT/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
