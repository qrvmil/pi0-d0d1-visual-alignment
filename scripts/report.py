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
    return f'Фактическая ставка аренды${rate:.3f}/h ([billing log](logs/cost_rate.json)); от первого compatibility smoke до финального evaluation{hours:.2f}h, примерно${value:.2f}. Это оплачиваемое walltime инстанса, включая ожидание/подготовку между стадиями; начальный setup и публикация исключены.'


def main():
    r=json.loads((ROOT/'evaluation/results.json').read_text())
    original=json.loads((ROOT/'evaluation/B-original_historical.json').read_text())
    original={x['task']:x for x in original['rows']}
    metrics={}
    for arm in ('B-control','B-align'):
        metrics[arm]=json.loads((ROOT/'logs'/f'{arm}_complete.json').read_text())
        if metrics[arm]['steps']!=3001:raise ValueError('Training incomplete')
    figures=ROOT/'figures';figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for arm in metrics:
        rows=[json.loads(s) for s in (ROOT/'logs'/f'{arm}.jsonl').read_text().splitlines()]
        steps=np.array([x['step'] for x in rows]);flow=np.array([x['flow'] for x in rows]);align=np.array([x['align'] for x in rows])
        # Show all measured points, plus an explicit rolling mean.
        for ax,y,label in [(axes[0],flow,'L_flow'),(axes[1],align,'L_align')]:
            ax.plot(steps,y,alpha=.2,lw=.4)
            width=min(50,len(y));avg=np.convolve(y,np.ones(width)/width,mode='valid')
            ax.plot(steps[width-1:],avg,label=arm)
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
                sr=k/len(rows);lo,hi=cp(k,len(rows),.025)
            means.append(sr*100);lower.append((sr-lo)*100);upper.append((hi-sr)*100)
        ax.errorbar(xx+(j-1)*.15,means,yerr=[lower,upper],fmt='o',capsize=4,label=arm)
    ax.set_xticks(xx,['D0','D1','D2 OOD']);ax.set_ylabel('Success rate, % (95% CI)');ax.set_ylim(-4,104)
    ax.legend();fig.tight_layout();fig.savefig(figures/'success_rates.png',dpi=160);plt.close(fig)
    def pct(x):return f'{100*x:.1f}'
    d0=r['D0'];d2=r['D2']
    helped='доказанное улучшение' if d2['status']=='ПРОЙДЕН' else 'улучшение не доказано'
    preserved='сохранение в пределах2 п.п. подтверждено' if d0['status']=='ПРОЙДЕН' else 'сохранение в пределах2 п.п. не подтверждено'
    lines=[f'# Visual alignment π0 / LIBERO', '',
        f'Alignment на D2: {helped}; Δ={pct(d2["delta"])} п.п., 95% CI [{pct(d2["ci95"][0])}; {pct(d2["ci95"][1])}]. '
        f'D0: {preserved}; Δ={pct(d0["delta"])} п.п., односторонняя нижняя simultaneous95% граница {pct(d0["one_sided95_lower_simultaneous"])} п.п. '
        'Обе новые модели обучены на одинаковых D0+D1 из одной исходной π0; выбран заранее последний update3001. '
        'D2 не участвовала в обучении, calibration, λ, normalization, debugging или выборе checkpoint.', '',
        '| Task | B-original, reference | B-control | B-align | Δ, п.п. | paired95% CI, п.п. | Статус |',
        '|---|---:|---:|---:|---:|---|---|']
    for t,v in r.items():lines.append(f'|{t}, n={v["n"]}|{pct(float(original[t]["success_rate"]))}% (50)|{pct(v["control_sr"])}%|{pct(v["align_sr"])}%|{pct(v["delta"])}|[{pct(v["ci95"][0])}; {pct(v["ci95"][1])}]|{v["status"]}|')
    lines.extend(['',f'**Прямой ответ:** D2 — {helped}. D0 — {preserved}. Статистически неопределённые результаты не объявляем успехом.',
        'B-original — исторический reference из pinned source, а не causal control. D0 критерий: нижняя граница строго>−2 п.п.; D2: нижняя paired95% CI строго>0. D1 показана как exploratory superiority.',
        '', 'Обучены новые B-control и B-align: LoRA32 в PaliGemma/action expert плюс параметры, которые upstream low-memory recipe оставляет trainable (включая SigLIP). Отличие arms — frozen-projector cosine regularizer. '
        'Перенесены frozen C-RADIO, middle backbone patch alignment и предварительно обученный LN+MLP с последующим замораживанием. Адаптации: flow matching вместо CE, девятый PaliGemma block, 2 камеры, bilinear14×14→16×16. '
        'Middle state вычисляется prefix-only первыми9 блоками; тест доказывает совпадение с prefix state при action suffix, все слои не материализуются.',
        '',f'Конфигурация: seed0, 3001 updates, effectivebatch8; {json.loads((ROOT/"configs/memory_recipe.json").read_text())}; '
        f'λ={json.loads((ROOT/"configs/lambda_decision.json").read_text())["selected"]}. Projector200 updates, pilot20, K=4. '
        f'Демонстрации:50 D0+50 D1; {json.loads((ROOT/"configs/normalization_provenance.json").read_text())["frames"]} кадров. '
        'Нормализация all-seen, chunk-weighted, next-action shift; отличается от исторического pipeline обработкой последнего неполного batch, это зафиксированная адаптация.',
        '', '| Run | Wall time (process), h | Peak VRAM (NVML), GiB | Peak JAX allocations, GiB |', '|---|---:|---:|---:|'])
    for arm,v in metrics.items():
        memory=json.loads((ROOT/'logs'/f'{arm}_vram.json').read_text())
        lines.append(f'|{arm}|{memory["wall_seconds"]/3600:.2f}|{memory["peak_process_vram_bytes"]/2**30:.2f}|{v["memory_stats"]["peak_bytes_in_use"]/2**30:.2f}|')
    lines.extend(['', 'GPU: фактически A100-SXM4-80GB. Blackwell smoke не проверен на этой машине. Driver580.159.03; JAX0.5.3, CUDA runtime12.8. '+cost_text(),
        '', '![Losses](figures/losses.png)', '![Success rates](figures/success_rates.png)', '',
        'Проблемы: CUDA12.6 обновлена до12.8; локальный HF dynamic loader и per-tensor loading RADIO несовместимы с выбранным Transformers, использована прямая загрузка полного pinned safetensors state_dict с strict=True. '
        'Доверительные границы консервативные exact CP с marginal Bonferroni; D0 error budget распределён на2 заранее заданных просмотра. '
        f'При n50 интервал может быть широким. D0: фактически{d0["n"]} пар; расширение до500 '+('выполнено по заранее записанному правилу.' if d0['n']==500 else 'не потребовалось по заранее записанному правилу.')+' Single trainingseed ограничивает перенос вывода на другие seeds.',
        '', 'Все числа: [paired results](evaluation/results.json), [CSV](evaluation/summary.csv), [logs](logs/), [pins](configs/pins.json), [дизайн](docs/DESIGN_RU.md). '
        'Исторический reference: [pinned table](https://github.com/qrvmil/maniskill_myws/blob/c6b77f51e3e5bd8873b812d258744c43380fdfeb/docs/multitask_sft/success_rates.csv). '
        '[Статья](https://arxiv.org/html/2510.25616).',
        '', 'Воспроизведение (полные команды и environment lock в README):', '```bash',
        'bash scripts/setup.sh', 'source scripts/env.sh', '$ALIGN_PY -m pytest -q', '$ALIGN_PY -m alignment.smoke',
        '$ALIGN_PY scripts/fetch.py', '$ALIGN_PY -m alignment.build_cache --samples 2',
        '$ALIGN_PY -m alignment.train smoke', 'bash scripts/pipeline.sh', '```'])
    (ROOT/'REPORT_RU.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
