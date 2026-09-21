#!/usr/bin/env python3
"""Saved-data-only breakdowns from paired same-call resident GPU captures."""
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import cjxl_gjxl_stage_breakdown as legacy


GPU_NAMES = tuple('GPU: ' + name for name in legacy.GPU_GROUPS)
PIPELINE_REMAINDER = 'Pipeline orchestration / gaps'
OUTER = 'Outer publication / teardown'
STAGES = (*legacy.INPUT_GROUPS, 'Other input preparation', *GPU_NAMES,
          PIPELINE_REMAINDER, *legacy.HOST_GROUPS['serializer'], 'Other serializer',
          'Other workflow', OUTER)
KINDS = ('flat', 'scaled', 'gpu')
SEMANTICS = {
    'flat': 'Same-call host timers and nonoverlapping GPU intervals. Replace the quantization '
            'parent with measured GPU stages plus an explicit elapsed-time residual. '
            'The partition sums to the externally measured complete profiled API call.',
    'scaled': 'Estimated attribution: multiply every flat segment by the paired ordinary / '
              'profiled complete-call ratio before averaging. This preserves the ordinary '
              'total, but does not correct stage-specific instrumentation bias.',
    'gpu': 'GPU timestamp intervals from the same profiled calls. Empty indirect stages '
           'with invalid timestamps are verified zero work. Input preparation remains '
           'in its measured host wall-time segment; no GPU interval is invented for it.',
    'aggregation': 'Average samples within each image/setting, then give each tuple equal '
                   'weight. Percentages divide summed means. Require all declared tuples.',
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def partition(row, sample):
    p = row['phase_nanoseconds']
    flat = legacy.host_partitions(dict(backend='metal', phase_nanoseconds=p))['flat']
    parent = flat.pop(legacy.UNRESOLVED_PIPELINE)
    exact = legacy.gpu_stages(sample)
    grouped = dict.fromkeys(GPU_NAMES, 0)
    for name, ns in exact.items():
        grouped['GPU: ' + legacy.gpu_group(name)] += ns
    remaining = parent - sum(grouped.values())
    outer = row['complete_call_nanoseconds'] - p['total']
    if remaining < 0 or outer < 0:
        raise ValueError('Same-call GPU/host partition exceeds its parent')
    flat.update(grouped)
    flat[PIPELINE_REMAINDER] = remaining
    flat[OUTER] = outer
    if abs(sum(flat.values()) - row['complete_call_nanoseconds']) > 1:
        raise ValueError('Breakdown does not sum to the complete profiled call')
    return flat, exact


def load_profiles(config_path):
    path = Path(config_path).expanduser().resolve()
    root = path.parent
    c = read(path)
    if c.get('capture_kind') != 'paired-same-call-v1' or c.get('schema_version') != 1:
        raise ValueError('Expected paired same-call capture configuration')
    if c['samples'] < 1 or c['samples'] % 2 or c['cpu_threads'] != 8 or c['collect_final_score']:
        raise ValueError('Unsupported capture protocol')
    for name, sha in c['frozen_sha256'].items():
        if digest(root/name) != sha:
            raise ValueError('Frozen capture artifact changed: ' + name)
    if digest(root/'bin/capture') != c['binary_sha256']:
        raise ValueError('Capture binary mismatch')
    images = {i['image_id']: i for i in c['images']}
    if len(images) != len(c['images']) or len({i['input_path'] for i in images.values()}) != len(images):
        raise ValueError('Duplicate image identity')
    for image in images.values():
        if digest(image['input_path']) != image['input_sha256']:
            raise ValueError('Input hash mismatch: ' + image['image_id'])
    expected = {(image, setting, effort) for image in images
                for setting in c['settings'] for effort in c['efforts']}
    jobs = {(j['image_id'], j['setting'], j['effort']): j for j in c['jobs']}
    if len(jobs) != len(c['jobs']) or set(jobs) != expected:
        raise ValueError('Job grid does not cover the declared cohort exactly')
    rows, exact_rows, overhead, rejected, ready = [], [], [], [], set()
    for key, j in jobs.items():
        case = root/'cases'/j['case_id']
        if not (case/'complete.json').is_file():
            continue
        done = read(case/'complete.json')
        attempt = case/done['attempt']
        summary = done['summary']
        for name in ('gpu', 'paired'):
            if digest(attempt/(name+'.json')) != summary[name+'_sha256']:
                raise ValueError('Raw capture hash mismatch: ' + j['case_id'])
        if digest(attempt/'command.json') != done['command_sha256']:
            raise ValueError('Command hash mismatch')
        command = read(attempt/'command.json')
        argv = command['argv']
        if command['returncode'] != 0 or command['interference']:
            raise ValueError('Rejected collection attempt marked complete')
        for flag, value in (('--input', j['input_path']), ('--effort', str(j['effort'])),
                            ('--cpu-threads', str(c['cpu_threads'])), ('--samples', str(c['samples'])),
                            ('--warmups', str(c['warmups'])), ('--distance', str(j['distance']))):
            if argv.count(flag) != 1 or argv[argv.index(flag)+1] != value:
                raise ValueError('Command does not match the declared setting')
        gpu, paired = read(attempt/'gpu.json'), read(attempt/'paired.json')
        if (gpu.get('execution_path') != 'production-aligned-resident-v1' or
                gpu.get('mode') != 'stage' or gpu.get('schema_version') != 4 or
                gpu.get('gpu_aq') != 'fully-resident' or gpu.get('collect_final_score') is not False or
                paired.get('capture_kind') != 'paired-same-call-v1'):
            raise ValueError('Historical or incompatible profiler data')
        if (gpu['sample_count'] != c['samples'] or gpu['warmups'] != c['warmups'] or
                not np.isclose(gpu['distance'], j['distance'], rtol=1e-6) or len(gpu['workloads']) != 1):
            raise ValueError('Profile settings mismatch')
        work = gpu['workloads'][0]
        image = images[j['image_id']]
        for item in (work, paired):
            if (item['source_width'], item['source_height']) != (image['width'], image['height']):
                raise ValueError('Profile geometry mismatch')
        raw = paired['rows']
        keys = {(r['sample_index'], r['mode']) for r in raw}
        expected_keys = {(i, m) for i in range(-c['warmups'], c['samples'])
                         for m in ('ordinary', 'profiled')}
        if keys != expected_keys or len(raw) != len(keys):
            raise ValueError('Incomplete paired timing rows')
        if (not all(r['byte_equal'] is True and r['summary_equal'] is True for r in raw)
                or len({r['committed_submissions'] for r in raw}) != 1):
            raise ValueError('Output or submission equivalence failed')
        paired_rows = {(r['sample_index'], r['mode']): r for r in raw}
        samples = work['samples']
        if len(samples) != c['samples'] or {s['sample_index'] for s in samples} != set(range(c['samples'])):
            raise ValueError('Missing or duplicate GPU sample')
        case_rows, case_exact, case_overhead = [], [], []
        try:
            for sample in samples:
                i = sample['sample_index']
                row = paired_rows[i, 'profiled']
                ordinary = legacy.duration(paired_rows[i, 'ordinary']['complete_call_nanoseconds'])
                total = legacy.duration(row['complete_call_nanoseconds'])
                if ordinary <= 0 or total <= 0:
                    raise ValueError('Empty complete-call timing')
                flat, exact = partition(row, sample)
                base = dict(image_id=j['image_id'], setting=j['setting'], effort=j['effort'],
                            resolution_class=image['resolution_class'], sample_index=i)
                for kind, values in (('flat', flat), ('scaled', {s: v*ordinary/total for s,v in flat.items()})):
                    case_rows.extend(dict(base, kind=kind, stage=s, ms=values.get(s, 0)/1e6) for s in STAGES)
                grouped = dict.fromkeys(legacy.GPU_GROUPS, 0)
                for name, ns in exact.items():
                    grouped[legacy.gpu_group(name)] += ns
                    case_exact.append(dict(base, stage=name, ms=ns/1e6))
                case_rows.extend(dict(base, kind='gpu', stage=s, ms=v/1e6) for s,v in grouped.items())
                case_overhead.append(dict(base, ordinary_ms=ordinary/1e6, profiled_ms=total/1e6,
                    delta_ms=(total-ordinary)/1e6, overhead_percent=100*(total/ordinary-1)))
        except ValueError as error:
            rejected.append(dict(image_id=j['image_id'], effort=j['effort'], reason=str(error)))
            continue
        rows.extend(case_rows);exact_rows.extend(case_exact);overhead.extend(case_overhead);ready.add(key)
    coverage = []
    for res in dict.fromkeys(i['resolution_class'] for i in images.values()):
        cohort = [i for i in images if images[i]['resolution_class'] == res]
        for effort in c['efforts']:
            missing = [f'{i}/{s}' for i in cohort for s in c['settings'] if (i,s,effort) not in ready]
            for kind in KINDS:
                coverage.append(dict(resolution_class=res, effort=effort, kind=kind,
                    expected_tuples=len(cohort)*len(c['settings']), complete_tuples=len(cohort)*len(c['settings'])-len(missing),
                    status='incomplete' if missing else 'complete', missing_tuples='; '.join(missing)))
    coverage = pd.DataFrame(coverage)
    columns = ['image_id','setting','effort','resolution_class','sample_index','kind','stage','ms']
    samples = pd.DataFrame(rows, columns=columns)
    eligible = coverage.query("status == 'complete'")[['resolution_class','effort','kind']]
    complete = samples.merge(eligible, on=['resolution_class','effort','kind'])
    keys = ['resolution_class','effort','kind','image_id','setting','stage']
    tuples = complete.groupby(keys, as_index=False)['ms'].mean()
    means = tuples.groupby(['resolution_class','effort','kind','stage'], as_index=False)['ms'].mean()
    means['percent'] = 100*means['ms']/means.groupby(['resolution_class','effort','kind'])['ms'].transform('sum')
    manifest = dict(c, samples_per_capture=c['samples'],
                    settings_label=c.get('quality_policy', ', '.join(c['settings'])))
    return dict(manifest=manifest, manifest_path=str(path), samples=samples, means=means, coverage=coverage,
                gpu_stages=pd.DataFrame(exact_rows), rejected=pd.DataFrame(rejected), overhead=pd.DataFrame(overhead))


def plot_breakdown(report, resolution, normalize=True, kind='flat'):
    c = report['manifest'];part = report['means'].query('resolution_class == @resolution and kind == @kind')
    efforts = c['efforts'];stages = legacy.GPU_GROUPS if kind == 'gpu' else STAGES
    colors = dict(zip(stages, (list(plt.get_cmap('tab20').colors) + list(plt.get_cmap('Set3').colors))[:len(stages)]))
    for s in ('Other input preparation', PIPELINE_REMAINDER, 'Other serializer', 'Other workflow', OUTER):
        colors[s] = '#B7BEC7'
    fig, ax = plt.subplots(figsize=(15, 8.4))
    fig.subplots_adjust(left=.075, right=.68, top=.77, bottom=.20)
    heights = np.zeros(len(efforts));handles=[]
    for stage in stages:
        v = part[part.stage == stage].set_index('effort')['percent' if normalize else 'ms'].reindex(efforts).fillna(0).to_numpy()
        if not v.any():continue
        ax.bar(efforts, v, bottom=heights, width=.72, color=colors[stage], edgecolor='white', linewidth=.2)
        handles.append(Patch(facecolor=colors[stage], label=stage))
        if normalize:
            for e,b,h in zip(efforts,heights,v):
                if h >= 9: ax.text(e,b+h/2,f'{h:.0f}%',ha='center',va='center',fontsize=8,
                                   color='white' if np.dot(legacy.to_rgb(colors[stage]), [.299,.587,.114]) < .56 else '#192635')
        heights += v
    totals = part.groupby('effort')['ms'].sum()
    marker_max = 0
    if kind == 'flat' and not report['overhead'].empty:
        o = report['overhead'].query('resolution_class == @resolution')
        ordinary = o.groupby(['image_id','setting','effort'])['ordinary_ms'].mean().groupby('effort').mean()
        ordinary = ordinary.reindex(totals.index)
        y = ordinary/totals*100 if normalize else ordinary
        marker_max = y.max()
        ax.plot(totals.index,y,'D',color='#172B40',markerfacecolor='white',markersize=5,zorder=5)
        handles.append(Line2D([],[],marker='D',linestyle='none',color='#172B40',markerfacecolor='white',label='Ordinary complete-call total'))
    for e in efforts:
        if e not in totals:ax.text(e,.03,'missing',rotation=90,ha='center',transform=ax.get_xaxis_transform(),color='#7C8792')
    ymax = max(115, marker_max*1.10) if normalize else max(1, max(heights.max(),marker_max)*1.18)
    if normalize:
        for e,t in totals.items():ax.text(e,ymax*.94,f'{t:.1f} ms',ha='center',fontsize=8)
    ax.set(xlabel='Effort', ylabel=('Measured GPU stage share (%)' if kind=='gpu' else
        'Estimated ordinary-call allocation (%)' if kind=='scaled' else 'Profiled complete-call share (%)')
        if normalize else 'Mean ms / encode', xticks=efforts, xlim=(.35,10.65), ylim=(0,ymax))
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for side in ('top','right'):ax.spines[side].set_visible(False)
    ax.legend(handles=handles,loc='center left',bbox_to_anchor=(1.015,.5),frameon=False,fontsize=8)
    title = 'GJXL runtime breakdown' if kind=='flat' else 'GJXL estimated runtime allocation' if kind=='scaled' else 'GJXL measured GPU stages'
    label=legacy.RESOLUTION_LABELS.get(resolution,resolution)
    fig.suptitle(f'{title} · {label}',x=.075,y=.955,ha='left',fontsize=18)
    n=sum(i['resolution_class']==resolution for i in c['images'])
    settings = f'Q{c["quality"]} / distance {c["distance"]}' if 'quality' in c and 'distance' in c else c['settings_label']
    effort_label = '1–10' if efforts == list(range(1,11)) else ', '.join(map(str,efforts))
    fig.text(.075,.88,f'{n} image(s) · {settings} · efforts {effort_label} · {c["samples"]} paired repetitions · {c["cpu_threads"]} CPU threads',fontsize=11)
    fig.text(.075,.835,f'Production-aligned resident Metal · source {c["source_revision"][:7]}',fontsize=10,color='#536473')
    caption = ('Estimated only: each sample is scaled by its paired ordinary / profiled complete-call ratio.' if kind=='scaled' else
               'GPU counters come from the same profiled calls; their sum excludes input preparation and gaps.' if kind=='gpu' else
               'One complete-call partition: host phases, measured GPU stages, and explicit residuals. Diamonds: ordinary totals.')
    fig.text(.075,.115,caption,fontsize=9)
    fig.text(.075,.075,'Nominal settings are not matched quality. Profiling perturbs execution; no universal offset correction is assumed.',fontsize=9)
    fig.text(.075,.035,'Saved data only. Missing cohort members suppress the corresponding bar.',fontsize=9,color='#536473')
    return fig


def generate_breakdowns(config_path, output_dir, *, resolution=None, normalize=True,
                        formats=('png','svg'), show=False, gpu_diagnostics=False, scaled=False):
    report = load_profiles(config_path)
    out=Path(output_dir).expanduser();out.mkdir(parents=True,exist_ok=True)
    for name in ('samples','means','coverage','gpu_stages','rejected','overhead'):
        report[name].to_csv(out/f'gjxl-stage-{name.replace("_","-")}.csv',index=False)
    (out/'gjxl-stage-methodology.json').write_text(json.dumps(dict(config=report['manifest'],
        config_sha256=digest(config_path),semantics=SEMANTICS,normalize=normalize,scaled=scaled),indent=2)+'\n')
    resolutions=list(report['coverage']['resolution_class'].unique())
    if resolution is not None:
        if resolution not in resolutions:raise ValueError('Unknown resolution class')
        resolutions=[resolution]
    report['figures']={};report['gpu_figures']={}
    for res in resolutions:
        for kind in (('scaled' if scaled else 'flat'), *(('gpu',) if gpu_diagnostics else ())):
            figure=plot_breakdown(report,res,normalize,kind)
            report['gpu_figures' if kind=='gpu' else 'figures'][res]=figure
            prefix='gjxl-gpu-stage-diagnostic' if kind=='gpu' else 'gjxl-stage-scaled' if kind=='scaled' else 'gjxl-stage-breakdown'
            for ext in formats:figure.savefig(out/f'{prefix}-{res}.{ext}',dpi=180)
            if show:plt.show()
            plt.close(figure)
    return report
