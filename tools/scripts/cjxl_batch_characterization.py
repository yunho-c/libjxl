#!/usr/bin/env python3
"""Bounded, resumable B1/B4 collection and saved-data analysis for the paper."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import statistics
import subprocess
import time

import cjxl_throughput_table as throughput


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def ledger(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['status'])
        writer.writeheader()
        writer.writerows(rows)


def environment():
    result = {'recorded_at': now()}
    for name, command in {'power': ['pmset', '-g', 'batt'], 'memory': ['vm_stat'],
                          'swap': ['sysctl', '-n', 'vm.swapusage'],
                          'processes': ['ps', '-axo', 'pid,pgid,pcpu,rss,comm'],
                          'thermal': ['pmset', '-g', 'therm']}.items():
        result[name] = subprocess.run(command, capture_output=True, text=True).stdout
    return result


def swap_mib():
    text = subprocess.check_output(['sysctl', '-n', 'vm.swapusage'], text=True)
    return float(re.search(r'used = ([0-9.]+)M', text)[1])


def init_run(args):
    studies, host = throughput._load_runs(args.libjxl_run, args.gjxl_run)
    all_images = throughput._cohort(studies, 1.0, None)
    if args.max_megapixels is not None:
        all_images = [image for image in all_images if image['pixels'] <= args.max_megapixels * 1e6]
    if not all_images:
        raise ValueError('No images remain in the resolution range')
    qualities = [90] if args.pilot else list(throughput.DEFAULT_QUALITIES)
    efforts = [3, 7, 9] if args.pilot else list(range(1, 11))
    if args.pilot:
        small = min(all_images, key=lambda x: x['pixels'])
        images = [small, *[x for x in all_images if x['image_id'] in
                          ('unsplash/alpine_lake/12mp', 'unsplash/alpine_lake/48mp')]]
        if len(images) != 3:
            raise ValueError('Pilot requires the selected 2/12/48 MP images')
    else:
        images = all_images
    if args.image_ids:
        selected = set(args.image_ids)
        images = [image for image in images if image['image_id'] in selected]
        if {image['image_id'] for image in images} != selected:
            raise ValueError('Requested images are outside the selected cohort')
    repetitions = args.repetitions or (2 if args.pilot else 5)
    build_path = args.build_manifest.resolve()
    build = json.loads(build_path.read_text())
    expected_revisions = {'gjxl': studies[1]['config']['encoder_revision'],
                          'libjxl': studies[0]['config']['libjxl_revision']}
    if build['codec_revisions'] != expected_revisions:
        raise ValueError('Batch codec revisions differ from the paper table')
    baseline = []
    for item in studies:
        grouped = defaultdict(list)
        for row in item['records']:
            if row['image_id'] in {i['image_id'] for i in all_images} and row['quality'] in throughput.DEFAULT_QUALITIES:
                grouped[(row['image_id'], row['effort'], row['quality'])].append(row)
        for (image_id, effort, quality), rows in grouped.items():
            if len({(r['output_sha256'], r['encoded_bytes']) for r in rows}) != 1:
                raise ValueError('Baseline codestream differs between repetitions')
            baseline.append({'encoder': item['codec'], 'image_id': image_id, 'effort': effort,
                             'quality': quality, 'output_sha256': rows[0]['output_sha256'],
                             'encoded_bytes': rows[0]['encoded_bytes'], 'samples': len(rows),
                             'seconds': statistics.median(r['elapsed_nanoseconds'] / 1e9 for r in rows)})
    for image in images:
        if sha(image['pfm_path']) != image['pfm_sha256']:
            raise ValueError('Input hash mismatch')
    config = {'schema_version': 1, 'created_at': now(), 'pilot': args.pilot,
              'images': images, 'full_images': all_images, 'baseline': baseline,
              'encoders': args.encoders, 'max_megapixels': args.max_megapixels,
              'largest_first': args.largest_first,
              'efforts': efforts, 'qualities': qualities, 'repetitions': repetitions,
              'warmups': 1, 'samples_per_process': 1, 'batch_size': 4, 'cpu_budget': 8,
              'quality_to_distance': studies[1]['distances'], 'build': build,
              'build_manifest': str(build_path), 'build_manifest_sha256': sha(build_path),
              'source_runs': [item['run'] for item in studies],
              'source_metadata_sha256': [item['metadata_sha256'] for item in studies],
              'timing_sources': [item['ledger'] for item in studies], 'host': host,
              'protocol': 'four copies per image; one B1 control and one B4 call per pair; one warm pair and one measured pair per fresh process; alternating order',
              'thread_policy': {'gjxl': 'shared CPU participant cap 8; per-image cap 8',
                                'libjxl': '8 inner workers for B1, 2 per image for B4; outer callers are separate OS threads'},
              'max_rss_gib': args.max_rss_gib, 'max_swap_growth_gib': args.max_swap_growth_gib}
    baseline_keys = {(r['encoder'], r['image_id'], r['effort'], r['quality']) for r in baseline}
    for job in jobs(config):
        if tuple(job[k] for k in ('encoder', 'image_id', 'effort', 'quality')) not in baseline_keys:
            raise ValueError('Missing original codestream identity for ' + str(job))
    config['configuration_id'] = identity(config)
    args.run.mkdir(parents=True, exist_ok=False)
    write_json(args.run / 'metadata.json', config)
    for name in ('cjxl_batch_characterization.py', 'cjxl_throughput_table.py', 'cjxl_quality_characterization.py'):
        shutil.copyfile(Path(__file__).parent / name, args.run / name)
    write_json(args.run / 'collector-hashes.json', {name: sha(args.run / name) for name in
               ('cjxl_batch_characterization.py', 'cjxl_throughput_table.py', 'cjxl_quality_characterization.py')})
    print(f'Initialized {args.run}: {len(jobs(config))} fresh-process pairs; no collection started')
    summarize(args.run, config)


def load_config(run):
    config = json.loads((run / 'metadata.json').read_text())
    expected = dict(config)
    identifier = expected.pop('configuration_id')
    if identity(expected) != identifier:
        raise ValueError('Study configuration changed')
    return config


def jobs(config):
    result = []
    cases = [(i, e, q) for i in sorted(config['images'], key=lambda x: x['pixels'],
                                      reverse=config.get('largest_first', False))
             for e in config['efforts'] for q in config['qualities']]
    for repetition in range(config['repetitions']):
        for n, (image, effort, quality) in enumerate(cases):
            codecs = list(config.get('encoders', ('gjxl', 'libjxl')))
            if (n + repetition) % 2:
                codecs.reverse()
            for codec in codecs:
                row = {'encoder': codec, 'image_id': image['image_id'], 'effort': effort,
                       'quality': quality, 'repetition': repetition,
                       'batch_first': bool((n + repetition) % 2)}
                row['job_id'] = identity(row)[:24]
                result.append(row)
    return result


def checked_records(run, config):
    rows = ledger(run / 'samples.jsonl')
    expected = {job['job_id']: job for job in jobs(config)}
    seen = set()
    for row in rows:
        if row['configuration_id'] != config['configuration_id'] or row['job_id'] not in expected:
            raise ValueError('Mixed or unexpected sample')
        if row['job_id'] in seen:
            raise ValueError('Duplicate sample')
        seen.add(row['job_id'])
        if any(row[key] != value for key, value in expected[row['job_id']].items()):
            raise ValueError('Sample identity mismatch')
        if sha(row['raw_path']) != row['raw_sha256']:
            raise ValueError('Saved raw timing changed')
        with Path(row['raw_path']).open() as f:
            if list(csv.DictReader(f)) != [row['raw']]:
                raise ValueError('Saved sample differs from raw timing')
    return rows


def validate_raw(path, reference, job, config):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError('Expected exactly one independently warmed pair')
    row = rows[0]
    image = next(i for i in config['images'] if i['image_id'] == job['image_id'])
    expected = {'codec': job['encoder'], 'width': str(image['width']), 'height': str(image['height']),
                'effort': str(job['effort']), 'batch_size': '4', 'serial_image_count': '1',
                'sample': '0', 'cpu_budget': '8', 'source': image['pfm_path'],
                'order': 'batch-first' if job['batch_first'] else 'serial-first',
                'timing_boundary': 'linear_rgb_to_in_memory_codestream'}
    for key, value in expected.items():
        if row[key] != value:
            raise ValueError('Raw timing setting mismatch: ' + key)
    if not math.isclose(float(row['distance']), config['quality_to_distance'][str(job['quality'])], rel_tol=1e-6):
        raise ValueError('Raw distance mismatch')
    codec = job['encoder']
    if codec == 'gjxl':
        if row['backend'] != 'metal' or row['aq_mode'] != 'fully-resident' or row['cpu_threads_per_image'] != '8':
            raise ValueError('GJXL backend/thread policy mismatch')
        if not 0 < int(row['domain_peak_cpu_participants']) <= 8:
            raise ValueError('GJXL exceeded its CPU participation limit')
    elif row['cpu_threads_per_image'] != '2' or row['backend'] != 'cpu':
        raise ValueError('libjxl worker policy mismatch')
    for key in ('serial_ns', 'batch_ns'):
        if int(row[key]) <= 0:
            raise ValueError('Invalid timing duration')
    baseline = next(r for r in config['baseline'] if all(r[k] == job[k] for k in
                    ('encoder', 'image_id', 'effort', 'quality')))
    if sha(reference) != baseline['output_sha256'] or int(row['encoded_bytes_per_image']) != baseline['encoded_bytes']:
        raise ValueError('Batch reference differs from the pinned single-image codestream')
    return row


def command_for(job, config, folder):
    image = next(i for i in config['images'] if i['image_id'] == job['image_id'])
    codec = job['encoder']
    command = [config['build']['binaries'][codec]['path'], '--input', image['pfm_path'],
               '--effort', str(job['effort']), '--distance', str(config['quality_to_distance'][str(job['quality'])]),
               '--batch-sizes', '4', '--single-image-control', '--warmups', '1', '--samples', '1',
               '--raw-samples', str(folder / 'raw.csv'), '--reference-output', str(folder / 'reference.jxl')]
    command += ['--cpu-budget', '8', '--cpu-threads', '8'] if codec == 'gjxl' else ['--total-workers', '8']
    if job['batch_first']:
        command.append('--batch-first')
    return command


def execute(command, folder, deadline, config, initial_swap):
    started = time.monotonic()
    process = None
    peak_rss = 0
    with (folder / 'stdout.log').open('w') as out, (folder / 'stderr.log').open('w') as err:
        try:
            process = subprocess.Popen(['/usr/bin/time', '-l', *command], stdout=out, stderr=err, start_new_session=True)
            next_check = 0.0
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Collection/job wall budget exhausted')
                if time.monotonic() >= next_check:
                    text = subprocess.check_output(['ps', '-axo', 'pgid=,rss='], text=True)
                    rss = sum(int(fields[1]) * 1024 for line in text.splitlines()
                              if len(fields := line.split()) == 2 and int(fields[0]) == process.pid)
                    peak_rss = max(peak_rss, rss)
                    swap = swap_mib()
                    append(folder / 'resources.jsonl', {'recorded_at': now(),
                           'group_rss_bytes': rss, 'system_swap_mib': swap,
                           'swap_growth_mib': swap - initial_swap})
                    if rss > config['max_rss_gib'] * 2**30:
                        raise RuntimeError('Process-group RSS guard exceeded')
                    if swap - initial_swap > config['max_swap_growth_gib'] * 1024:
                        raise RuntimeError('System swap growth guard exceeded')
                    next_check = time.monotonic() + 2
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError('Benchmark failed with exit ' + str(process.returncode))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    text = (folder / 'stderr.log').read_text()
    match = re.search(r'(\d+)\s+maximum resident set size', text)
    final_swap = swap_mib()
    append(folder / 'resources.jsonl', {'recorded_at': now(), 'phase': 'after_process',
           'system_swap_mib': final_swap, 'swap_growth_mib': final_swap - initial_swap})
    if final_swap - initial_swap > config['max_swap_growth_gib'] * 1024:
        raise RuntimeError('System swap growth guard exceeded after encode')
    return {'wall_seconds': time.monotonic() - started,
            'max_rss_bytes': int(match[1]) if match else peak_rss,
            'max_group_rss_bytes': peak_rss, 'system_swap_mib_at_end': final_swap,
            'swap_growth_mib_at_end': final_swap - initial_swap}


def run_collection(args):
    run = args.run.resolve()
    config = load_config(run)
    with (run.parent / '.batch-paper-collection.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        records = checked_records(run, config)
        done = {r['job_id'] for r in records}
        pending = [job for job in jobs(config) if job['job_id'] not in done]
        if args.dry_run:
            print(f'{len(done)} complete; {len(pending)} pending fresh-process pairs; no encodes')
            return
        if not pending:
            print(f'All {len(done)} pairs are complete; no encodes or study changes')
            return
        for name, expected in json.loads((run / 'collector-hashes.json').read_text()).items():
            if sha(run / name) != expected:
                raise ValueError('Frozen collector changed: ' + name)
        for path, expected in config['build']['libraries'].items():
            if sha(path) != expected:
                raise ValueError('Pinned library changed: ' + path)
        for binary in config['build']['binaries'].values():
            if sha(binary['path']) != binary['sha256']:
                raise ValueError('Batch binary changed')
        for image in config['images']:
            if sha(image['pfm_path']) != image['pfm_sha256']:
                raise ValueError('Input changed')
        before = environment()
        competing = [line for line in before['processes'].splitlines()
                     if re.search(r'/(?:gjxl_[^/ ]*benchmark|jxl_image_batch_benchmark|gjxl_batch|libjxl_batch|cjxl|djxl)$', line)]
        if competing:
            raise RuntimeError('Another codec workload is active: ' + '\n'.join(competing))
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        write_json(run / ('environment-before-' + stamp + '.json'), before)
        initial_swap = swap_mib()
        deadline = time.monotonic() + args.budget_seconds
        state = {'state': 'running', 'pid': os.getpid(), 'started_at': now(), 'complete': False,
                 'expected': len(jobs(config)), 'completed': len(done), 'budget_seconds': args.budget_seconds}
        write_json(run / 'status.json', state)
        try:
            for job in pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Collection wall budget exhausted')
                if shutil.disk_usage(run).free < 6 * 2**30:
                    raise RuntimeError('Less than 6 GiB free disk space')
                folder = run / 'attempts' / (job['job_id'] + '-' + stamp)
                folder.mkdir(parents=True, exist_ok=False)
                command = command_for(job, config, folder)
                write_json(folder / 'command.json', command)
                state.update(current=job, current_started_at=now())
                write_json(run / 'status.json', state)
                metrics = execute(command, folder, min(deadline, time.monotonic() + args.job_timeout), config, initial_swap)
                raw = validate_raw(folder / 'raw.csv', folder / 'reference.jxl', job, config)
                record = {**job, **metrics, 'configuration_id': config['configuration_id'], 'recorded_at': now(),
                          'raw': raw, 'raw_path': str(folder / 'raw.csv'), 'raw_sha256': sha(folder / 'raw.csv'),
                          'output_sha256': sha(folder / 'reference.jxl')}
                append(run / 'samples.jsonl', record)
                # Reference codestreams already exist in the frozen source study.
                (folder / 'reference.jxl').unlink()
                state['completed'] += 1
                write_json(run / 'status.json', state)
                print(f"{state['completed']}/{state['expected']} {job['encoder']} e{job['effort']} Q{job['quality']} "
                      f"{job['image_id']} r{job['repetition']} B4/B1={4*int(raw['serial_ns'])/int(raw['batch_ns']):.2f} "
                      f"RSS={metrics['max_rss_bytes']/2**30:.2f} GiB wall={metrics['wall_seconds']:.1f}s", flush=True)
            state.update(state='complete', complete=True)
        except (Exception, KeyboardInterrupt) as error:
            state.update(state='stopped-incomplete', error=str(error) or 'interrupted')
            raise
        finally:
            state['finished_at'] = now()
            write_json(run / 'status.json', state)
            write_json(run / ('environment-after-' + stamp + '.json'), environment())
            summarize(run, config)


def analyze(config, records):
    """Compute fixed-cohort rates without writing files or starting encoders."""
    grouped = defaultdict(list)
    for row in records:
        grouped[(row['encoder'], row['image_id'], row['effort'], row['quality'])].append(row)
    cells = []
    for codec in config.get('encoders', ('gjxl', 'libjxl')):
        for image in config['images']:
            for effort in config['efforts']:
                for quality in config['qualities']:
                    rows = grouped[(codec, image['image_id'], effort, quality)]
                    complete = len(rows) == config['repetitions']
                    t1 = statistics.median(int(r['raw']['serial_ns']) / 1e9 for r in rows) if complete else None
                    t4 = statistics.median(int(r['raw']['batch_ns']) / 1e9 for r in rows) if complete else None
                    cells.append({'encoder': codec, 'image_id': image['image_id'], 'resolution_class': image['resolution_class'],
                                  'megapixels': image['pixels']/1e6, 'effort': effort, 'quality': quality,
                                  'samples': len(rows), 'expected_samples': config['repetitions'], 'complete': complete,
                                  'b1_seconds': t1, 'b4_seconds': t4,
                                  'batch_gain': 4*t1/t4 if complete else None,
                                  'max_rss_gib': max((r['max_rss_bytes']/2**30 for r in rows), default=0)})
    points = []
    for codec in config.get('encoders', ('gjxl', 'libjxl')):
        for effort in config['efforts']:
            subset = [r for r in cells if r['encoder'] == codec and r['effort'] == effort]
            complete = all(r['complete'] for r in subset)
            rates = {}
            for batch in (1, 4):
                rates[batch] = statistics.mean(
                    sum(r['megapixels'] * batch for r in subset if r['quality'] == q) /
                    sum(r[f'b{batch}_seconds'] for r in subset if r['quality'] == q)
                    for q in config['qualities']) if complete else None
            points.append({'encoder': codec, 'effort': effort, 'b1_mp_s': rates[1], 'b4_mp_s': rates[4],
                           'batch_gain': rates[4]/rates[1] if complete else None,
                           'complete': complete, 'complete_cells': sum(r['complete'] for r in subset),
                           'expected_cells': len(subset)})
    return cells, points


def summarize(run, config=None):
    config = config or load_config(run)
    records = checked_records(run, config)
    cells, points = analyze(config, records)
    output = run / 'summary'
    output.mkdir(exist_ok=True)
    write_csv(output / 'image-tuples.csv', cells)
    write_csv(output / 'throughput.csv', points)
    write_json(output / 'coverage.json', {'expected_pairs': len(jobs(config)), 'completed_pairs': len(records),
                                        'complete': len(records) == len(jobs(config)), 'pilot': config['pilot']})
    estimate = []
    images_by_id = {image['image_id']: image for image in config['full_images']}
    baseline_by_key = {(r['encoder'], r['image_id'], r['effort'], r['quality']): r
                       for r in config['baseline']}
    observed_groups = defaultdict(list)
    for record in records:
        observed_groups[(record['encoder'], record['image_id'], record['effort'], record['quality'])].append(record)
    observed_points = [{**rows[0], 'wall_seconds': statistics.median(r['wall_seconds'] for r in rows)}
                       for rows in observed_groups.values()]
    for codec in config.get('encoders', ('gjxl', 'libjxl')):
        observed = [r for r in records if r['encoder'] == codec]
        if not observed:
            continue
        points_for_codec = [r for r in observed_points if r['encoder'] == codec]
        seconds = 0.0
        for baseline in config['baseline']:
            if baseline['encoder'] != codec:
                continue
            image = images_by_id[baseline['image_id']]
            compatible = [r for r in points_for_codec if (r['effort'] >= 8) == (baseline['effort'] >= 8)]
            if not compatible:
                break
            def distance(r):
                other = images_by_id[r['image_id']]
                return (abs(math.log(other['pixels']/image['pixels'])) + .05*abs(r['effort']-baseline['effort'])
                        + .0001*abs(r['quality']-baseline['quality']))
            nearest = min(compatible, key=distance)
            anchor = baseline_by_key[tuple(nearest[k] for k in ('encoder', 'image_id', 'effort', 'quality'))]
            # Observed process wall time already includes validation and both warm/timed pairs.
            seconds += 5 * baseline['seconds'] * nearest['wall_seconds'] / anchor['seconds']
        else:
            estimate.append({'encoder': codec, 'estimated_full_grid_hours': seconds/3600,
                             'observed_pairs': len(observed),
                             'configured_pairs': sum(job['encoder'] == codec for job in jobs(config)),
                             'coverage_complete': len(records) == len(jobs(config)),
                             'resource_feasibility': 'not established by time extrapolation',
                             'method': 'five fresh processes; scale saved per-tuple encode time by nearest pilot size/effort process-wall multiplier',
                             'limitations': 'Q90 pilot gains transferred to Q30-Q95; content and effort interpolation; libjxl e10 baseline incomplete; diagnostic desktop conditions'})
    write_json(output / 'estimate.json', estimate)
    return cells, points


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init')
    init.add_argument('--run', type=Path, required=True)
    init.add_argument('--libjxl-run', type=Path, required=True)
    init.add_argument('--gjxl-run', type=Path, required=True)
    init.add_argument('--build-manifest', type=Path, required=True)
    init.add_argument('--pilot', action='store_true')
    init.add_argument('--image-ids', nargs='+', help='Explicit subset of the selected cohort')
    init.add_argument('--encoders', nargs='+', choices=('gjxl', 'libjxl'), default=['gjxl', 'libjxl'])
    init.add_argument('--max-megapixels', type=float, help='Inclusive upper bound on original input pixels')
    init.add_argument('--largest-first', action='store_true', help='Exercise the largest images first in every round')
    init.add_argument('--repetitions', type=int)
    init.add_argument('--max-rss-gib', type=float, default=24)
    init.add_argument('--max-swap-growth-gib', type=float, default=2)
    run = sub.add_parser('run')
    run.add_argument('--run', type=Path, required=True)
    run.add_argument('--budget-seconds', type=float, default=1800)
    run.add_argument('--job-timeout', type=float, default=1200)
    run.add_argument('--dry-run', action='store_true')
    summary = sub.add_parser('summarize')
    summary.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        if len(set(args.encoders)) != len(args.encoders):
            parser.error('encoders must be distinct')
        if args.max_megapixels is not None and (not math.isfinite(args.max_megapixels) or args.max_megapixels < 1):
            parser.error('max-megapixels must be finite and at least 1')
        if args.repetitions is not None and args.repetitions < 1:
            parser.error('repetitions must be positive')
        if any(not math.isfinite(x) or x <= 0 for x in (args.max_rss_gib, args.max_swap_growth_gib)):
            parser.error('memory guards must be finite and positive')
        init_run(args)
    elif args.command == 'run':
        if any(not math.isfinite(x) or x <= 0 for x in (args.budget_seconds, args.job_timeout)):
            parser.error('time budgets must be finite and positive')
        run_collection(args)
    else:
        summarize(args.run)


if __name__ == '__main__':
    main()
