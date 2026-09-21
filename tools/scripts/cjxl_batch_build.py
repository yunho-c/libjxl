#!/usr/bin/env python3
"""Build batch drivers against the paper's existing frozen Release libraries."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cjxl_quality_characterization as study


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(libjxl_run, gjxl_run, output, gjxl_harness):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    lib = study.load_config(Path(libjxl_run).resolve())
    gjxl = study.load_config(Path(gjxl_run).resolve())
    source_metadata = Path(lib['source_run']) / 'metadata.json'
    study.verify_file(source_metadata, lib['source_metadata_sha256'])
    ordinary = json.loads(source_metadata.read_text())['configuration']['build_records']['ordinary']['content']
    gbuild = json.loads((Path(gjxl_run) / 'encoder-build.json').read_text())
    gs, gb = Path(gbuild['source']), Path(gbuild['CMakeCache.txt']['path']).parent
    ls, lb = Path(ordinary['source']), Path(ordinary['build'])
    for root, revision in ((gs, gjxl['encoder_revision']), (ls, lib['libjxl_revision'])):
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != revision:
            raise ValueError('Frozen source revision mismatch: ' + str(root))
        if subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'src', 'lib'], text=True):
            raise ValueError('Frozen codec sources are dirty: ' + str(root))
    study.verify_file(gbuild['CMakeCache.txt']['path'], gbuild['CMakeCache.txt']['sha256'])
    for path, sha in lib['tool_hashes'].items():
        if str(lb) in path:
            study.verify_file(path, sha)
    gsrc = output / 'gjxl_image_batch_benchmark.cpp'
    lsrc = output / 'image_batch_benchmark.cc'
    shutil.copyfile(gjxl_harness, gsrc)
    shutil.copyfile(gs / 'benchmarks/synthetic_images.h', output / 'synthetic_images.h')
    shutil.copyfile(Path(__file__).parents[1] / 'benchmark/image_batch_benchmark.cc', lsrc)
    glibs = [gb / ('lib' + name + '.a') for name in
             ('gjxl_codestream', 'gjxl_metal', 'gjxl_pfm_io', 'gjxl_gpu_butteraugli',
              'gjxl_gpu_ops', 'gjxl_gpu', 'gjxl_codec')]
    llibs = [lb / 'lib/libjxl_extras_codec.a', lb / 'lib/libjxl_threads.dylib',
             lb / 'tools/libjxl_tool.a', lb / 'lib/libjxl.dylib', lb / 'lib/libjxl_cms.dylib',
             Path('/opt/homebrew/lib/libgif.dylib'), Path('/opt/homebrew/lib/libjpeg.dylib'),
             Path('/opt/homebrew/lib/libpng.dylib'), lb / 'third_party/highway/libhwy.a']
    commands = {
        'gjxl': ['/usr/bin/c++', '-O3', '-DNDEBUG', '-std=c++20', '-arch', 'arm64',
                 '-I' + str(gs / 'src'), str(gsrc), '-o', str(output / 'gjxl_batch'),
                 *map(str, glibs), '-framework', 'Metal', '-framework', 'Foundation',
                 '-framework', 'CoreGraphics'],
        'libjxl': ['/usr/bin/c++', '-O3', '-DNDEBUG', '-fno-rtti', '-std=c++17', '-arch', 'arm64',
                   *['-I' + str(p) for p in (ls, ls / 'lib/include', lb / 'lib/include', ls / 'third_party/highway')],
                   str(lsrc), '-o', str(output / 'libjxl_batch'), *map(str, llibs), '-lz',
                   '-Wl,-rpath,' + str(lb / 'lib')],
    }
    libraries = {str(p.resolve()): digest(p) for p in [*glibs, *llibs]}
    for codec, command in commands.items():
        subprocess.run(command, check=True)
    manifest = {
        'schema_version': 1, 'codec_revisions': {'gjxl': gjxl['encoder_revision'], 'libjxl': lib['libjxl_revision']},
        'libraries': libraries, 'commands': commands,
        'sources': {str(p): digest(p) for p in (gsrc, lsrc, output / 'synthetic_images.h')},
        'binaries': {c: {'path': str(output / (c + '_batch')), 'sha256': digest(output / (c + '_batch'))}
                    for c in commands},
        'gjxl_source': str(gs), 'libjxl_source': str(ls),
    }
    (output / 'build.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output / 'build.json'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--libjxl-run', type=Path, required=True)
    parser.add_argument('--gjxl-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gjxl-harness', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.libjxl_run, args.gjxl_run, args.output, args.gjxl_harness))
