#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""CUDA initialization for the existing fixed and matched quality collector.

Only initialization uses this module. Frozen collectors contain all collection
and analysis code and never depend on this mutable initialization helper.
"""

import json
import platform
from pathlib import Path
import shutil
import subprocess


def initialize_cuda(args, images, study):
    if not images:
        raise study.StudyError("CUDA study requires a nonempty image cohort")
    source = args.gjxl_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    benchmark = args.benchmark.resolve()
    record = json.loads(args.build_record.read_text())
    if record["revision"] != revision or Path(record["source"]).resolve() != source:
        raise study.StudyError("CUDA build record identifies a different source revision")
    study.verify_file(benchmark, record["benchmark_sha256"])
    for path, expected in record["artifacts"].items():
        study.verify_file(path, expected)
    for name, expected in record["source_hashes"].items():
        study.verify_file(source / name, expected)
    version = json.loads(subprocess.check_output([str(benchmark), "--version"], text=True))
    if version != {"encoder": "gjxl", "schema_version": 1,
                   "revision": revision, "shader_source": "compiled-cuda"}:
        raise study.StudyError("CUDA harness version does not match the frozen build")
    scorer = args.scorer.resolve()
    metric_version = json.loads(subprocess.check_output([str(scorer), "--version"], text=True))
    if metric_version.get("fast_ssim2_revision") != study.METRIC_REVISION:
        raise study.StudyError("Use the pinned fast-ssim2 quality_metric adapter")
    decoder = args.decoder.resolve()
    decoder_hash = study.digest(decoder)
    for image in images:
        if image["color_encoding"] != study.COLOR:
            raise study.StudyError("Only canonical linear-sRGB PFM inputs are supported")
        study.verify_file(image["pfm_path"], image["pfm_sha256"])
    seeds = study.seed_snapshot(args, images, metric_version, decoder_hash,
                                record["benchmark_sha256"])
    if args.run.exists():
        raise study.StudyError("Run directory already exists; resume it instead")
    args.run.mkdir(parents=True)
    frozen = args.run / benchmark.name
    shutil.copy2(benchmark, frozen)
    files = {str(frozen): study.digest(frozen), str(decoder): decoder_hash,
             str(scorer): study.digest(scorer)}
    for dll in record.get("runtime_dlls", []):
        path = Path(dll)
        destination = args.run / path.name
        shutil.copy2(path, destination)
        study.verify_file(destination, record["artifacts"][str(path)])
        files[str(destination)] = study.digest(destination)
    # Decoder runtimes are frozen in the decoder's own directory as well.
    for dll in decoder.parent.glob("*.dll"):
        files[str(dll.resolve())] = study.digest(dll)
    study.write_json(args.run / "encoder-build.json", record)
    config = {
        "schema_version": study.SCHEMA, "encoder": "gjxl", "metric": "fast-ssim2",
        "metric_version": metric_version, "preview_available": False,
        "intensity_target": args.intensity_target, "targets": args.targets,
        "tolerance": args.tolerance, "images": images,
        "efforts": args.measurement_efforts, "measurement_efforts": args.measurement_efforts,
        "scorer": str(scorer), "djxl": str(decoder), "benchmark": str(frozen),
        "encoder_revision": revision, "tool_hashes": files,
        "encoder_build_sha256": study.digest(args.run / "encoder-build.json"),
        "corpus_sha256": study.digest(args.corpus),
        "num_threads": 8, "thread_semantics": "maximum-participating-cpu-threads",
        "backend": "cuda", "gpu_aq_mode": "fully-resident", "density": "default",
        "continue_cuda_oom": True, "terminal_calibration_outcomes": True,
        "collection_lock": str(args.build_record.resolve().parent / ".gpu-collection.lock"),
        "compression": "automatic", "collect_final_score": False,
        "repetitions": 5, "warmups": 1, "seed": 20260908,
        "minimum_distance": args.minimum_distance, "maximum_distance": args.maximum_distance,
        "max_evaluations": args.max_evaluations, "timeout": args.timeout,
        "pilot": args.pilot, "collector_snapshot_sha256": study.digest(study.__file__),
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "gpu": record["gpu"]},
    }
    if args.mode == "fixed":
        helper_path = Path(study.__file__).with_name("cjxl_sweep_common.py")
        config.update(collection_mode="fixed", qualities=args.qualities,
                      preview_available=True, sweep_helpers_sha256=study.digest(helper_path))
        config["quality_to_distance"] = {
            str(q): study.sweep_helpers().quality_to_distance(q) for q in args.qualities
        }
        shutil.copy2(helper_path, args.run / helper_path.name)
    if seeds is not None:
        study.write_json(args.run / "seed-scores.json", seeds)
        study.write_csv(args.run / "source-tuples.csv", seeds["tuples"])
        config.update(seed_snapshot_sha256=study.digest(args.run / "seed-scores.json"),
                      tuple_snapshot_sha256=study.digest(args.run / "source-tuples.csv"),
                      preview_available=True)
    config["configuration_id"] = study.identity(config)
    (args.run / "collector.py").write_bytes(Path(study.__file__).read_bytes())
    study.write_json(args.run / "metadata.json", config)
    if seeds is not None:
        for original in seeds["scores"]:
            row = {k: v for k, v in original.items()
                   if k not in ("configuration_id", "recorded_at", "schema_version")}
            row["seed_configuration_id"] = seeds["configuration_id"]
            study.save_record(args.run, "scores", config, row)
    study.append(args.run / "execution-events.jsonl", {"event": "initialized",
                 "configuration_id": config["configuration_id"]})
    print(f"Initialized CUDA {args.run}: {len(images)} images; no collection started")
