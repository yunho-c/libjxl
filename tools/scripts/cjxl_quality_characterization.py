#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Resumable perceptual scoring, matched-quality search, and CSV summaries.

Collection is opt-in and separate from rendering. See doc/runtime-quality.md.
The standard-library CLI uses the existing uninstrumented API harness and
exports data only. Rendering lives in cjxl_runtime_characterization_notebook.
No work is started by importing this module.
"""

import argparse
import collections
import csv
import datetime
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import select
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

SCHEMA = 1
COLOR = "RGB_D65_SRG_Rel_Lin"
METRIC_REVISION = "c3867954c7bec8a761951df9256354b305fd0cff"
SOURCE_FIELDS = ("image_id", "corpus", "resolution_class", "width", "height")
METRICS = {
    "fast-ssim2": {"label": "fast-ssim2", "higher_is_better": True},
    "butteraugli": {"label": "Butteraugli distance", "higher_is_better": False},
}


def encoder_name(config):
    name = config.get("encoder", "libjxl")
    if name not in ("libjxl", "gjxl"):
        raise StudyError(f"Unknown encoder: {name}")
    return name


def is_fixed(config):
    return config.get("collection_mode") == "fixed"


def sweep_helpers(run=None, config=None):
    """Frozen runs load their own hashed copy, never mutable checkout helpers."""
    path = (
        Path(run) if run is not None else Path(__file__).parent
    ) / "cjxl_sweep_common.py"
    if config is not None:
        verify_file(path, config["sweep_helpers_sha256"])
    spec = importlib.util.spec_from_file_location("quality_sweep_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_snapshot(args, images, metric_version, decoder_hash, benchmark_hash):
    """Import immutable observations, not a live dependency on a changing ledger."""
    if args.seed_run is None:
        return None
    source = args.seed_run.resolve()
    config = load_config(source, verify=True)
    if not is_fixed(config) or encoder_name(config) != "gjxl":
        raise StudyError("--seed-run requires a GJXL fixed sweep")
    if (
        config["metric_version"] != metric_version
        or metric_name(config) != "fast-ssim2"
        or config["intensity_target"] != args.intensity_target
        or config["tool_hashes"][config["djxl"]] != decoder_hash
        or config["tool_hashes"][config["benchmark"]] != benchmark_hash
        or (config["num_threads"], config["warmups"], config["repetitions"])
        != (8, 1, 5)
    ):
        raise StudyError(
            "Seed sweep uses different binaries, metric or timing protocol"
        )
    selected = {i["image_id"]: i for i in images}
    for image in config["images"]:
        if image["image_id"] in selected and any(
            image[key] != selected[image["image_id"]][key]
            for key in (*SOURCE_FIELDS, "pfm_sha256", "color_encoding")
        ):
            raise StudyError("Seed sweep reference or geometry differs")
    scores = [
        r
        for r in read_records(source, "scores", config, verify=True)
        if r["image_id"] in selected and r["effort"] in args.measurement_efforts
    ]
    if not scores:
        raise StudyError("Seed sweep has no scored observations for the selected grid")
    return {
        "source_run": str(source),
        "configuration_id": config["configuration_id"],
        "metadata_sha256": digest(source / "metadata.json"),
        "scores": scores,
        "tuples": [
            r
            for r in fixed_rows(source, config)
            if r["job_id"] in {s["source_job_id"] for s in scores}
        ],
    }


def initialize_gjxl(args, images):
    """New GJXL study: independent probes, shared pinned decoder and metric."""
    metadata_path = args.source_run / "metadata.json"
    source_metadata = json.loads(metadata_path.read_text())
    ordinary = source_metadata["configuration"]["build_records"]["ordinary"]["content"]
    decoder = ordinary["binaries"]["djxl"]
    verify_file(decoder["path"], decoder["sha256"])
    files = {decoder["path"]: decoder["sha256"]}
    for pattern in ("*.dylib", "*.so*"):
        for library in (Path(ordinary["build"]) / "lib").glob(pattern):
            if library.is_file():
                files[str(library.resolve())] = digest(library)
    scorer = args.scorer.resolve()
    metric_version = json.loads(
        subprocess.check_output([str(scorer), "--version"], text=True)
    )
    if metric_version.get("fast_ssim2_revision") != METRIC_REVISION:
        raise StudyError("Use the pinned quality_metric adapter")
    files[str(scorer)] = digest(scorer)
    benchmark = args.benchmark.resolve()
    version = json.loads(
        subprocess.check_output([str(benchmark), "--version"], text=True)
    )
    source = args.gjxl_source.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if version != {
        "encoder": "gjxl",
        "schema_version": 1,
        "revision": revision,
        "shader_source": "embedded-production",
    }:
        raise StudyError("GJXL harness version does not match its source checkout")
    # The harness uses embedded shaders and statically linked GJXL. Refuse an
    # unexpected external runtime rather than silently lose it when snapshotting.
    dependencies = subprocess.check_output(["otool", "-L", str(benchmark)], text=True)
    if any(
        not line.strip().startswith(("/usr/lib/", "/System/Library/"))
        for line in dependencies.splitlines()[1:]
    ):
        raise StudyError(
            "GJXL harness must use only embedded/static GJXL and system libraries"
        )
    tracked = (
        subprocess.check_output(["git", "-C", str(source), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    tracked.append("benchmarks/quality_benchmark.cpp")
    build_record = {
        "revision": revision,
        "source": str(source),
        "source_hashes": {
            name: digest(source / name)
            for name in sorted(set(tracked))
            if name and (source / name).is_file()
        },
        "source_diff": subprocess.check_output(
            ["git", "-C", str(source), "diff", "HEAD", "--binary"], text=True
        ),
        "submodules": subprocess.check_output(
            ["git", "-C", str(source), "submodule", "status", "--recursive"], text=True
        ),
        "original_benchmark": str(benchmark),
        "benchmark_sha256": digest(benchmark),
        "version": version,
        "dependencies": dependencies,
    }
    for name in ("CMakeCache.txt", "metal/gjxl.metallib"):
        artifact = benchmark.parent / name
        if not artifact.is_file():
            raise StudyError(f"Missing GJXL build provenance: {artifact}")
        build_record[name] = {"path": str(artifact), "sha256": digest(artifact)}
    if (
        "CMAKE_BUILD_TYPE:STRING=Release"
        not in (benchmark.parent / "CMakeCache.txt").read_text()
    ):
        raise StudyError("Use a Release GJXL harness")
    for image in images:
        if image["color_encoding"] != COLOR:
            raise StudyError("Only canonical linear-sRGB PFM inputs are supported")
        verify_file(image["pfm_path"], image["pfm_sha256"])
    seeds = seed_snapshot(
        args,
        images,
        metric_version,
        decoder["sha256"],
        build_record["benchmark_sha256"],
    )
    if args.run.exists():
        raise StudyError("Run directory already exists; resume it instead")
    args.run.mkdir(parents=True)
    frozen = args.run / "gjxl_quality_benchmark"
    shutil.copy2(benchmark, frozen)
    verify_file(frozen, build_record["benchmark_sha256"])
    files[str(frozen)] = build_record["benchmark_sha256"]
    write_json(args.run / "encoder-build.json", build_record)
    config = {
        "schema_version": SCHEMA,
        "encoder": "gjxl",
        "metric": "fast-ssim2",
        "metric_version": metric_version,
        "preview_available": False,
        "intensity_target": args.intensity_target,
        "targets": args.targets,
        "tolerance": args.tolerance,
        "images": images,
        "efforts": args.measurement_efforts,
        "measurement_efforts": args.measurement_efforts,
        "scorer": str(scorer),
        "djxl": decoder["path"],
        "benchmark": str(frozen),
        "encoder_revision": revision,
        "tool_hashes": files,
        "encoder_build_sha256": digest(args.run / "encoder-build.json"),
        "source_run": str(args.source_run.resolve()),
        "source_metadata_sha256": digest(metadata_path),
        "corpus_sha256": digest(args.corpus),
        "num_threads": 8,
        "thread_semantics": "maximum-participating-cpu-threads",
        "backend": "metal",
        "metal_aq_mode": "fully-resident",
        "density": "default",
        "compression": "automatic",
        "collect_final_score": False,
        "repetitions": 5,
        "warmups": 1,
        "seed": 20260908,
        "minimum_distance": args.minimum_distance,
        "maximum_distance": args.maximum_distance,
        "max_evaluations": args.max_evaluations,
        "timeout": args.timeout,
        "pilot": args.pilot,
        "collector_snapshot_sha256": digest(__file__),
    }
    if args.mode == "fixed":
        helper_path = Path(__file__).with_name("cjxl_sweep_common.py")
        config.update(
            collection_mode="fixed",
            qualities=args.qualities,
            preview_available=True,
            sweep_helpers_sha256=digest(helper_path),
        )
        config["quality_to_distance"] = {
            str(q): sweep_helpers().quality_to_distance(q) for q in args.qualities
        }
        shutil.copy2(helper_path, args.run / helper_path.name)
    if seeds is not None:
        write_json(args.run / "seed-scores.json", seeds)
        write_csv(args.run / "source-tuples.csv", seeds["tuples"])
        config.update(
            seed_snapshot_sha256=digest(args.run / "seed-scores.json"),
            tuple_snapshot_sha256=digest(args.run / "source-tuples.csv"),
            preview_available=True,
        )
    config["configuration_id"] = identity(config)
    (args.run / "collector.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.run / "metadata.json", config)
    if seeds is not None:
        for original in seeds["scores"]:
            row = {
                k: v
                for k, v in original.items()
                if k not in ("configuration_id", "recorded_at", "schema_version")
            }
            row["seed_configuration_id"] = seeds["configuration_id"]
            save_record(args.run, "scores", config, row)
    append(
        args.run / "execution-events.jsonl",
        {"event": "initialized", "configuration_id": config["configuration_id"]},
    )
    print(f"Initialized {args.run}: {len(images)} images; no collection started")


def metric_name(config):
    # Backward-compatible with the original fast-ssim2 run/test fixtures.
    name = config.get("metric", "fast-ssim2")
    if name not in METRICS:
        raise StudyError(f"Unknown metric: {name}")
    return name


def require_encoding_allowed(config):
    if config.get("preview_only") or metric_name(config) == "butteraugli":
        raise StudyError(
            "Butteraugli is preview-only: no calibration or timing encodes"
        )


def parse_butteraugli(stdout):
    """Keep the conventional distance distinct from the auxiliary 3-norm."""
    lines = stdout.strip().splitlines()
    if len(lines) != 2 or not re.fullmatch(r"3-norm: \S+", lines[1]):
        raise StudyError("Unexpected Butteraugli output")
    try:
        distance, pnorm = float(lines[0]), float(lines[1].split()[1])
    except ValueError as error:
        raise StudyError("Invalid Butteraugli distance") from error
    if not all(math.isfinite(v) and v >= 0 for v in (distance, pnorm)):
        raise StudyError("Butteraugli distances must be finite and nonnegative")
    return {
        "score": distance,
        "butteraugli_pnorm3": pnorm,
        "metric_stdout": stdout.strip(),
    }


class StudyError(Exception):
    pass


class BudgetExpired(Exception):
    pass


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def append(path, value):
    value = dict(value, schema_version=SCHEMA, recorded_at=utc())
    with Path(path).open("a") as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def ledger(path):
    if not Path(path).exists():
        return []
    try:
        return [json.loads(line) for line in Path(path).read_text().splitlines()]
    except (ValueError, UnicodeError) as error:
        raise StudyError(
            f"Invalid ledger {path}; preserve and repair its incomplete tail before resuming"
        ) from error


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.with_suffix(".csv.tmp").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or ["status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: json.dumps(v, sort_keys=True)
                    if isinstance(v, (dict, list))
                    else v
                    for k, v in row.items()
                }
            )
    path.with_suffix(".csv.tmp").replace(path)


def verify_file(path, expected):
    if not Path(path).is_file() or digest(path) != expected:
        raise StudyError(f"Missing or changed artifact: {path}")


class Budget:
    def __init__(self, seconds=None, jobs=None):
        self.deadline = time.monotonic() + seconds if seconds is not None else math.inf
        self.jobs = jobs
        self.completed = 0

    def check(self):
        if time.monotonic() >= self.deadline or (
            self.jobs is not None and self.completed >= self.jobs
        ):
            raise BudgetExpired()

    def timeout(self, maximum):
        self.check()
        return min(maximum, self.deadline - time.monotonic())


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def execute(command, run, budget, timeout):
    budget.check()
    command = list(map(str, command))
    append(run / "commands.jsonl", {"command": command})
    with tempfile.TemporaryFile(mode="w+") as stderr:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = process.communicate(timeout=budget.timeout(timeout))
        except BaseException:
            stop_process(process)
            raise
        finally:
            process.stdout.close()
        if process.returncode:
            stderr.seek(0)
            raise StudyError(
                f"Command failed ({process.returncode}): {command}\n{stderr.read()[-4000:]}"
            )
    return stdout


class Scorer:
    """A single live reference cache, no concurrent metric work during timing."""

    def __init__(self, config, image, run, budget):
        self.config, self.image, self.run, self.budget = config, image, run, budget
        self.process = None

    def __enter__(self):
        if metric_name(self.config) == "butteraugli":
            self.info = {"mode": "pairwise-linear-srgb"}
            return self
        command = [self.config["scorer"], self.image["pfm_path"], "auto"]
        append(self.run / "commands.jsonl", {"command": command})
        self.errors = tempfile.TemporaryFile(mode="w+")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.errors,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        try:
            self.info = self.response()
            if (self.info.get("width"), self.info.get("height")) != (
                self.image["width"],
                self.image["height"],
            ) or self.info.get("fast_ssim2_revision") != METRIC_REVISION:
                raise StudyError("Unexpected metric dimensions or version")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def response(self):
        if not select.select(
            [self.process.stdout], [], [], self.budget.timeout(self.config["timeout"])
        )[0]:
            self.budget.check()
            raise StudyError("Metric subprocess timed out")
        line = self.process.stdout.readline()
        if not line:
            self.errors.seek(0)
            raise StudyError("Metric exited: " + self.errors.read()[-2000:])
        result = json.loads(line)
        if "error" in result:
            raise StudyError("Metric: " + result["error"])
        return result

    def score(self, codestream):
        with tempfile.TemporaryDirectory(prefix="cjxl-quality-decode-") as temp:
            decoded = Path(temp) / "decoded.pfm"
            execute(
                [self.config["djxl"], codestream, decoded, "--color_space=" + COLOR],
                self.run,
                self.budget,
                self.config["timeout"],
            )
            if metric_name(self.config) == "butteraugli":
                result = parse_butteraugli(
                    execute(
                        [
                            self.config["scorer"],
                            self.image["pfm_path"],
                            decoded,
                            "--colorspace",
                            COLOR,
                            "--intensity_target",
                            self.config["intensity_target"],
                            "--pnorm",
                            "3",
                        ],
                        self.run,
                        self.budget,
                        self.config["timeout"],
                    )
                )
            else:
                self.process.stdin.write(json.dumps({"path": str(decoded)}) + "\n")
                self.process.stdin.flush()
                result = self.response()
            score = result["score"]
            if not isinstance(score, (float, int)) or not math.isfinite(score):
                raise StudyError("Metric produced a non-finite score")
            return {
                **{
                    key: result[key]
                    for key in ("score", "butteraugli_pnorm3", "metric_stdout")
                    if key in result
                },
                "metric": metric_name(self.config),
                "decoded_sha256": digest(decoded),
                "scoring_mode": self.info["mode"],
            }

    def __exit__(self, *exc):
        if self.process:
            stop_process(self.process)
            self.process.stdin.close()
            self.process.stdout.close()
        if hasattr(self, "errors"):
            self.errors.close()


def numbers(text, kind=int):
    values = sorted(set(kind(v) for v in text.split(",")))
    if not values or not all(math.isfinite(v) for v in values):
        raise argparse.ArgumentTypeError(
            "Expected a nonempty comma-separated numeric list"
        )
    return values


def initialize(args):
    corpus = json.loads(args.corpus.read_text())
    images = sorted(corpus["images"], key=lambda r: r["image_id"])
    if args.pilot:
        images = [r for r in images if r["corpus"] == "kodak"][:3]
        if len(images) != 3:
            raise StudyError("Pilot requires three Kodak images")
    elif args.images:
        selected = set(args.images.split(","))
        images = [r for r in images if r["image_id"] in selected]
        if selected != {r["image_id"] for r in images}:
            raise StudyError("Unknown image IDs")
    elif args.all_images:
        pass
    else:
        raise StudyError("Specify --pilot or an explicit --images list")
    if args.encoder == "gjxl":
        return initialize_gjxl(args, images)
    metadata_path = args.source_run / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    ordinary = metadata["configuration"]["build_records"]["ordinary"]["content"]
    binaries = ordinary["binaries"]
    for key in ("benchmark", "djxl"):
        verify_file(binaries[key]["path"], binaries[key]["sha256"])
    # Include the local dynamic libraries as well as executable hashes.
    libraries = sorted((Path(ordinary["build"]) / "lib").glob("*.dylib"))
    libraries += sorted((Path(ordinary["build"]) / "lib").glob("*.so*"))
    files = {str(p.resolve()): digest(p) for p in libraries if p.is_file()}
    for key in ("benchmark", "djxl"):
        files[binaries[key]["path"]] = binaries[key]["sha256"]
    scorer = args.scorer.resolve()
    if args.metric == "butteraugli":
        expected_scorer = Path(binaries["djxl"]["path"]).with_name("butteraugli_main")
        if scorer != expected_scorer.resolve():
            raise StudyError("Use butteraugli_main from the frozen decoder build")
        version = {
            "implementation": "libjxl/butteraugli_main",
            "source_build_revision": ordinary["libjxl_revision"],
            "sha256": digest(scorer),
            "primary_score": "distance",
            "auxiliary_score": "3-norm",
            "color_space": COLOR,
            "intensity_target": args.intensity_target,
            "hf_asymmetry": 1.0,
            "xmul": 1.0,
        }
    else:
        version = json.loads(
            subprocess.check_output([str(scorer), "--version"], text=True)
        )
        if version.get("fast_ssim2_revision") != METRIC_REVISION:
            raise StudyError("Use the pinned quality_metric adapter")
    files[str(scorer)] = digest(scorer)
    with args.tuples.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["job_id"] for r in rows}) != len(rows):
        raise StudyError("Duplicate tuple IDs")
    selected_rows = [
        r for r in rows if r["image_id"] in {i["image_id"] for i in images}
    ]
    if not images or {r["image_id"] for r in selected_rows} != {
        i["image_id"] for i in images
    }:
        raise StudyError("Selected images must all have source tuples")
    if any(int(r["thread_count"]) != 8 for r in selected_rows):
        raise StudyError("Source timings must use the same eight-worker configuration")
    for image in images:
        if image["color_encoding"] != COLOR:
            raise StudyError("Only canonical linear-sRGB PFM inputs are supported")
        verify_file(image["pfm_path"], image["pfm_sha256"])
    efforts = sorted({int(r["effort"]) for r in selected_rows})
    if args.pilot and efforts != list(range(1, 11)):
        raise StudyError("Pilot requires existing outputs at efforts 1-10")
    measured = [] if args.metric == "butteraugli" else args.measurement_efforts
    if not set(measured) <= set(efforts):
        raise StudyError("Measurement efforts are missing from the source tuples")
    config = {
        "schema_version": SCHEMA,
        "metric": args.metric,
        "metric_version": version,
        "preview_only": args.metric == "butteraugli",
        "intensity_target": args.intensity_target,
        "targets": args.targets,
        "tolerance": args.tolerance,
        "images": images,
        "efforts": efforts,
        "measurement_efforts": measured,
        "scorer": str(scorer),
        "djxl": binaries["djxl"]["path"],
        "benchmark": binaries["benchmark"]["path"],
        "libjxl_revision": ordinary["libjxl_revision"],
        "tool_hashes": files,
        "source_run": str(args.source_run.resolve()),
        "source_metadata_sha256": digest(metadata_path),
        "tuple_snapshot_sha256": digest(args.tuples),
        "source_tuples": str(args.tuples.resolve()),
        "corpus_sha256": digest(args.corpus),
        "num_threads": 8,
        "repetitions": 5,
        "warmups": 1,
        "seed": 20260908,
        "minimum_distance": args.minimum_distance,
        "maximum_distance": args.maximum_distance,
        "max_evaluations": args.max_evaluations,
        "timeout": args.timeout,
        "pilot": args.pilot,
        "collector_snapshot_sha256": digest(__file__),
    }
    config["configuration_id"] = identity(config)
    if args.run.exists():
        raise StudyError(
            "Run directory already exists; use its collection commands to resume"
        )
    args.run.mkdir(parents=True)
    write_json(args.run / "metadata.json", config)
    # Snapshot the input bytes, rather than depending on a changing partial CSV.
    (args.run / "source-tuples.csv").write_bytes(args.tuples.read_bytes())
    (args.run / "collector.py").write_bytes(Path(__file__).read_bytes())
    append(
        args.run / "execution-events.jsonl",
        {"event": "initialized", "configuration_id": config["configuration_id"]},
    )
    print(f"Initialized {args.run}: {len(images)} images; no collection started")


def load_config(run, verify=False):
    config = json.loads((run / "metadata.json").read_text())
    expected = dict(config)
    identifier = expected.pop("configuration_id")
    if identity(expected) != identifier:
        raise StudyError("Configuration changed; initialize a separate run")
    if encoder_name(config) == "libjxl":
        verify_file(run / "source-tuples.csv", config["tuple_snapshot_sha256"])
    else:
        verify_file(run / "encoder-build.json", config["encoder_build_sha256"])
        if config.get("tuple_snapshot_sha256"):
            verify_file(run / "source-tuples.csv", config["tuple_snapshot_sha256"])
    if config.get("sweep_helpers_sha256"):
        verify_file(run / "cjxl_sweep_common.py", config["sweep_helpers_sha256"])
    if config.get("seed_snapshot_sha256"):
        verify_file(run / "seed-scores.json", config["seed_snapshot_sha256"])
    if verify:
        if config.get("collector_snapshot_sha256"):
            verify_file(run / "collector.py", config["collector_snapshot_sha256"])
        for path, sha in config["tool_hashes"].items():
            verify_file(path, sha)
        for image in config["images"]:
            verify_file(image["pfm_path"], image["pfm_sha256"])
    return config


def read_records(run, name, config, verify=False):
    records = ledger(run / (name + ".jsonl"))
    seen = set()
    for row in records:
        if row.get("configuration_id") != config["configuration_id"]:
            raise StudyError("Mixed configurations in " + name)
        if verify and row.get("output_path"):
            verify_file(row["output_path"], row["output_sha256"])
        if verify and row.get("raw_path"):
            verify_file(row["raw_path"], row["raw_sha256"])
        # Calibration can contain repeated failure outcomes; observations cannot.
        key = None
        if name == "scores":
            key = row.get("source_job_id")
        elif name == "measurements":
            key = (row.get("match_id"), row.get("repetition"))
        elif name == "probes":
            key = (row.get("image_id"), row.get("effort"), row.get("distance"))
        elif name == "warmup-checks":
            key = (row.get("image_id"), row.get("warmups"), row.get("repetition"))
        elif name == "timings":
            key = row["sample_id"]
        if key is not None:
            if key in seen:
                raise StudyError("Duplicate observation in " + name)
            seen.add(key)
    return records


def save_record(run, name, config, row):
    append(
        run / (name + ".jsonl"), dict(row, configuration_id=config["configuration_id"])
    )


def source_rows(run, config):
    if is_fixed(config):
        # Freeze a score's associated timing statistics only after all repetitions.
        return [row for row in fixed_rows(run, config) if row["timing_complete"]]
    if encoder_name(config) == "gjxl" and not config.get("tuple_snapshot_sha256"):
        return []
    with (run / "source-tuples.csv").open(newline="") as f:
        return [
            r
            for r in csv.DictReader(f)
            if r["image_id"] in {i["image_id"] for i in config["images"]}
        ]


def base_row(image, effort, distance, config=None):
    encoder = encoder_name(config or {})
    return {
        **{key: image[key] for key in SOURCE_FIELDS},
        "encoder": encoder,
        "pixels": image["width"] * image["height"],
        "effort": effort,
        "distance": distance,
        "reference_sha256": image["pfm_sha256"],
        # The frozen libjxl baseline automatically resamples at distance >= 10.
        "resampling": None
        if distance is None
        else (2 if encoder == "libjxl" and distance >= 10 else 1),
    }


def collect_scores(run, config, budget, efforts=None):
    existing = read_records(run, "scores", config, verify=True)
    done = {r["source_job_id"] for r in existing}
    inputs = source_rows(run, config)
    for image in config["images"]:
        jobs = [
            r
            for r in inputs
            if r["image_id"] == image["image_id"]
            and r["job_id"] not in done
            and (efforts is None or int(r["effort"]) in efforts)
        ]
        if not jobs:
            continue
        budget.check()
        with Scorer(config, image, run, budget) as scorer:
            for row in sorted(
                jobs, key=lambda r: (int(r["effort"]), float(r["distance"]))
            ):
                budget.check()
                verify_file(row["output_path"], row["output_sha256"])
                result = {
                    **base_row(
                        image, int(row["effort"]), float(row["distance"]), config
                    ),
                    **scorer.score(row["output_path"]),
                    "source_job_id": row["job_id"],
                    "output_path": row["output_path"],
                    "output_sha256": row["output_sha256"],
                    "encoded_bytes": Path(row["output_path"]).stat().st_size,
                    "elapsed_ms": float(row["complete_encode_median_ms"]),
                    "timing_sample_count": int(row["timing_sample_count"]),
                    "requested_quality": float(row["quality"]),
                }
                save_record(run, "scores", config, result)
                budget.completed += 1
                print(f"score {row['job_id']}: {result['score']:.4f}", flush=True)


def interpolate(points, target, higher_is_better=True):
    """Local monotone interpolation; never sort by score and hide reversals."""
    points = sorted(points, key=lambda p: p["distance"])
    if any(a["distance"] == b["distance"] for a, b in zip(points, points[1:])):
        return None, "duplicate-distance"
    exact = [p for p in points if abs(p["score"] - target) < 1e-9]
    if exact:
        best = min(exact, key=lambda p: (p["encoded_bytes"], p["distance"]))
        return {**best, "bracket": [best["distance"]]}, "exact-existing"
    candidates = []
    for a, b in zip(points, points[1:]):
        direction = 1 if higher_is_better else -1
        if direction * a["score"] > direction * target > direction * b["score"]:
            if a["resampling"] != b["resampling"]:
                continue
            if a["encoded_bytes"] < b["encoded_bytes"]:
                continue
            weight = (target - a["score"]) / (b["score"] - a["score"])
            result = {
                **a,
                "score": target,
                "distance": a["distance"] + weight * (b["distance"] - a["distance"]),
                "bracket": [a["distance"], b["distance"]],
                "timing_sample_count": min(
                    a["timing_sample_count"], b["timing_sample_count"]
                ),
            }
            for key in ("encoded_bytes", "elapsed_ms"):
                if min(a[key], b[key]) <= 0:
                    return None, "nonpositive-input"
                result[key] = math.exp(
                    math.log(a[key]) * (1 - weight) + math.log(b[key]) * weight
                )
            candidates.append(result)
    if len(candidates) == 1:
        return candidates[0], "interpolated"
    return None, "ambiguous-curve" if candidates else "no-safe-bracket"


def search(points, target, tolerance, evaluate, minimum, maximum, max_evaluations):
    """Budget counts NEW probes in this attempt. Every probe is persisted by evaluate."""
    by_distance = {
        p["distance"]: p for p in points if minimum <= p["distance"] <= maximum
    }
    calls = 0
    while True:
        values = sorted(by_distance.values(), key=lambda p: p["distance"])
        matches = [p for p in values if abs(p["score"] - target) <= tolerance]
        if matches:
            return (
                min(
                    matches,
                    key=lambda p: (
                        abs(p["score"] - target),
                        p["encoded_bytes"],
                        p["distance"],
                    ),
                ),
                "matched",
                calls,
            )
        if calls >= max_evaluations:
            return None, "evaluation-budget-exhausted", calls
        brackets = [
            (a, b)
            for a, b in zip(values, values[1:])
            if (a["score"] - target) * (b["score"] - target) < 0
            and b["distance"] - a["distance"] > 1e-6
        ]
        if brackets:
            a, b = min(
                brackets,
                key=lambda ab: (
                    ab[1]["distance"] - ab[0]["distance"],
                    abs(ab[0]["score"] - target) + abs(ab[1]["score"] - target),
                    ab[0]["distance"],
                ),
            )
            candidate = (a["distance"] + b["distance"]) / 2
        else:
            # Cover endpoints, then deterministic dyadic probes for nonmonotonic curves.
            choices = [minimum, maximum]
            choices += [minimum + (maximum - minimum) * i / 8 for i in range(1, 8)]
            # Compare the same decimal representation passed to the encoder.
            # E.g. 3.1337500000000005 otherwise looks new despite a saved
            # 3.13375 probe, and prematurely stops the fallback search.
            choices = [float(format(d, ".12g")) for d in choices]
            choices = [d for d in choices if d not in by_distance]
            if not choices:
                return None, "no-resolved-bracket", calls
            candidate = choices[0]
        candidate = float(format(candidate, ".12g"))
        if candidate in by_distance:
            return None, "unresolved-discontinuity", calls
        by_distance[candidate] = evaluate(candidate)
        calls += 1


def harness(config, image, effort, distance, directory, run, budget, warmups):
    raw, output = directory / "raw.json", directory / "output.jxl"
    execute(
        [
            config["benchmark"],
            "--input",
            image["pfm_path"],
            "--raw-samples",
            raw,
            "--distance",
            format(distance, ".12g"),
            "--effort",
            effort,
            "--num-threads",
            config["num_threads"],
            "--warmups",
            warmups,
            "--samples",
            1,
            "--output",
            output,
        ],
        run,
        budget,
        config["timeout"],
    )
    document = json.loads(raw.read_text())
    if (
        document.get("stage_profile_enabled")
        or document.get("schema_version") != 1
        or document.get("timing_semantics") != "complete-encode-wall-time"
        or document.get("revision")
        != config.get("encoder_revision", config.get("libjxl_revision"))
        or document.get("thread_count") != config["num_threads"]
        or document.get("effort") != effort
        or document.get("input_width") != image["width"]
        or document.get("input_height") != image["height"]
        or not math.isclose(document["requested_distance"], distance, rel_tol=2e-7)
        or len(document["samples"]) != 1
    ):
        raise StudyError("Unexpected uninstrumented harness schema/configuration")
    if encoder_name(config) == "gjxl" and any(
        document.get(key) != value
        for key, value in {
            "encoder": "gjxl",
            "stage_profile_enabled": False,
            "backend": "metal",
            "metal_aq_mode": "fully-resident",
            "density": "default",
            "compression": "automatic",
            "collect_final_score": False,
            "thread_semantics": "maximum-participating-cpu-threads",
            "resampling": 1,
            "input_layout": "planar-linear-srgb-f32",
            "validation_encodes": 1,
            "warmups": warmups,
            "sample_count": 1,
        }.items()
    ):
        raise StudyError("Unexpected GJXL backend, policy or warmup protocol")
    sample = document["samples"][0]
    if encoder_name(config) == "gjxl" and (
        sample.get("sample_index") != 0
        or type(sample.get("elapsed_nanoseconds")) is not int
        or type(sample.get("encoded_bytes")) is not int
    ):
        raise StudyError("GJXL requires integer raw timing samples")
    if (
        sample["encoded_bytes"] != output.stat().st_size
        or sample["elapsed_nanoseconds"] <= 0
    ):
        raise StudyError("Invalid measured output size/time")
    return output, raw, sample


def fixed_key(image_id, quality, effort):
    return f"{image_id}|quality={quality}|effort={effort}"


def fixed_rows(run, config):
    """Original fixed-sweep timing columns, plus explicit encoder/coverage data."""
    helper = sweep_helpers(run, config)
    groups = collections.defaultdict(list)
    images = {i["image_id"]: i for i in config["images"]}
    expected = collections.Counter(
        (i["corpus"], i["resolution_class"]) for i in images.values()
    )
    for row in read_records(run, "timings", config):
        if (
            row["image_id"] not in images
            or row["quality"] not in config["qualities"]
            or row["effort"] not in config["efforts"]
            or row["job_id"]
            != fixed_key(row["image_id"], row["quality"], row["effort"])
            or not 0 <= row["repetition"] < config["repetitions"]
            or row["sample_id"] != f"{row['job_id']}|repetition={row['repetition']}"
        ):
            raise StudyError("Timing observation is outside the fixed grid")
        groups[row["job_id"]].append(row)
    rows = []
    for key, samples in sorted(groups.items()):
        try:
            row = helper.summarize_timing_samples(samples)
        except helper.StudyError as error:
            raise StudyError(str(error)) from error
        image = images[row["image_id"]]
        row.update(
            encoder=encoder_name(config),
            resampling=1,
            reference_sha256=image["pfm_sha256"],
            expected_image_count=expected[(image["corpus"], image["resolution_class"])],
            timing_complete=len(samples) == config["repetitions"],
            thread_semantics=config["thread_semantics"],
        )
        rows.append(row)
    return rows


def collect_fixed_timings(run, config, budget, efforts=None):
    """Five fresh processes per tuple, with output parity and raw report retention."""
    require_encoding_allowed(config)
    helper = sweep_helpers(run, config)
    existing = read_records(run, "timings", config, verify=True)
    # Also check tuple/repetition membership and cross-repetition consistency.
    fixed_rows(run, config)
    done = {r["sample_id"] for r in existing}
    selected = sorted(e for e in config["efforts"] if efforts is None or e in efforts)
    for effort in selected:
        for repetition in range(config["repetitions"]):
            jobs = helper.shuffled_tuple_jobs(
                config["images"],
                config["qualities"],
                (effort,),
                config["seed"],
                "timing",
                repetition,
                helper.SCHEDULE_EFFORT_MAJOR,
            )
            for position, (image, quality, _) in enumerate(jobs):
                key = fixed_key(image["image_id"], quality, effort)
                sample_id = f"{key}|repetition={repetition}"
                if sample_id in done:
                    continue
                budget.check()
                distance = config["quality_to_distance"][str(quality)]
                retained = (
                    run
                    / "outputs"
                    / image["image_id"]
                    / f"q{quality:03d}-e{effort:02d}.jxl"
                )
                with tempfile.TemporaryDirectory(prefix="gjxl-fixed-") as temp:
                    output, raw, sample = harness(
                        config,
                        image,
                        effort,
                        distance,
                        Path(temp),
                        run,
                        budget,
                        config["warmups"],
                    )
                    if retained.exists():
                        verify_file(output, digest(retained))
                    else:
                        retained.parent.mkdir(parents=True, exist_ok=True)
                        output.replace(retained)
                    raw_path = (
                        run / "raw" / ("fixed-" + identity(sample_id)[:24] + ".json")
                    )
                    raw_path.parent.mkdir(exist_ok=True)
                    raw.replace(raw_path)
                    row = {
                        **base_row(image, effort, distance, config),
                        "job_id": key,
                        "sample_id": sample_id,
                        "quality": quality,
                        "repetition": repetition,
                        "thread_count": config["num_threads"],
                        "elapsed_nanoseconds": sample["elapsed_nanoseconds"],
                        "encoded_bytes": sample["encoded_bytes"],
                        "output_path": str(retained),
                        "output_sha256": digest(retained),
                        "raw_path": str(raw_path),
                        "raw_sha256": digest(raw_path),
                        "harness_revision": config["encoder_revision"],
                        "schedule_position": position,
                        "schedule_policy": helper.SCHEDULE_EFFORT_MAJOR,
                        "schedule_seed": helper.phase_shuffle_seed(
                            config["seed"], "timing", repetition, effort
                        ),
                    }
                    save_record(run, "timings", config, row)
                    done.add(sample_id)
                    budget.completed += 1
                    print(f"fixed {sample_id}", flush=True)


def fixed_completion(run, config):
    rows = fixed_rows(run, config)
    expected = len(config["images"]) * len(config["efforts"]) * len(config["qualities"])
    scored = {r["source_job_id"] for r in read_records(run, "scores", config)}
    complete = sum(r["timing_complete"] for r in rows)
    score_count = sum(r["job_id"] in scored for r in rows)
    return {
        "expected_tuples": expected,
        "timed_tuples": complete,
        "timing_samples": sum(r["timing_sample_count"] for r in rows),
        "expected_timing_samples": expected * config["repetitions"],
        "scored_tuples": score_count,
        "complete": complete == score_count == expected,
    }


def match_key(image_id, effort, target):
    return f"{image_id}|effort={effort}|target={target:g}"


def collect_calibration(run, config, budget, efforts=None):
    require_encoding_allowed(config)
    if is_fixed(config):
        raise StudyError(
            "Fixed sweeps do not calibrate; initialize a matched run with --seed-run"
        )
    scores = read_records(run, "scores", config, verify=True)
    probes = read_records(run, "probes", config, verify=True)
    outcomes = read_records(run, "calibration", config, verify=True)
    done = {r["match_id"] for r in outcomes if r["status"] == "matched"}
    for effort in config["measurement_efforts"]:
        if efforts is not None and effort not in efforts:
            continue
        for image in config["images"]:
            pending = [
                t
                for t in config["targets"]
                if match_key(image["image_id"], effort, t) not in done
            ]
            if not pending:
                continue
            budget.check()
            with Scorer(config, image, run, budget) as scorer:
                for target in pending:
                    budget.check()
                    key = match_key(image["image_id"], effort, target)
                    candidates = [
                        p
                        for p in scores + probes
                        if p["image_id"] == image["image_id"] and p["effort"] == effort
                    ]

                    def evaluate(distance):
                        budget.check()
                        token = identity([image["image_id"], effort, distance])[:24]
                        destination = run / "encodes" / (token + ".jxl")
                        destination.parent.mkdir(exist_ok=True)
                        with tempfile.TemporaryDirectory(
                            prefix="cjxl-quality-probe-"
                        ) as temp:
                            output, raw, sample = harness(
                                config,
                                image,
                                effort,
                                distance,
                                Path(temp),
                                run,
                                budget,
                                0,
                            )
                            result = {
                                **base_row(image, effort, distance, config),
                                **scorer.score(output),
                                "output_sha256": digest(output),
                                "output_path": str(destination),
                                "encoded_bytes": sample["encoded_bytes"],
                                "match_id": key,
                            }
                            output.replace(destination)
                            (run / "raw").mkdir(exist_ok=True)
                            raw.replace(run / "raw" / (token + ".json"))
                            save_record(run, "probes", config, result)
                            probes.append(result)
                            return result

                    # Count previously attempted probes for this target, preserving its total cap on resume.
                    used = sum(p.get("match_id") == key for p in probes)
                    best, status, _ = search(
                        candidates,
                        target,
                        config["tolerance"],
                        evaluate,
                        config["minimum_distance"],
                        config["maximum_distance"],
                        max(0, config["max_evaluations"] - used),
                    )
                    outcome = {
                        "match_id": key,
                        "image_id": image["image_id"],
                        "effort": effort,
                        "target": target,
                        "status": status,
                    }
                    if best:
                        # Retain one named-by-match artifact even when a seed from
                        # the original sweep already meets the target.
                        budget.check()
                        matched_dir = run / "matched"
                        matched_dir.mkdir(exist_ok=True)
                        token = identity([key, best["output_sha256"]])[:24]
                        matched_output = matched_dir / (token + ".jxl")
                        if not matched_output.exists():
                            temporary = matched_output.with_suffix(".jxl.tmp")
                            shutil.copyfile(best["output_path"], temporary)
                            verify_file(temporary, best["output_sha256"])
                            temporary.replace(matched_output)
                        verify_file(matched_output, best["output_sha256"])
                        outcome.update(best)
                        outcome.update(
                            match_id=key,
                            target=target,
                            status=status,
                            score_error=best["score"] - target,
                            output_path=str(matched_output),
                        )
                    save_record(run, "calibration", config, outcome)
                    budget.completed += 1
                    print(
                        f"calibrate {key}: {status}"
                        + (f" score={best['score']:.4f}" if best else ""),
                        flush=True,
                    )


def collect_measurements(run, config, budget, efforts=None):
    require_encoding_allowed(config)
    matches = {
        r["match_id"]: r for r in read_records(run, "calibration", config, verify=True)
    }
    selected = [r for r in matches.values() if r["status"] == "matched"]
    existing = read_records(run, "measurements", config, verify=True)
    done = {(r["match_id"], r["repetition"]) for r in existing}
    images = {i["image_id"]: i for i in config["images"]}
    for repetition in range(config["repetitions"]):
        for effort in config["measurement_efforts"]:
            if efforts is not None and effort not in efforts:
                continue
            jobs = sorted(
                (m for m in selected if m["effort"] == effort),
                key=lambda m: m["match_id"],
            )
            random.Random(config["seed"] + repetition * 100 + effort).shuffle(jobs)
            for match in jobs:
                if (match["match_id"], repetition) in done:
                    continue
                budget.check()
                if abs(match["score"] - match["target"]) > config["tolerance"]:
                    raise StudyError(
                        "Out-of-tolerance calibration cannot be measured as matched"
                    )
                with tempfile.TemporaryDirectory(
                    prefix="cjxl-quality-measure-"
                ) as temp:
                    output, raw, sample = harness(
                        config,
                        images[match["image_id"]],
                        effort,
                        match["distance"],
                        Path(temp),
                        run,
                        budget,
                        config["warmups"],
                    )
                    verify_file(output, match["output_sha256"])
                    token = identity([match["match_id"], repetition])[:24]
                    (run / "raw").mkdir(exist_ok=True)
                    raw.replace(run / "raw" / ("timing-" + token + ".json"))
                    row = {
                        **match,
                        "repetition": repetition,
                        "elapsed_ms": sample["elapsed_nanoseconds"] / 1e6,
                        "encoded_bytes": sample["encoded_bytes"],
                    }
                    save_record(run, "measurements", config, row)
                    budget.completed += 1
                    print(
                        f"measure {match['match_id']} repetition={repetition + 1}",
                        flush=True,
                    )


def per_image_rows(run, config, mode):
    scores = read_records(run, "scores", config)
    measurements = read_records(run, "measurements", config)
    matches = {r["match_id"]: r for r in read_records(run, "calibration", config)}
    efforts = config["efforts"] if mode == "preview" else config["measurement_efforts"]
    rows = []
    for image in config["images"]:
        for target in config["targets"]:
            for effort in efforts:
                row = {
                    **base_row(image, effort, None, config),
                    "mode": mode,
                    "target": target,
                    "metric": metric_name(config),
                    "status": "missing",
                    "tolerance": config["tolerance"],
                }
                if mode == "preview":
                    points = [
                        r
                        for r in scores
                        if r["image_id"] == image["image_id"] and r["effort"] == effort
                    ]
                    result, status = interpolate(
                        points, target, METRICS[metric_name(config)]["higher_is_better"]
                    )
                    if result:
                        row.update(
                            {
                                k: result[k]
                                for k in (
                                    "encoded_bytes",
                                    "elapsed_ms",
                                    "score",
                                    "distance",
                                    "timing_sample_count",
                                    "bracket",
                                    "resampling",
                                )
                            }
                        )
                        row["status"] = "ready"
                        row["preview_method"] = status
                        row["provisional_timing"] = (
                            result["timing_sample_count"] < config["repetitions"]
                        )
                    else:
                        row["status"] = status
                else:
                    key = match_key(image["image_id"], effort, target)
                    match = matches.get(key)
                    if match:
                        row["status"] = match["status"]
                    if match and match["status"] == "matched":
                        samples = [r for r in measurements if r["match_id"] == key]
                        if len({r["repetition"] for r in samples}) != len(samples):
                            raise StudyError("Duplicate measurement repetitions")
                        for sample in samples:
                            if (
                                sample["output_sha256"] != match["output_sha256"]
                                or sample["distance"] != match["distance"]
                            ):
                                raise StudyError(
                                    "Measurements refer to a different calibration"
                                )
                        row.update(
                            {
                                k: match[k]
                                for k in (
                                    "distance",
                                    "score",
                                    "score_error",
                                    "encoded_bytes",
                                    "resampling",
                                )
                            }
                        )
                        row["timing_sample_count"] = len(samples)
                        if (
                            set(r["repetition"] for r in samples)
                            == set(range(config["repetitions"]))
                            and abs(match["score"] - target) <= config["tolerance"]
                        ):
                            row["elapsed_ms"] = statistics.median(
                                r["elapsed_ms"] for r in samples
                            )
                            row["status"] = "ready"
                        else:
                            row["status"] = "incomplete-timing"
                rows.append(row)
    return rows


def aggregate_points(rows):
    """Fixed manifest cohort. No intersection that silently drops hard images."""
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[
            (
                metric_name(row),
                row["mode"],
                row["corpus"],
                row["resolution_class"],
                row["target"],
                row["encoder"],
                row["effort"],
            )
        ].append(row)
    result = []
    for key, items in sorted(grouped.items()):
        if len({r["image_id"] for r in items}) != len(items):
            raise StudyError("Duplicate images in aggregate")
        out = dict(
            zip(
                (
                    "metric",
                    "mode",
                    "corpus",
                    "resolution_class",
                    "target",
                    "encoder",
                    "effort",
                ),
                key,
            )
        )
        ready = [r for r in items if r["status"] == "ready"]
        out.update(
            image_count=len(items),
            ready_count=len(ready),
            tolerance=items[0].get("tolerance", 0.5),
            cohort=[r["image_id"] for r in items],
            status="missing-coverage",
        )
        if len(ready) == len(items):
            pixels = sum(r["pixels"] for r in items)
            elapsed = sum(r["elapsed_ms"] for r in items)
            if pixels <= 0 or elapsed <= 0:
                raise StudyError("Nonpositive aggregate size/time")
            out.update(
                status="ready",
                bits_per_pixel=8 * sum(r["encoded_bytes"] for r in items) / pixels,
                throughput_mp_s=pixels / 1000 / elapsed,
                min_score=min(r["score"] for r in items),
                max_score=max(r["score"] for r in items),
                max_abs_score_error=max(abs(r["score"] - r["target"]) for r in items),
                provisional_timing=any(
                    r.get("provisional_timing", False) for r in items
                ),
            )
        result.append(out)
    return result


def encoder_comparison_points(libjxl_run, gjxl_run, image_ids=None):
    """Measured encoder comparison, not the identical-codestream metric study."""
    runs = [Path(libjxl_run), Path(gjxl_run)]
    configs = [load_config(run) for run in runs]
    a, b = configs
    if any(is_fixed(config) for config in configs):
        raise StudyError(
            "Measured encoder comparison requires matched runs, not a fixed-sweep preview"
        )
    if [encoder_name(config) for config in configs] != ["libjxl", "gjxl"]:
        raise StudyError("Encoder comparison requires a libjxl run and a GJXL run")
    if metric_name(a) != metric_name(b) or a["metric_version"] != b["metric_version"]:
        raise StudyError("Encoder comparison requires the same metric implementation")
    for key in (
        "targets",
        "tolerance",
        "intensity_target",
        "repetitions",
        "warmups",
        "num_threads",
    ):
        if a[key] != b[key]:
            raise StudyError(f"Encoder comparison has incompatible {key}")
    if a["tool_hashes"][a["djxl"]] != b["tool_hashes"][b["djxl"]]:
        raise StudyError("Encoder comparison requires the same pinned decoder")
    selected = (
        list(image_ids)
        if image_ids is not None
        else [i["image_id"] for i in b["images"]]
    )
    if not selected or len(set(selected)) != len(selected):
        raise StudyError("Select a nonempty, unique comparison cohort")
    references = []
    for config in configs:
        images = {i["image_id"]: i for i in config["images"]}
        if not set(selected) <= images.keys():
            raise StudyError("Comparison cohort is absent from one of the runs")
        references.append(
            [
                tuple(images[name][key] for key in (*SOURCE_FIELDS, "pfm_sha256"))
                for name in selected
            ]
        )
    if references[0] != references[1]:
        raise StudyError("Encoder comparison reference hashes or geometry differ")
    efforts = set(b["measurement_efforts"])
    if not efforts <= set(a["measurement_efforts"]):
        raise StudyError("Comparison efforts are absent from the libjxl study")
    points = []
    for run, config in zip(runs, configs):
        rows = [
            row
            for row in per_image_rows(run, config, "measured")
            if row["image_id"] in selected and row["effort"] in efforts
        ]
        points.extend(aggregate_points(rows))
    return points


def comparison_studies(run, compare_run):
    """Read-only comparison of the same references, encodes, and source timings."""
    studies = []
    for path in (Path(run), Path(compare_run)):
        config = load_config(path)
        scores = read_records(path, "scores", config)
        studies.append((path, config, scores))
    _, a, scores_a = studies[0]
    _, b, scores_b = studies[1]
    if metric_name(a) == metric_name(b):
        raise StudyError("Comparison requires two different metrics")

    def cohort(config):
        return sorted(
            (i["image_id"], i["pfm_sha256"], i["width"], i["height"])
            for i in config["images"]
        )

    if (
        cohort(a) != cohort(b)
        or a["tuple_snapshot_sha256"] != b["tuple_snapshot_sha256"]
    ):
        raise StudyError(
            "Metric comparison requires identical cohorts and source timings"
        )
    rows_b = {r["source_job_id"]: r for r in scores_b}
    if {r["source_job_id"] for r in scores_a if r["effort"] in (4, 5)} != {
        r["source_job_id"] for r in scores_b if r["effort"] in (4, 5)
    }:
        raise StudyError(
            "Metric comparison requires matching e4/e5 scoring coverage; resume score"
        )
    for row in scores_a:
        other = rows_b.get(row["source_job_id"])
        if other is not None and any(
            row[k] != other[k]
            for k in (
                "output_sha256",
                "reference_sha256",
                "decoded_sha256",
                "encoded_bytes",
                "elapsed_ms",
                "timing_sample_count",
            )
        ):
            raise StudyError(
                "Metric comparison observations refer to different artifacts"
            )
    return sorted(studies, key=lambda study: metric_name(study[1]) == "butteraugli")


def effort_comparison_rows(studies):
    """Within each metric, report e5 relative to e4; never equate metric scales."""
    output = []
    for run, config, _ in studies:
        matches = per_image_rows(run, config, "preview")
        pooled = aggregate_points(matches)
        for image in [i["image_id"] for i in config["images"]] + ["pooled"]:
            for target in config["targets"]:
                source = (
                    pooled
                    if image == "pooled"
                    else [r for r in matches if r["image_id"] == image]
                )
                pairs = {r["effort"]: r for r in source if r["target"] == target}
                # A pooled row must describe exactly one corpus/resolution panel.
                if (
                    image == "pooled"
                    and len({(r["corpus"], r["resolution_class"]) for r in source}) != 1
                ):
                    continue
                out = {
                    "metric": metric_name(config),
                    "image_id": image,
                    "target": target,
                    "mode": "interpolated-preview",
                    "status": "missing-coverage",
                }
                if all(e in pairs and pairs[e]["status"] == "ready" for e in (4, 5)):
                    a, b = pairs[4], pairs[5]
                    size_key = (
                        "bits_per_pixel" if image == "pooled" else "encoded_bytes"
                    )
                    time_ratio = (
                        a["throughput_mp_s"] / b["throughput_mp_s"]
                        if image == "pooled"
                        else b["elapsed_ms"] / a["elapsed_ms"]
                    )
                    out.update(
                        status="ready",
                        e5_size_change_percent=100 * (b[size_key] / a[size_key] - 1),
                        e5_encode_time_ratio=time_ratio,
                        provisional_timing=a.get("provisional_timing", False)
                        or b.get("provisional_timing", False),
                    )
                output.append(out)
    return output


def summarize(run, config, modes=("preview", "measured"), compare_run=None):
    """Export CSV/JSON summaries; never import or invoke plotting code."""
    summary = run / "summary"
    summary.mkdir(exist_ok=True)
    if is_fixed(config):
        write_csv(summary / "image-tuples.csv", fixed_rows(run, config))
        write_json(summary / "fixed-coverage.json", fixed_completion(run, config))
    totals = {}
    for mode in modes:
        if mode == "measured" and is_fixed(config):
            continue
        if mode == "preview" and not config.get("preview_available", True):
            continue
        if mode == "measured" and config.get("preview_only"):
            continue
        rows = per_image_rows(run, config, mode)
        points = aggregate_points(rows)
        write_csv(summary / f"{mode}-matches.csv", rows)
        write_csv(summary / f"{mode}-points.csv", points)
        totals[mode] = {
            "ready_matches": sum(r["status"] == "ready" for r in rows),
            "total_matches": len(rows),
            "ready_points": sum(r["status"] == "ready" for r in points),
            "total_points": len(points),
        }
    scores = read_records(run, "scores", config)
    write_csv(summary / "quality-scores.csv", scores)
    write_json(summary / "coverage.json", totals)
    if compare_run is not None:
        studies = comparison_studies(run, compare_run)
        write_csv(summary / "effort4-vs5-preview.csv", effort_comparison_rows(studies))
    print(json.dumps(totals, indent=2))
    return totals


def collect_warmup_check(run, config, budget):
    """Opt-in sentinel study; never mixed with calibrated measurement records."""
    if encoder_name(config) != "gjxl":
        raise StudyError("The warmup check is currently a GJXL-only diagnostic")
    images = config["images"]
    sentinels = {images[0]["image_id"]: images[0]}
    large = max(images, key=lambda image: image["width"] * image["height"])
    sentinels[large["image_id"]] = large
    effort, distance = max(config["measurement_efforts"]), 1.2
    records = read_records(run, "warmup-checks", config, verify=True)
    done = {(r["image_id"], r["warmups"], r["repetition"]) for r in records}
    folder = run / "warmup-check"
    folder.mkdir(exist_ok=True)
    for repetition in range(5):
        for image in sentinels.values():
            for warmups in (1, 3) if repetition % 2 == 0 else (3, 1):
                if (image["image_id"], warmups, repetition) in done:
                    continue
                budget.check()
                with tempfile.TemporaryDirectory(prefix="gjxl-warmup-check-") as temp:
                    output, raw, sample = harness(
                        config,
                        image,
                        effort,
                        distance,
                        Path(temp),
                        run,
                        budget,
                        warmups,
                    )
                    retained = folder / (identity(image["image_id"])[:16] + ".jxl")
                    if retained.exists():
                        verify_file(output, digest(retained))
                    else:
                        output.replace(retained)
                    token = identity([image["image_id"], warmups, repetition])[:24]
                    raw.replace(folder / (token + ".json"))
                    row = dict(
                        base_row(image, effort, distance, config),
                        warmups=warmups,
                        repetition=repetition,
                        elapsed_ms=sample["elapsed_nanoseconds"] / 1e6,
                        output_path=str(retained),
                        output_sha256=digest(retained),
                    )
                    save_record(run, "warmup-checks", config, row)
                    records.append(row)
                    budget.completed += 1
                    print(
                        f"warmup-check {image['image_id']} warmups={warmups} repetition={repetition + 1}",
                        flush=True,
                    )
    summary = []
    for image_id in sentinels:
        medians = {
            warmups: statistics.median(
                r["elapsed_ms"]
                for r in records
                if r["image_id"] == image_id and r["warmups"] == warmups
            )
            for warmups in (1, 3)
        }
        change = 100 * (medians[3] / medians[1] - 1)
        summary.append(
            dict(
                image_id=image_id,
                effort=effort,
                distance=distance,
                median_ms_by_warmups=medians,
                three_vs_one_change_percent=change,
                needs_review=abs(change) > 5,
            )
        )
    write_json(
        run / "warmup-check-summary.json",
        {
            "configuration_id": config["configuration_id"],
            "results": summary,
            "interpretation": "Separate 5-pair diagnostic; changes over 5% flag review, not an automatic protocol change.",
        },
    )


def collect_full_run(run, config, budget):
    """Effort-major scoring -> calibration -> matched-output timing, resumable."""
    require_encoding_allowed(config)
    for effort in sorted(config["measurement_efforts"]):
        phases = (
            ("score", collect_scores),
            ("calibrate", collect_calibration),
            ("measure", collect_measurements),
        )
        if is_fixed(config):
            phases = (
                ("fixed-timing", collect_fixed_timings),
                ("score", collect_scores),
            )
        for name, collector in phases:
            if (
                name == "score"
                and encoder_name(config) == "gjxl"
                and not is_fixed(config)
            ):
                continue
            budget.check()
            append(
                run / "execution-events.jsonl",
                {"event": "phase_started", "phase": name, "effort": effort},
            )
            print(f"phase effort={effort} {name}", flush=True)
            collector(run, config, budget, efforts={effort})
            append(
                run / "execution-events.jsonl",
                {"event": "phase_completed", "phase": name, "effort": effort},
            )
            # Publish partial data summaries after each phase.
            summarize(run, config)
            write_json(run / "progress.json", run_completion(run, config))


def run_completion(run, config):
    if is_fixed(config):
        return fixed_completion(run, config)
    outcomes = {r["match_id"]: r for r in read_records(run, "calibration", config)}
    measurements = read_records(run, "measurements", config)
    expected = (
        len(config["images"])
        * len(config["measurement_efforts"])
        * len(config["targets"])
    )
    matched = sum(r["status"] == "matched" for r in outcomes.values())
    samples = collections.Counter(r["match_id"] for r in measurements)
    complete = sum(
        r["status"] == "matched" and samples[key] == config["repetitions"]
        for key, r in outcomes.items()
    )
    return {
        "expected_matches": expected,
        "calibration_outcomes": len(outcomes),
        "matched": matched,
        "unmatched": len(outcomes) - matched,
        "timed_matches": complete,
        "timing_samples": len(measurements),
        "complete": matched == expected and complete == expected,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="freeze inputs/configuration; does not collect")
    init.add_argument("--run", type=Path, required=True)
    init.add_argument("--corpus", type=Path, required=True)
    init.add_argument("--tuples", type=Path)
    init.add_argument("--encoder", choices=("libjxl", "gjxl"), default="libjxl")
    init.add_argument(
        "--mode",
        choices=("matched", "fixed"),
        default="matched",
        help="GJXL fixed nominal-quality grid or matched perceptual quality",
    )
    init.add_argument("--qualities", type=numbers, default=[10, 30, 50, 70, 80, 90, 95])
    init.add_argument(
        "--seed-run",
        type=Path,
        help="GJXL fixed sweep whose scored outputs seed a new matched run",
    )
    init.add_argument(
        "--benchmark", type=Path, help="GJXL quality benchmark executable"
    )
    init.add_argument(
        "--gjxl-source", type=Path, help="source checkout used for the GJXL build"
    )
    init.add_argument("--source-run", type=Path, required=True)
    init.add_argument("--scorer", type=Path, required=True)
    init.add_argument("--metric", choices=METRICS, default="fast-ssim2")
    init.add_argument("--intensity-target", type=float, default=80.0)
    selection = init.add_mutually_exclusive_group(required=True)
    selection.add_argument("--pilot", action="store_true")
    selection.add_argument("--images", help="comma-separated exact corpus IDs")
    selection.add_argument(
        "--all-images", action="store_true", help="select the full corpus manifest"
    )
    init.add_argument(
        "--targets",
        type=lambda s: numbers(s, float),
    )
    init.add_argument("--tolerance", type=float)
    init.add_argument(
        "--measurement-efforts", "--efforts", type=numbers, default=[3, 5, 7]
    )
    init.add_argument("--max-evaluations", type=int, default=12)
    init.add_argument("--timeout", type=float, default=300)
    init.add_argument("--minimum-distance", type=float, default=0.55)
    init.add_argument("--maximum-distance", type=float, default=15.266666666667)
    for name in (
        "score",
        "calibrate",
        "measure",
        "pilot",
        "run",
        "preview",
        "summarize",
    ):
        p = sub.add_parser(name)
        p.add_argument("--run", type=Path, required=True)
        p.add_argument("--dry-run", action="store_true")
        if name in ("preview", "summarize"):
            p.add_argument(
                "--compare-run",
                type=Path,
                help="saved alternative-metric run for e4/e5 diagnostics",
            )
        if name in ("score", "calibrate", "measure", "pilot", "run"):
            p.add_argument(
                "--budget-seconds", type=float, default=900 if name == "pilot" else None
            )
            p.add_argument("--max-jobs", type=int)
        if name == "run":
            p.add_argument(
                "--check-warmups",
                action="store_true",
                help="GJXL-only: run a resumable sentinel warmup check within the same budget",
            )
    args = parser.parse_args(argv)
    args.run = args.run.resolve()
    if args.command == "init":
        if (
            args.mode == "fixed" or args.seed_run is not None
        ) and args.encoder != "gjxl":
            parser.error("--mode fixed and --seed-run currently require --encoder gjxl")
        if args.mode == "fixed" and args.seed_run is not None:
            parser.error("--seed-run is for a new matched run, not a fixed sweep")
        if args.mode == "fixed" and not set(args.qualities) <= set(range(1, 100)):
            parser.error("Fixed qualities must be 1-99; lossless is not included")
        if args.encoder == "libjxl" and args.tuples is None:
            parser.error("libjxl init requires --tuples")
        if args.encoder == "gjxl" and (
            args.benchmark is None
            or args.gjxl_source is None
            or args.metric != "fast-ssim2"
        ):
            parser.error(
                "GJXL init requires --benchmark, --gjxl-source and --metric fast-ssim2"
            )
        if args.targets is None:
            args.targets = (
                [1.0, 2.0, 3.0] if args.metric == "butteraugli" else [60.0, 70.0, 85.0]
            )
        if args.tolerance is None:
            args.tolerance = 0.05 if args.metric == "butteraugli" else 0.5
        if args.metric == "butteraugli" and min(args.targets) < 0:
            parser.error("Butteraugli targets must be nonnegative")
    for key in (
        "tolerance",
        "timeout",
        "budget_seconds",
        "max_jobs",
        "max_evaluations",
        "intensity_target",
    ):
        value = getattr(args, key, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{key.replace('_', '-')} must be finite and positive")
    if args.command == "init" and (
        not args.measurement_efforts
        or not set(args.measurement_efforts) <= set(range(1, 11))
    ):
        parser.error("Efforts must be 1-10")
    if args.command == "init" and not (
        0 < args.minimum_distance < args.maximum_distance <= 25
    ):
        parser.error("Distance bounds must satisfy 0 < minimum < maximum <= 25")
    if args.command == "init" and args.pilot and args.measurement_efforts != [3, 5, 7]:
        parser.error("The bounded pilot only calibrates efforts 3,5,7")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "init":
        initialize(args)
        return 0
    collecting = args.command in ("score", "calibrate", "measure", "pilot", "run")
    config = load_config(args.run, verify=collecting and not args.dry_run)
    if args.command in ("calibrate", "measure", "pilot", "run"):
        require_encoding_allowed(config)
    if args.command == "pilot" and not config["pilot"]:
        raise StudyError("pilot requires a --pilot configuration")
    if is_fixed(config) and args.command in ("calibrate", "pilot"):
        raise StudyError(
            "Use run/measure/score for fixed sweeps; calibration requires a separate matched run"
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "command": args.command,
                    "images": [i["image_id"] for i in config["images"]],
                    "score_jobs": len(source_rows(args.run, config)),
                    "efforts": config["efforts"],
                    "measurement_efforts": config["measurement_efforts"],
                    "targets": config["targets"],
                    **(
                        {
                            "max_matched_jobs": len(config["images"])
                            * len(config["measurement_efforts"])
                            * len(config["targets"])
                        }
                        if not is_fixed(config)
                        else {}
                    ),
                    "budget_seconds": getattr(args, "budget_seconds", None),
                    **(
                        {
                            "collection_mode": "fixed",
                            "qualities": config["qualities"],
                            "fixed_grid": fixed_completion(args.run, config),
                        }
                        if is_fixed(config)
                        else {}
                    ),
                },
                indent=2,
            )
        )
        return 0
    with (args.run / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise StudyError("Another command is using this quality run") from error
        if not collecting:
            summarize(
                args.run,
                config,
                ("preview",) if args.command == "preview" else ("preview", "measured"),
                compare_run=args.compare_run,
            )
            return 0
        # Don't overlap new timing with the existing sweep.
        processes = subprocess.check_output(["ps", "-axo", "command"], text=True)
        if any(
            "cjxl_runtime_characterization.py run" in line
            for line in processes.splitlines()
        ):
            raise StudyError(
                "Pause the original runtime sweep before quality collection"
            )
        budget = Budget(args.budget_seconds, args.max_jobs)
        start, status = time.monotonic(), "complete"
        write_json(
            args.run / "runner.json",
            {
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "started_at": utc(),
                "command": args.command,
                "state": "running",
            },
        )
        append(
            args.run / "execution-events.jsonl",
            {"event": "collection_started", "command": args.command},
        )
        try:
            if getattr(args, "check_warmups", False):
                collect_warmup_check(args.run, config, budget)
            commands = (
                ("score", "calibrate", "measure")
                if args.command == "pilot"
                else (args.command,)
            )
            for command in commands:
                {
                    "score": collect_scores,
                    "calibrate": collect_calibration,
                    "measure": collect_fixed_timings
                    if is_fixed(config)
                    else collect_measurements,
                    "run": collect_full_run,
                }[command](args.run, config, budget)
        except (BudgetExpired, subprocess.TimeoutExpired):
            status = "budget-or-timeout"
            print("Stopped at budget/timeout; completed records are resumable.")
        except KeyboardInterrupt:
            status = "paused"
            print("Paused; completed records are resumable.")
        except BaseException:
            status = "error"
            raise
        finally:
            if args.command == "run" or is_fixed(config):
                report = run_completion(args.run, config)
                write_json(args.run / "progress.json", report)
                if (
                    args.command == "run"
                    and status == "complete"
                    and not report["complete"]
                ):
                    status = (
                        "incomplete-fixed-sweep"
                        if is_fixed(config)
                        else "incomplete-targets"
                    )
                    print(
                        "Pass finished with missing timings/scores; see progress.json."
                        if is_fixed(config)
                        else "Pass finished, but some targets remain unmatched; see progress.json."
                    )
            append(
                args.run / "execution-events.jsonl",
                {
                    "event": "collection_stopped",
                    "command": args.command,
                    "status": status,
                    "active_seconds": time.monotonic() - start,
                    "completed_jobs": budget.completed,
                },
            )
            write_json(
                args.run / "runner.json",
                {
                    "pid": os.getpid(),
                    "pgid": os.getpgrp(),
                    "stopped_at": utc(),
                    "command": args.command,
                    "state": status,
                },
            )
            if is_fixed(config):
                # Publish useful partial snapshots after a bounded interruption too.
                summarize(args.run, config)
        if args.command == "pilot":
            summarize(args.run, config)
        return 0 if status == "complete" else 130


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StudyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
