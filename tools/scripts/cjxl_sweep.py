#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Run reproducible cjxl quality/distance versus effort sweeps.

The raw output is append-only JSON Lines with one record per cjxl invocation.
The summary is a chart-ready CSV with one row per parameter/effort cell and
statistics computed from complete dataset passes. Separate resumable size
probes add encoded bytes, bits per pixel, and PNG-to-JXL ratios without adding
output-file I/O to the timed measurements.

Example:

  python3 tools/scripts/cjxl_sweep.py run \
      --cjxl build-samply/tools/cjxl \
      --dataset /path/to/PhotoCD_PCD0992 \
      --axis distance --values 0.5,0.75,1,1.5,2,3 \
      --efforts 1-10 --repetitions 5 --num-threads 0 \
      --output build-samply/sweeps/kodak-distance-effort

Runs are resumable when invoked again with the same output directory and
configuration. Use the summarize subcommand to regenerate summary.csv from
metadata.json, raw.jsonl, and optional sizes.jsonl. To enrich an existing run:

  python3 tools/scripts/cjxl_sweep.py sizes \
      --input build-samply/sweeps/kodak-distance-effort
"""

import argparse
import csv
import dataclasses
import datetime
import decimal
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time


SCHEMA_VERSION = 1
RAW_FILENAME = "raw.jsonl"
SIZE_RAW_FILENAME = "sizes.jsonl"
METADATA_FILENAME = "metadata.json"
SUMMARY_FILENAME = "summary.csv"
RESERVED_CJXL_OPTIONS = (
    "--disable_output",
    "--distance",
    "--effort",
    "--num_threads",
    "--quality",
    "--quiet",
)


class SweepError(Exception):
    """An expected configuration, input, or subprocess error."""


@dataclasses.dataclass(frozen=True)
class SweepValue:
    text: str
    number: float


@dataclasses.dataclass(frozen=True)
class InputImage:
    path: pathlib.Path
    relative_path: str
    size_bytes: int
    sha256: str
    width: int
    height: int

    @property
    def pixels(self):
        return self.width * self.height


@dataclasses.dataclass(frozen=True)
class Job:
    axis: str
    value: SweepValue
    effort: int
    repetition: int
    image: InputImage

    @property
    def sample_id(self):
        return "%s=%s|effort=%d|repetition=%d|image=%s" % (
            self.axis,
            self.value.text,
            self.effort,
            self.repetition,
            self.image.relative_path,
        )

    @property
    def cell_id(self):
        return "%s=%s|effort=%d" % (self.axis, self.value.text, self.effort)


@dataclasses.dataclass(frozen=True)
class SizeJob:
    axis: str
    value: SweepValue
    effort: int
    image: InputImage

    @property
    def size_id(self):
        return "%s=%s|effort=%d|image=%s" % (
            self.axis,
            self.value.text,
            self.effort,
            self.image.relative_path,
        )

    @property
    def cell_id(self):
        return "%s=%s|effort=%d" % (self.axis, self.value.text, self.effort)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def canonical_decimal(value):
    """Returns a stable non-exponent decimal spelling."""
    try:
        parsed = decimal.Decimal(value)
    except decimal.InvalidOperation as error:
        raise SweepError("Invalid numeric value: %s" % value) from error
    if not parsed.is_finite():
        raise SweepError("Numeric values must be finite: %s" % value)
    text = format(parsed, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        text = "0"
    return text


def parse_values(text, axis):
    values = []
    seen = set()
    lower, upper = (0.0, 25.0) if axis == "distance" else (0.0, 100.0)
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        canonical = canonical_decimal(item)
        number = float(decimal.Decimal(canonical))
        if number < lower or number > upper:
            raise SweepError(
                "%s value %s is outside [%g, %g]" % (axis, canonical, lower, upper)
            )
        if canonical not in seen:
            values.append(SweepValue(canonical, number))
            seen.add(canonical)
    if not values:
        raise SweepError("At least one %s value is required" % axis)
    return values


def parse_efforts(text):
    efforts = []
    seen = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", item)
        if not match:
            raise SweepError("Invalid effort or effort range: %s" % item)
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first > last:
            raise SweepError("Descending effort ranges are not supported: %s" % item)
        for effort in range(first, last + 1):
            if effort < 1 or effort > 10:
                raise SweepError("Effort %d is outside [1, 10]" % effort)
            if effort not in seen:
                efforts.append(effort)
                seen.add(effort)
    if not efforts:
        raise SweepError("At least one effort is required")
    return efforts


def validate_extra_args(extra_args):
    for argument in extra_args:
        for reserved in RESERVED_CJXL_OPTIONS:
            if argument == reserved or argument.startswith(reserved + "="):
                raise SweepError("%s is controlled by the sweep harness" % reserved)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def read_png_dimensions(path):
    with path.open("rb") as source:
        header = source.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise SweepError("Input is not a PNG file: %s" % path)
    if header[12:16] != b"IHDR":
        raise SweepError("PNG has no leading IHDR chunk: %s" % path)
    width = int.from_bytes(header[16:20], "big")
    height = int.from_bytes(header[20:24], "big")
    if width <= 0 or height <= 0:
        raise SweepError("PNG has invalid dimensions: %s" % path)
    return width, height


def discover_inputs(dataset, pattern):
    dataset = dataset.resolve()
    if dataset.is_file():
        paths = [dataset]
        root = dataset.parent
    else:
        if not dataset.is_dir():
            raise SweepError(
                "Dataset does not exist or is not a directory: %s" % dataset
            )
        paths = sorted(path for path in dataset.glob(pattern) if path.is_file())
        root = dataset
    if not paths:
        raise SweepError("No inputs matched %s under %s" % (pattern, dataset))
    images = []
    for path in paths:
        width, height = read_png_dimensions(path)
        images.append(
            InputImage(
                path=path.resolve(),
                relative_path=path.resolve().relative_to(root).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                width=width,
                height=height,
            )
        )
    return root, images


def dataset_manifest_hash(images):
    digest = hashlib.sha256()
    for image in images:
        digest.update(image.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(image.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(image.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def run_text_command(command, cwd=None):
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def git_metadata(repo_root):
    commit = run_text_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = run_text_command(["git", "status", "--porcelain=v1"], cwd=repo_root)
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def cjxl_version(cjxl):
    output = run_text_command([str(cjxl), "--version"])
    if output is None:
        raise SweepError("Could not execute %s --version" % cjxl)
    return output


def metadata_image(image):
    return {
        "path": image.relative_path,
        "size_bytes": image.size_bytes,
        "sha256": image.sha256,
        "width": image.width,
        "height": image.height,
        "pixels": image.pixels,
    }


def load_metadata_images(config):
    dataset_root = pathlib.Path(config["dataset_root"]).resolve()
    images = []
    for stored in config["images"]:
        relative_path = stored["path"]
        path = (dataset_root / relative_path).resolve()
        try:
            path.relative_to(dataset_root)
        except ValueError as error:
            raise SweepError(
                "Stored input escapes the dataset root: %s" % relative_path
            ) from error
        if not path.is_file():
            raise SweepError("Stored input no longer exists: %s" % path)
        width, height = read_png_dimensions(path)
        size_bytes = path.stat().st_size
        digest = sha256_file(path)
        if (
            width != stored["width"]
            or height != stored["height"]
            or size_bytes != stored["size_bytes"]
            or digest != stored["sha256"]
        ):
            raise SweepError("Stored input has changed since the timing run: %s" % path)
        images.append(
            InputImage(
                path=path,
                relative_path=relative_path,
                size_bytes=size_bytes,
                sha256=digest,
                width=width,
                height=height,
            )
        )
    if not images:
        raise SweepError("Stored timing run has no input images")
    if dataset_manifest_hash(images) != config["dataset_manifest_sha256"]:
        raise SweepError("Stored dataset manifest does not match its input metadata")
    return images


def create_metadata(args, cjxl, dataset_root, images, values, efforts, repo_root):
    run_config = {
        "axis": args.axis,
        "values": [dataclasses.asdict(value) for value in values],
        "efforts": efforts,
        "repetitions": args.repetitions,
        "warmups_per_cell": args.warmups,
        "num_threads": args.num_threads,
        "seed": args.seed,
        "extra_cjxl_args": args.cjxl_arg,
        "cjxl_path": str(cjxl),
        "cjxl_sha256": sha256_file(cjxl),
        "dataset_root": str(dataset_root),
        "dataset_pattern": args.pattern,
        "dataset_manifest_sha256": dataset_manifest_hash(images),
        "images": [metadata_image(image) for image in images],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "measurement_boundary": (
            "One cjxl process per image. Wall and child CPU time include process "
            "startup, PNG input decode, packed-plane conversion, and encoding. "
            "--disable_output excludes output-file writes but retains bitstream "
            "generation."
        ),
        "cjxl_version": cjxl_version(cjxl),
        "repository": git_metadata(repo_root),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "run_config": run_config,
    }


def write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_json(path):
    try:
        with path.open(encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError("Could not read %s: %s" % (path, error)) from error


def ensure_compatible_metadata(path, metadata):
    if not path.exists():
        write_json_atomic(path, metadata)
        return metadata
    existing = load_json(path)
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise SweepError("Unsupported metadata schema in %s" % path)
    if existing.get("run_config") != metadata.get("run_config") or existing.get(
        "host"
    ) != metadata.get("host"):
        raise SweepError(
            "Existing output configuration or host differs from this invocation. "
            "Choose a new --output directory to avoid mixing measurements."
        )
    return existing


def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SweepError(
                    "Invalid JSON in %s:%d: %s" % (path, line_number, error)
                ) from error
    return records


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as output:
        output.write(line)
        output.flush()
        os.fsync(output.fileno())


def build_schedule(axis, values, efforts, repetitions, images, seed):
    """Interleaves all cells for each image in every repetition."""
    cells = [(value, effort) for value in values for effort in efforts]
    rng = random.Random(seed)
    jobs = []
    for repetition in range(repetitions):
        repetition_images = list(images)
        rng.shuffle(repetition_images)
        for image in repetition_images:
            image_cells = list(cells)
            rng.shuffle(image_cells)
            jobs.extend(
                Job(axis, value, effort, repetition, image)
                for value, effort in image_cells
            )
    return jobs


def build_size_schedule(axis, values, efforts, images, seed):
    """Interleaves one size probe for each image and parameter cell."""
    rng = random.Random(seed ^ 0x51AE5)
    shuffled_images = list(images)
    rng.shuffle(shuffled_images)
    jobs = []
    cells = [(value, effort) for value in values for effort in efforts]
    for image in shuffled_images:
        image_cells = list(cells)
        rng.shuffle(image_cells)
        jobs.extend(
            SizeJob(axis, value, effort, image) for value, effort in image_cells
        )
    return jobs


def build_cjxl_command(cjxl, job, num_threads, extra_args):
    return (
        [
            str(cjxl),
            "--quiet",
            "--disable_output",
            "--%s=%s" % (job.axis, job.value.text),
            "--effort=%d" % job.effort,
            "--num_threads=%d" % num_threads,
        ]
        + list(extra_args)
        + [str(job.image.path)]
    )


def build_size_cjxl_command(cjxl, job, num_threads, extra_args, output_path):
    return (
        [
            str(cjxl),
            "--quiet",
            "--%s=%s" % (job.axis, job.value.text),
            "--effort=%d" % job.effort,
            "--num_threads=%d" % num_threads,
        ]
        + list(extra_args)
        + [str(job.image.path), str(output_path)]
    )


def usage_delta(before, after, attribute):
    return max(
        0,
        int(
            round(
                (getattr(after, attribute) - getattr(before, attribute)) * 1_000_000_000
            )
        ),
    )


def execute_command(command, timeout):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start_time = utc_now()
    start_ns = time.perf_counter_ns()
    status = "ok"
    exit_code = None
    stderr = ""
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        exit_code = result.returncode
        stderr = result.stderr
        if exit_code != 0:
            status = "error"
    except subprocess.TimeoutExpired as error:
        status = "timeout"
        stderr = error.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
    except OSError as error:
        status = "error"
        stderr = str(error)
    end_ns = time.perf_counter_ns()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "status": status,
        "exit_code": exit_code,
        "started_at": start_time,
        "finished_at": utc_now(),
        "wall_time_ns": end_ns - start_ns,
        "user_cpu_time_ns": usage_delta(before, after, "ru_utime"),
        "system_cpu_time_ns": usage_delta(before, after, "ru_stime"),
        "stderr": stderr[-16384:],
    }


def execute_job(cjxl, job, num_threads, extra_args, timeout):
    command = build_cjxl_command(cjxl, job, num_threads, extra_args)
    result = execute_command(command, timeout)
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_id": job.sample_id,
        "cell_id": job.cell_id,
        "axis": job.axis,
        "parameter_text": job.value.text,
        "parameter_value": job.value.number,
        "effort": job.effort,
        "repetition": job.repetition,
        "image": job.image.relative_path,
        "width": job.image.width,
        "height": job.image.height,
        "pixels": job.image.pixels,
        "command": command,
        **result,
    }


def execute_size_job(
    cjxl, job, num_threads, extra_args, timeout, temporary_output_path
):
    try:
        temporary_output_path.unlink()
    except FileNotFoundError:
        pass
    command = build_size_cjxl_command(
        cjxl, job, num_threads, extra_args, temporary_output_path
    )
    result = execute_command(command, timeout)
    encoded_size_bytes = None
    if result["status"] == "ok":
        try:
            encoded_size_bytes = temporary_output_path.stat().st_size
        except OSError as error:
            result["status"] = "error"
            result["stderr"] = "cjxl produced no readable output: %s" % error
        else:
            if encoded_size_bytes <= 0:
                result["status"] = "error"
                result["stderr"] = "cjxl produced an empty output"
                encoded_size_bytes = None
    try:
        temporary_output_path.unlink()
    except FileNotFoundError:
        pass
    return {
        "schema_version": SCHEMA_VERSION,
        "size_id": job.size_id,
        "cell_id": job.cell_id,
        "axis": job.axis,
        "parameter_text": job.value.text,
        "parameter_value": job.value.number,
        "effort": job.effort,
        "image": job.image.relative_path,
        "input_size_bytes": job.image.size_bytes,
        "width": job.image.width,
        "height": job.image.height,
        "pixels": job.image.pixels,
        "encoded_size_bytes": encoded_size_bytes,
        "command": command[:-1] + ["<temporary-output.jxl>"],
        "probe_wall_time_ns": result.pop("wall_time_ns"),
        "probe_user_cpu_time_ns": result.pop("user_cpu_time_ns"),
        "probe_system_cpu_time_ns": result.pop("system_cpu_time_ns"),
        **result,
    }


def completed_sample_ids(records):
    return {
        record.get("sample_id")
        for record in records
        if record.get("status") == "ok" and record.get("sample_id")
    }


def completed_size_ids(records):
    return {
        record.get("size_id")
        for record in records
        if record.get("status") == "ok"
        and record.get("size_id")
        and isinstance(record.get("encoded_size_bytes"), int)
        and record["encoded_size_bytes"] > 0
    }


def format_duration(seconds):
    if seconds < 60:
        return "%.1f s" % seconds
    if seconds < 3600:
        return "%.1f min" % (seconds / 60)
    return "%.2f h" % (seconds / 3600)


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def statistics_fields(prefix, values):
    if not values:
        return {
            prefix + "_median": None,
            prefix + "_mean": None,
            prefix + "_stdev": None,
            prefix + "_min": None,
            prefix + "_p10": None,
            prefix + "_p90": None,
            prefix + "_max": None,
        }
    return {
        prefix + "_median": statistics.median(values),
        prefix + "_mean": statistics.mean(values),
        prefix + "_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        prefix + "_min": min(values),
        prefix + "_p10": percentile(values, 0.10),
        prefix + "_p90": percentile(values, 0.90),
        prefix + "_max": max(values),
    }


def build_summary_rows(metadata, records, size_records=None):
    config = metadata["run_config"]
    expected_images = {image["path"] for image in config["images"]}
    expected_per_cell = len(expected_images) * config["repetitions"]
    expected_size_per_cell = len(expected_images)
    dataset_input_bytes = sum(image["size_bytes"] for image in config["images"])
    latest_success = {}
    for record in records:
        if record.get("status") == "ok" and record.get("sample_id"):
            latest_success[record["sample_id"]] = record
    latest_sizes = {}
    for record in size_records or []:
        if (
            record.get("status") == "ok"
            and record.get("size_id")
            and isinstance(record.get("encoded_size_bytes"), int)
            and record["encoded_size_bytes"] > 0
        ):
            latest_sizes[record["size_id"]] = record

    rows = []
    for value in config["values"]:
        for effort in config["efforts"]:
            cell_records = [
                record
                for record in latest_success.values()
                if record.get("axis") == config["axis"]
                and record.get("parameter_text") == value["text"]
                and record.get("effort") == effort
            ]
            by_repetition = {}
            for record in cell_records:
                by_repetition.setdefault(record["repetition"], {})[record["image"]] = (
                    record
                )

            complete_passes = []
            for repetition in range(config["repetitions"]):
                image_records = by_repetition.get(repetition, {})
                if set(image_records) != expected_images:
                    continue
                ordered_records = [
                    image_records[path] for path in sorted(expected_images)
                ]
                wall_ms = (
                    sum(record["wall_time_ns"] for record in ordered_records)
                    / 1_000_000
                )
                cpu_ms = (
                    sum(
                        record["user_cpu_time_ns"] + record["system_cpu_time_ns"]
                        for record in ordered_records
                    )
                    / 1_000_000
                )
                pixels = sum(record["pixels"] for record in ordered_records)
                complete_passes.append((wall_ms, cpu_ms, pixels))

            wall_values = [item[0] for item in complete_passes]
            cpu_values = [item[1] for item in complete_passes]
            wall_per_mp = [
                wall / (pixels / 1_000_000) for wall, _, pixels in complete_passes
            ]
            cpu_per_mp = [
                cpu / (pixels / 1_000_000) for _, cpu, pixels in complete_passes
            ]
            dataset_pixels = sum(image["pixels"] for image in config["images"])
            cell_size_records = [
                record
                for record in latest_sizes.values()
                if record.get("axis") == config["axis"]
                and record.get("parameter_text") == value["text"]
                and record.get("effort") == effort
                and record.get("image") in expected_images
            ]
            size_images = {record["image"] for record in cell_size_records}
            size_complete = size_images == expected_images
            dataset_encoded_bytes = (
                sum(record["encoded_size_bytes"] for record in cell_size_records)
                if size_complete
                else None
            )
            row = {
                "schema_version": SCHEMA_VERSION,
                "axis": config["axis"],
                "parameter_text": value["text"],
                "parameter_value": value["number"],
                "effort": effort,
                "num_threads": config["num_threads"],
                "sample_count": len(cell_records),
                "expected_sample_count": expected_per_cell,
                "missing_sample_count": expected_per_cell - len(cell_records),
                "complete": len(cell_records) == expected_per_cell,
                "complete_repetitions": len(complete_passes),
                "expected_repetitions": config["repetitions"],
                "image_count": len(expected_images),
                "dataset_pixels": dataset_pixels,
                "size_sample_count": len(size_images),
                "expected_size_sample_count": expected_size_per_cell,
                "missing_size_sample_count": expected_size_per_cell - len(size_images),
                "size_complete": size_complete,
                "dataset_input_bytes": dataset_input_bytes,
                "dataset_encoded_bytes": dataset_encoded_bytes,
                "bits_per_pixel": (
                    dataset_encoded_bytes * 8 / dataset_pixels
                    if dataset_encoded_bytes is not None and dataset_pixels
                    else None
                ),
                "png_to_jxl_ratio": (
                    dataset_input_bytes / dataset_encoded_bytes
                    if dataset_encoded_bytes
                    else None
                ),
            }
            row.update(statistics_fields("dataset_wall_ms", wall_values))
            row.update(statistics_fields("dataset_cpu_ms", cpu_values))
            row.update(statistics_fields("wall_ms_per_megapixel", wall_per_mp))
            row.update(statistics_fields("cpu_ms_per_megapixel", cpu_per_mp))
            rows.append(row)
    return rows


def csv_fieldnames():
    base = [
        "schema_version",
        "axis",
        "parameter_text",
        "parameter_value",
        "effort",
        "num_threads",
        "sample_count",
        "expected_sample_count",
        "missing_sample_count",
        "complete",
        "complete_repetitions",
        "expected_repetitions",
        "image_count",
        "dataset_pixels",
        "size_sample_count",
        "expected_size_sample_count",
        "missing_size_sample_count",
        "size_complete",
        "dataset_input_bytes",
        "dataset_encoded_bytes",
        "bits_per_pixel",
        "png_to_jxl_ratio",
    ]
    statistics_names = []
    for prefix in (
        "dataset_wall_ms",
        "dataset_cpu_ms",
        "wall_ms_per_megapixel",
        "cpu_ms_per_megapixel",
    ):
        statistics_names.extend(
            prefix + suffix
            for suffix in ("_median", "_mean", "_stdev", "_min", "_p10", "_p90", "_max")
        )
    return base + statistics_names


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=csv_fieldnames())
            writer.writeheader()
            for row in rows:
                serialized = dict(row)
                serialized["complete"] = "true" if row["complete"] else "false"
                serialized["size_complete"] = (
                    "true" if row["size_complete"] else "false"
                )
                writer.writerow(serialized)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def summarize_directory(output_directory, summary_path=None):
    metadata_path = output_directory / METADATA_FILENAME
    raw_path = output_directory / RAW_FILENAME
    size_raw_path = output_directory / SIZE_RAW_FILENAME
    metadata = load_json(metadata_path)
    records = read_jsonl(raw_path)
    size_records = read_jsonl(size_raw_path)
    rows = build_summary_rows(metadata, records, size_records)
    if summary_path is None:
        summary_path = output_directory / SUMMARY_FILENAME
    write_summary_csv(summary_path, rows)
    complete = sum(1 for row in rows if row["complete"])
    print("Wrote %s (%d/%d complete cells)" % (summary_path, complete, len(rows)))
    return rows


def print_dry_run(jobs, cjxl, args):
    print("Planned measured invocations: %d" % len(jobs))
    print(
        "Warm-up invocations: %d" % (len({job.cell_id for job in jobs}) * args.warmups)
    )
    for job in jobs[:5]:
        print(
            "  "
            + " ".join(build_cjxl_command(cjxl, job, args.num_threads, args.cjxl_arg))
        )
    if len(jobs) > 5:
        print("  ...")


def run_sweep(args):
    validate_extra_args(args.cjxl_arg)
    if args.repetitions < 1:
        raise SweepError("--repetitions must be positive")
    if args.warmups < 0:
        raise SweepError("--warmups must be non-negative")
    if args.num_threads < -1:
        raise SweepError("--num-threads must be -1, 0, or positive")
    if args.timeout is not None and args.timeout <= 0:
        raise SweepError("--timeout must be positive")
    if args.max_jobs < 0:
        raise SweepError("--max-jobs must be non-negative")
    if args.progress_every < 0:
        raise SweepError("--progress-every must be non-negative")

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    cjxl = pathlib.Path(args.cjxl).resolve()
    if not cjxl.is_file() or not os.access(cjxl, os.X_OK):
        raise SweepError("cjxl is not an executable file: %s" % cjxl)
    values = parse_values(args.values, args.axis)
    efforts = parse_efforts(args.efforts)
    if (args.axis == "distance" and any(value.number == 0 for value in values)) or (
        args.axis == "quality" and any(value.number == 100 for value in values)
    ):
        print(
            "warning: lossless mode is a codec-mode discontinuity; use a "
            "separate surface when comparing lossy VarDCT results",
            file=sys.stderr,
        )
    dataset_root, images = discover_inputs(pathlib.Path(args.dataset), args.pattern)
    jobs = build_schedule(
        args.axis, values, efforts, args.repetitions, images, args.seed
    )
    output_directory = pathlib.Path(args.output).resolve()

    if args.dry_run:
        print_dry_run(jobs, cjxl, args)
        return 0

    metadata = create_metadata(
        args, cjxl, dataset_root, images, values, efforts, repo_root
    )
    metadata_path = output_directory / METADATA_FILENAME
    raw_path = output_directory / RAW_FILENAME
    if raw_path.exists() and not metadata_path.exists():
        raise SweepError(
            "%s exists without %s; refusing to mix unknown results"
            % (raw_path, metadata_path)
        )
    metadata = ensure_compatible_metadata(metadata_path, metadata)
    records = read_jsonl(raw_path)
    completed = completed_sample_ids(records)
    pending = [job for job in jobs if job.sample_id not in completed]
    if args.max_jobs:
        pending = pending[: args.max_jobs]

    print(
        "Grid: %d %s values x %d efforts; %d images x %d repetitions"
        % (len(values), args.axis, len(efforts), len(images), args.repetitions)
    )
    print(
        "Measured invocations: %d total, %d already complete, %d pending now"
        % (len(jobs), len(completed), len(pending))
    )
    if pending:
        existing_wall = [
            record["wall_time_ns"] / 1_000_000_000
            for record in records
            if record.get("status") == "ok"
        ]
        if existing_wall:
            estimate = statistics.median(existing_wall) * len(pending)
            print(
                "Estimated pending runtime from existing samples: %s"
                % format_duration(estimate)
            )

    pending_cells = {}
    for job in pending:
        pending_cells.setdefault(job.cell_id, job)
    if args.warmups and pending_cells:
        warmup_jobs = list(pending_cells.values())
        random.Random(args.seed ^ 0xC0DEC0DE).shuffle(warmup_jobs)
        print(
            "Running %d warm-ups per pending cell (%d invocations)"
            % (args.warmups, len(warmup_jobs) * args.warmups)
        )
        for warmup in range(args.warmups):
            for job in warmup_jobs:
                command = build_cjxl_command(cjxl, job, args.num_threads, args.cjxl_arg)
                result = execute_command(command, args.timeout)
                if result["status"] != "ok":
                    raise SweepError(
                        "Warm-up failed for %s: %s"
                        % (job.cell_id, result["stderr"].strip())
                    )

    failure = False
    recent_wall = []
    for index, job in enumerate(pending, 1):
        record = execute_job(cjxl, job, args.num_threads, args.cjxl_arg, args.timeout)
        append_jsonl(raw_path, record)
        recent_wall.append(record["wall_time_ns"] / 1_000_000_000)
        recent_wall = recent_wall[-25:]
        if args.progress_every and (
            index == 1 or index == len(pending) or index % args.progress_every == 0
        ):
            remaining = len(pending) - index
            estimate = statistics.median(recent_wall) * remaining
            print(
                "[%d/%d] %s (%s remaining)"
                % (index, len(pending), job.sample_id, format_duration(estimate))
            )
        if record["status"] != "ok":
            failure = True
            print(
                "cjxl failed for %s: %s" % (job.sample_id, record["stderr"].strip()),
                file=sys.stderr,
            )
            if not args.keep_going:
                break

    summarize_directory(output_directory)
    return 1 if failure else 0


def run_size_probes(args):
    if args.timeout is not None and args.timeout <= 0:
        raise SweepError("--timeout must be positive")
    if args.max_jobs < 0:
        raise SweepError("--max-jobs must be non-negative")
    if args.progress_every < 0:
        raise SweepError("--progress-every must be non-negative")

    output_directory = pathlib.Path(args.input).resolve()
    metadata_path = output_directory / METADATA_FILENAME
    raw_path = output_directory / RAW_FILENAME
    size_raw_path = output_directory / SIZE_RAW_FILENAME
    if not metadata_path.is_file() or not raw_path.is_file():
        raise SweepError(
            "%s must contain both %s and %s"
            % (output_directory, METADATA_FILENAME, RAW_FILENAME)
        )
    metadata = load_json(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise SweepError("Unsupported metadata schema in %s" % metadata_path)
    config = metadata["run_config"]
    cjxl = pathlib.Path(args.cjxl or config["cjxl_path"]).resolve()
    if not cjxl.is_file() or not os.access(cjxl, os.X_OK):
        raise SweepError("cjxl is not an executable file: %s" % cjxl)
    if sha256_file(cjxl) != config["cjxl_sha256"]:
        raise SweepError(
            "Size probes must use the exact cjxl binary from the timing run"
        )

    images = load_metadata_images(config)
    values = [SweepValue(value["text"], value["number"]) for value in config["values"]]
    efforts = list(config["efforts"])
    extra_args = list(config["extra_cjxl_args"])
    validate_extra_args(extra_args)
    jobs = build_size_schedule(config["axis"], values, efforts, images, config["seed"])
    records = read_jsonl(size_raw_path)
    completed = completed_size_ids(records)
    pending = [job for job in jobs if job.size_id not in completed]
    if args.max_jobs:
        pending = pending[: args.max_jobs]

    print(
        "Size probes: %d total, %d already complete, %d pending now"
        % (len(jobs), len(completed), len(pending))
    )
    if pending:
        existing_wall = [
            record["probe_wall_time_ns"] / 1_000_000_000
            for record in records
            if record.get("status") == "ok" and record.get("probe_wall_time_ns")
        ]
        if existing_wall:
            estimate = statistics.median(existing_wall) * len(pending)
            print(
                "Estimated pending runtime from existing probes: %s"
                % format_duration(estimate)
            )

    if args.dry_run:
        for job in pending[:5]:
            command = build_size_cjxl_command(
                cjxl,
                job,
                config["num_threads"],
                extra_args,
                pathlib.Path("<temporary-output.jxl>"),
            )
            print("  " + " ".join(str(argument) for argument in command))
        if len(pending) > 5:
            print("  ...")
        return 0

    failure = False
    recent_wall = []
    with tempfile.TemporaryDirectory(prefix="cjxl-size-probes-") as temporary:
        temporary_output_path = pathlib.Path(temporary) / "probe.jxl"
        for index, job in enumerate(pending, 1):
            record = execute_size_job(
                cjxl,
                job,
                config["num_threads"],
                extra_args,
                args.timeout,
                temporary_output_path,
            )
            append_jsonl(size_raw_path, record)
            recent_wall.append(record["probe_wall_time_ns"] / 1_000_000_000)
            recent_wall = recent_wall[-25:]
            if args.progress_every and (
                index == 1 or index == len(pending) or index % args.progress_every == 0
            ):
                remaining = len(pending) - index
                estimate = statistics.median(recent_wall) * remaining
                print(
                    "[%d/%d] %s (%s remaining)"
                    % (index, len(pending), job.size_id, format_duration(estimate))
                )
            if record["status"] != "ok":
                failure = True
                print(
                    "cjxl size probe failed for %s: %s"
                    % (job.size_id, record["stderr"].strip()),
                    file=sys.stderr,
                )
                if not args.keep_going:
                    break

    summarize_directory(output_directory)
    return 1 if failure else 0


def add_run_arguments(parser):
    parser.add_argument("--cjxl", required=True, help="Path to the cjxl executable.")
    parser.add_argument(
        "--dataset", required=True, help="PNG file or root directory containing inputs."
    )
    parser.add_argument(
        "--pattern", default="**/*.png", help="Dataset glob relative to --dataset."
    )
    parser.add_argument(
        "--axis",
        choices=("distance", "quality"),
        default="distance",
        help="Parameter to sweep; cjxl makes these exclusive.",
    )
    parser.add_argument(
        "--values", required=True, help="Comma-separated distance or quality values."
    )
    parser.add_argument(
        "--efforts",
        default="1-10",
        help="Comma-separated efforts and ranges (default 1-10).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Measured complete dataset passes per cell.",
    )
    parser.add_argument(
        "--warmups", type=int, default=1, help="Unmeasured warm-ups per pending cell."
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=0,
        help="cjxl worker count: -1 auto, 0 none, or positive.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260827,
        help="Seed for deterministic balanced scheduling.",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for metadata, JSONL, and CSV."
    )
    parser.add_argument("--timeout", type=float, help="Per-cjxl timeout in seconds.")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Limit new measured jobs; 0 means no limit.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N jobs; 0 disables it.",
    )
    parser.add_argument(
        "--cjxl-arg",
        action="append",
        default=[],
        help="Additional non-reserved cjxl argument; repeatable.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failed measured invocations.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without writing files.",
    )


def add_size_arguments(parser):
    parser.add_argument(
        "--input", required=True, help="Existing sweep output directory."
    )
    parser.add_argument(
        "--cjxl",
        help="Relocated cjxl executable; its hash must match the timing binary.",
    )
    parser.add_argument("--timeout", type=float, help="Per-cjxl timeout in seconds.")
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Limit new size probes; 0 means no limit.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N probes; 0 disables it.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failed size probes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print pending probes without writing files.",
    )


def create_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run or resume a parameter sweep.")
    add_run_arguments(run_parser)
    size_parser = subparsers.add_parser(
        "sizes", help="Run or resume untimed encoded-size probes."
    )
    add_size_arguments(size_parser)
    summarize_parser = subparsers.add_parser(
        "summarize", help="Regenerate chart-ready CSV from raw results."
    )
    summarize_parser.add_argument(
        "--input", required=True, help="Sweep output directory."
    )
    summarize_parser.add_argument(
        "--output", help="CSV path; defaults to INPUT/summary.csv."
    )
    return parser


def main(argv=None):
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_sweep(args)
        if args.command == "sizes":
            return run_size_probes(args)
        output_directory = pathlib.Path(args.input).resolve()
        summary_path = pathlib.Path(args.output).resolve() if args.output else None
        summarize_directory(output_directory, summary_path)
        return 0
    except SweepError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
