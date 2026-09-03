#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Prepare and run a repeatable libjxl runtime-characterization study.

The experiment deliberately keeps four measurements separate:

* complete-encode wall time from an uninstrumented API harness;
* exact wall-clock serializer phases from the opt-in stage profiler;
* aggregate worker time from that profiler (not latency);
* sampled thread-CPU attribution from Samply (not wall time).

The input is canonical three-channel, linear-sRGB, 32-bit-float PFM. Reading
the PFM and writing the final JXL are outside the API harness's timed region.
Runs are append-only and resumable. A unique codestream is retained for every
image/quality/effort tuple. By default, complete timing, stage, and sampled
profile results are collected for each effort before advancing to the next.
"""

import argparse
import csv
import dataclasses
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import zipfile


SCHEMA_VERSION = 1
DEFAULT_QUALITIES = (10, 30, 50, 70, 80, 90, 95)
DEFAULT_EFFORTS = tuple(range(1, 11))
PROFILE_DIAGNOSTIC_QUALITIES = frozenset((30, 90))
PROFILE_DIAGNOSTIC_EFFORTS = frozenset((3, 7, 10))
PFM_COLOR_ENCODING = "RGB_D65_SRG_Rel_Lin"
SCHEDULE_EFFORT_MAJOR = "effort-major"
SCHEDULE_PHASE_MAJOR_SHUFFLED = "phase-major-shuffled"
PHASE_SEED_OFFSETS = {
    "timing": 0,
    "stages": 100_000,
    "profiles": 200_000,
}

CLIC_2024_TEST_URL = "https://downloads.compression.cc/clic2024_image_test.zip"
CLIC_2024_TEST_SHA256 = (
    "f61a2ee3646d010f1abcfd090ae3403546499b6e95924def4a34ece76dba0379"
)
UNSPLASH_LICENSE_URL = "https://unsplash.com/license"
UNSPLASH_SOURCES = (
    {
        "id": "campus_interior",
        "photographer": "Ricardo Gomez Angel",
        "page_url": "https://unsplash.com/photos/jbCLsTtsP3s",
        "download_url": (
            "https://images.unsplash.com/photo-1680535131131-4d79b312d5d0"
            "?fm=jpg&q=100"
        ),
    },
    {
        "id": "forest_stream",
        "photographer": "Karthik Sreenivas",
        "page_url": "https://unsplash.com/photos/5CQ8kGiouQo",
        "download_url": (
            "https://images.unsplash.com/photo-1668709096799-5d34284a7502"
            "?fm=jpg&q=100"
        ),
    },
    {
        "id": "alpine_lake",
        "photographer": "Philipp",
        "page_url": "https://unsplash.com/photos/-hvC60Fen7I",
        "download_url": (
            "https://images.unsplash.com/photo-1660937472479-f0d9608e8370"
            "?fm=jpg&q=100"
        ),
    },
)
HIGH_RESOLUTION_TARGETS = (12_000_000, 24_000_000, 48_000_000)


class StudyError(Exception):
    """An expected input, configuration, or subprocess failure."""


@dataclasses.dataclass
class JobBudget:
    limit: object
    executed: int = 0

    @property
    def exhausted(self):
        return self.limit is not None and self.executed >= self.limit

    def completed_one(self):
        self.executed += 1


@dataclasses.dataclass(frozen=True)
class CorpusImage:
    image_id: str
    corpus: str
    resolution_class: str
    source_path: pathlib.Path
    pfm_path: pathlib.Path
    width: int
    height: int
    pfm_sha256: str
    source_sha256: str

    @property
    def pixels(self):
        return self.width * self.height


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        while True:
            chunk = source.read(4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_jsonl(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_jsonl(path):
    path = pathlib.Path(path)
    if not path.is_file():
        return []
    values = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise StudyError(
                    "Malformed JSON at %s:%d: %s" % (path, line_number, error)
                ) from error
    return values


def run_command(command, *, capture=True):
    try:
        return subprocess.run(
            [str(item) for item in command],
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        if isinstance(error, subprocess.CalledProcessError):
            details = "\nstdout:\n%s\nstderr:\n%s" % (
                error.stdout or "",
                error.stderr or "",
            )
        else:
            details = ""
        raise StudyError(
            "Command failed: %s%s" % (" ".join(str(item) for item in command), details)
        ) from error


def require_executable(path, label):
    resolved = shutil.which(str(path)) if "/" not in str(path) else str(path)
    if not resolved or not pathlib.Path(resolved).is_file():
        raise StudyError("%s does not exist: %s" % (label, path))
    return pathlib.Path(resolved).resolve()


def read_pfm_dimensions(path):
    try:
        with pathlib.Path(path).open("rb") as source:
            non_comments = []
            while len(non_comments) < 3:
                line = source.readline()
                if not line:
                    break
                text = line.decode("ascii").strip()
                if text and not text.startswith("#"):
                    non_comments.append(text)
        if len(non_comments) != 3 or non_comments[0] != "PF":
            raise ValueError("not a three-channel PFM")
        width, height = (int(item) for item in non_comments[1].split())
        scale = float(non_comments[2])
        if width <= 0 or height <= 0 or scale == 0 or not math.isfinite(scale):
            raise ValueError("invalid dimensions or scale")
        return width, height
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise StudyError("Invalid PFM %s: %s" % (path, error)) from error


def image_dimensions(magick, path):
    result = run_command(
        (magick, str(path), "-auto-orient", "-format", "%w %h", "info:")
    )
    try:
        width, height = (int(value) for value in result.stdout.split())
    except ValueError as error:
        raise StudyError("Could not read dimensions for %s" % path) from error
    if width <= 0 or height <= 0:
        raise StudyError("Invalid dimensions for %s" % path)
    return width, height


def target_dimensions(width, height, target_pixels):
    if width * height < target_pixels:
        raise StudyError(
            "Source %dx%d is too small for %.1f MP without upsampling"
            % (width, height, target_pixels / 1_000_000)
        )
    scale = math.sqrt(target_pixels / (width * height))
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    return target_width, target_height


def convert_to_pfm(magick, source, destination, dimensions=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=destination.stem + "-", suffix=".pfm", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = pathlib.Path(temporary.name)
    temporary_path.unlink()
    command = [
        str(magick),
        str(source),
        "-auto-orient",
        "-background",
        "white",
        "-alpha",
        "remove",
        "-alpha",
        "off",
        "-colorspace",
        "RGB",
    ]
    if dimensions is not None:
        command.extend(("-filter", "Lanczos", "-resize", "%dx%d!" % dimensions))
    command.extend(
        (
            "-define",
            "quantum:format=floating-point",
            "-depth",
            "32",
            str(temporary_path),
        )
    )
    try:
        run_command(command)
        actual = read_pfm_dimensions(temporary_path)
        if dimensions is not None and actual != dimensions:
            raise StudyError(
                "ImageMagick produced %s rather than %s for %s"
                % (actual, dimensions, source)
            )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def download(curl, url, destination):
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".download-%d" % os.getpid())
    temporary.unlink(missing_ok=True)
    try:
        run_command(
            (curl, "-L", "--fail", "--retry", "3", "--output", temporary, url),
            capture=False,
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def verify_clic_archive(directory, archive):
    if sha256_file(archive) != CLIC_2024_TEST_SHA256:
        raise StudyError("CLIC archive SHA-256 does not match the pinned archive")
    pngs = sorted(directory.glob("*.png"))
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(
            info.filename
            for info in zipped.infolist()
            if not info.is_dir() and info.filename.lower().endswith(".png")
        )
        if [path.name for path in pngs] != members:
            raise StudyError("CLIC directory filenames do not match the official archive")
        for path, member in zip(pngs, members):
            if hashlib.sha256(zipped.read(member)).hexdigest() != sha256_file(path):
                raise StudyError("CLIC file differs from official archive: %s" % path)
    return pngs


def progress_by_id(path):
    return {value["image_id"]: value for value in read_jsonl(path)}


def prepare_image(progress_path, prior, specification, magick):
    image_id = specification["image_id"]
    source = pathlib.Path(specification["source_path"]).resolve()
    destination = pathlib.Path(specification["pfm_path"]).resolve()
    source_hash = sha256_file(source)
    existing = prior.get(image_id)
    if (
        existing
        and existing.get("source_sha256") == source_hash
        and destination.is_file()
        and destination.stat().st_size == existing.get("pfm_size_bytes")
    ):
        return existing

    dimensions = specification.get("target_dimensions")
    dimensions = tuple(dimensions) if dimensions else None
    convert_to_pfm(magick, source, destination, dimensions)
    width, height = read_pfm_dimensions(destination)
    record = {
        **specification,
        "schema_version": SCHEMA_VERSION,
        "prepared_at": utc_now(),
        "source_path": str(source),
        "source_sha256": source_hash,
        "pfm_path": str(destination),
        "pfm_sha256": sha256_file(destination),
        "pfm_size_bytes": destination.stat().st_size,
        "width": width,
        "height": height,
        "pixels": width * height,
        "megapixels": width * height / 1_000_000,
        "input_layout": "interleaved-linear-srgb-f32",
        "color_encoding": PFM_COLOR_ENCODING,
    }
    append_jsonl(progress_path, record)
    prior[image_id] = record
    return record


def command_prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    magick = require_executable(args.magick, "ImageMagick")
    curl = require_executable(args.curl, "curl")
    kodak_dir = args.kodak_dir.resolve()
    clic_dir = args.clic_dir.resolve()
    clic_archive = args.clic_archive.resolve()
    if not kodak_dir.is_dir() or not clic_dir.is_dir():
        raise StudyError("Kodak and CLIC input directories must exist")
    if not clic_archive.is_file():
        download(curl, CLIC_2024_TEST_URL, clic_archive)
    clic_pngs = verify_clic_archive(clic_dir, clic_archive)
    kodak_pngs = sorted(kodak_dir.glob("*.png"))
    if len(kodak_pngs) != 24:
        raise StudyError("Expected 24 Kodak PNGs, found %d" % len(kodak_pngs))

    downloads = output / "downloads" / "unsplash"
    high_sources = []
    for metadata in UNSPLASH_SOURCES:
        source_path = downloads / (metadata["id"] + ".jpg")
        download(curl, metadata["download_url"], source_path)
        width, height = image_dimensions(magick, source_path)
        high_sources.append((metadata, source_path, width, height))

    progress_path = output / "corpus-progress.jsonl"
    prior = progress_by_id(progress_path)
    specifications = []
    for path in kodak_pngs:
        specifications.append(
            {
                "image_id": "kodak/%s" % path.stem.lower(),
                "corpus": "kodak",
                "resolution_class": "kodak_0_4mp",
                "source_path": str(path),
                "pfm_path": str(output / "pfm" / "kodak" / (path.stem + ".pfm")),
                "source_provenance": {
                    "repository": args.kodak_repository,
                    "revision": args.kodak_revision,
                },
            }
        )
    for path in clic_pngs:
        specifications.append(
            {
                "image_id": "clic2024_test/%s" % path.stem,
                "corpus": "clic2024_test",
                "resolution_class": "clic_1_8_to_3_4mp",
                "source_path": str(path),
                "pfm_path": str(output / "pfm" / "clic2024_test" / (path.stem + ".pfm")),
                "source_provenance": {
                    "archive_url": CLIC_2024_TEST_URL,
                    "archive_sha256": CLIC_2024_TEST_SHA256,
                },
            }
        )
    for metadata, path, width, height in high_sources:
        for target in HIGH_RESOLUTION_TARGETS:
            target_width, target_height = target_dimensions(width, height, target)
            label = "%dmp" % (target // 1_000_000)
            specifications.append(
                {
                    "image_id": "unsplash/%s/%s" % (metadata["id"], label),
                    "corpus": "unsplash_controlled",
                    "resolution_class": label,
                    "source_path": str(path),
                    "pfm_path": str(
                        output / "pfm" / "unsplash" / metadata["id"] / (label + ".pfm")
                    ),
                    "target_dimensions": [target_width, target_height],
                    "source_provenance": {
                        **metadata,
                        "license_url": UNSPLASH_LICENSE_URL,
                        "native_width": width,
                        "native_height": height,
                    },
                }
            )

    records = []
    for index, specification in enumerate(specifications, 1):
        print("prepare %d/%d %s" % (index, len(specifications), specification["image_id"]))
        records.append(prepare_image(progress_path, prior, specification, magick))
    records.sort(key=lambda value: value["image_id"])
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "image_count": len(records),
        "input_layout": "interleaved-linear-srgb-f32",
        "color_encoding": PFM_COLOR_ENCODING,
        "image_magick": run_command((magick, "-version")).stdout.splitlines()[0],
        "images": records,
    }
    atomic_json(output / "corpus.json", manifest)
    print("prepared %d images in %s" % (len(records), output / "corpus.json"))


def parse_integer_list(text, lower, upper, label):
    values = []
    seen = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            pieces = part.split("-", 1)
            try:
                first, last = (int(item) for item in pieces)
            except ValueError as error:
                raise StudyError("Invalid %s range: %s" % (label, part)) from error
            candidates = range(first, last + 1)
        else:
            try:
                candidates = (int(part),)
            except ValueError as error:
                raise StudyError("Invalid %s: %s" % (label, part)) from error
        for value in candidates:
            if value < lower or value > upper:
                raise StudyError("%s %d is outside [%d, %d]" % (label, value, lower, upper))
            if value not in seen:
                values.append(value)
                seen.add(value)
    if not values:
        raise StudyError("At least one %s is required" % label)
    return tuple(values)


def quality_to_distance(quality):
    if quality >= 100:
        return 0.0
    if quality >= 30:
        value = 0.1 + (100.0 - quality) * 0.09
    else:
        value = 53.0 / 3000.0 * quality * quality - 23.0 / 20.0 * quality + 25.0
    # Avoid exposing binary floating-point artifacts such as 0.9999999999999999
    # in identifiers and reports. Twelve decimal digits are more precise than
    # the public API's float input.
    return round(value, 12)


def load_corpus(path, validate_hashes=True):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError("Could not read corpus manifest %s: %s" % (path, error)) from error
    images = []
    for record in value.get("images", []):
        image = CorpusImage(
            image_id=record["image_id"],
            corpus=record["corpus"],
            resolution_class=record["resolution_class"],
            source_path=pathlib.Path(record["source_path"]),
            pfm_path=pathlib.Path(record["pfm_path"]),
            width=record["width"],
            height=record["height"],
            pfm_sha256=record["pfm_sha256"],
            source_sha256=record["source_sha256"],
        )
        if not image.pfm_path.is_file():
            raise StudyError("Corpus PFM is missing: %s" % image.pfm_path)
        if read_pfm_dimensions(image.pfm_path) != (image.width, image.height):
            raise StudyError("Corpus PFM dimensions changed: %s" % image.pfm_path)
        if validate_hashes and sha256_file(image.pfm_path) != image.pfm_sha256:
            raise StudyError("Corpus PFM hash changed: %s" % image.pfm_path)
        images.append(image)
    if not images:
        raise StudyError("Corpus manifest contains no images")
    return value, images


def file_identity(path):
    path = pathlib.Path(path).resolve()
    result = {"path": str(path), "sha256": sha256_file(path)}
    try:
        completed = run_command((path, "--version"))
        result["version"] = (completed.stdout + completed.stderr).strip().splitlines()[0]
    except StudyError:
        pass
    return result


def optional_command_output(command):
    try:
        completed = run_command(command)
        return (completed.stdout + completed.stderr).strip()
    except StudyError as error:
        return "unavailable: %s" % error


def environment_snapshot():
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "macos": optional_command_output(("sw_vers",)),
        "hardware_model": optional_command_output(("sysctl", "-n", "hw.model")),
        "memory_bytes": optional_command_output(("sysctl", "-n", "hw.memsize")),
        "cpu_brand": optional_command_output(
            ("sysctl", "-n", "machdep.cpu.brand_string")
        ),
        "power": optional_command_output(("pmset", "-g", "batt")),
        "thermal": optional_command_output(("pmset", "-g", "therm")),
        "highest_cpu_processes": optional_command_output(
            ("ps", "-Ao", "pid,%cpu,%mem,comm", "-r")
        ).splitlines()[:21],
    }


def load_optional_json(path):
    if path is None:
        return None
    path = path.resolve()
    try:
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "content": json.loads(path.read_text(encoding="utf-8")),
        }
    except (OSError, json.JSONDecodeError) as error:
        raise StudyError("Could not read build manifest %s: %s" % (path, error)) from error


def output_path(root, image, quality, effort):
    return root / "outputs" / image.image_id / ("q%03d-e%02d.jxl" % (quality, effort))


def job_id(image, quality, effort):
    return "%s|quality=%d|effort=%d" % (image.image_id, quality, effort)


def load_or_create_metadata(args, corpus_value, tools):
    path = args.output / "metadata.json"
    configuration = {
        "schema_version": SCHEMA_VERSION,
        "corpus_manifest": str(args.corpus.resolve()),
        "corpus_manifest_sha256": sha256_file(args.corpus),
        "qualities": list(args.qualities),
        "quality_to_distance": {
            str(quality): quality_to_distance(quality) for quality in args.qualities
        },
        "efforts": list(args.efforts),
        "thread_count": args.num_threads,
        "warmups_per_process": args.warmups,
        "timing_repetitions": args.repetitions,
        "stage_samples_per_process": args.stage_samples,
        "stage_warmups_per_process": args.stage_warmups,
        "samply_rate_hz": args.samply_rate,
        "profile_policy": args.profile_policy,
        "tools": tools,
        "build_records": {
            "ordinary": load_optional_json(args.ordinary_build_manifest),
            "stage": load_optional_json(args.stage_build_manifest),
            "samply_cjxl_cmake_cache": (
                None
                if args.cjxl_cmake_cache is None
                else {
                    "path": str(args.cjxl_cmake_cache.resolve()),
                    "sha256": sha256_file(args.cjxl_cmake_cache.resolve()),
                }
            ),
        },
    }
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("configuration") != configuration:
            raise StudyError(
                "Run metadata does not match this invocation; use another output directory"
            )
        return current
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "host": environment_snapshot(),
        "methodology": {
            "primary_timing": "uninstrumented complete-encode wall time",
            "timed_boundary": (
                "JxlEncoder creation/configuration, image submission, and output processing; "
                "PFM read and JXL filesystem write excluded"
            ),
            "process_sampling": "one measured sample per independent harness process",
            "stage_phase_nanoseconds": "wall-clock barrier time",
            "stage_work_nanoseconds": "aggregate worker time, not latency",
            "samply": "sampled thread CPU, not wall time",
            "execution_order": (
                "Scheduling policy, seed, filters, and invocation boundaries are "
                "recorded in execution-events.jsonl"
            ),
            "input_layout": corpus_value["input_layout"],
            "color_encoding": corpus_value["color_encoding"],
        },
        "configuration": configuration,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(path, metadata)
    return metadata


def filtered_images(images, expression):
    if not expression:
        return images
    selected = [image for image in images if expression in image.image_id]
    if not selected:
        raise StudyError("No image ID contains --image-filter %r" % expression)
    return selected


def all_tuple_jobs(images, qualities, efforts):
    return [
        (image, quality, effort)
        for image in images
        for quality in qualities
        for effort in efforts
    ]


def phase_shuffle_seed(base_seed, phase, repetition=0, effort=None):
    seed = base_seed + PHASE_SEED_OFFSETS[phase] + repetition
    if effort is not None:
        seed += effort * 1_000
    return seed


def shuffled_tuple_jobs(
    images, qualities, efforts, base_seed, phase, repetition, schedule
):
    if schedule == SCHEDULE_PHASE_MAJOR_SHUFFLED:
        seed = phase_shuffle_seed(base_seed, phase, repetition)
        jobs = all_tuple_jobs(images, qualities, efforts)
        random.Random(seed).shuffle(jobs)
        return jobs

    jobs = []
    for effort in efforts:
        block = all_tuple_jobs(images, qualities, (effort,))
        seed = phase_shuffle_seed(base_seed, phase, repetition, effort)
        random.Random(seed).shuffle(block)
        jobs.extend(block)
    return jobs


def execution_plan(phase, schedule, efforts):
    phases = (
        ("timing", "stages", "profiles") if phase == "all" else (phase,)
    )
    if schedule == SCHEDULE_EFFORT_MAJOR:
        return [
            (selected_phase, (effort,))
            for effort in efforts
            for selected_phase in phases
        ]
    return [(selected_phase, tuple(efforts)) for selected_phase in phases]


def completed_ids(path, field):
    return {record[field] for record in read_jsonl(path)}


def benchmark_command(binary, image, raw, quality, effort, threads, warmups, samples):
    return [
        str(binary),
        "--input",
        str(image.pfm_path),
        "--raw-samples",
        str(raw),
        "--distance",
        "%.12g" % quality_to_distance(quality),
        "--effort",
        str(effort),
        "--num-threads",
        str(threads),
        "--warmups",
        str(warmups),
        "--samples",
        str(samples),
    ]


def command_timing(args, images, benchmark, efforts, budget):
    records_path = args.output / "timings.jsonl"
    existing_records = read_jsonl(records_path)
    done = {record["sample_id"] for record in existing_records}
    output_hashes = {}
    for record in existing_records:
        output_hashes.setdefault(record["job_id"], record["output_sha256"])
    for repetition in range(args.repetitions):
        jobs = shuffled_tuple_jobs(
            images,
            args.qualities,
            efforts,
            args.shuffle_seed,
            "timing",
            repetition,
            args.schedule,
        )
        for position, (image, quality, effort) in enumerate(jobs):
            sample_id = "%s|repetition=%d" % (job_id(image, quality, effort), repetition)
            if sample_id in done:
                continue
            if budget.exhausted:
                return False
            final_output = output_path(args.output, image, quality, effort)
            final_output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="cjxl-timing-") as temporary:
                temporary = pathlib.Path(temporary)
                raw = temporary / "raw.json"
                staged_output = temporary / "output.jxl"
                command = benchmark_command(
                    benchmark,
                    image,
                    raw,
                    quality,
                    effort,
                    args.num_threads,
                    args.warmups,
                    1,
                )
                if repetition == 0:
                    command.extend(("--output", str(staged_output)))
                print("timing %s" % sample_id, flush=True)
                run_command(command, capture=False)
                document = json.loads(raw.read_text(encoding="utf-8"))
                if len(document["samples"]) != 1:
                    raise StudyError("Timing harness did not emit exactly one sample")
                if repetition == 0:
                    staged_output.replace(final_output)
                if not final_output.is_file():
                    raise StudyError("Final codestream is missing: %s" % final_output)
                identifier = job_id(image, quality, effort)
                output_hash = output_hashes.get(identifier)
                if output_hash is None:
                    output_hash = sha256_file(final_output)
                    output_hashes[identifier] = output_hash
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "recorded_at": utc_now(),
                    "sample_id": sample_id,
                    "job_id": identifier,
                    "repetition": repetition,
                    "schedule_position": position,
                    "schedule_position_scope": (
                        "effort_repetition"
                        if args.schedule == SCHEDULE_EFFORT_MAJOR
                        else "repetition"
                    ),
                    "schedule_policy": args.schedule,
                    "schedule_seed": phase_shuffle_seed(
                        args.shuffle_seed,
                        "timing",
                        repetition,
                        (
                            effort
                            if args.schedule == SCHEDULE_EFFORT_MAJOR
                            else None
                        ),
                    ),
                    "image_id": image.image_id,
                    "corpus": image.corpus,
                    "resolution_class": image.resolution_class,
                    "width": image.width,
                    "height": image.height,
                    "pixels": image.pixels,
                    "quality": quality,
                    "distance": quality_to_distance(quality),
                    "effort": effort,
                    "thread_count": args.num_threads,
                    "elapsed_nanoseconds": document["samples"][0]["elapsed_nanoseconds"],
                    "encoded_bytes": document["samples"][0]["encoded_bytes"],
                    "output_path": str(final_output.resolve()),
                    "output_sha256": output_hash,
                    "harness_revision": document["revision"],
                }
                append_jsonl(records_path, record)
                done.add(sample_id)
                budget.completed_one()
    return True


def median_mapping(samples, key):
    names = samples[0][key].keys()
    return {
        name: statistics.median(sample[key][name] for sample in samples)
        for name in names
    }


def command_stages(args, images, benchmark, efforts, budget):
    records_path = args.output / "stage-profiles.jsonl"
    done = completed_ids(records_path, "job_id")
    jobs = shuffled_tuple_jobs(
        images,
        args.qualities,
        efforts,
        args.shuffle_seed,
        "stages",
        0,
        args.schedule,
    )
    for position, (image, quality, effort) in enumerate(jobs):
        identifier = job_id(image, quality, effort)
        if identifier in done:
            continue
        if budget.exhausted:
            return False
        final_output = output_path(args.output, image, quality, effort)
        if not final_output.is_file():
            raise StudyError("Run timing phase first; missing %s" % final_output)
        with tempfile.TemporaryDirectory(prefix="cjxl-stage-") as temporary:
            temporary = pathlib.Path(temporary)
            raw = temporary / "raw.json"
            profiled_output = temporary / "output.jxl"
            command = benchmark_command(
                benchmark,
                image,
                raw,
                quality,
                effort,
                args.num_threads,
                args.stage_warmups,
                args.stage_samples,
            )
            command.extend(("--stage-profile", "--output", str(profiled_output)))
            print("stage %s" % identifier, flush=True)
            run_command(command, capture=False)
            document = json.loads(raw.read_text(encoding="utf-8"))
            if not document.get("stage_profile_enabled"):
                raise StudyError("Stage benchmark did not enable profiling")
            if sha256_file(profiled_output) != sha256_file(final_output):
                raise StudyError("Instrumented codestream differs for %s" % identifier)
            samples = document["samples"]
            representative = sorted(
                samples, key=lambda sample: sample["elapsed_nanoseconds"]
            )[len(samples) // 2]
            record = {
                "schema_version": SCHEMA_VERSION,
                "recorded_at": utc_now(),
                "job_id": identifier,
                "schedule_position": position,
                "schedule_position_scope": (
                    "effort"
                    if args.schedule == SCHEDULE_EFFORT_MAJOR
                    else "phase"
                ),
                "schedule_policy": args.schedule,
                "schedule_seed": phase_shuffle_seed(
                    args.shuffle_seed,
                    "stages",
                    effort=(
                        effort if args.schedule == SCHEDULE_EFFORT_MAJOR else None
                    ),
                ),
                "image_id": image.image_id,
                "corpus": image.corpus,
                "resolution_class": image.resolution_class,
                "width": image.width,
                "height": image.height,
                "pixels": image.pixels,
                "quality": quality,
                "distance": quality_to_distance(quality),
                "effort": effort,
                "thread_count": args.num_threads,
                "output_sha256": sha256_file(final_output),
                "output_matches_uninstrumented": True,
                "timing_semantics": document["timing_semantics"],
                "harness_revision": document["revision"],
                "samples": samples,
                "median_elapsed_nanoseconds": statistics.median(
                    sample["elapsed_nanoseconds"] for sample in samples
                ),
                "representative_sample_index": representative["sample_index"],
                "representative_frontend_residual_nanoseconds": (
                    representative["elapsed_nanoseconds"]
                    - representative["phase_nanoseconds"]["complete_serializer"]
                ),
                "representative_phase_nanoseconds": representative[
                    "phase_nanoseconds"
                ],
                "representative_work_nanoseconds": representative[
                    "work_nanoseconds"
                ],
                "median_phase_nanoseconds": median_mapping(samples, "phase_nanoseconds"),
                "median_work_nanoseconds": median_mapping(samples, "work_nanoseconds"),
            }
            append_jsonl(records_path, record)
            done.add(identifier)
            budget.completed_one()
    return True


def representative_images(images):
    by_class = {}
    for image in images:
        by_class.setdefault(image.resolution_class, []).append(image)
    selected = []
    for resolution_class, candidates in sorted(by_class.items()):
        if resolution_class.startswith("kodak"):
            preferred = [image for image in candidates if image.image_id.endswith("kodim13")]
            selected.append(preferred[0] if preferred else candidates[len(candidates) // 2])
        elif resolution_class.startswith("clic"):
            ordered = sorted(candidates, key=lambda image: (image.pixels, image.image_id))
            selected.append(ordered[len(ordered) // 2])
        else:
            preferred = [image for image in candidates if "forest_stream" in image.image_id]
            selected.append(preferred[0] if preferred else sorted(candidates, key=lambda x: x.image_id)[0])
    return selected


def profile_sidecar(path):
    suffix = ".json.gz"
    text = str(path)
    if not text.endswith(suffix):
        raise StudyError("Samply output must end in %s" % suffix)
    return pathlib.Path(text[: -len(suffix)] + ".json.syms.json")


def command_profiles(args, images, cjxl, samply, efforts, budget):
    if args.profile_policy == "none":
        return True
    efforts = tuple(efforts)
    candidates = images if args.profile_policy == "all" else representative_images(images)
    modes = [("workers", args.num_threads, args.qualities, efforts)]
    if args.profile_no_worker_diagnostics:
        modes.append(
            (
                "no_workers_diagnostic",
                0,
                tuple(q for q in args.qualities if q in PROFILE_DIAGNOSTIC_QUALITIES),
                tuple(e for e in efforts if e in PROFILE_DIAGNOSTIC_EFFORTS),
            )
        )
    jobs = []
    for mode, threads, qualities, mode_efforts in modes:
        for image in candidates:
            for quality in qualities:
                for effort in mode_efforts:
                    jobs.append((mode, threads, image, quality, effort))
    schedule_effort = (
        efforts[0]
        if args.schedule == SCHEDULE_EFFORT_MAJOR and len(efforts) == 1
        else None
    )
    random.Random(
        phase_shuffle_seed(
            args.shuffle_seed, "profiles", effort=schedule_effort
        )
    ).shuffle(jobs)
    for mode, threads, image, quality, effort in jobs:
        profile = (
            args.output
            / "samply"
            / mode
            / image.image_id
            / ("q%03d-e%02d.json.gz" % (quality, effort))
        )
        sidecar = profile_sidecar(profile)
        if profile.is_file() and profile.stat().st_size and sidecar.is_file() and sidecar.stat().st_size:
            continue
        if budget.exhausted:
            return False
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        command = [
            str(samply),
            "record",
            "--save-only",
            "--unstable-presymbolicate",
            "--rate",
            str(args.samply_rate),
            "--output",
            str(profile),
            "--",
            str(cjxl),
            str(image.pfm_path),
            "--disable_output",
            "--quiet",
            "--quality=%d" % quality,
            "--effort=%d" % effort,
            "--num_threads=%d" % threads,
            "-x",
            "color_space=%s" % PFM_COLOR_ENCODING,
        ]
        print(
            "samply %s threads=%d %s"
            % (mode, threads, job_id(image, quality, effort)),
            flush=True,
        )
        run_command(command, capture=False)
        if not profile.is_file() or not sidecar.is_file():
            raise StudyError("Samply did not emit both capture and symbol sidecar")
        budget.completed_one()
    return True


def has_existing_measurements(output):
    for name in ("timings.jsonl", "stage-profiles.jsonl"):
        path = output / name
        if path.is_file() and path.stat().st_size:
            return True
    samply = output / "samply"
    return samply.is_dir() and next(samply.rglob("*.json.gz"), None) is not None


def record_execution_invocation(args, image_count):
    path = args.output / "execution-events.jsonl"
    if not path.is_file() and has_existing_measurements(args.output):
        append_jsonl(
            path,
            {
                "schema_version": SCHEMA_VERSION,
                "recorded_at": utc_now(),
                "event": "legacy_results_detected",
                "schedule_policy": SCHEDULE_PHASE_MAJOR_SHUFFLED,
                "shuffle_seed_persisted": False,
                "note": (
                    "Existing results predate execution event logging; their records "
                    "retain repetition and schedule_position"
                ),
            },
        )
    append_jsonl(
        path,
        {
            "schema_version": SCHEMA_VERSION,
            "recorded_at": utc_now(),
            "event": "run_invocation",
            "phase": args.phase,
            "schedule_policy": args.schedule,
            "shuffle_seed": args.shuffle_seed,
            "effort_order": list(args.efforts),
            "image_filter": args.image_filter,
            "selected_image_count": image_count,
            "max_jobs_per_phase": args.max_jobs,
            "power": optional_command_output(("pmset", "-g", "batt")),
            "thermal": optional_command_output(("pmset", "-g", "therm")),
        },
    )


def command_run(args):
    args.corpus = args.corpus.resolve()
    args.output = args.output.resolve()
    corpus_value, all_images = load_corpus(args.corpus)
    images = filtered_images(all_images, args.image_filter)
    ordinary = require_executable(args.benchmark, "ordinary benchmark")
    stage = require_executable(args.stage_benchmark, "stage benchmark")
    cjxl = require_executable(args.cjxl, "cjxl")
    samply = require_executable(args.samply, "Samply")
    tools = {
        "ordinary_benchmark": file_identity(ordinary),
        "stage_benchmark": file_identity(stage),
        "cjxl": file_identity(cjxl),
        "samply": file_identity(samply),
    }
    load_or_create_metadata(args, corpus_value, tools)
    record_execution_invocation(args, len(images))
    budgets = {
        selected_phase: JobBudget(args.max_jobs)
        for selected_phase in ("timing", "stages", "profiles")
    }
    for selected_phase, efforts in execution_plan(
        args.phase, args.schedule, args.efforts
    ):
        print(
            "schedule %s phase=%s efforts=%s"
            % (args.schedule, selected_phase, ",".join(map(str, efforts))),
            flush=True,
        )
        if selected_phase == "timing":
            complete = command_timing(
                args, images, ordinary, efforts, budgets[selected_phase]
            )
        elif selected_phase == "stages":
            complete = command_stages(
                args, images, stage, efforts, budgets[selected_phase]
            )
        else:
            complete = command_profiles(
                args, images, cjxl, samply, efforts, budgets[selected_phase]
            )
        if not complete:
            return


def percentile(values, fraction):
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def tuple_summary(timing_records, stage_records):
    timings = {}
    for record in timing_records:
        timings.setdefault(record["job_id"], []).append(record)
    stages = {record["job_id"]: record for record in stage_records}
    rows = []
    for identifier, samples in sorted(timings.items()):
        first = samples[0]
        elapsed = [sample["elapsed_nanoseconds"] for sample in samples]
        encoded_sizes = {sample["encoded_bytes"] for sample in samples}
        output_hashes = {sample["output_sha256"] for sample in samples}
        if len(encoded_sizes) != 1 or len(output_hashes) != 1:
            raise StudyError("Codestream changed across timing samples: %s" % identifier)
        median_ns = statistics.median(elapsed)
        row = {
            "job_id": identifier,
            "image_id": first["image_id"],
            "corpus": first["corpus"],
            "resolution_class": first["resolution_class"],
            "width": first["width"],
            "height": first["height"],
            "megapixels": first["pixels"] / 1_000_000,
            "quality": first["quality"],
            "distance": first["distance"],
            "effort": first["effort"],
            "thread_count": first["thread_count"],
            "timing_sample_count": len(elapsed),
            "complete_encode_median_ms": median_ns / 1_000_000,
            "complete_encode_min_ms": min(elapsed) / 1_000_000,
            "complete_encode_p10_ms": percentile(elapsed, 0.10) / 1_000_000,
            "complete_encode_p90_ms": percentile(elapsed, 0.90) / 1_000_000,
            "complete_encode_max_ms": max(elapsed) / 1_000_000,
            "complete_encode_stdev_ms": (
                statistics.stdev(elapsed) / 1_000_000 if len(elapsed) > 1 else 0.0
            ),
            "complete_encode_ms_per_mp": median_ns / 1_000_000 / (first["pixels"] / 1_000_000),
            "encoded_bytes": next(iter(encoded_sizes)),
            "bits_per_pixel": next(iter(encoded_sizes)) * 8 / first["pixels"],
            "output_sha256": next(iter(output_hashes)),
            "output_path": first["output_path"],
        }
        stage = stages.get(identifier)
        if stage:
            row["profiled_complete_wall_ms"] = stage["median_elapsed_nanoseconds"] / 1_000_000
            phases = stage["representative_phase_nanoseconds"]
            row["frontend_residual_wall_ms"] = stage[
                "representative_frontend_residual_nanoseconds"
            ] / 1_000_000
            for name, value in phases.items():
                row["phase_wall_%s_ms" % name] = value / 1_000_000
            for name, value in stage["representative_work_nanoseconds"].items():
                row["work_aggregate_worker_%s_ms" % name] = value / 1_000_000
        rows.append(row)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    try:
        with temporary.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def aggregate_summary(rows):
    groups = {}
    for row in rows:
        key = (row["corpus"], row["resolution_class"], row["quality"], row["effort"])
        groups.setdefault(key, []).append(row)
    aggregated = []
    for (corpus, resolution, quality, effort), members in sorted(groups.items()):
        total_pixels = sum(row["megapixels"] * 1_000_000 for row in members)
        total_ms = sum(row["complete_encode_median_ms"] for row in members)
        result = {
            "corpus": corpus,
            "resolution_class": resolution,
            "quality": quality,
            "distance": members[0]["distance"],
            "effort": effort,
            "image_count": len(members),
            "total_megapixels": total_pixels / 1_000_000,
            "total_complete_encode_median_ms": total_ms,
            "complete_encode_ms_per_mp": total_ms / (total_pixels / 1_000_000),
            "total_encoded_bytes": sum(row["encoded_bytes"] for row in members),
            "aggregate_bits_per_pixel": sum(row["encoded_bytes"] for row in members) * 8 / total_pixels,
        }
        stage_fields = [field for field in members[0] if field.startswith(("profiled_", "frontend_", "phase_", "work_"))]
        for field in stage_fields:
            if all(field in row for row in members):
                result["total_%s" % field] = sum(row[field] for row in members)
        aggregated.append(result)
    return aggregated


def load_profile_module(path):
    specification = importlib.util.spec_from_file_location("cjxl_samply_profile", path)
    if specification is None or specification.loader is None:
        raise StudyError("Could not load Samply parser: %s" % path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def summarize_profiles(run, parser_path):
    profiles = sorted((run / "samply").glob("**/*.json.gz"))
    if not profiles:
        return []
    parser = load_profile_module(parser_path)
    rows = []
    for profile in profiles:
        relative = profile.relative_to(run / "samply")
        parts = relative.parts
        mode = parts[0]
        filename = parts[-1]
        image_id = "/".join(parts[1:-1])
        quality = int(filename[1:4])
        effort = int(filename[6:8])
        analysis = parser.analyze_profiles([profile], "current")
        value = parser.analysis_as_json_value(analysis, 20)
        for operation in value["operations"]:
            rows.append(
                {
                    "mode": mode,
                    "image_id": image_id,
                    "quality": quality,
                    "effort": effort,
                    "operation": operation["operation"],
                    "sampled_cpu_delta_us": operation["cpu_delta_us"],
                    "whole_command_percent": operation["whole_command_percent"],
                    "encoder_plus_input_percent": operation["encoder_plus_input_percent"],
                    "semantics": "sampled thread CPU, not wall time",
                }
            )
    return rows


def command_summarize(args):
    run = args.run.resolve()
    timings = read_jsonl(run / "timings.jsonl")
    stages = read_jsonl(run / "stage-profiles.jsonl")
    rows = tuple_summary(timings, stages)
    summary = run / "summary"
    write_csv(summary / "image-tuples.csv", rows)
    aggregated = aggregate_summary(rows)
    write_csv(summary / "aggregated.csv", aggregated)
    profile_rows = summarize_profiles(run, args.samply_parser.resolve())
    write_csv(summary / "samply-operations.csv", profile_rows)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    configuration = metadata["configuration"]
    expected_tuples = (
        json.loads(pathlib.Path(configuration["corpus_manifest"]).read_text(encoding="utf-8"))["image_count"]
        * len(configuration["qualities"])
        * len(configuration["efforts"])
    )
    completed_tuples = len({record["job_id"] for record in timings})
    timing_samples = len(timings)
    report = """# libjxl runtime characterization

Generated: {generated}

## Progress

- Image/quality/effort tuples with timing: {completed_tuples}/{expected_tuples}
- Independent timing samples: {timing_samples}/{expected_samples}
- Tuples with exact stage profiles: {stage_count}/{expected_tuples}
- Samply captures parsed: {profile_count}

## Measurement boundaries

The primary result is the median uninstrumented complete-encode wall time from
independent processes. Each process records one sample after one validation
encode and {warmups} warmup encode(s). PFM input reading and filesystem output
are outside the timed region. The exact phase columns cover the serializer
portion of the instrumented sample whose complete elapsed time is the stage
run's median. `frontend_residual_wall_ms` is that sample's complete wall time
minus its complete serializer wall time.

`phase_wall_*` values are mutually exclusive wall-clock phases.
`work_aggregate_worker_*` values are aggregate worker time and must not be
summed or interpreted as latency. Samply columns are sampled thread-CPU
attribution and are likewise not wall-clock timings.

## Outputs

- `image-tuples.csv`: per-image distributions, size, exact wall phases, and
  aggregate worker measurements.
- `aggregated.csv`: sums across each corpus/resolution/quality/effort cell.
- `samply-operations.csv`: operation attribution for stratified captures.
- `../execution-events.jsonl`: append-only scheduling and invocation history.
- `../outputs`: one deterministic JXL codestream per unique tuple.
""".format(
        generated=utc_now(),
        completed_tuples=completed_tuples,
        expected_tuples=expected_tuples,
        timing_samples=timing_samples,
        expected_samples=expected_tuples * configuration["timing_repetitions"],
        stage_count=len(stages),
        profile_count=len(
            {
                (row["mode"], row["image_id"], row["quality"], row["effort"])
                for row in profile_rows
            }
        ),
        warmups=configuration["warmups_per_process"],
    )
    (summary / "REPORT.md").write_text(report, encoding="utf-8")
    print("wrote summaries under %s" % summary)


def command_verify(args):
    run = args.run.resolve()
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    configuration = metadata["configuration"]
    _, images = load_corpus(
        pathlib.Path(configuration["corpus_manifest"]), validate_hashes=False
    )
    timing_records = read_jsonl(run / "timings.jsonl")
    stage_records = read_jsonl(run / "stage-profiles.jsonl")
    by_sample = {record["sample_id"]: record for record in timing_records}
    stages = {record["job_id"]: record for record in stage_records}
    expected_outputs = []
    errors = []
    for image, quality, effort in all_tuple_jobs(
        images, configuration["qualities"], configuration["efforts"]
    ):
        identifier = job_id(image, quality, effort)
        output = output_path(run, image, quality, effort)
        expected_outputs.append((identifier, output))
        samples = [
            by_sample.get("%s|repetition=%d" % (identifier, repetition))
            for repetition in range(configuration["timing_repetitions"])
        ]
        if any(sample is None for sample in samples):
            errors.append("missing timing sample: %s" % identifier)
        if not output.is_file():
            errors.append("missing output: %s" % output)
        elif samples[0] and sha256_file(output) != samples[0]["output_sha256"]:
            errors.append("output hash mismatch: %s" % output)
        stage = stages.get(identifier)
        if stage is None:
            errors.append("missing stage profile: %s" % identifier)
        elif not stage.get("output_matches_uninstrumented"):
            errors.append("stage output mismatch: %s" % identifier)
    if errors:
        raise StudyError("Verification failed:\n" + "\n".join(errors[:50]))

    djxl = require_executable(args.djxl, "djxl")
    decoded_path = run / "verified-decodes.jsonl"
    decoded = completed_ids(decoded_path, "job_id")
    for index, (identifier, output) in enumerate(expected_outputs, 1):
        if identifier in decoded:
            continue
        print("decode %d/%d %s" % (index, len(expected_outputs), identifier), flush=True)
        run_command((djxl, output, "--disable_output", "--quiet"), capture=False)
        append_jsonl(
            decoded_path,
            {"job_id": identifier, "decoded_at": utc_now(), "output_sha256": sha256_file(output)},
        )
    atomic_json(
        run / "verification.json",
        {
            "schema_version": SCHEMA_VERSION,
            "verified_at": utc_now(),
            "tuple_count": len(expected_outputs),
            "timing_sample_count": len(timing_records),
            "stage_profile_count": len(stage_records),
            "decoded_output_count": len(expected_outputs),
            "status": "complete",
        },
    )
    print("verification complete")


def add_run_arguments(parser):
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--stage-benchmark", required=True)
    parser.add_argument("--cjxl", required=True)
    parser.add_argument("--samply", default="samply")
    parser.add_argument("--ordinary-build-manifest", type=pathlib.Path)
    parser.add_argument("--stage-build-manifest", type=pathlib.Path)
    parser.add_argument("--cjxl-cmake-cache", type=pathlib.Path)
    parser.add_argument("--phase", choices=("timing", "stages", "profiles", "all"), default="all")
    parser.add_argument("--qualities", default=",".join(map(str, DEFAULT_QUALITIES)))
    parser.add_argument("--efforts", default="1-10")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--stage-warmups", type=int, default=1)
    parser.add_argument("--stage-samples", type=int, default=3)
    parser.add_argument("--samply-rate", type=int, default=1000)
    parser.add_argument("--profile-policy", choices=("none", "stratified", "all"), default="stratified")
    parser.add_argument("--profile-no-worker-diagnostics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--schedule",
        choices=(SCHEDULE_EFFORT_MAJOR, SCHEDULE_PHASE_MAJOR_SHUFFLED),
        default=SCHEDULE_EFFORT_MAJOR,
        help=(
            "effort-major completes timing, stages, and profiles for each effort "
            "before advancing; phase-major-shuffled preserves the original policy"
        ),
    )
    parser.add_argument("--shuffle-seed", type=int, default=20260903)
    parser.add_argument("--image-filter", help="run only image IDs containing this text")
    parser.add_argument("--max-jobs", type=int, help="execute at most this many pending jobs in each selected phase")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="prepare canonical PFM corpus")
    prepare.add_argument("--output", type=pathlib.Path, required=True)
    prepare.add_argument("--kodak-dir", type=pathlib.Path, required=True)
    prepare.add_argument("--clic-dir", type=pathlib.Path, required=True)
    prepare.add_argument("--clic-archive", type=pathlib.Path, required=True)
    prepare.add_argument("--kodak-repository", required=True)
    prepare.add_argument("--kodak-revision", required=True)
    prepare.add_argument("--magick", default="magick")
    prepare.add_argument("--curl", default="curl")
    prepare.set_defaults(function=command_prepare)

    run = subparsers.add_parser("run", help="run resumable measurement phases")
    add_run_arguments(run)
    run.set_defaults(function=command_run)

    summarize = subparsers.add_parser("summarize", help="write CSV and Markdown summaries")
    summarize.add_argument("--run", type=pathlib.Path, required=True)
    summarize.add_argument(
        "--samply-parser",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("cjxl_samply_profile.py"),
    )
    summarize.set_defaults(function=command_summarize)

    verify = subparsers.add_parser("verify", help="verify completeness, hashes, and decodability")
    verify.add_argument("--run", type=pathlib.Path, required=True)
    verify.add_argument("--djxl", required=True)
    verify.set_defaults(function=command_verify)

    args = parser.parse_args(argv)
    if args.command == "run":
        args.qualities = parse_integer_list(args.qualities, 0, 99, "quality")
        args.efforts = parse_integer_list(args.efforts, 1, 10, "effort")
        for name in ("num_threads", "warmups", "repetitions", "stage_warmups", "stage_samples", "samply_rate"):
            if getattr(args, name) < (1 if name in ("repetitions", "stage_samples", "samply_rate") else 0):
                parser.error("--%s has an invalid value" % name.replace("_", "-"))
        if args.max_jobs is not None and args.max_jobs < 1:
            parser.error("--max-jobs must be positive")
    return args


def main(argv=None):
    try:
        args = parse_args(argv)
        args.function(args)
    except StudyError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
