# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Pure fixed-sweep helpers shared by runtime and quality-study collectors."""

import math
import random
import statistics


class StudyError(Exception):
    """Invalid timing observations."""


DEFAULT_QUALITIES = (10, 30, 50, 70, 80, 90, 95)

SCHEDULE_EFFORT_MAJOR = "effort-major"

SCHEDULE_PHASE_MAJOR_SHUFFLED = "phase-major-shuffled"

PHASE_SEED_OFFSETS = {
    "timing": 0,
    "stages": 100_000,
    "profiles": 200_000,
}


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
    phases = ("timing", "stages", "profiles") if phase == "all" else (phase,)
    if schedule == SCHEDULE_EFFORT_MAJOR:
        return [
            (selected_phase, (effort,))
            for effort in efforts
            for selected_phase in phases
        ]
    return [(selected_phase, tuple(efforts)) for selected_phase in phases]


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


def summarize_timing_samples(samples):
    identifier = samples[0]["job_id"]
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
        "complete_encode_ms_per_mp": median_ns
        / 1_000_000
        / (first["pixels"] / 1_000_000),
        "encoded_bytes": next(iter(encoded_sizes)),
        "bits_per_pixel": next(iter(encoded_sizes)) * 8 / first["pixels"],
        "output_sha256": next(iter(output_hashes)),
        "output_path": first["output_path"],
    }
    return row
