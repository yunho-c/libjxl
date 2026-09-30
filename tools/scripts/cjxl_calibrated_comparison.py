#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Direct size/time comparisons from saved calibrated encodes; no interpolation."""

from collections import Counter
import math
from pathlib import Path
import statistics

import cjxl_bd_rate as bd


def load_studies(runs):
    """Reuse calibrated timing/identity checks and retain unresolved outcomes."""
    studies = bd.load_calibrated_studies(runs)
    for item in studies:
        run, config = Path(item["run"]), item["config"]
        item["outcomes"] = bd.study.read_records(run, "calibration", config)
        item["metadata_sha256"] = bd.study.digest(run / "metadata.json")
    return studies


def analyze(studies, *, targets=(60, 70, 85), efforts=None, image_ids=None,
            baseline_encoder="libjxl", baseline_effort=7):
    """Select one complete, unresampled cohort across ALL panels and series.

    This is an explicitly reported common-subset analysis, not an estimate for
    excluded images. There is no per-marker cohort reduction or interpolation.
    """
    if {bd.study.encoder_name(item["config"]) for item in studies} != {"libjxl", "gjxl"}:
        raise ValueError("Select one calibrated study for each encoder")
    images = bd._validate_studies(studies, baseline_encoder, baseline_effort, image_ids)
    targets = list(targets)
    if (not targets or len(set(targets)) != len(targets)
            or any(not isinstance(t, (int, float)) or not math.isfinite(t) for t in targets)):
        raise ValueError("Select finite, unique quality targets")
    targets.sort()
    selected_efforts = sorted(set(e for item in studies
                                  for e in item["config"]["measurement_efforts"])) if efforts is None else list(efforts)
    if (not selected_efforts or len(set(selected_efforts)) != len(selected_efforts)
            or any(type(e) is not int or not 1 <= e <= 10 for e in selected_efforts)):
        raise ValueError("Select unique efforts in 1..10")
    selected_efforts.sort()
    tolerance = studies[0]["config"]["tolerance"]
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid calibration tolerance")
    requested = {image["image_id"] for image in images}
    observations, outcomes, configs = {}, {}, {}
    for item in studies:
        config = item["config"]
        encoder = bd.study.encoder_name(config)
        configs[encoder] = config
        if bd.study.is_fixed(config) or item.get("observation_source") != "calibrated":
            raise ValueError("Only calibrated observations can supply these panels")
        if config["tolerance"] != tolerance:
            raise ValueError("Calibration tolerances differ")
        required_efforts = set(selected_efforts)
        if encoder == baseline_encoder:
            required_efforts.add(baseline_effort)
        if (not required_efforts <= set(config["measurement_efforts"])
                or not set(targets) <= set(config["targets"])):
            raise ValueError("Requested effort or target is absent from a study")
        for row in item.get("outcomes", []):
            outcomes[encoder, row["image_id"], row["effort"], row["target"]] = row["status"]
        for row in item["scores"]:
            if row["target"] not in config["targets"]:
                raise ValueError("Observation target is outside its study manifest")
            key = encoder, row["image_id"], row["effort"], row["target"]
            if key in observations:
                raise ValueError("Duplicate calibrated observation")
            if (row.get("status") != "matched" or not math.isfinite(row["score"])
                    or abs(row["score"] - row["target"]) > tolerance
                    or not math.isfinite(row["encoded_bytes"]) or row["encoded_bytes"] <= 0
                    or row.get("resampling") not in (1, 2)):
                raise ValueError("Invalid accepted calibrated observation")
            if row["timing_sample_count"] > config["repetitions"]:
                raise ValueError("Too many calibrated timing samples")
            if (row["timing_sample_count"] == config["repetitions"]
                    and (row.get("elapsed_ms") is None
                         or not math.isfinite(row["elapsed_ms"]) or row["elapsed_ms"] <= 0)):
                raise ValueError("Invalid calibrated encode time")
            observations[key] = row

    cohort, excluded = [], []
    for image in images:
        image_id = image["image_id"]
        reasons = []
        for encoder, config in configs.items():
            required_efforts = sorted(set(selected_efforts) |
                                      ({baseline_effort} if encoder == baseline_encoder else set()))
            for effort in required_efforts:
                for target in targets:
                    key = encoder, image_id, effort, target
                    row = observations.get(key)
                    reason = (outcomes.get(key, "missing-calibration") if row is None else
                              "resampled" if row["resampling"] != 1 else
                              "incomplete-timing" if row["timing_sample_count"] != config["repetitions"] else None)
                    if reason:
                        reasons.append(dict(encoder=encoder, effort=effort, target=target, reason=reason))
        if reasons:
            excluded.append({"image_id": image_id, "reasons": reasons})
        else:
            cohort.append(image_id)

    points, per_image = [], []
    for target in targets:
        for encoder in configs:
            for effort in selected_efforts:
                rows = []
                for image_id in cohort:
                    row = observations[encoder, image_id, effort, target]
                    anchor = observations[baseline_encoder, image_id, baseline_effort, target]
                    result = dict(
                        encoder=encoder, effort=effort, target=target, image_id=image_id,
                        encoded_bytes=row["encoded_bytes"], baseline_bytes=anchor["encoded_bytes"],
                        elapsed_ms=row["elapsed_ms"], baseline_elapsed_ms=anchor["elapsed_ms"],
                        score=row["score"], baseline_score=anchor["score"],
                        score_error=row["score"] - target,
                        paired_score_difference=row["score"] - anchor["score"],
                        size_difference_pct=100 * (row["encoded_bytes"] / anchor["encoded_bytes"] - 1),
                        output_sha256=row["output_sha256"], baseline_output_sha256=anchor["output_sha256"],
                        reference_sha256=row["reference_sha256"],
                    )
                    rows.append(result)
                    per_image.append(result)
                point = dict(target=target, encoder=encoder, effort=effort,
                             image_count=len(cohort), status="ready" if cohort else "empty-cohort")
                if rows:
                    point.update(
                        mean_encode_ms=statistics.mean(r["elapsed_ms"] for r in rows),
                        mean_size_difference_pct=statistics.mean(r["size_difference_pct"] for r in rows),
                        mean_score_error=statistics.mean(r["score_error"] for r in rows),
                        max_abs_score_error=max(abs(r["score_error"]) for r in rows),
                        mean_paired_score_difference=statistics.mean(r["paired_score_difference"] for r in rows),
                        max_abs_paired_score_difference=max(abs(r["paired_score_difference"]) for r in rows),
                    )
                points.append(point)
    return dict(
        schema_version=1, metric="fast-ssim2", targets=targets, tolerance=tolerance,
        efforts=selected_efforts, baseline=dict(encoder=baseline_encoder, effort=baseline_effort),
        cohort_policy="One common complete cohort across both encoders, all selected efforts and targets; resampling=1 only",
        requested_cohort=[image["image_id"] for image in images], cohort=cohort,
        requested_image_count=len(requested), image_count=len(cohort), excluded_images=excluded,
        exclusion_image_counts=dict(Counter(reason for image in excluded
                                            for reason in {r["reason"] for r in image["reasons"]})),
        rate_method="Arithmetic mean of per-image size differences against the baseline at the same target; not BD-rate",
        timing_method="Arithmetic mean of per-image median measured calibrated encode times; no interpolation",
        timing_boundary="Warm complete encode call; excludes startup, file I/O, decode, scoring and calibration search",
        quality_matching="Each encode is within target +/- tolerance; paired scores can differ by up to twice tolerance; no correction or interpolation",
        sources=[dict(
            run=item["run"], encoder=bd.study.encoder_name(item["config"]),
            configuration_id=item["config"]["configuration_id"],
            metadata_sha256=item.get("metadata_sha256"),
            encoder_revision=item["config"].get("encoder_revision", item["config"].get("libjxl_revision")),
            encoder_build_sha256=item["config"].get("encoder_build_sha256"),
            metric_version=item["config"]["metric_version"],
            decoder_sha256=item["config"]["tool_hashes"][item["config"]["djxl"]],
            repetitions=item["config"]["repetitions"], warmups=item["config"]["warmups"],
            num_threads=item["config"]["num_threads"],
            backend=item["config"].get("backend", "metal" if bd.study.encoder_name(item["config"]) == "gjxl" else "cpu"),
            thread_semantics=item["config"].get("thread_semantics", "libjxl-worker-threads"),
            scores_sha256=item.get("scores_sha256"), timing_source=item.get("timing_source"),
            observation_source="calibrated",
        ) for item in studies], points=points, images=per_image,
    )
