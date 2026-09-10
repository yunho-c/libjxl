#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Saved-sweep BD-rate analysis. No encoding, scoring, or study writes.

Integrate log(rate) against measured fast-ssim2 on an explicit common interval.
Average per-image BD percentages, never fit a pooled image RD curve. Timings
are log-linearly interpolated on a common score grid, then averaged in ms.
"""

from collections import Counter, defaultdict
import math
from pathlib import Path
import statistics

import numpy as np
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator

import cjxl_quality_characterization as study


def _interval(quality_range):
    if len(quality_range) != 2:
        raise ValueError("quality_range must contain two scores")
    low, high = map(float, quality_range)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("quality_range must be finite and increasing")
    return low, high


def bd_rate(anchor_scores, anchor_rates, test_scores, test_rates,
            quality_range=(75, 85), method="pchip"):
    """BD-rate percent on a fixed interval; reject extrapolation and reversals."""
    low, high = _interval(quality_range)
    if method not in ("pchip", "akima"):
        raise ValueError("method must be pchip or akima")
    interpolator = PchipInterpolator if method == "pchip" else Akima1DInterpolator
    integrals = []
    for scores, rates in ((anchor_scores, anchor_rates), (test_scores, test_rates)):
        scores, rates = np.asarray(scores, dtype=float), np.asarray(rates, dtype=float)
        if (scores.ndim != 1 or rates.shape != scores.shape or len(scores) < 4
                or not np.isfinite(scores).all() or not np.isfinite(rates).all()
                or (rates <= 0).any()):
            raise ValueError("Need at least four finite score/rate pairs with positive rates")
        if (np.diff(scores) <= 0).any() or (np.diff(rates) <= 0).any():
            raise ValueError("Scores and rates must increase in encoder setting order")
        if scores[0] > low or scores[-1] < high:
            raise ValueError("Curve does not bracket quality_range; extrapolation forbidden")
        curve = interpolator(scores, np.log(rates), extrapolate=False)
        integrals.append(float(curve.integrate(low, high)))
    result = 100 * math.expm1((integrals[1] - integrals[0]) / (high - low))
    if not math.isfinite(result):
        raise ValueError("Nonfinite BD-rate")
    return result


def _refresh_timings(run, config, scores):
    """Join saved repetitions by codestream identity, allowing paused runs to resume.

    Score records freeze their timing count when scored. Reading the original
    timing ledger avoids requiring rescoring after additional repetitions.
    """
    if study.encoder_name(config) == "libjxl":
        timing_run = Path(config["source_run"])
        study.verify_file(timing_run / "metadata.json", config["source_metadata_sha256"])
        samples = study.ledger(timing_run / "timings.jsonl")
    elif study.is_fixed(config):
        timing_run = run
        samples = study.read_records(run, "timings", config)
    else:
        raise ValueError("For GJXL BD-rate, select the fixed-quality run")
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["job_id"]].append(sample)
    refreshed = []
    for score in scores:
        row = dict(score)
        values = grouped.get(row["source_job_id"], [])
        repetitions = [value["repetition"] for value in values]
        if len(set(repetitions)) != len(repetitions):
            raise ValueError("Duplicate timing repetitions for " + row["source_job_id"])
        if not set(repetitions) <= set(range(config["repetitions"])):
            raise ValueError("Unexpected timing repetition")
        for value in values:
            for key in ("image_id", "effort", "distance", "encoded_bytes",
                        "output_sha256", "width", "height", "pixels"):
                if value[key] != row[key]:
                    raise ValueError("Timing/score identity mismatch: " + key)
            if value["thread_count"] != config["num_threads"]:
                raise ValueError("Timing thread configuration mismatch")
            elapsed = value["elapsed_nanoseconds"]
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise ValueError("Invalid complete-encode timing")
        row["timing_sample_count"] = len(values)
        row["elapsed_ms"] = (statistics.median(v["elapsed_nanoseconds"] for v in values)
                             / 1e6 if values else None)
        refreshed.append(row)
    return refreshed, {
        "path": str(timing_run / "timings.jsonl"),
        "sha256": study.digest(timing_run / "timings.jsonl"),
    }


def load_studies(runs):
    """Load immutable configurations and current saved ledgers, without collectors."""
    result = []
    for path in runs:
        run = Path(path).expanduser().resolve()
        config = study.load_config(run)
        scores = study.read_records(run, "scores", config)
        scores, timing_source = _refresh_timings(run, config, scores)
        result.append({
            "run": str(run), "config": config, "scores": scores,
            "scores_sha256": study.digest(run / "scores.jsonl")
            if (run / "scores.jsonl").exists() else None,
            "timing_source": timing_source,
        })
    return result


def _validate_studies(studies, baseline_encoder, baseline_effort, image_ids):
    if not studies:
        raise ValueError("Select at least one saved study")
    encoders = [study.encoder_name(item["config"]) for item in studies]
    if len(set(encoders)) != len(encoders):
        raise ValueError("Select one run per encoder")
    if baseline_encoder not in encoders:
        raise ValueError("Baseline encoder is absent from the selected studies")
    baseline = studies[encoders.index(baseline_encoder)]["config"]
    if baseline_effort not in baseline["efforts"]:
        raise ValueError("Baseline effort is absent from the study configuration")
    selected = (list(image_ids) if image_ids is not None
                else [image["image_id"] for image in baseline["images"]])
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Select a nonempty, unique image cohort")
    references = []
    for item in studies:
        config = item["config"]
        if config.get("metric", "fast-ssim2") != "fast-ssim2":
            raise ValueError("BD-rate currently supports measured fast-ssim2 scores only")
        for key in ("metric_version", "intensity_target", "num_threads",
                    "repetitions", "warmups"):
            if config[key] != baseline[key]:
                raise ValueError("Incompatible study configuration: " + key)
        if config["tool_hashes"][config["djxl"]] != baseline["tool_hashes"][baseline["djxl"]]:
            raise ValueError("Studies require the same pinned decoder")
        images = {image["image_id"]: image for image in config["images"]}
        if len(images) != len(config["images"]) or not set(selected) <= images.keys():
            raise ValueError("Fixed image cohort is missing or duplicated in a study")
        references.append([
            tuple(images[name][key] for key in (*study.SOURCE_FIELDS, "pfm_sha256"))
            for name in selected
        ])
        for row in item["scores"]:
            image = images.get(row["image_id"])
            if image is None or row["effort"] not in config["efforts"]:
                raise ValueError("Score observation is outside its study manifest")
            if (row.get("encoder") != study.encoder_name(config)
                    or row.get("metric") != "fast-ssim2"
                    or row["reference_sha256"] != image["pfm_sha256"]
                    or any(row[key] != image[key] for key in study.SOURCE_FIELDS)
                    or row["pixels"] != image["width"] * image["height"]):
                raise ValueError("Score reference, geometry, or encoder identity mismatch")
    if any(reference != references[0] for reference in references[1:]):
        raise ValueError("Studies require identical reference hashes and geometry")
    images = {image["image_id"]: image for image in baseline["images"]}
    return [images[name] for name in selected]


def _curve(rows, quality_range, repetitions, grid):
    detail = {"sample_count": len(rows), "excluded_resampled_count": 0}
    if not rows:
        return {**detail, "status": "missing-scores"}
    if any(row.get("resampling") not in (1, 2) for row in rows):
        return {**detail, "status": "unknown-resampling"}
    detail["excluded_resampled_count"] = sum(row["resampling"] != 1 for row in rows)
    rows = [row for row in rows if row["resampling"] == 1]
    detail["sample_count"] = len(rows)
    if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key])
           for row in rows for key in ("distance", "score", "encoded_bytes")):
        return {**detail, "status": "invalid-score-or-rate"}
    rows = sorted(rows, key=lambda row: row["distance"], reverse=True)
    if len(rows) < 4:
        return {**detail, "status": "insufficient-samples"}
    if len({row["distance"] for row in rows}) != len(rows):
        return {**detail, "status": "duplicate-distance"}
    scores = np.array([row["score"] for row in rows])
    rates = np.array([row["encoded_bytes"] for row in rows], dtype=float)
    detail.update(score_min=float(min(scores)), score_max=float(max(scores)))
    if (rates <= 0).any() or (np.diff(scores) <= 0).any() or (np.diff(rates) <= 0).any():
        return {**detail, "status": "nonmonotone-curve"}
    low, high = quality_range
    if scores[0] > low or scores[-1] < high:
        return {**detail, "status": "missing-quality-bracket"}
    detail.update(status="ready", rate_ready=True, scores=scores, rates=rates)
    if any(row.get("timing_sample_count") != repetitions for row in rows):
        return {**detail, "status": "incomplete-timing"}
    times = [row.get("elapsed_ms") for row in rows]
    if any(value is None or not math.isfinite(value) or value <= 0 for value in times):
        return {**detail, "status": "invalid-timing"}
    detail["mean_encode_ms"] = float(np.exp(np.interp(grid, scores, np.log(times))).mean())
    return detail


def analyze(studies, baseline_encoder="libjxl", baseline_effort=7,
            quality_range=(75, 85), efforts=None, image_ids=None, grid_size=101):
    """Return JSON-ready per-image results and fixed-cohort coverage diagnostics."""
    quality_range = _interval(quality_range)
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer >= 2")
    images = _validate_studies(studies, baseline_encoder, baseline_effort, image_ids)
    selected_efforts = list(efforts) if efforts is not None else sorted({
        effort for item in studies for effort in item["config"]["efforts"]
    })
    if (not selected_efforts or len(set(selected_efforts)) != len(selected_efforts)
            or any(not isinstance(effort, int) or not 1 <= effort <= 10
                   for effort in selected_efforts)):
        raise ValueError("efforts must be unique integers in 1..10")
    selected_efforts.sort()
    grid = np.linspace(*quality_range, grid_size)
    curves = {}
    for item in studies:
        grouped = defaultdict(list)
        for row in item["scores"]:
            grouped[row["image_id"], row["effort"]].append(row)
        encoder = study.encoder_name(item["config"])
        for image in images:
            for effort in set(selected_efforts + [baseline_effort]):
                curves[encoder, image["image_id"], effort] = _curve(
                    grouped[image["image_id"], effort], quality_range,
                    item["config"]["repetitions"], grid,
                )
    rows = []
    for item in studies:
        encoder = study.encoder_name(item["config"])
        for effort in selected_efforts:
            for image in images:
                test = curves[encoder, image["image_id"], effort]
                anchor = curves[baseline_encoder, image["image_id"], baseline_effort]
                row = {
                    "encoder": encoder, "effort": effort,
                    "image_id": image["image_id"], "corpus": image["corpus"],
                    "resolution_class": image["resolution_class"],
                    **{key: value for key, value in test.items()
                       if key not in ("scores", "rates", "rate_ready")},
                    "baseline_status": anchor["status"],
                }
                if not anchor.get("rate_ready"):
                    row["status"] = "baseline-" + anchor["status"]
                elif test.get("rate_ready"):
                    for method in ("pchip", "akima"):
                        row["bd_rate_" + method] = bd_rate(
                            anchor["scores"], anchor["rates"], test["scores"],
                            test["rates"], quality_range, method,
                        )
                    row["interpolation_delta_pp"] = row["bd_rate_akima"] - row["bd_rate_pchip"]
                rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        for scope in ("all", row["resolution_class"]):
            groups[scope, row["encoder"], row["effort"]].append(row)
    points = []
    for (scope, encoder, effort), group in sorted(groups.items()):
        ready = [row for row in group if row["status"] == "ready"]
        rate_ready = [row for row in group if "bd_rate_pchip" in row]
        point = {
            "scope": scope, "encoder": encoder, "effort": effort,
            "image_count": len(group), "ready_count": len(ready),
            "rate_ready_count": len(rate_ready),
            "status": "ready" if len(ready) == len(group) else "missing-coverage",
            "missing_reasons": dict(Counter(row["status"] for row in group
                                            if row["status"] != "ready")),
            "cohort": [row["image_id"] for row in group],
        }
        if len(rate_ready) == len(group):
            for method in ("pchip", "akima"):
                point["bd_rate_" + method] = statistics.mean(row["bd_rate_" + method]
                                                            for row in group)
            point["interpolation_delta_pp"] = point["bd_rate_akima"] - point["bd_rate_pchip"]
            point["max_image_interpolation_delta_pp"] = max(
                abs(row["interpolation_delta_pp"]) for row in group)
        if point["status"] == "ready":
            point["mean_encode_ms"] = statistics.mean(row["mean_encode_ms"] for row in group)
        points.append(point)
    return {
        "schema_version": 1,
        "baseline": {"encoder": baseline_encoder, "effort": baseline_effort},
        "quality_range": list(quality_range), "quality_grid_size": grid_size,
        "metric": "fast-ssim2", "quality_axis": "raw measured fast-ssim2 score",
        "rate_method": "PCHIP log(bytes), Akima sensitivity, no extrapolation",
        "aggregation": "arithmetic mean of per-image BD-rate percentages; fixed cohort",
        "timing_method": "log-linear interpolation on a uniform score grid; arithmetic mean ms",
        "timing_boundary": "warm complete encode call; excludes startup, file I/O, decode and scoring",
        "resampling": "only resampling=1; at least four monotone measured points per curve",
        "timing_coverage": "all retained unresampled curve points require the configured repetitions",
        "efforts": selected_efforts,
        "sources": [{
            "run": item["run"], "encoder": study.encoder_name(item["config"]),
            "configuration_id": item["config"]["configuration_id"],
            "metric_version": item["config"]["metric_version"],
            "repetitions": item["config"]["repetitions"],
            "warmups": item["config"]["warmups"],
            "num_threads": item["config"]["num_threads"],
            "scores_sha256": item.get("scores_sha256"),
            "timing_source": item.get("timing_source"),
        } for item in studies],
        "points": points, "images": rows,
    }
