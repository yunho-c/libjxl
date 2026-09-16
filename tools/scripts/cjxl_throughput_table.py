#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Matched-nominal-setting throughput tables from saved timing ledgers only."""

from collections import defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

import pandas as pd

import cjxl_quality_characterization as study


DEFAULT_QUALITIES = (30, 50, 70, 80, 90, 95)
TABLE_COLUMNS = ["effort", "libjxl_mp_s", "gjxl_mp_s", "gjxl_over_libjxl"]


def _selection(values, name):
    values = list(values)
    if (not values or len(set(values)) != len(values)
            or any(isinstance(x, bool) or not isinstance(x, int) for x in values)):
        raise ValueError(name + " must be nonempty, unique integers")
    return sorted(values)


def _snapshot(path):
    """Hash the exact bytes analyzed, including when a collection is paused."""
    data = path.read_bytes() if path.exists() else b""
    rows = [json.loads(line) for line in data.splitlines()]
    return rows, {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                  "exists": path.exists(), "record_count": len(rows)}


def _load_runs(libjxl_run, gjxl_run):
    runs = [Path(p).expanduser().resolve() for p in (libjxl_run, gjxl_run)]
    configs = [study.load_config(p) for p in runs]
    lib, gjxl = configs
    if study.encoder_name(lib) != "libjxl":
        raise ValueError("libjxl_run must select a libjxl quality-study manifest")
    if study.encoder_name(gjxl) != "gjxl" or not study.is_fixed(gjxl):
        raise ValueError("gjxl_run must select a GJXL fixed nominal-quality study")
    if gjxl.get("backend") != "metal" or gjxl.get("metal_aq_mode") != "fully-resident":
        raise ValueError("This table requires fully-resident Metal GJXL")
    for key in ("num_threads", "repetitions", "warmups", "intensity_target"):
        if lib[key] != gjxl[key]:
            raise ValueError("Incompatible timing protocol: " + key)
    if lib["repetitions"] < 1:
        raise ValueError("Timing repetitions must be positive")
    timing_run = Path(lib["source_run"]).expanduser().resolve()
    source_meta = timing_run / "metadata.json"
    study.verify_file(source_meta, lib["source_metadata_sha256"])
    original = json.loads(source_meta.read_text())
    source = original["configuration"]
    for field, key in (("thread_count", "num_threads"),
                       ("timing_repetitions", "repetitions"),
                       ("warmups_per_process", "warmups")):
        if source[field] != lib[key]:
            raise ValueError("libjxl source timing protocol mismatch: " + field)
    revision = source["build_records"]["ordinary"]["content"]["libjxl_revision"]
    if revision != lib["libjxl_revision"]:
        raise ValueError("libjxl source revision mismatch")
    result = []
    for codec, run, config, timing in zip(("libjxl", "gjxl"), runs, configs,
                                         (timing_run, runs[1])):
        records, ledger = _snapshot(timing / "timings.jsonl")
        result.append({"codec": codec, "run": str(run), "config": config,
                       "qualities": source["qualities"] if codec == "libjxl" else config["qualities"],
                       "distances": source["quality_to_distance"] if codec == "libjxl" else config["quality_to_distance"],
                       "records": records, "ledger": ledger,
                       "metadata_sha256": study.digest(run / "metadata.json")})
    return result, original.get("host", {})


def _cohort(studies, min_megapixels, image_ids):
    if not math.isfinite(min_megapixels) or min_megapixels < 0:
        raise ValueError("min_megapixels must be finite and nonnegative")
    manifests = []
    for item in studies:
        images = item["config"]["images"]
        lookup = {image["image_id"]: image for image in images}
        if len(lookup) != len(images):
            raise ValueError("Duplicate manifest image")
        manifests.append(lookup)
    names = list(manifests[0]) if image_ids is None else list(image_ids)
    if not names or len(set(names)) != len(names):
        raise ValueError("Select a nonempty, unique image cohort")
    if any(not set(names) <= images.keys() for images in manifests):
        raise ValueError("Image cohort is missing from a study manifest")
    selected = []
    for name in names:
        image = manifests[0][name]
        for field in (*study.SOURCE_FIELDS, "pixels", "pfm_sha256", "color_encoding"):
            if image[field] != manifests[1][name][field]:
                raise ValueError("Reference or geometry mismatch: " + field)
        if image["pixels"] != image["width"] * image["height"]:
            raise ValueError("Invalid manifest pixel count")
        if image["pixels"] / 1e6 >= min_megapixels:
            selected.append(image)
    if not selected:
        raise ValueError("No images pass the selected resolution threshold")
    return selected


def _tuple_rows(item, images, efforts, qualities):
    config, codec = item["config"], item["codec"]
    manifest = {image["image_id"]: image for image in images}
    grouped, seen = defaultdict(dict), set()
    revisions = {e: config.get("encoder_revision", config.get("libjxl_revision"))
                 for e in efforts}
    for origin in config.get("composite_sources", []):
        for effort in origin["efforts"]:
            revisions[effort] = origin["encoder_revision"]
    for row in item["records"]:
        key = (row["image_id"], row["effort"], row["quality"])
        name, effort, quality = key
        if name not in manifest or effort not in efforts or quality not in qualities:
            continue
        image = manifest[name]
        expected_job = study.fixed_key(name, quality, effort)
        repetition = row["repetition"]
        if (type(repetition) is not int or repetition not in range(config["repetitions"])
                or row["job_id"] != expected_job
                or row["sample_id"] != f"{expected_job}|repetition={repetition}"):
            raise ValueError("Invalid timing identity or repetition: " + expected_job)
        if row["sample_id"] in seen or repetition in grouped[key]:
            raise ValueError("Duplicate timing repetition: " + expected_job)
        seen.add(row["sample_id"])
        if (row["harness_revision"] != revisions[effort]
                or row["thread_count"] != config["num_threads"]
                or row["distance"] != item["distances"][str(quality)]
                or any(row[field] != image[field] for field in (*study.SOURCE_FIELDS, "pixels"))):
            raise ValueError("Timing configuration or geometry mismatch: " + expected_job)
        if codec == "gjxl" and (
                row.get("configuration_id") != config["configuration_id"]
                or row.get("encoder") != codec
                or row.get("reference_sha256") != image["pfm_sha256"]
                or row.get("resampling") != 1):
            raise ValueError("GJXL timing identity or resampling mismatch: " + expected_job)
        if row.get("resampling", 1) != 1:
            raise ValueError("Resampled timing is not eligible: " + expected_job)
        elapsed = row["elapsed_nanoseconds"]
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("Invalid complete-encode duration: " + expected_job)
        if grouped[key]:
            previous = next(iter(grouped[key].values()))
            if any(row[field] != previous[field]
                   for field in ("output_sha256", "encoded_bytes")):
                raise ValueError("Codestream changed between repetitions: " + expected_job)
        grouped[key][repetition] = row
    rows = []
    for image in images:
        for effort in efforts:
            for quality in qualities:
                samples = grouped[(image["image_id"], effort, quality)]
                complete = len(samples) == config["repetitions"]
                rows.append({
                    "encoder": codec, "image_id": image["image_id"],
                    "resolution_class": image["resolution_class"],
                    "megapixels": image["pixels"] / 1e6, "effort": effort,
                    "quality": quality, "distance": item["distances"][str(quality)],
                    "timing_sample_count": len(samples), "timing_complete": complete,
                    "median_seconds": statistics.median(
                        row["elapsed_nanoseconds"] / 1e9 for row in samples.values()
                    ) if complete else None,
                })
    return rows


def _aggregate(rows, efforts, qualities, repetitions, resolution="all"):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["encoder"], row["effort"], row["quality"])].append(row)
    by_quality, coverage, table = [], [], []
    for effort in efforts:
        result = {"effort": effort}
        for codec in ("libjxl", "gjxl"):
            rates = []
            for quality in qualities:
                group = groups[(codec, effort, quality)]
                missing = [row for row in group if not row["timing_complete"]]
                rate = (sum(row["megapixels"] for row in group)
                        / sum(row["median_seconds"] for row in group)) if not missing else None
                rates.append(rate)
                base = {"resolution_class": resolution, "encoder": codec,
                        "effort": effort, "quality": quality}
                by_quality.append({**base, "distance": group[0]["distance"],
                                   "throughput_mp_s": rate})
                coverage.append({
                    **base, "status": "incomplete" if missing else "complete",
                    "expected_images": len(group), "complete_images": len(group) - len(missing),
                    "expected_samples": len(group) * repetitions,
                    "timing_samples": sum(row["timing_sample_count"] for row in group),
                    "missing_samples": sum(repetitions - row["timing_sample_count"] for row in group),
                    "missing_image_ids": ";".join(row["image_id"] for row in missing),
                })
            result[codec + "_mp_s"] = statistics.mean(rates) if all(
                rate is not None for rate in rates) else None
        result["gjxl_over_libjxl"] = (result["gjxl_mp_s"] / result["libjxl_mp_s"]
                                      if all(result[c + "_mp_s"] is not None
                                             for c in ("libjxl", "gjxl")) else None)
        table.append(result)
    return table, by_quality, coverage


def build_tables(libjxl_run, gjxl_run, *, min_megapixels=1.0,
                 qualities=DEFAULT_QUALITIES, efforts=None, image_ids=None):
    """Read current fixed-sweep timings; never drop missing image/quality cells."""
    studies, host = _load_runs(libjxl_run, gjxl_run)
    qualities = _selection(qualities, "qualities")
    efforts = _selection(studies[0]["config"]["efforts"] if efforts is None else efforts,
                         "efforts")
    for item in studies:
        if not set(efforts) <= set(item["config"]["efforts"]):
            raise ValueError("Requested effort is absent from a study manifest")
        if not set(qualities) <= set(item["qualities"]):
            raise ValueError("Requested quality is absent from a study manifest")
    for q in qualities:
        distance = studies[0]["distances"][str(q)]
        if distance != studies[1]["distances"][str(q)]:
            raise ValueError("Requested distance mappings differ")
        # The retained libjxl protocol selects automatic downsampling at d >= 10.
        if not math.isfinite(distance) or not 0 < distance < 10:
            raise ValueError("Select lossy distances below 10 to exclude automatic resampling (Q10)")
    images = _cohort(studies, min_megapixels, image_ids)
    tuples = [row for item in studies for row in _tuple_rows(item, images, efforts, qualities)]
    repetitions = studies[0]["config"]["repetitions"]
    table, by_quality, coverage = _aggregate(tuples, efforts, qualities, repetitions)
    by_resolution, weights = [], []
    total_mp = sum(image["pixels"] / 1e6 for image in images)
    for resolution in sorted({image["resolution_class"] for image in images}):
        selected = [row for row in tuples if row["resolution_class"] == resolution]
        values, _, _ = _aggregate(selected, efforts, qualities, repetitions, resolution)
        by_resolution.extend({"resolution_class": resolution, **row} for row in values)
        subset = [image for image in images if image["resolution_class"] == resolution]
        mp = sum(image["pixels"] / 1e6 for image in subset)
        weights.append({"resolution_class": resolution, "image_count": len(subset),
                        "megapixels": mp, "pixel_share": mp / total_mp})
    caption = (
        f"Warm encoding throughput at matched nominal distance and effort, on {len(images)} "
        f"images with at least {min_megapixels:g} MP. Rates pool original megapixels over "
        f"summed per-image median encode times ({repetitions} repetitions), then average "
        "equally across " + ", ".join(f"Q{q}" for q in qualities) + ". "
        "GJXL uses fully-resident Metal. Speedup is GJXL/libjxl; a dash denotes incomplete "
        "timing coverage. This compares nominal presets, not matched decoded quality."
    )
    provenance = []
    for item in studies:
        config = item["config"]
        provenance.append({
            "encoder": item["codec"], "run": item["run"],
            "metadata_sha256": item["metadata_sha256"], "timing_ledger": item["ledger"],
            "revision": config.get("encoder_revision", config.get("libjxl_revision")),
            "composite_sources": config.get("composite_sources", []),
            "configuration_id": config["configuration_id"],
            "num_threads": config["num_threads"],
            "thread_semantics": config.get("thread_semantics", "libjxl worker threads"),
            "warmups": config["warmups"], "repetitions": repetitions,
            "source_metadata_sha256": config.get("source_metadata_sha256"),
            "benchmark": config.get("benchmark"),
            "benchmark_sha256": config["tool_hashes"].get(config.get("benchmark")),
            "encoder_build_sha256": config.get("encoder_build_sha256"),
        })
    report = {
        "caption": caption, "schema_version": 1, "batch_size": 1,
        "min_megapixels": min_megapixels, "qualities": qualities, "efforts": efforts,
        "quality_to_distance": {str(q): studies[0]["distances"][str(q)] for q in qualities},
        "aggregation": "arithmetic mean over qualities of sum(input MP) / sum(per-image median seconds)",
        "speedup": "ratio of encoder aggregate throughputs, not mean of per-image ratios",
        "timing_boundary": "warm uninstrumented complete encode call; CPU/GPU work, transfers and synchronization included; startup, input preparation, file I/O and scoring excluded",
        "resource_note": "Equal numeric thread settings have different participation semantics; GJXL additionally uses Metal",
        "images": [{k: image[k] for k in (*study.SOURCE_FIELDS, "pixels", "pfm_sha256")}
                   for image in images],
        "resolution_weights": weights, "sources": provenance, "libjxl_host": host,
    }
    return {"table": pd.DataFrame(table, columns=TABLE_COLUMNS),
            "by_quality": pd.DataFrame(by_quality),
            "by_resolution": pd.DataFrame(by_resolution),
            "coverage": pd.DataFrame(coverage), "tuples": pd.DataFrame(tuples),
            "methodology": report}


def _number(value, ratio=False):
    if pd.isna(value):
        return "—"
    return f"{value:.2f}×" if ratio else (f"{value:.1f}" if value >= 10 else f"{value:.2f}")


def table_html(tables):
    labels = dict(zip(TABLE_COLUMNS, ("Effort", "libjxl (MP/s)", "GJXL (MP/s)", "GJXL/libjxl")))
    frame = tables["table"].rename(columns=labels)
    return ("<div style='max-width:760px'><p>" + html.escape(tables["methodology"]["caption"])
            + "</p>" + frame.to_html(index=False, border=0, na_rep="—", justify="right",
                                    formatters={"libjxl (MP/s)": _number,
                                                "GJXL (MP/s)": _number,
                                                "GJXL/libjxl": lambda x: _number(x, True)}) + "</div>")


def write_tables(tables, output_dir):
    """Export analysis artifacts only; no writes to source study directories."""
    output = Path(output_dir).expanduser().resolve()
    for source in tables["methodology"]["sources"]:
        for root in (Path(source["run"]), Path(source["timing_ledger"]["path"]).parent):
            if output == root or root in output.parents:
                raise ValueError("Export outside the source study directories")
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, suffix in (("table", ""), ("by_quality", "-by-quality"),
                        ("by_resolution", "-by-resolution"), ("coverage", "-coverage"),
                        ("tuples", "-image-tuples")):
        path = output / ("encoding-throughput" + suffix + ".csv")
        tables[key].to_csv(path, index=False)
        paths[key] = str(path)
    report = output / "encoding-throughput-methodology.json"
    report.write_text(json.dumps(tables["methodology"], indent=2, allow_nan=False) + "\n")
    paths["methodology"] = str(report)
    tex = [r"% Requires \usepackage{booktabs}", r"\begin{table}[t]", r"\centering",
           r"\begin{tabular}{rrrr}", r"\toprule",
           r"Effort & libjxl (MP/s) & GJXL (MP/s) & GJXL/libjxl \\", r"\midrule"]
    for row in tables["table"].itertuples(index=False):
        values = [str(row.effort), _number(row.libjxl_mp_s), _number(row.gjxl_mp_s),
                  _number(row.gjxl_over_libjxl, True)]
        tex.append(" & ".join(value.replace("—", "--").replace("×", r"$\times$")
                              for value in values) + r" \\")
    tex.extend([r"\bottomrule", r"\end{tabular}",
                r"\caption{" + tables["methodology"]["caption"] + "}",
                r"\label{tab:encoding-throughput}", r"\end{table}"])
    latex = output / "encoding-throughput.tex"
    latex.write_text("\n".join(tex) + "\n")
    paths["latex"] = str(latex)
    page = output / "encoding-throughput.html"
    page.write_text("<!doctype html><meta charset='utf-8'><title>Encoding throughput</title>"
                    "<style>body{font:16px Georgia,serif;margin:2rem}table{border-collapse:collapse}"
                    "th,td{padding:.35rem .8rem;text-align:right}thead{border-block:1px solid}"
                    "tbody{border-bottom:1px solid}</style>" + table_html(tables))
    paths["html"] = str(page)
    return paths
