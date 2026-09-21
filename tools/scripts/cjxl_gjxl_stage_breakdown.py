#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
"""Saved-data-only, flat GJXL wall breakdown and optional GPU diagnostics.

Input contract and collection limitations: doc/runtime-gjxl-profile.md.
This module never starts a benchmark, builds code, or modifies input data.
"""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


HOST_GROUPS = {
    "workflow": {
        "Input preparation": ("input_preparation",),
        "Quantization pipeline (host + GPU waits)": ("quantization_pipeline",),
        "CPU serializer": ("codestream_encoding",),
    },
    "serializer": {
        "Validation": ("codestream_validation",),
        "DC tokenization": ("codestream_dc_tokenization",),
        "AC tokenization": ("codestream_ac_tokenization",),
        "Entropy optimization": ("codestream_entropy_optimization",),
        "Section writing": ("codestream_section_writing",),
        "Assembly": ("codestream_assembly",),
    },
}
INPUT_GROUPS = {
    "Input geometry / storage": "input_geometry_and_storage",
    "Input color transform": "input_color_transform",
    "Matrix-scale statistics": "input_matrix_scale_stats",
    "Resident input preparation": "input_resident_preparation",
    "Quantization setup": "input_quantization_preparation",
}
UNRESOLVED_PIPELINE = "Quantization pipeline (unresolved)"
FLAT_STAGES = (*INPUT_GROUPS, "Other input preparation", UNRESOLVED_PIPELINE,
               *HOST_GROUPS["serializer"], "Other serializer", "Other workflow")
GPU_GROUPS = (
    "Reference features", "Initial quantization", "Quantizer adjustment",
    "AC strategy search", "Other frontend", "Forward transform",
    "Trial coefficients", "Trial reconstruction", "Loop filtering",
    "Butteraugli comparison", "AQ policy", "Final CfL", "DC quantization",
    "Final coefficients", "Other GPU stages",
)
COLORS = {
    "flat": ("#8CA8BB", "#7CBFB6", "#578C89", "#B5C9D7", "#7796AD", "#D9E1E7",
             "#396E9E", "#AAB7C4", "#287568", "#84B4A0", "#79629F", "#D3A454",
             "#B98269", "#BBC1CB", "#E7E9EC"),
    "gpu": tuple(plt.get_cmap("tab20").colors[:len(GPU_GROUPS)]),
}
KINDS = ("flat", "workflow", "serializer", "gpu")
RESOLUTION_LABELS = {"kodak_0_4mp": "Kodak", "clic_1_8_to_3_4mp": "CLIC test",
                     "12mp": "Unsplash 12 MP", "24mp": "Unsplash 24 MP",
                     "48mp": "Unsplash 48 MP"}
SEMANTICS = {
    "flat": "One exclusive partition of each host sample's internal workflow total. "
            "Replace input preparation and CPU serializer parents with their children and "
            "explicit residuals before averaging. Quantization remains unresolved. No GPU "
            "counter durations are inserted, rescaled, or added to this partition.",
    "workflow": "Internal profiled wall time; excludes outer teardown/publication. "
                "Quantization includes host orchestration and waits.",
    "serializer": "Exclusive host phases plus an explicit residual, within the same host capture.",
    "gpu": "Separate instrumented invocation. Sum of validated nonoverlapping stage counters, "
           "not complete GPU time or workflow wall time. Resident input preparation is absent. "
           "GPU counters must not be stacked with host times or scaled into quantization time.",
    "aggregation": "Average repetitions within each image/setting tuple, then average tuples. "
                   "Percentages are ratios of summed tuple means, not mean percentages. "
                   "Require the entire declared image/setting cohort per resolution/effort/kind.",
    "instrumentation": "The historical preview profiler changes encoder boundaries and disables combined "
                       "deferred ACS/AQ. It is diagnostic, not production-path attribution.",
}


def gpu_group(name):
    """Keep the exact stage IDs in exports; these groups only simplify the plot."""
    if name.startswith("frontend.prepare_aq.reference"):
        return "Reference features"
    if name.startswith("frontend.initial_quantization"):
        return "Initial quantization"
    if name.startswith("frontend.quant_adjustment"):
        return "Quantizer adjustment"
    if name.startswith("frontend."):
        return "AC strategy search" if "strategy" in name else "Other frontend"
    if name.endswith(".final_cfl"):
        return "Final CfL"
    if name.endswith(".dc_quantization"):
        return "DC quantization"
    if name.startswith("aq.reconstruction.forward."):
        return "Forward transform"
    if name.startswith("aq.reconstruction.coefficients."):
        return "Trial coefficients"
    if name.startswith("aq.reconstruction."):
        return "Trial reconstruction"
    if name.startswith(("aq.epf.", "aq.epf_linear.")):
        return "Loop filtering"
    if name.startswith("butteraugli."):
        return "Butteraugli comparison"
    if name.startswith("aq.policy_"):
        return "AQ policy"
    if name.startswith("aq.final_frame."):
        return "Final coefficients"
    return "Other GPU stages"


def duration(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid nonnegative duration")
    return value


def host_partitions(sample):
    if sample["backend"] != "metal":
        raise ValueError("Expected Metal host sample")
    p = sample["phase_nanoseconds"]
    result = {}
    for kind, groups in HOST_GROUPS.items():
        total = duration(p["total" if kind == "workflow" else "codestream_encoding"])
        values = {label: sum(duration(p[key]) for key in keys)
                  for label, keys in groups.items()}
        residual = total - sum(values.values())
        if residual < 0 or total <= 0:
            raise ValueError("Host partition exceeds its measured total")
        values["Other " + kind] = residual
        result[kind] = values
    # Flatten within each sample: never add parent timers to their children,
    # nor normalize serializer children to the serializer's smaller total.
    inputs = {name: duration(p[key]) for name, key in INPUT_GROUPS.items()}
    residual = p["input_preparation"] - sum(inputs.values())
    if residual < 0:
        raise ValueError("Input partition exceeds its measured total")
    flat = dict(inputs)
    flat["Other input preparation"] = residual
    flat[UNRESOLVED_PIPELINE] = p["quantization_pipeline"]
    flat.update(result["serializer"])
    flat["Other workflow"] = result["workflow"]["Other workflow"]
    if not math.isclose(sum(flat.values()), p["total"], rel_tol=1e-12, abs_tol=1):
        raise ValueError("Flat partition does not sum to workflow total")
    result["flat"] = flat
    return result


def gpu_stages(sample):
    caps = sample["capabilities"]
    if not (caps["timestamp_counter"] and caps["stage_boundary"]):
        raise ValueError("GPU stage counters unavailable")
    stages = defaultdict(int)
    previous_end = 0
    for submission in sample["submissions"]:
        for stage in submission["stages"]:
            begin, end = stage["begin_timestamp"], stage["end_timestamp"]
            elapsed = duration(stage["gpu_nanoseconds"])
            if stage.get("timestamp_valid") is False:
                dispatches = stage.get("dispatches", [])
                if (begin != 0 or end != 0 or elapsed != 0 or not dispatches
                        or not all(d.get("kind") == "indirect_threadgroups"
                                   and len(d.get("grid", [])) == 3 and 0 in d["grid"]
                                   for d in dispatches)):
                    raise ValueError("Unexplained missing GPU timestamp")
                stages[stage["stage_id"]] += 0
                continue
            if (not math.isfinite(begin) or not math.isfinite(end) or begin <= 0
                    or end < begin or abs(end - begin - elapsed) > 1):
                raise ValueError("Invalid GPU counter interval")
            if begin < previous_end:
                raise ValueError("Overlapping GPU counter intervals")
            previous_end = end
            stages[stage["stage_id"]] += elapsed
    if not stages or sum(stages.values()) <= 0:
        raise ValueError("No measured GPU stages")
    return dict(stages)


def read_capture(root, ref, capture, manifest, kind):
    path = (root / ref["path"]).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise ValueError("Profile hash mismatch: " + str(path))
    if ref["binary_sha256"] != manifest["binary_sha256"]:
        raise ValueError("Mixed benchmark binaries")
    argv = ref["argv"]
    image = next(i for i in manifest["images"] if i["image_id"] == capture["image_id"])
    if argv.count("--input") != 1 or argv.index("--input") + 1 == len(argv):
        raise ValueError("Capture requires exactly one --input")
    source = (root / argv[argv.index("--input") + 1]).resolve()
    if source != (root / image["input_path"]).resolve():
        raise ValueError("Capture input disagrees with declared image")
    if (int(argv[argv.index("--effort") + 1]) != capture["effort"]
            or int(argv[argv.index("--cpu-threads") + 1]) != manifest["cpu_threads"]
            or not math.isclose(float(argv[argv.index("--distance") + 1]),
                                capture["distance"], rel_tol=1e-6)):
        raise ValueError("Command disagrees with effort/distance/CPU settings")
    data = json.loads(raw)
    required = {"schema_version": 17 if kind == "host" else 4,
                "scope": "metal-public-workflow", "gpu_aq": "fully-resident",
                "collect_final_score": False, "warmups": manifest["warmups"],
                "sample_count": manifest["samples_per_capture"]}
    if kind == "host":
        required.update(effort=capture["effort"], cpu_threads=manifest["cpu_threads"],
                        density="default", compression="automatic", validation="metal-only")
    else:
        required["mode"] = "stage"
    if any(data.get(k) != v for k, v in required.items()):
        raise ValueError("Incompatible profile metadata: " + str(path))
    if not math.isclose(data["distance"], capture["distance"], rel_tol=1e-6):
        raise ValueError("Profile distance mismatch")
    if len(data["workloads"]) != 1:
        raise ValueError("One image per capture is required")
    workload = data["workloads"][0]
    if (workload["source_width"], workload["source_height"]) != (image["width"], image["height"]):
        raise ValueError("Image dimensions disagree")
    samples = workload["samples"]
    indices = [s["sample_index"] for s in samples]
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate sample indices")
    if any(i not in range(manifest["samples_per_capture"]) for i in indices):
        raise ValueError("Unexpected sample index")
    return samples


def load_profiles(manifest_path):
    """Validate raw captures and keep incomplete efforts out of measured bars."""
    path = Path(manifest_path).expanduser().resolve()
    m = json.loads(path.read_text())
    if m["schema_version"] != 2:
        raise ValueError("GJXL stage manifest schema 2 with input paths and hashes is required")
    for field in ("label", "source_revision", "source_description", "binary_sha256", "settings_label"):
        if not m.get(field):
            raise ValueError("Missing provenance: " + field)
    efforts, settings = m["efforts"], m["settings"]
    images = {i["image_id"]: i for i in m["images"]}
    if (not images or len(images) != len(m["images"]) or not settings
            or len(set(settings)) != len(settings) or not efforts
            or len(set(efforts)) != len(efforts) or any(e not in range(1, 11) for e in efforts)
            or m["samples_per_capture"] < 1):
        raise ValueError("Invalid declared cohort")
    input_paths = set()
    for image in images.values():
        if not image.get("input_path") or not image.get("input_sha256"):
            raise ValueError("Missing input identity: " + image["image_id"])
        source = (path.parent / image["input_path"]).resolve()
        if source in input_paths:
            raise ValueError("One input path is assigned to multiple image IDs")
        input_paths.add(source)
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != image["input_sha256"]:
            raise ValueError("Input hash mismatch: " + image["image_id"])
    rows, exact_rows, rejected, seen, counts = [], [], [], set(), {}
    capture_paths = set()
    for c in m["captures"]:
        key = (c["image_id"], c["setting"], c["effort"])
        if key in seen:
            raise ValueError("Duplicate image/setting/effort capture")
        seen.add(key)
        if c["image_id"] not in images or c["setting"] not in settings or c["effort"] not in efforts:
            raise ValueError("Capture outside declared cohort")
        base = dict(image_id=c["image_id"], setting=c["setting"], effort=c["effort"],
                    resolution_class=images[c["image_id"]]["resolution_class"])
        for mode in ("host", "gpu"):
            ref = c.get(mode)
            if ref is None:
                continue
            capture_path = (path.parent / ref["path"]).resolve()
            if capture_path in capture_paths:
                raise ValueError("Profile file reused for multiple captures")
            capture_paths.add(capture_path)
            samples = read_capture(path.parent, ref, c, m, mode)
            for sample in samples:
                sample_base = dict(base, sample_index=sample["sample_index"])
                try:
                    if mode == "host":
                        partitions = host_partitions(sample)
                    else:
                        exact = gpu_stages(sample)
                        grouped = defaultdict(int)
                        for name, ns in exact.items():
                            grouped[gpu_group(name)] += ns
                        partitions = {"gpu": dict(grouped)}
                except ValueError as error:
                    rejected.append(dict(sample_base, mode=mode, reason=str(error)))
                    continue
                if mode == "gpu":
                    exact_rows.extend(dict(sample_base, stage=name, ms=ns / 1e6)
                                      for name, ns in exact.items())
                for kind, values in partitions.items():
                    counts[key + (kind,)] = counts.get(key + (kind,), 0) + 1
                    # Explicit measured zeros preserve equal weighting across repetitions.
                    names = GPU_GROUPS if kind == "gpu" else values
                    rows.extend(dict(sample_base, kind=kind, stage=name,
                                     ms=values.get(name, 0) / 1e6) for name in names)
    columns = ["image_id", "setting", "effort", "resolution_class", "sample_index", "kind", "stage", "ms"]
    samples = pd.DataFrame(rows, columns=columns)
    resolutions = list(dict.fromkeys(i["resolution_class"] for i in m["images"]))
    coverage = []
    for resolution in resolutions:
        cohort = [i for i in images if images[i]["resolution_class"] == resolution]
        for effort in efforts:
            for kind in KINDS:
                missing = [f"{image}/{setting}" for image in cohort for setting in settings
                           if counts.get((image, setting, effort, kind), 0) != m["samples_per_capture"]]
                coverage.append(dict(resolution_class=resolution, effort=effort, kind=kind,
                                     expected_tuples=len(cohort) * len(settings),
                                     complete_tuples=len(cohort) * len(settings) - len(missing),
                                     status="incomplete" if missing else "complete",
                                     missing_tuples="; ".join(missing)))
    coverage = pd.DataFrame(coverage)
    eligible = coverage.query("status == 'complete'")[["resolution_class", "effort", "kind"]]
    complete = samples.merge(eligible, on=["resolution_class", "effort", "kind"])
    tuple_keys = ["resolution_class", "effort", "kind", "image_id", "setting", "stage"]
    tuples = complete.groupby(tuple_keys, as_index=False)["ms"].mean()
    means = tuples.groupby(["resolution_class", "effort", "kind", "stage"], as_index=False)["ms"].mean()
    if not means.empty:
        totals = means.groupby(["resolution_class", "effort", "kind"])["ms"].transform("sum")
        means["percent"] = 100 * means["ms"] / totals
    else:
        means["percent"] = pd.Series(dtype=float)
    return dict(manifest=m, manifest_path=str(path), samples=samples, means=means,
                coverage=coverage, gpu_stages=pd.DataFrame(exact_rows), rejected=pd.DataFrame(rejected))


def plot_breakdown(report, resolution, normalize=True, *, kind="flat"):
    """One bar per effort with a single denominator; GPU is a separate figure."""
    if kind not in ("flat", "gpu"):
        raise ValueError("Plot kind must be flat or gpu")
    m = report["manifest"]
    efforts = m["efforts"]
    part = report["means"].query("resolution_class == @resolution and kind == @kind")
    fig, ax = plt.subplots(figsize=(13, 6.6))
    fig.subplots_adjust(left=.085, right=.67, top=.74, bottom=.25)
    stages = FLAT_STAGES if kind == "flat" else GPU_GROUPS
    present = part.groupby("stage")["ms"].sum()
    names = [name for name in stages if present.get(name, 0) > 0]
    colors = dict(zip(stages, COLORS[kind]))
    heights = np.zeros(len(efforts))
    handles = []
    for name in names:
        vals = part[part["stage"] == name].set_index("effort")["percent" if normalize else "ms"]
        vals = vals.reindex(efforts).fillna(0).to_numpy()
        hatch = "///" if name == UNRESOLVED_PIPELINE else None
        ax.bar(efforts, vals, bottom=heights, color=colors[name], width=.72,
               hatch=hatch, edgecolor="white", linewidth=.25)
        handles.append(Patch(facecolor=colors[name], edgecolor="white", hatch=hatch, label=name))
        if normalize:
            for effort, bottom, value in zip(efforts, heights, vals):
                if value >= 8:
                    luminance = np.dot(to_rgb(colors[name]), [.299, .587, .114])
                    ax.text(effort, bottom + value / 2, f"{value:.0f}%", ha="center", va="center",
                            color="white" if luminance < .56 else "#172B40", fontsize=8,
                            bbox=dict(facecolor=colors[name], edgecolor="none", pad=1.5)
                            if hatch else None)
        heights += vals
    complete = set(part["effort"])
    for effort in efforts:
        if effort not in complete:
            ax.text(effort, .035, "missing", rotation=90, ha="center", va="bottom",
                    color="#8A9299", fontsize=8, transform=ax.get_xaxis_transform())
    if normalize:
        for effort, total in part.groupby("effort")["ms"].sum().items():
            ax.text(effort, 103, f"{total:.1f} ms", ha="center", fontsize=9)
    ax.set_ylabel(("Profiled workflow wall time (%)" if kind == "flat" else "Measured GPU stage time (%)")
                  if normalize else "Mean ms / encode")
    ax.set_ylim(0, 115 if normalize else max(1, heights.max() * 1.18))
    ax.set_xlim(min(efforts) - .65, max(efforts) + .65)
    ax.set_xticks(efforts)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if handles:
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, .5), frameon=False, fontsize=9)
    ax.set_xlabel("Effort")
    cohort = [i for i in m["images"] if i["resolution_class"] == resolution]
    label = RESOLUTION_LABELS.get(resolution, resolution)
    title = "GJXL runtime breakdown" if kind == "flat" else "GJXL GPU stages · separate diagnostic"
    fig.suptitle(f"{title} · {label}\n{m['label']}", x=.085, ha="left", fontsize=16, y=.965)
    fig.text(.085, .83, f"{len(cohort)} images · {m['settings_label']} · {m['samples_per_capture']} samples/tuple · "
             f"{m['cpu_threads']} CPU-thread cap", fontsize=10, color="#556270")
    if kind == "flat":
        fig.text(.085, .13, "One total: internal workflow wall time. Input and serializer parents are replaced by their measured substages.", fontsize=9)
        fig.text(.085, .09, "Hatched quantization time is unresolved. Finer GPU attribution requires new instrumentation and captures.", fontsize=9)
        fig.text(.085, .05, "Outer teardown/publication excluded. Missing efforts are unmeasured; only saved data is used.", fontsize=9)
    else:
        fig.text(.085, .13, "Separate GPU-instrumented invocations; denominator is the sum of recorded stage intervals, not workflow time.", fontsize=9)
        fig.text(.085, .09, "Profiling changes execution structure and omits resident input preparation. These counters cannot split the wall-time bar.", fontsize=9)
        fig.text(.085, .05, "Missing efforts have no complete capture cohort. This diagnostic is not production-path attribution.", fontsize=9)
    return fig


def generate_breakdowns(manifest_path, output_dir, *, resolution=None, normalize=True,
                        formats=("png", "svg"), show=False, gpu_diagnostics=False):
    if manifest_path is None or not Path(manifest_path).expanduser().is_file():
        print("No GJXL stage manifest. New workflow/GPU captures are required; see doc/runtime-gjxl-profile.md.")
        return None
    report = load_profiles(manifest_path)
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("samples", "means", "coverage", "gpu_stages", "rejected"):
        report[name].to_csv(output / f"gjxl-stage-{name.replace('_', '-')}.csv", index=False)
    methodology = dict(manifest=report["manifest"], manifest_path=report["manifest_path"],
                       semantics=SEMANTICS, normalize=normalize, plot_kind="flat",
                       gpu_diagnostics=gpu_diagnostics,
                       manifest_sha256=hashlib.sha256(Path(report["manifest_path"]).read_bytes()).hexdigest())
    (output / "gjxl-stage-methodology.json").write_text(json.dumps(methodology, indent=2) + "\n")
    resolutions = list(report["coverage"]["resolution_class"].unique())
    if resolution is not None:
        if resolution not in resolutions:
            raise ValueError("Resolution not in the GJXL profile cohort: " + resolution)
        resolutions = [resolution]
    report["figures"] = {}
    report["gpu_figures"] = {}
    for res in resolutions:
        for kind in (("flat", "gpu") if gpu_diagnostics else ("flat",)):
            figure = plot_breakdown(report, res, normalize, kind=kind)
            report["figures" if kind == "flat" else "gpu_figures"][res] = figure
            name = "gjxl-stage-breakdown" if kind == "flat" else "gjxl-gpu-stage-diagnostic"
            for suffix in formats:
                figure.savefig(output / f"{name}-{res}.{suffix}", dpi=180)
            if show:
                plt.show()
            plt.close(figure)
    ready = report["coverage"].query("status == 'complete'")
    for kind in KINDS:
        found = sorted(int(e) for e in ready.loc[ready["kind"] == kind, "effort"].unique())
        print(f"GJXL {kind}: complete efforts in at least one resolution = {found}; see coverage for the full grid.")
    return report
