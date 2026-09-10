#!/usr/bin/env -S uv run --script
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8,<4",
#   "numpy>=1.26,<3",
#   "pandas>=2.2,<3",
# ]
# ///

# %% [markdown]
# # libjxl runtime-characterization figures
#
# This notebook reads `image-tuples.csv` (or an explicitly labeled partial
# snapshot) produced by `cjxl_runtime_characterization.py summarize`.
#
# Measurement semantics are deliberately kept separate:
#
# - `complete_encode_*` is uninstrumented complete-encode wall time.
# - `phase_wall_*` contains mutually exclusive serializer wall phases.
# - `frontend_residual_wall_ms` is complete profiled wall time minus the
#   serializer phase union.
# - `work_aggregate_worker_*` is overlapping worker CPU time and is therefore
#   not plotted as latency here.
#
# Partial timing cells are drawn with open markers. Stage figures omit cells
# without exhaustive stage data.
# Expanded wall-v2 data uses exclusive durations for additive charts and
# inclusive durations only for the separate refinement breakdown.

# %%
import argparse
import math
import os
import pathlib
import sys

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %% [markdown]
# ## Notebook configuration
#
# Edit these paths when opening this file as a Jupytext notebook. When invoked
# as a script, use `--input` and `--output-dir` instead.
# The default input points at the refreshed, paused wall-v2 partial snapshot:
# expanded stage wall times joined to the original uninstrumented timings.
# Refreshed 2026-09-08: 4,469/4,550 stage tuples; efforts 1-9 complete.
# Missing stage captures stay empty; pooled stage bars omit incomplete efforts.

# %%
DEFAULT_INPUT_CSV = pathlib.Path(
    "/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/"
    "run-e8ff0976-wall-v2/summary/image-tuples.partial.csv"
)
INPUT_CSV = pathlib.Path(
    os.environ.get("CJXL_IMAGE_TUPLES_CSV", DEFAULT_INPUT_CSV)
).expanduser()
OUTPUT_DIR = pathlib.Path(
    os.environ.get("CJXL_CHARACTERIZATION_PLOT_DIR", "characterization-plots")
).expanduser()
EXPECTED_TIMING_SAMPLES = 5
SAVE_FORMATS = ("png", "svg")
QUALITY_RUN = pathlib.Path(
    os.environ.get(
        "CJXL_QUALITY_RUN",
        "/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/quality-full-20260908",
    )
).expanduser()
GJXL_RUN = pathlib.Path(
    os.environ.get(
        "CJXL_GJXL_RUN",
        "/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/quality-gjxl-pilot-20260909",
    )
).expanduser()
BUTTERAUGLI_RUN = pathlib.Path(
    os.environ.get(
        "CJXL_BUTTERAUGLI_RUN",
        "/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/quality-butteraugli-pilot-20260908",
    )
).expanduser()
# Keep the two-metric comparison on the same three-image pilot cohort.
BUTTERAUGLI_COMPARE_RUN = pathlib.Path(
    os.environ.get(
        "CJXL_BUTTERAUGLI_COMPARE_RUN",
        "/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/quality-pilot-20260908",
    )
).expanduser()


# %% [markdown]
# ## Load and validate the per-tuple table

# %%
REQUIRED_COLUMNS = frozenset(
    (
        "job_id",
        "image_id",
        "corpus",
        "resolution_class",
        "megapixels",
        "quality",
        "effort",
        "timing_sample_count",
        "complete_encode_median_ms",
        "complete_encode_p10_ms",
        "complete_encode_p90_ms",
        "complete_encode_stdev_ms",
        "bits_per_pixel",
        "profiled_complete_wall_ms",
        "frontend_residual_wall_ms",
        "phase_wall_coefficient_tokenization_ms",
        "phase_wall_entropy_model_construction_ms",
        "phase_wall_model_and_token_emission_ms",
        "phase_wall_framing_and_assembly_ms",
    )
)

STAGE_COMPONENTS = (
    ("frontend_residual_wall_ms", "Frontend and other encoder work", "#5B6770"),
    (
        "phase_wall_coefficient_tokenization_ms",
        "Coefficient tokenization",
        "#4477AA",
    ),
    (
        "phase_wall_entropy_model_construction_ms",
        "Entropy-model construction",
        "#EE6677",
    ),
    (
        "phase_wall_model_and_token_emission_ms",
        "Model and token emission",
        "#228833",
    ),
    (
        "phase_wall_framing_and_assembly_ms",
        "Framing and assembly",
        "#CCBB44",
    ),
)

QUALITY_COLORS = {
    quality: plt.get_cmap("viridis")(position)
    for quality, position in zip(
        (10, 30, 50, 70, 80, 90, 95), np.linspace(0.05, 0.95, 7)
    )
}

RESOLUTION_NAMES = {
    "kodak_0_4mp": "Kodak",
    "clic_1_8_to_3_4mp": "CLIC test",
    "12mp": "Unsplash 12 MP",
    "24mp": "Unsplash 24 MP",
    "48mp": "Unsplash 48 MP",
}


def load_image_tuples(path):
    path = pathlib.Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("image-tuples CSV does not exist: %s" % path)
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError("CSV is missing required columns: %s" % ", ".join(missing))
    if frame.empty:
        raise ValueError("CSV contains no tuple rows: %s" % path)
    if frame["job_id"].duplicated().any():
        duplicates = frame.loc[frame["job_id"].duplicated(), "job_id"].head(3)
        raise ValueError("CSV contains duplicate job_id values: %s" % list(duplicates))

    numeric = (
        "megapixels",
        "quality",
        "effort",
        "timing_sample_count",
        "complete_encode_median_ms",
        "complete_encode_p10_ms",
        "complete_encode_p90_ms",
        "complete_encode_stdev_ms",
        "bits_per_pixel",
        "profiled_complete_wall_ms",
        *(name for name, _, _ in STAGE_COMPONENTS),
    )
    for column in dict.fromkeys(numeric):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in (
        "megapixels",
        "quality",
        "effort",
        "timing_sample_count",
        "complete_encode_median_ms",
        "complete_encode_p10_ms",
        "complete_encode_p90_ms",
        "complete_encode_stdev_ms",
        "bits_per_pixel",
    ):
        if frame[column].isna().any():
            raise ValueError("CSV has missing or invalid values in %s" % column)
    if (frame["megapixels"] <= 0).any() or (
        frame["complete_encode_median_ms"] <= 0
    ).any():
        raise ValueError("megapixels and runtime values must be positive")

    frame["quality"] = frame["quality"].astype(int)
    frame["effort"] = frame["effort"].astype(int)
    frame["timing_sample_count"] = frame["timing_sample_count"].astype(int)
    frame.attrs["source_path"] = path
    return frame


def resolution_order(frame):
    return (
        frame.groupby("resolution_class", observed=True)["megapixels"]
        .median()
        .sort_values()
        .index.tolist()
    )


def resolution_label(frame, resolution_class):
    values = frame.loc[frame["resolution_class"] == resolution_class, "megapixels"]
    median_mp = values.median()
    name = RESOLUTION_NAMES.get(resolution_class, resolution_class)
    if median_mp < 1:
        return "%s (%.2f MP)" % (name, median_mp)
    return "%s (~%.1f MP)" % (name, median_mp)


def aggregate_cells(frame, expected_timing_samples=5):
    """Aggregate image rows without averaging already-normalized ms/MP values."""
    records = []
    keys = ("corpus", "resolution_class", "quality", "effort")
    for key, group in frame.groupby(list(keys), observed=True, sort=True):
        corpus, resolution_class, quality, effort = key
        total_mp = group["megapixels"].sum()
        total_runtime = group["complete_encode_median_ms"].sum()
        record = {
            "corpus": corpus,
            "resolution_class": resolution_class,
            "quality": int(quality),
            "effort": int(effort),
            "image_count": group["image_id"].nunique(),
            "mean_megapixels": group["megapixels"].mean(),
            "total_megapixels": total_mp,
            "mean_image_runtime_ms": total_runtime / len(group),
            "runtime_ms_per_mp": total_runtime / total_mp,
            "bits_per_pixel": (
                (group["bits_per_pixel"] * group["megapixels"]).sum() / total_mp
            ),
            "timing_sample_min": int(group["timing_sample_count"].min()),
            "timing_sample_max": int(group["timing_sample_count"].max()),
            "timing_complete": bool(
                group["timing_sample_count"].ge(expected_timing_samples).all()
            ),
            "relative_p10_p90_percent": float(
                (
                    (group["complete_encode_p90_ms"] - group["complete_encode_p10_ms"])
                    / group["complete_encode_median_ms"]
                    * 100
                ).median()
            ),
        }
        stage_columns = [
            "profiled_complete_wall_ms",
            *(column for column, _, _ in STAGE_COMPONENTS),
        ]
        stage_complete = bool(group[stage_columns].notna().all().all())
        record["stage_complete"] = stage_complete
        if stage_complete:
            record["profiled_complete_wall_ms"] = group[
                "profiled_complete_wall_ms"
            ].sum()
            for column, _, _ in STAGE_COMPONENTS:
                record[column] = group[column].sum()
        else:
            record["profiled_complete_wall_ms"] = math.nan
            for column, _, _ in STAGE_COMPONENTS:
                record[column] = math.nan
        records.append(record)
    cells = pd.DataFrame.from_records(records)
    cells.attrs["source_path"] = frame.attrs.get("source_path")
    return cells


# %% [markdown]
# ## Shared presentation helpers


# %%
def configure_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    matplotlib.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def subplot_grid(count, columns=3, width=5.1, height=3.7):
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(columns * width, rows * height),
        constrained_layout=True,
        squeeze=False,
    )
    flattened = list(axes.flat)
    for axis in flattened[count:]:
        axis.set_visible(False)
    return figure, flattened[:count]


def save_figure(figure, output_dir, name, formats=("png", "svg")):
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        extension = extension.lstrip(".").lower()
        if extension not in ("png", "svg", "pdf"):
            raise ValueError("unsupported output format: %s" % extension)
        path = output_dir / (name + "." + extension)
        figure.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def incomplete_marker_legend():
    return mlines.Line2D(
        [],
        [],
        color="#555555",
        marker="o",
        markerfacecolor="none",
        linestyle="none",
        label="Partial timing cell",
    )


# %% [markdown]
# ## 1. Normalized runtime versus effort
#
# Each resolution class is aggregated by summing image runtimes and pixels,
# then dividing the sums. This avoids giving a small image the same pixel
# weight as a large image.


# %%
def plot_runtime_vs_effort(frame, cells):
    resolutions = resolution_order(frame)
    figure, axes = subplot_grid(len(resolutions))
    qualities = sorted(cells["quality"].unique())
    for axis, resolution in zip(axes, resolutions):
        subset = cells[cells["resolution_class"] == resolution]
        for quality in qualities:
            line = subset[subset["quality"] == quality].sort_values("effort")
            color = QUALITY_COLORS.get(quality, "#555555")
            axis.plot(
                line["effort"],
                line["runtime_ms_per_mp"],
                color=color,
                linewidth=1.5,
                label="Q%d" % quality,
            )
            complete = line[line["timing_complete"]]
            partial = line[~line["timing_complete"]]
            axis.scatter(
                complete["effort"],
                complete["runtime_ms_per_mp"],
                color=[color],
                s=25,
                zorder=3,
            )
            axis.scatter(
                partial["effort"],
                partial["runtime_ms_per_mp"],
                facecolors="none",
                edgecolors=[color],
                s=34,
                linewidths=1.4,
                zorder=3,
            )
        axis.set_title(resolution_label(frame, resolution))
        axis.set_xlabel("Effort")
        axis.set_ylabel("Complete encode (ms/MP)")
        axis.set_xticks(sorted(cells["effort"].unique()))
        axis.set_yscale("log")
    handles = [
        mlines.Line2D([], [], color=QUALITY_COLORS[q], marker="o", label="Q%d" % q)
        for q in qualities
    ]
    handles.append(incomplete_marker_legend())
    figure.legend(handles=handles, loc="outside lower center", ncols=8)
    figure.suptitle("libjxl complete-encode runtime versus effort")
    return figure


# %% [markdown]
# ## 2. Quality–effort runtime surfaces


# %%
def plot_quality_effort_heatmaps(frame, cells):
    resolutions = resolution_order(frame)
    qualities = sorted(cells["quality"].unique())
    efforts = sorted(cells["effort"].unique())
    positive = cells.loc[cells["runtime_ms_per_mp"] > 0, "runtime_ms_per_mp"]
    normalization = mcolors.LogNorm(vmin=positive.min(), vmax=positive.max())
    figure, axes = subplot_grid(len(resolutions), width=5.2, height=3.8)
    image = None
    for axis, resolution in zip(axes, resolutions):
        subset = cells[cells["resolution_class"] == resolution]
        values = subset.pivot(
            index="quality", columns="effort", values="runtime_ms_per_mp"
        )
        values = values.reindex(index=qualities, columns=efforts)
        image = axis.imshow(
            values.to_numpy(),
            aspect="auto",
            origin="lower",
            cmap="magma",
            norm=normalization,
        )
        complete = subset.pivot(
            index="quality", columns="effort", values="timing_complete"
        )
        complete = complete.reindex(index=qualities, columns=efforts).fillna(False)
        for row, quality in enumerate(qualities):
            for column, effort in enumerate(efforts):
                if not bool(complete.loc[quality, effort]):
                    axis.scatter(
                        column,
                        row,
                        marker="s",
                        s=52,
                        facecolors="none",
                        edgecolors="white",
                        linewidths=1.2,
                    )
        axis.set_title(resolution_label(frame, resolution))
        axis.set_xlabel("Effort")
        axis.set_ylabel("Quality")
        axis.set_xticks(range(len(efforts)), efforts)
        axis.set_yticks(range(len(qualities)), qualities)
    if image is not None:
        colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
        colorbar.set_label("Complete encode (ms/MP, log scale)")
    figure.suptitle("Quality–effort runtime surface; white squares are partial")
    return figure


# %% [markdown]
# ## 3. Resolution scaling
#
# The default panels use efforts 1, 5, and the highest effort with complete
# timing coverage. Values are the average per-image runtime within each
# resolution class, shown on log–log axes.


# %%
def select_scaling_efforts(cells):
    completeness = cells.groupby("effort", observed=True)["timing_complete"].all()
    complete = sorted(int(effort) for effort, value in completeness.items() if value)
    if not complete:
        return sorted(int(value) for value in cells["effort"].unique())[:3]
    candidates = (
        complete[0],
        5 if 5 in complete else complete[len(complete) // 2],
        complete[-1],
    )
    return list(dict.fromkeys(candidates))


def plot_resolution_scaling(frame, cells):
    efforts = select_scaling_efforts(cells)
    figure, axes = subplot_grid(
        len(efforts), columns=len(efforts), width=5.1, height=4.2
    )
    for axis, effort in zip(axes, efforts):
        subset = cells[(cells["effort"] == effort) & cells["timing_complete"]]
        for quality in sorted(subset["quality"].unique()):
            line = subset[subset["quality"] == quality].sort_values("mean_megapixels")
            axis.plot(
                line["mean_megapixels"],
                line["mean_image_runtime_ms"],
                color=QUALITY_COLORS.get(quality, "#555555"),
                marker="o",
                linewidth=1.4,
                markersize=4,
                label="Q%d" % quality,
            )
        axis.set_title("Effort %d" % effort)
        axis.set_xlabel("Mean image resolution (MP)")
        axis.set_ylabel("Mean complete encode (ms/image)")
        axis.set_xscale("log")
        axis.set_yscale("log")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=7)
    figure.suptitle("libjxl runtime scaling with resolution")
    return figure


# %% [markdown]
# ## 4. Rate–runtime tradeoff
#
# Each line holds effort constant and walks through quality. Lower and farther
# left is better, subject to decoded-quality requirements.


# %%
def plot_rate_runtime_tradeoff(frame, cells):
    resolutions = resolution_order(frame)
    figure, axes = subplot_grid(len(resolutions), width=5.1, height=3.9)
    efforts = sorted(cells["effort"].unique())
    effort_colors = {
        effort: plt.get_cmap("plasma")(position)
        for effort, position in zip(efforts, np.linspace(0.05, 0.95, len(efforts)))
    }
    for axis, resolution in zip(axes, resolutions):
        subset = cells[cells["resolution_class"] == resolution]
        for effort in efforts:
            line = subset[subset["effort"] == effort].sort_values("quality")
            color = effort_colors[effort]
            axis.plot(
                line["bits_per_pixel"],
                line["runtime_ms_per_mp"],
                color=color,
                linewidth=1.25,
                alpha=0.85,
                label="E%d" % effort,
            )
            complete = line[line["timing_complete"]]
            partial = line[~line["timing_complete"]]
            axis.scatter(
                complete["bits_per_pixel"],
                complete["runtime_ms_per_mp"],
                color=[color],
                s=20,
                zorder=3,
            )
            axis.scatter(
                partial["bits_per_pixel"],
                partial["runtime_ms_per_mp"],
                facecolors="none",
                edgecolors=[color],
                s=28,
                linewidths=1.2,
                zorder=3,
            )
        axis.set_title(resolution_label(frame, resolution))
        axis.set_xlabel("Bits per pixel")
        axis.set_ylabel("Complete encode (ms/MP)")
        axis.set_xscale("log")
        axis.set_yscale("log")
    handles = [
        mlines.Line2D([], [], color=effort_colors[e], marker="o", label="E%d" % e)
        for e in efforts
    ]
    handles.append(incomplete_marker_legend())
    figure.legend(handles=handles, loc="outside lower center", ncols=6)
    figure.suptitle("Encoded rate versus complete-encode runtime")
    return figure


# %% [markdown]
# ## 5. Instrumented wall-time composition
#
# The mutually exclusive serializer phases are stacked with the frontend
# residual. Aggregate worker substages are intentionally excluded because they
# overlap and are not latency.


# %%
def plot_stage_breakdown(frame, cells):
    if "wall_exclusive_frame_setup_other_ms" in frame.columns:
        return plot_expanded_wall_breakdown(frame)
    resolutions = resolution_order(frame)
    figure, axes = subplot_grid(len(resolutions), width=5.1, height=3.9)
    for axis, resolution in zip(axes, resolutions):
        subset = cells[
            (cells["resolution_class"] == resolution) & cells["stage_complete"]
        ]
        grouped = subset.groupby("effort", observed=True, sort=True)
        efforts = np.array(sorted(int(value) for value in subset["effort"].unique()))
        bottoms = np.zeros(len(efforts))
        totals = grouped["profiled_complete_wall_ms"].sum().reindex(efforts).to_numpy()
        for column, label, color in STAGE_COMPONENTS:
            values = grouped[column].sum().reindex(efforts).to_numpy()
            percentages = values / totals * 100
            axis.bar(
                efforts,
                percentages,
                bottom=bottoms,
                color=color,
                width=0.78,
                label=label,
            )
            bottoms += percentages
        axis.set_title(resolution_label(frame, resolution))
        axis.set_xlabel("Effort")
        axis.set_ylabel("Profiled complete wall time (%)")
        axis.set_xticks(efforts)
        axis.set_ylim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3)
    figure.suptitle(
        "libjxl instrumented wall-time composition (pooled across images and qualities)"
    )
    return figure


# %% [markdown]
# ### Expanded frontend wall profiles (version 2)
#
# Only fully populated resolution/quality/effort cells enter these plots.
# The exclusive stages, including the outside-frame API remainder, sum to
# complete profiled encode wall time. DC preparation includes trial passes;
# the inclusive quantizer figure below includes those within refinement.


# %%
def expanded_wall_rows(frame):
    columns = [
        c
        for c in frame.columns
        if c.startswith("wall_exclusive_") and c.endswith("_ms")
    ]
    if not columns:
        return frame.iloc[:0], columns
    valid = frame[columns].notna().all(axis=1)
    # These figures pool all qualities. Require the whole resolution/effort
    # group so a partially collected effort cannot silently change that mix.
    keys = ["resolution_class", "effort"]
    complete = valid.groupby([frame[k] for k in keys]).transform("all")
    rows = frame[complete].copy()
    if len(rows) and not np.allclose(
        rows[columns].sum(axis=1), rows["profiled_complete_wall_ms"], atol=1e-6
    ):
        raise ValueError(
            "Expanded exclusive wall columns do not sum to complete encode"
        )
    return rows, columns


def plot_expanded_wall_breakdown(frame):
    rows, columns = expanded_wall_rows(frame)
    resolutions = resolution_order(frame)
    figure, axes = subplot_grid(len(resolutions), width=5.4, height=4.1)
    groups = {
        "Input unpack / color": ("input_unpack", "color_conversion"),
        "Downsampling": ("downsampling",),
        "Initial AQ": ("initial_aq",),
        "Inverse Gaborish": ("inverse_gaborish",),
        "Feature search": ("feature_search",),
        "AC/CfL tile heuristics": (
            "heuristics_setup",
            "ac_cfl_tiles",
            "heuristics_finalize",
        ),
        "Perceptual refinement": tuple(
            c[len("wall_exclusive_") : -3]
            for c in columns
            if c.startswith("wall_exclusive_refinement_")
        )
        + ("quantizer_refinement",),
        "Final coefficients": ("final_coefficients",),
        "AR filter selection": ("ar_selection",),
        "DC / metadata / ordering": (
            "dc_preparation",
            "ac_metadata",
            "block_context",
            "coefficient_order",
            "modular_tree",
        ),
        "Tokenization": ("tokenization",),
        "Entropy model": ("entropy_model",),
        "Model/token emission": ("emission",),
        "Assembly": ("assembly",),
        "Frame / API remainder": ("frame_setup_other", "encode_api_other"),
    }
    used = {
        "wall_exclusive_" + name + "_ms" for names in groups.values() for name in names
    }
    if used != set(columns):
        raise ValueError("Expanded wall plot groups do not cover the schema exactly")
    colors = plt.get_cmap("tab20").colors
    for axis, resolution in zip(axes, resolutions):
        subset = rows[rows["resolution_class"] == resolution]
        grouped = subset.groupby("effort", observed=True)[
            columns + ["profiled_complete_wall_ms"]
        ].sum()
        bottom = np.zeros(len(grouped))
        for (label, names), color in zip(groups.items(), colors):
            values = grouped[["wall_exclusive_" + name + "_ms" for name in names]].sum(
                axis=1
            )
            share = 100 * values / grouped["profiled_complete_wall_ms"]
            axis.bar(grouped.index, share, bottom=bottom, color=color, label=label)
            bottom += share.to_numpy()
        axis.set_title(resolution_label(frame, resolution))
        axis.set_xlabel("Effort")
        axis.set_ylabel("Complete profiled wall time (%)")
        axis.set_xticks(grouped.index)
        axis.set_ylim(0, 100)
        if grouped.empty:
            axis.text(
                0.5,
                0.5,
                "No complete wall-v2 efforts yet",
                ha="center",
                transform=axis.transAxes,
            )
    figure.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=4
    )
    figure.suptitle(
        "Expanded wall-time composition (pooled across images and qualities)"
    )
    return figure


def plot_refinement_wall(frame):
    rows, _ = expanded_wall_rows(frame)
    figure, axes = subplot_grid(len(resolution_order(frame)), width=5.1, height=3.9)
    fields = {
        "Reference preparation": "refinement_reference",
        "Candidate roundtrip": "refinement_roundtrip",
        "Butteraugli comparison": "refinement_compare",
        "Quant-field update": "refinement_update",
    }
    for axis, resolution in zip(axes, resolution_order(frame)):
        subset = rows[rows["resolution_class"] == resolution]
        grouped = subset.groupby("effort", observed=True).sum(numeric_only=True)
        bottom = np.zeros(len(grouped))
        for label, stage in fields.items():
            values = grouped["wall_inclusive_" + stage + "_ms"] / grouped["megapixels"]
            axis.bar(grouped.index, values, bottom=bottom, label=label)
            bottom += values.to_numpy()
        total = (
            grouped["wall_inclusive_quantizer_refinement_ms"] / grouped["megapixels"]
        )
        if np.any(total.to_numpy() - bottom < -1e-6):
            raise ValueError("Refinement children exceed their parent")
        axis.bar(
            grouped.index,
            np.maximum(total.to_numpy() - bottom, 0),
            bottom=bottom,
            label="Other refinement work",
        )
        axis.set_title(resolution_label(frame, resolution))
        axis.set_xlabel("Effort")
        axis.set_ylabel("Quantizer refinement wall time (ms/MP)")
        axis.set_xticks(grouped.index)
    figure.legend(
        *axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3
    )
    figure.suptitle(
        "Inclusive refinement wall time, partitioned into sequential children"
    )
    return figure


# %% [markdown]
# ## 6. Timing variability
#
# This shows the median per-image `(p90 - p10) / median` spread and its
# interquartile range. Only tuples with all expected independent samples are
# included.


# %%
def plot_timing_variability(frame, expected_timing_samples=5):
    complete = frame[frame["timing_sample_count"] >= expected_timing_samples].copy()
    complete["relative_spread_percent"] = (
        (complete["complete_encode_p90_ms"] - complete["complete_encode_p10_ms"])
        / complete["complete_encode_median_ms"]
        * 100
    )
    complete = complete[complete["relative_spread_percent"] > 0]
    figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for color, resolution in zip(colors, resolution_order(frame)):
        subset = complete[complete["resolution_class"] == resolution]
        grouped = subset.groupby("effort", observed=True)["relative_spread_percent"]
        summary = grouped.agg(
            median="median",
            lower=lambda values: values.quantile(0.25),
            upper=lambda values: values.quantile(0.75),
        ).sort_index()
        x = summary.index.to_numpy(dtype=float)
        axis.plot(
            x,
            summary["median"],
            color=color,
            marker="o",
            label=RESOLUTION_NAMES.get(resolution, resolution),
        )
        axis.fill_between(
            x,
            summary["lower"].to_numpy(dtype=float),
            summary["upper"].to_numpy(dtype=float),
            color=color,
            alpha=0.13,
        )
    axis.set_xlabel("Effort")
    axis.set_ylabel("Per-image (p90 − p10) / median (%, log scale)")
    axis.set_xticks(sorted(complete["effort"].unique()))
    axis.set_yscale("log")
    axis.legend(ncols=2)
    axis.set_title("Independent-process timing variability (median and IQR)")
    return figure


# %% [markdown]
# ## Generate the complete figure set


# %%
def generate_characterization(
    input_csv,
    output_dir,
    expected_timing_samples=5,
    formats=("png", "svg"),
    show=False,
):
    configure_style()
    frame = load_image_tuples(input_csv)
    cells = aggregate_cells(frame, expected_timing_samples)
    figures = {
        "runtime-vs-effort": plot_runtime_vs_effort(frame, cells),
        "quality-effort-heatmaps": plot_quality_effort_heatmaps(frame, cells),
        "resolution-scaling": plot_resolution_scaling(frame, cells),
        "rate-runtime-tradeoff": plot_rate_runtime_tradeoff(frame, cells),
        "stage-wall-breakdown": plot_stage_breakdown(frame, cells),
        "timing-variability": plot_timing_variability(frame, expected_timing_samples),
    }
    if "wall_inclusive_quantizer_refinement_ms" in frame.columns:
        figures["quantizer-refinement-wall"] = plot_refinement_wall(frame)
    written = []
    for name, figure in figures.items():
        written.extend(save_figure(figure, output_dir, name, formats))
    complete_timing = int(
        frame["timing_sample_count"].ge(expected_timing_samples).sum()
    )
    complete_stage = int(frame["profiled_complete_wall_ms"].notna().sum())
    print("input: %s" % frame.attrs["source_path"])
    print("tuple rows: %d" % len(frame))
    print(
        "complete timing tuples: %d/%d; stage tuples: %d/%d"
        % (complete_timing, len(frame), complete_stage, len(frame))
    )
    print(
        "wrote %d files under %s" % (len(written), pathlib.Path(output_dir).resolve())
    )
    if show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)
    return frame, cells, figures, written


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate static characterization figures from image-tuples.csv."
    )
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path)
    parser.add_argument("--expected-timing-samples", type=int, default=5)
    parser.add_argument(
        "--formats",
        default="png,svg",
        help="comma-separated output formats: png, svg, and/or pdf",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--quality-run", type=pathlib.Path,
                        help="optional saved quality study; never starts collection")
    parser.add_argument("--butteraugli-run", type=pathlib.Path,
                        help="optional saved Butteraugli preview; compare with --quality-run")
    parser.add_argument("--gjxl-run", type=pathlib.Path,
                        help="optional saved GJXL study; compare with --quality-run")
    args = parser.parse_args(argv)
    if args.gjxl_run and not args.quality_run:
        parser.error("--gjxl-run requires --quality-run")
    if args.expected_timing_samples < 1:
        parser.error("--expected-timing-samples must be positive")
    args.formats = tuple(
        value.strip() for value in args.formats.split(",") if value.strip()
    )
    if not args.formats:
        parser.error("--formats must contain at least one output format")
    if args.output_dir is None:
        args.output_dir = args.input.resolve().parent / (args.input.stem + ".plots")
    return args


def main(argv=None):
    args = parse_args(argv)
    generate_characterization(
        args.input,
        args.output_dir,
        args.expected_timing_samples,
        args.formats,
        args.show,
    )
    if args.quality_run:
        generate_quality_figures(args.quality_run, args.output_dir, args.formats, args.show)
    if args.butteraugli_run:
        generate_quality_figures(args.butteraugli_run, args.output_dir, args.formats,
                                 args.show, compare_run=args.quality_run, prefix="butteraugli-")
    if args.gjxl_run:
        generate_encoder_comparison(args.quality_run, args.gjxl_run, args.output_dir,
                                    args.formats, args.show)
    return 0


# %% [markdown]
# ## Quality-study drawing functions
#
# Pareto, rate-quality, and metric-comparison plots live here alongside the
# runtime and stage plots. The quality script supplies saved-data loading and
# aggregation only; rendering never starts collection.

# %%
# Presentation labels are independent of the collector's metric implementation.
QUALITY_METRICS = {
    "fast-ssim2": {"label": "fast-ssim2", "higher_is_better": True},
    "butteraugli": {"label": "Butteraugli distance", "higher_is_better": False},
}


def quality_metric_name(config):
    name = config.get("metric", "fast-ssim2")
    if name not in QUALITY_METRICS:
        raise ValueError(f"Unknown metric: {name}")
    return name


def frontier(points):
    return [
        p
        for p in points
        if not any(
            q["bits_per_pixel"] <= p["bits_per_pixel"]
            and q["throughput_mp_s"] >= p["throughput_mp_s"]
            and (
                q["bits_per_pixel"] < p["bits_per_pixel"]
                or q["throughput_mp_s"] > p["throughput_mp_s"]
            )
            for q in points
        )
    ]


def plot_pareto(points, mode):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, FuncFormatter

    metrics = {quality_metric_name(r) for r in points}
    if len(metrics) > 1:
        raise ValueError(
            "Plot each metric separately; their scales are not interchangeable"
        )
    label_metric = QUALITY_METRICS[next(iter(metrics), "fast-ssim2")]["label"]
    encoders = sorted({r["encoder"] for r in points})
    panels = sorted({(r["corpus"], r["resolution_class"], r["target"]) for r in points})
    fig, axes = plt.subplots(
        max(1, math.ceil(len(panels) / 3)),
        min(3, max(1, len(panels))),
        figsize=(
            5.4 * min(3, max(1, len(panels))),
            4.4 * max(1, math.ceil(len(panels) / 3)),
        ),
        squeeze=False,
        layout="constrained",
    )
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    for ax, panel in zip(axes.flat, panels):
        entries = [
            r
            for r in points
            if (r["corpus"], r["resolution_class"], r["target"]) == panel
        ]
        ready = [r for r in entries if r["status"] == "ready"]
        if ready:
            if len({tuple(sorted(r["cohort"])) for r in ready}) != 1:
                raise ValueError("Plot points have different cohorts")
            for encoder in sorted({r["encoder"] for r in ready}):
                line = sorted(
                    [r for r in ready if r["encoder"] == encoder],
                    key=lambda r: r["effort"],
                )
                color = "#2065ac" if encoder == "libjxl" else "#d55e00"
                # Missing efforts break the connecting line.
                all_efforts = sorted(
                    r["effort"] for r in entries if r["encoder"] == encoder
                )
                lookup = {r["effort"]: r for r in line}
                ax.plot(
                    [
                        lookup[e]["bits_per_pixel"] if e in lookup else math.nan
                        for e in all_efforts
                    ],
                    [
                        lookup[e]["throughput_mp_s"] if e in lookup else math.nan
                        for e in all_efforts
                    ],
                    color=color,
                    alpha=0.45,
                    lw=1,
                )
                offsets = [
                    (6, 10),
                    (6, -15),
                    (6, 10),
                    (6, -15),
                    (6, 10),
                    (12, -18),
                    (-22, 14),
                    (-26, -14),
                    (-26, 10),
                    (8, -23),
                ]
                for row in line:
                    ax.plot(
                        row["bits_per_pixel"],
                        row["throughput_mp_s"],
                        marker=markers[(row["effort"] - 1) % len(markers)],
                        ms=8,
                        markerfacecolor="white" if mode == "preview" else color,
                        markeredgecolor=color,
                        linestyle="none",
                    )
                    ax.annotate(
                        f"e{row['effort']}",
                        (row["bits_per_pixel"], row["throughput_mp_s"]),
                        xytext=offsets[(row["effort"] - 1) % len(offsets)],
                        textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "lw": 0.5, "color": color},
                        fontsize=9,
                        color=color,
                    )
            edge = sorted(frontier(ready), key=lambda r: r["bits_per_pixel"])
            ax.plot(
                [r["bits_per_pixel"] for r in edge],
                [r["throughput_mp_s"] for r in edge],
                "--",
                color="#333333",
                lw=1.2,
                zorder=0,
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No complete matched-quality points",
                ha="center",
                transform=ax.transAxes,
            )
        missing = [
            (r["encoder"] + " " if len(encoders) > 1 else "")
            + f"e{r['effort']} ({r['ready_count']}/{r['image_count']})"
            for r in entries
            if r["status"] != "ready"
        ]
        if missing:
            ax.text(
                0.98,
                0.98,
                "Missing: " + ", ".join(missing),
                fontsize=8,
                transform=ax.transAxes,
                wrap=True,
                ha="right",
                va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        count = entries[0]["image_count"] if entries else 0
        tolerance = entries[0].get("tolerance", 0.5) if entries else 0.5
        resolution = {
            "kodak_0_4mp": "Kodak (0.39 MP)",
            "clic_1_8_to_3_4mp": "CLIC test (~2.8 MP)",
        }.get(panel[1], panel[1].replace("_", " "))
        quality_label = (
            f"target {panel[2]:g} (estimated)"
            if mode == "preview"
            else f"{panel[2]:g} ±{tolerance:g}"
        )
        ax.set_title(
            f"{resolution}\n{label_metric} {quality_label} · {count} images",
            fontsize=11,
        )
        ax.set(
            xlabel="Compressed size (bits/original pixel)",
            ylabel="Complete encode throughput (MP/s)",
        )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.grid(True, alpha=0.2)
        ax.margins(x=0.18, y=0.22)
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    label = "INTERPOLATED PREVIEW" if mode == "preview" else "MEASURED"
    fig.suptitle(
        f"{label}: matched-quality speed–compression\nUpper-left is better", fontsize=14
    )
    if any(r.get("provisional_timing") for r in points):
        fig.text(
            0.99,
            0.01,
            "Preview includes incomplete source timing repetitions",
            ha="right",
            fontsize=8,
            color="#8c510a",
        )
    fig.legend(
        handles=[
            *[Line2D([], [], color="#2065ac" if name == "libjxl" else "#d55e00",
                     label=("libjxl: effort order" if name == "libjxl" else
                            "gjxl (fully-resident Metal): effort order"))
              for name in (encoders or ["libjxl"])],
            Line2D(
                [], [], color="#333333", linestyle="--", label="Nondominated frontier"
            ),
        ],
        loc="outside lower center",
        ncols=2,
    )
    return fig


# %%
def plot_curves(scores, config):
    import matplotlib.pyplot as plt

    images = config["images"]
    fig, axes = plt.subplots(
        math.ceil(len(images) / 3),
        min(3, len(images)),
        squeeze=False,
        figsize=(5 * min(3, len(images)), 3.8 * math.ceil(len(images) / 3)),
        layout="constrained",
    )
    for ax, image in zip(axes.flat, images):
        for effort in config["efforts"]:
            rows = sorted(
                [
                    r
                    for r in scores
                    if r["image_id"] == image["image_id"] and r["effort"] == effort
                ],
                key=lambda r: r["distance"],
            )
            ax.plot(
                [8 * r["encoded_bytes"] / r["pixels"] for r in rows],
                [r["score"] for r in rows],
                ".-",
                label=f"e{effort}",
                lw=0.9,
            )
        for target in config["targets"]:
            ax.axhline(target, color="gray", ls=":", lw=0.7)
        ax.set(
            title=image["image_id"],
            xlabel="bits/original pixel",
            ylabel=QUALITY_METRICS[quality_metric_name(config)]["label"]
            + (
                " (higher is better)"
                if QUALITY_METRICS[quality_metric_name(config)]["higher_is_better"]
                else " (lower is better)"
            ),
        )
        ax.grid(True, alpha=0.2)
    for ax in list(axes.flat)[len(images) :]:
        ax.set_visible(False)
    fig.legend(
        *axes.flat[0].get_legend_handles_labels(), loc="outside lower center", ncols=10
    )
    fig.suptitle(
        "Measured rate–quality samples (lines are diagnostic, not accepted interpolation)"
    )
    return fig


# %%
def plot_effort_comparison(studies):
    import matplotlib.pyplot as plt

    images = studies[0][1]["images"]
    fig, axes = plt.subplots(
        len(studies),
        len(images),
        squeeze=False,
        figsize=(4.7 * len(images), 3.6 * len(studies)),
        sharex="col",
        layout="constrained",
    )
    for row_index, (_, config, scores) in enumerate(studies):
        info = QUALITY_METRICS[quality_metric_name(config)]
        for ax, image in zip(axes[row_index], images):
            count = 0
            for effort, color, marker in ((4, "#2065ac", "o"), (5, "#d55e00", "v")):
                rows = sorted(
                    (
                        r
                        for r in scores
                        if r["image_id"] == image["image_id"] and r["effort"] == effort
                    ),
                    key=lambda r: r["distance"],
                )
                count += len(rows)
                # Show observations, breaking lines at the internal resampling boundary.
                for resampling in (1, 2):
                    segment = [r for r in rows if r["resampling"] == resampling]
                    ax.plot(
                        [8 * r["encoded_bytes"] / r["pixels"] for r in segment],
                        [r["score"] for r in segment],
                        color=color,
                        marker=marker,
                        linestyle="-" if effort == 4 else "--",
                        ms=4,
                        lw=1.2,
                        label=f"e{effort}" if resampling == 1 else None,
                    )
            direction = (
                "higher is better" if info["higher_is_better"] else "lower is better"
            )
            ax.set(
                title=f"{image['image_id']} · {count} scored outputs",
                xlabel="Compressed size (bits/original pixel)",
                ylabel=f"{info['label']}\n({direction})",
            )
            ax.grid(True, alpha=0.2)
            ax.legend()
    fig.suptitle(
        "e4 versus e5 under two metrics\n"
        "Identical retained encodes; lines are diagnostic, not calibrated matches",
        fontsize=13,
    )
    return fig


# %%
def load_quality_helpers():
    import importlib

    candidates = []
    if "__file__" in globals():
        candidates.append(pathlib.Path(__file__).resolve().parent)
    for parent in (pathlib.Path.cwd(), *pathlib.Path.cwd().parents):
        candidates.extend((parent, parent / "tools/scripts"))
    source = next(
        (folder / "cjxl_quality_characterization.py" for folder in candidates
         if (folder / "cjxl_quality_characterization.py").is_file()),
        None,
    )
    if source is None:
        raise FileNotFoundError(
            "Open Jupyter within the libjxl checkout so the quality data helpers can be found."
        )
    # Jupyter may start in the checkout root rather than tools/scripts.
    sys.path.insert(0, str(source.parent))
    try:
        study = importlib.import_module("cjxl_quality_characterization")
    finally:
        sys.path.pop(0)
    return study


def notebook_figures(run, compare_run=None):
    """Pure read/plot entrypoint: no collectors, binaries, or CSV regeneration."""
    run = pathlib.Path(run)
    if not (run / "metadata.json").is_file():
        print(
            f"No quality-study data at {run}; collection is never started by this notebook."
        )
        return {}
    study = load_quality_helpers()
    config = study.load_config(run)
    figures = {}
    modes = (("measured",) if not config.get("preview_available", True) else
             ("preview",) if config.get("preview_only") else ("preview", "measured"))
    for mode in modes:
        rows = study.per_image_rows(run, config, mode)
        points = study.aggregate_points(rows)
        figures["pareto-" + mode] = plot_pareto(points, mode)
    if config.get("preview_available", True):
        scores = study.read_records(run, "scores", config)
        figures["rate-quality-diagnostics"] = plot_curves(scores, config)
    if compare_run is not None:
        studies = study.comparison_studies(run, compare_run)
        figures["effort4-vs5-metrics"] = plot_effort_comparison(studies)
    return figures


# %%
def generate_quality_figures(run, output_dir, formats=SAVE_FORMATS, show=False,
                             compare_run=None, prefix=""):
    """Draw saved quality studies using the plotting functions above."""
    figures = notebook_figures(run, compare_run=compare_run)
    for name, figure in figures.items():
        save_figure(figure, pathlib.Path(output_dir), prefix + name, formats)
    if show:
        plt.show()
    else:
        for figure in figures.values():
            plt.close(figure)
    return figures


# %%
def generate_encoder_comparison(libjxl_run, gjxl_run, output_dir,
                                formats=SAVE_FORMATS, show=False):
    """Compare saved measured points on GJXL's explicit manifest cohort."""
    if not (pathlib.Path(gjxl_run) / "metadata.json").is_file():
        print("No saved GJXL study at %s; skipping encoder comparison." % gjxl_run)
        return {}
    study = load_quality_helpers()
    points = study.encoder_comparison_points(libjxl_run, gjxl_run)
    figure = plot_pareto(points, "measured")
    warmup_report = pathlib.Path(gjxl_run) / "warmup-check-summary.json"
    if warmup_report.is_file():
        import json
        if any(row["needs_review"] for row in json.loads(warmup_report.read_text())["results"]):
            figure.suptitle(
                "MEASURED PILOT: libjxl versus fully-resident Metal GJXL\n"
                "Matched perceptual quality; upper-left is better\n"
                "Warmup sensitivity flagged: consult warmup-check-summary.json",
                fontsize=13,
            )
    figure.text(0.99, 0.01,
                "Warm complete calls; libjxl: 8 workers; GJXL: CPU participant cap 8 + Metal",
                ha="right", fontsize=8)
    name = "pareto-measured-libjxl-vs-gjxl"
    save_figure(figure, pathlib.Path(output_dir), name, formats)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return {name: figure}


# %%
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    raise SystemExit(main())


# %% [markdown]
# ## Generate figures in Jupyter
#
# “Run All” reaches this cell after defining the analysis functions. Set
# `INPUT_CSV` and `OUTPUT_DIR` in the configuration cell before running it.
# The figures are both displayed inline and saved in each requested format.

# %%
if __name__ == "__main__" and "ipykernel" in sys.modules:
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            "Set INPUT_CSV in the notebook configuration cell; file not found: %s"
            % INPUT_CSV.resolve()
        )
    frame, cells, figures, written = generate_characterization(
        INPUT_CSV,
        OUTPUT_DIR,
        EXPECTED_TIMING_SAMPLES,
        SAVE_FORMATS,
        show=True,
    )


# %% [markdown]
# ## 7. Matched-quality speed–compression
#
# QUALITY_RUN points to a separate, resumable quality study. These panels
# target fast-ssim2 scores 60, 70, and 85 with a measured tolerance of ±0.5.
# Open markers are interpolated previews, not actual calibrated encodes.
# Filled markers require a verified score and five independent timing samples.
# Each point uses the same image cohort; unavailable efforts are identified.
# Dashed lines identify the nondominated frontier, not the effort-order curve.
# A separate per-image rate-quality diagnostic shows all scored baseline points,
# including intervals rejected by preview interpolation. Its lines are not
# calibrated matches. All figures are saved under OUTPUT_DIR.
#
# This cell only reads saved results and generates plots. It never decodes,
# scores, calibrates, or encodes. Missing measured data is shown explicitly.
# See doc/runtime-quality.md for opt-in collection and resume commands.
#
# The default now reads the paused full 65-image study (2026-09-09).
# Efforts 1–8 have completed timing rounds, with some unresolved quality targets.
# Effort 9 calibration is complete, but its timing rounds are only partial;
# effort 10 has not started. Incomplete efforts are not plotted as measured
# points, and unresolved images are never silently dropped from a cohort.

# %%
if __name__ == "__main__" and "ipykernel" in sys.modules:
    quality_figures = generate_quality_figures(QUALITY_RUN, OUTPUT_DIR, SAVE_FORMATS, show=True)


# %% [markdown]
# ## 8. Butteraugli preview and metric sensitivity
#
# This separate study rescores the same retained encodes with libjxl Butteraugli
# at 80 nits, explicitly interpreting our PFM inputs as linear sRGB. The primary
# score is conventional Butteraugli distance (lower is better), not its 3-norm
# auxiliary output and not the encoder's requested distance.
#
# All Butteraugli Pareto points are interpolated previews. The targets are on a
# different scale from fast-ssim2; panels across metrics are not equivalent
# perceptual-quality levels. Per-image e4/e5 curves compare the same codestreams
# under both metrics. Lines are diagnostic and break at the resampling boundary.
# A ranking reversal suggests metric sensitivity, not a universal quality winner.
# Separate per-image rate-quality diagnostics show the saved Butteraugli scores.
#
# Only saved results are read. No calibration, scoring or encoding is launched.
# The Butteraugli section remains a three-image pilot. Its metric comparison
# uses BUTTERAUGLI_COMPARE_RUN, not the full-corpus SSIMU2 study above.

# %%
if __name__ == "__main__" and "ipykernel" in sys.modules:
    butteraugli_figures = generate_quality_figures(
        BUTTERAUGLI_RUN, OUTPUT_DIR, SAVE_FORMATS, show=True,
        compare_run=(BUTTERAUGLI_COMPARE_RUN
                     if (BUTTERAUGLI_COMPARE_RUN / "metadata.json").is_file() else None),
        prefix="butteraugli-",
    )


# %% [markdown]
# ## 9. Measured libjxl versus GJXL
#
# GJXL_RUN selects a separate forced fully-resident Metal study. The comparison
# uses exactly its manifest image cohort and effort selections in both encoders;
# it never intersects away incomplete images. Quality targets and the external
# scorer/decoder must match, but requested distance and effort semantics need not.
# GJXL points require five complete timing samples at an accepted score.
# Its preview is intentionally unavailable: no original GJXL quality sweep is
# required. Warm public-call timing includes CPU/GPU work and excludes startup,
# file I/O and external scoring. The CPU thread settings have different semantics.
# Only saved data is read; this cell never starts collection.

# %%
if __name__ == "__main__" and "ipykernel" in sys.modules:
    encoder_comparison_figures = generate_encoder_comparison(
        QUALITY_RUN, GJXL_RUN, OUTPUT_DIR, SAVE_FORMATS, show=True)
