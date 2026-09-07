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
# The default input below temporarily points at the current partial snapshot.

# %%
DEFAULT_INPUT_CSV = pathlib.Path(
    "/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/"
    "run-e8ff0976/summary/image-tuples.partial.csv"
)
INPUT_CSV = pathlib.Path(
    os.environ.get("CJXL_IMAGE_TUPLES_CSV", DEFAULT_INPUT_CSV)
).expanduser()
OUTPUT_DIR = pathlib.Path(
    os.environ.get("CJXL_CHARACTERIZATION_PLOT_DIR", "characterization-plots")
).expanduser()
EXPECTED_TIMING_SAMPLES = 5
SAVE_FORMATS = ("png", "svg")


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
    args = parser.parse_args(argv)
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
    return 0


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
