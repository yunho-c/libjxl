#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.
#
# /// script
# requires-python = ">=3.9"
# dependencies = ["matplotlib>=3.8,<4"]
# ///

"""Render a publication-ready Matplotlib figure from cjxl Samply analysis.

Example:

  tools/scripts/cjxl_samply_profile.py \
      'build-samply/profiles/kodak-??-d1-e7-t1.json.gz' \
      --format json --output build-samply/profiles/kodak-t1-analysis.json

  uv run tools/scripts/cjxl_samply_plot.py \
      --input build-samply/profiles/kodak-t1-analysis.json \
      --output build-samply/profiles/kodak-stage-breakdown.svg

SVG, PDF, and PNG outputs are supported. The default scope shows encoder work
only, excluding process startup, input decoding, and input conversion. The
figure reports sampled thread CPU, not instrumented wall-clock stage time.
"""

import argparse
import json
import math
import pathlib
import sys


ANALYSIS_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_FILENAME = "stage-breakdown.svg"
STARTUP_OPERATION = "Process startup and static initializers"
PNG_OPERATION = "PNG input decode"
SRGB_OPERATION = "sRGB to XYB color transform"
PACKED_INPUT_OPERATION = "Packed input to float planes"
COMMAND_OVERHEAD_OPERATIONS = frozenset((STARTUP_OPERATION, PNG_OPERATION))
INPUT_CONVERSION_OPERATIONS = frozenset((SRGB_OPERATION, PACKED_INPUT_OPERATION))
ENCODER_EXCLUSIONS = COMMAND_OVERHEAD_OPERATIONS | INPUT_CONVERSION_OPERATIONS
OUTPUT_SUFFIXES = frozenset((".pdf", ".png", ".svg"))

STAGE_PRESENTATION = {
    "AC strategy search and candidate DCT/IDCT": (
        "AC strategy + candidate transforms",
        "#E64B35",
    ),
    "AR heuristics and roundtrip reconstruction/filtering": (
        "AR heuristics + reconstruction",
        "#287A78",
    ),
    "Chroma-from-luma heuristics": ("Chroma-from-luma", "#D19A28"),
    "Entropy modeling, tokenization, and bit writing": (
        "Entropy + bit writing",
        "#4B70AD",
    ),
    "Final coefficient DCT and quantization": (
        "Final DCT + quantization",
        "#8B6144",
    ),
    "Modular/DC side data": ("Modular / DC side data", "#6B795F"),
    "Adaptive quantization map": ("Adaptive quantization", "#8171A1"),
    "Patch dictionary search": ("Patch search", "#AD607C"),
    "Other CLI, encoder, and runtime": ("Other encoder / runtime", "#666664"),
    "Other lossy heuristics": ("Other lossy heuristics", "#8A725F"),
    SRGB_OPERATION: ("sRGB to XYB", "#438DA4"),
    PACKED_INPUT_OPERATION: ("Packed input to float", "#8D999C"),
    STARTUP_OPERATION: ("Process startup", "#BCB6AA"),
    PNG_OPERATION: ("PNG input decode", "#A29682"),
    "Unresolved or missing stack": ("Unresolved stack", "#3E3E3C"),
}

FALLBACK_COLORS = (
    "#A45F45",
    "#527D82",
    "#97873F",
    "#736B89",
    "#9E6B79",
    "#627258",
)


class PlotError(Exception):
    """An expected analysis input or plotting error."""


def _number(value, label, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotError("%s must be numeric" % label)
    if not math.isfinite(value) or value < 0:
        raise PlotError("%s must be finite and nonnegative" % label)
    if integer and int(value) != value:
        raise PlotError("%s must be an integer" % label)
    return int(value) if integer else float(value)


def load_analysis(input_path):
    path = pathlib.Path(input_path).expanduser().resolve()
    if not path.is_file():
        raise PlotError("Analysis JSON does not exist: %s" % path)
    try:
        with path.open("r", encoding="utf-8") as source:
            value = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise PlotError("Failed to read %s: %s" % (path, error)) from error
    if not isinstance(value, dict):
        raise PlotError("Analysis JSON root must be an object")
    if value.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise PlotError(
            "Analysis schema must be %d; regenerate it with "
            "cjxl_samply_profile.py" % ANALYSIS_SCHEMA_VERSION
        )
    return value


def build_plot_data(analysis):
    try:
        summary = analysis["summary"]
        captures = analysis["captures"]
        aggregate_rows = analysis["operations"]
        delta_attribution = analysis["delta_attribution"]
    except KeyError as error:
        raise PlotError("Analysis JSON is missing %s" % error) from error

    if not isinstance(captures, list) or not captures:
        raise PlotError("Analysis must contain at least one capture")
    if not isinstance(aggregate_rows, list) or not aggregate_rows:
        raise PlotError("Analysis must contain operation totals")

    capture_values = []
    for index, capture in enumerate(captures):
        label = "captures[%d]" % index
        if not isinstance(capture, dict):
            raise PlotError("%s must be an object" % label)
        try:
            path = str(capture["path"])
            sample_count = _number(
                capture["sample_count"], label + ".sample_count", integer=True
            )
            cpu_delta_us = _number(capture["cpu_delta_us"], label + ".cpu_delta_us")
        except KeyError as error:
            raise PlotError("%s is missing %s" % (label, error)) from error
        capture_values.append(
            {
                "path": path,
                "sample_count": sample_count,
                "cpu_ms": cpu_delta_us / 1000.0,
            }
        )

    aggregate = {}
    for index, row in enumerate(aggregate_rows):
        label = "operations[%d]" % index
        if not isinstance(row, dict):
            raise PlotError("%s must be an object" % label)
        try:
            operation = row["operation"]
            cpu_delta = _number(row["cpu_delta_us"], label + ".cpu_delta_us")
        except KeyError as error:
            raise PlotError("%s is missing %s" % (label, error)) from error
        if not isinstance(operation, str) or not operation:
            raise PlotError("%s.operation must be a nonempty string" % label)
        if operation in aggregate:
            raise PlotError("Duplicate aggregate operation: %s" % operation)
        aggregate[operation] = cpu_delta

    total_cpu_us = _number(summary.get("cpu_delta_us"), "summary.cpu_delta_us")
    capture_total_us = sum(capture["cpu_ms"] for capture in capture_values) * 1000
    if not math.isclose(total_cpu_us, capture_total_us, abs_tol=0.5):
        raise PlotError("Summary CPU does not match the capture total")
    if not math.isclose(total_cpu_us, sum(aggregate.values()), abs_tol=0.5):
        raise PlotError("Operation totals do not match summary CPU")

    capture_count = _number(
        summary.get("capture_count"), "summary.capture_count", integer=True
    )
    if capture_count != len(capture_values):
        raise PlotError("summary.capture_count does not match captures")

    stages = []
    for fallback_index, (operation, cpu_delta_us) in enumerate(
        sorted(aggregate.items(), key=lambda item: (-item[1], item[0]))
    ):
        short_name, color = STAGE_PRESENTATION.get(
            operation,
            (operation, FALLBACK_COLORS[fallback_index % len(FALLBACK_COLORS)]),
        )
        if operation in INPUT_CONVERSION_OPERATIONS:
            stage_scope = "input"
        elif operation in COMMAND_OVERHEAD_OPERATIONS:
            stage_scope = "overhead"
        else:
            stage_scope = "encoder"
        stages.append(
            {
                "name": operation,
                "short_name": short_name,
                "color": color,
                "scope": stage_scope,
                "aggregate_ms": cpu_delta_us / 1000.0,
                "mean_ms": cpu_delta_us / 1000.0 / capture_count,
            }
        )

    encoder_cpu_us = sum(
        value
        for operation, value in aggregate.items()
        if operation not in ENCODER_EXCLUSIONS
    )
    sample_count = _number(
        summary.get("sample_count"), "summary.sample_count", integer=True
    )
    resolution = _number(
        summary.get("resolved_leaf_cpu_percent"),
        "summary.resolved_leaf_cpu_percent",
    )
    return {
        "delta_attribution": str(delta_attribution),
        "summary": {
            "capture_count": capture_count,
            "sample_count": sample_count,
            "total_cpu_ms": total_cpu_us / 1000.0,
            "encoder_cpu_ms": encoder_cpu_us / 1000.0,
            "mean_cpu_ms": total_cpu_us / 1000.0 / capture_count,
            "mean_encoder_cpu_ms": encoder_cpu_us / 1000.0 / capture_count,
            "resolved_leaf_percent": resolution,
        },
        "captures": capture_values,
        "stages": stages,
    }


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as error:
        raise PlotError(
            "Matplotlib is required. Run this script with `uv run`, or install "
            "matplotlib>=3.8."
        ) from error
    return matplotlib, pyplot


def _format_ms(value):
    return ("%.2f" if value < 10 else "%.1f") % value


def render_figure(
    plot_data,
    output_path,
    scope="encoder",
    title=None,
    width_inches=7.2,
    dpi=200,
    transparent=False,
):
    if scope not in ("encoder", "full"):
        raise PlotError("Scope must be encoder or full")
    if not math.isfinite(width_inches) or width_inches <= 0:
        raise PlotError("Figure width must be positive")
    if dpi <= 0:
        raise PlotError("DPI must be positive")

    stages = [
        stage
        for stage in plot_data["stages"]
        if scope == "full" or stage["scope"] == "encoder"
    ]
    if not stages:
        raise PlotError("Selected scope contains no stages")

    matplotlib, pyplot = _load_matplotlib()
    output_path = pathlib.Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in OUTPUT_SUFFIXES:
        raise PlotError("Output must be SVG, PDF, or PNG")

    figure_height = max(3.0, 0.75 + 0.34 * len(stages))
    narrow = width_inches < 5.5
    label_size = 7.5 if narrow else 8.5

    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": label_size,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
    with matplotlib.rc_context(rc):
        figure, axis = pyplot.subplots(figsize=(width_inches, figure_height))

        if title:
            axis.set_title(
                title,
                loc="left",
                fontsize=9,
                fontweight="bold",
                pad=8,
                color="#202020",
            )

        aggregate_total = sum(stage["aggregate_ms"] for stage in stages)
        y_positions = list(range(len(stages)))
        mean_values = [stage["mean_ms"] for stage in stages]
        axis.barh(
            y_positions,
            mean_values,
            height=0.58,
            color=[stage["color"] for stage in stages],
            edgecolor="none",
            zorder=2,
        )
        for y_position, stage in zip(y_positions, stages):
            share = 100.0 * stage["aggregate_ms"] / aggregate_total
            axis.text(
                1.025,
                y_position,
                "%s ms (%.1f%%)" % (_format_ms(stage["mean_ms"]), share),
                transform=axis.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=label_size,
                color="#333333",
                clip_on=False,
            )

        maximum = max(mean_values)
        maximum = maximum * 1.08 if maximum > 0 else 1.0
        axis.set_yticks(
            y_positions,
            labels=[stage["short_name"] for stage in stages],
        )
        axis.invert_yaxis()
        axis.set_xlim(0, maximum)
        axis.set_xlabel("Mean sampled CPU (ms)", labelpad=7)
        axis.grid(
            axis="x",
            color="#E5E5E5",
            linewidth=0.65,
            zorder=0,
        )
        axis.tick_params(axis="x", colors="#666666", labelsize=7.5)
        axis.tick_params(axis="y", length=0, pad=9, labelsize=label_size)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_color("#777777")

        figure.subplots_adjust(
            left=0.38,
            right=0.77 if narrow else 0.82,
            top=0.91 if title else 0.98,
            bottom=0.15 if narrow else 0.13,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=dpi,
            transparent=transparent,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        pyplot.close(figure)


def resolve_output_path(input_path, output_path):
    if output_path:
        path = pathlib.Path(output_path).expanduser().resolve()
    else:
        path = (
            pathlib.Path(input_path).expanduser().resolve().parent
            / DEFAULT_OUTPUT_FILENAME
        )
    if path.suffix.lower() not in OUTPUT_SUFFIXES:
        raise PlotError("Output must be SVG, PDF, or PNG")
    return path


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="parser-generated analysis JSON")
    parser.add_argument(
        "--output",
        help="output SVG, PDF, or PNG path (default: stage-breakdown.svg)",
    )
    parser.add_argument(
        "--scope",
        choices=("encoder", "full"),
        default="encoder",
        help="show encoder-only work or the full command (default: encoder)",
    )
    parser.add_argument("--title", help="optional figure title (omitted by default)")
    parser.add_argument(
        "--width",
        type=float,
        default=7.2,
        help="figure width in inches (default: 7.2)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="PNG resolution in dots per inch (default: 200)",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="use a transparent figure background",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        analysis = load_analysis(args.input)
        plot_data = build_plot_data(analysis)
        output_path = resolve_output_path(args.input, args.output)
        render_figure(
            plot_data,
            output_path,
            scope=args.scope,
            title=args.title,
            width_inches=args.width,
            dpi=args.dpi,
            transparent=args.transparent,
        )
    except (OSError, PlotError) as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    print("Wrote %s" % output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
