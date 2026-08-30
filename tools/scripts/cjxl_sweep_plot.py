#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Generate an interactive 3D chart from a cjxl sweep summary.

The generated HTML embeds the sweep data and has no Python dependencies beyond
the standard library. By default it loads a pinned Plotly.js release from a
CDN. Pass a local Plotly.js file to --plotly-js for a fully offline chart.

Example:

    python3 tools/scripts/cjxl_sweep_plot.py \
        --input build-samply/sweeps/kodak-distance-effort \
        --output build-samply/sweeps/kodak-distance-effort/runtime-surface.html
"""

import argparse
import csv
import dataclasses
import html
import json
import math
import os
import pathlib
import re
import sys
import tempfile


PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@3.7.0/plotly.min.js"
DEFAULT_OUTPUT_FILENAME = "runtime-surface.html"
NUMERIC_FIELDS = (
    "dataset_wall_ms_median",
    "dataset_wall_ms_mean",
    "dataset_wall_ms_p10",
    "dataset_wall_ms_p90",
    "dataset_cpu_ms_median",
    "dataset_cpu_ms_mean",
    "dataset_cpu_ms_p10",
    "dataset_cpu_ms_p90",
)
INTEGER_FIELDS = (
    "sample_count",
    "expected_sample_count",
    "complete_repetitions",
    "expected_repetitions",
    "image_count",
)
REQUIRED_FIELDS = (
    "axis",
    "parameter_text",
    "parameter_value",
    "effort",
    "complete",
    *NUMERIC_FIELDS,
    *INTEGER_FIELDS,
)


class PlotError(Exception):
    """An expected chart input or output error."""


@dataclasses.dataclass(frozen=True)
class Cell:
    parameter_text: str
    parameter_value: float
    effort: int
    complete: bool
    values: dict


@dataclasses.dataclass(frozen=True)
class ChartData:
    axis: str
    parameters: tuple
    parameter_labels: tuple
    efforts: tuple
    cells: tuple
    complete_cells: int
    expected_repetitions: int
    image_count: int

    def as_json_value(self):
        return {
            "axis": self.axis,
            "parameters": list(self.parameters),
            "parameterLabels": list(self.parameter_labels),
            "efforts": list(self.efforts),
            "cells": [
                [cell.values if cell is not None else None for cell in row]
                for row in self.cells
            ],
            "completeCells": self.complete_cells,
            "totalCells": len(self.parameters) * len(self.efforts),
            "expectedRepetitions": self.expected_repetitions,
            "imageCount": self.image_count,
        }


def resolve_summary_path(input_path):
    path = pathlib.Path(input_path).expanduser().resolve()
    if path.is_dir():
        path = path / "summary.csv"
    if not path.is_file():
        raise PlotError("Sweep summary does not exist: %s" % path)
    return path


def parse_float(row, field, row_number, allow_empty):
    text = row[field].strip()
    if not text and allow_empty:
        return None
    try:
        value = float(text)
    except ValueError as error:
        raise PlotError(
            "Row %d has invalid %s: %s" % (row_number, field, text)
        ) from error
    if not math.isfinite(value):
        raise PlotError("Row %d has non-finite %s" % (row_number, field))
    if value < 0:
        raise PlotError("Row %d has negative %s" % (row_number, field))
    return value


def parse_int(row, field, row_number):
    text = row[field].strip()
    try:
        value = int(text)
    except ValueError as error:
        raise PlotError(
            "Row %d has invalid %s: %s" % (row_number, field, text)
        ) from error
    if value < 0:
        raise PlotError("Row %d has negative %s" % (row_number, field))
    return value


def parse_complete(text, row_number):
    normalized = text.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise PlotError("Row %d has invalid complete value: %s" % (row_number, text))


def load_chart_data(summary_path, allow_incomplete=False):
    path = resolve_summary_path(summary_path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing_fields = [
            field for field in REQUIRED_FIELDS if field not in reader.fieldnames
        ]
        if missing_fields:
            raise PlotError(
                "Summary is missing required columns: %s" % ", ".join(missing_fields)
            )
        parsed_cells = []
        axes = set()
        for row_number, row in enumerate(reader, 2):
            axis = row["axis"].strip()
            if axis not in ("distance", "quality"):
                raise PlotError("Row %d has unsupported axis: %s" % (row_number, axis))
            axes.add(axis)
            parameter_text = row["parameter_text"].strip()
            parameter_value = parse_float(
                row, "parameter_value", row_number, allow_empty=False
            )
            effort = parse_int(row, "effort", row_number)
            if effort < 1 or effort > 10:
                raise PlotError("Row %d has effort outside 1-10" % row_number)
            complete = parse_complete(row["complete"], row_number)
            values = {
                field: parse_float(row, field, row_number, allow_empty=not complete)
                for field in NUMERIC_FIELDS
            }
            values.update(
                {field: parse_int(row, field, row_number) for field in INTEGER_FIELDS}
            )
            values["parameterText"] = parameter_text
            values["complete"] = complete
            parsed_cells.append(
                Cell(parameter_text, parameter_value, effort, complete, values)
            )

    if not parsed_cells:
        raise PlotError("Summary has no data rows: %s" % path)
    if len(axes) != 1:
        raise PlotError("Summary must contain exactly one axis")
    axis = axes.pop()

    parameter_labels = {}
    efforts = set()
    by_key = {}
    for cell in parsed_cells:
        if cell.parameter_value in parameter_labels:
            if parameter_labels[cell.parameter_value] != cell.parameter_text:
                raise PlotError(
                    "Parameter %s has inconsistent labels" % cell.parameter_value
                )
        else:
            parameter_labels[cell.parameter_value] = cell.parameter_text
        efforts.add(cell.effort)
        key = (cell.effort, cell.parameter_value)
        if key in by_key:
            raise PlotError(
                "Summary has a duplicate cell for %s=%s, effort=%d"
                % (axis, cell.parameter_text, cell.effort)
            )
        by_key[key] = cell

    parameters = tuple(sorted(parameter_labels))
    sorted_efforts = tuple(sorted(efforts))
    cells = []
    incomplete = []
    for effort in sorted_efforts:
        cell_row = []
        for parameter in parameters:
            cell = by_key.get((effort, parameter))
            if cell is None or not cell.complete:
                incomplete.append((parameter, effort))
            cell_row.append(cell)
        cells.append(tuple(cell_row))
    if incomplete and not allow_incomplete:
        preview = ", ".join(
            "%s=%s/effort=%d" % (axis, parameter, effort)
            for parameter, effort in incomplete[:5]
        )
        suffix = "" if len(incomplete) <= 5 else ", ..."
        raise PlotError(
            "Summary has %d missing or incomplete cells: %s%s"
            % (len(incomplete), preview, suffix)
        )

    expected_repetitions = {
        cell.values["expected_repetitions"] for cell in parsed_cells
    }
    image_counts = {cell.values["image_count"] for cell in parsed_cells}
    if len(expected_repetitions) != 1 or len(image_counts) != 1:
        raise PlotError("Summary has inconsistent repetition or image counts")

    return ChartData(
        axis=axis,
        parameters=parameters,
        parameter_labels=tuple(parameter_labels[value] for value in parameters),
        efforts=sorted_efforts,
        cells=tuple(cells),
        complete_cells=sum(cell.complete for cell in parsed_cells),
        expected_repetitions=expected_repetitions.pop(),
        image_count=image_counts.pop(),
    )


def plotly_script_tag(plotly_js):
    if plotly_js.startswith(("https://", "http://")):
        return '<script src="%s"></script>' % html.escape(plotly_js, quote=True)
    path = pathlib.Path(plotly_js).expanduser().resolve()
    if not path.is_file():
        raise PlotError("Plotly JavaScript file does not exist: %s" % path)
    source = path.read_text(encoding="utf-8").replace("</script", "<\\/script")
    return "<script>\n%s\n</script>" % source


def generate_html(chart_data, title, plotly_js=PLOTLY_CDN):
    title_text = title or "cjxl %s versus effort runtime" % chart_data.axis
    escaped_title = html.escape(title_text)
    data_json = json.dumps(
        chart_data.as_json_value(), separators=(",", ":"), sort_keys=True
    ).replace("</", "<\\/")
    script_tag = plotly_script_tag(plotly_js)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>__TITLE__</title>
  __PLOTLY_SCRIPT__
  <style>
    :root {
      color-scheme: light dark;
      --page: #ffffff;
      --foreground: #172033;
      --muted: #5d6778;
      --border: #d9dee7;
      --control: #f6f7f9;
      --grid: #d8dee8;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --page: #11151d;
        --foreground: #edf1f7;
        --muted: #a9b2c1;
        --border: #343c49;
        --control: #1a202b;
        --grid: #343c49;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--page);
      color: var(--foreground);
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1180px, 100%);
      margin: 0 auto;
      padding: 24px clamp(12px, 3vw, 32px) 20px;
    }
    h1 {
      margin: 0 0 4px;
      font-size: clamp(20px, 2.5vw, 30px);
      font-weight: 650;
      letter-spacing: -0.025em;
    }
    #subtitle, #status { color: var(--muted); }
    #subtitle { margin: 0 0 18px; }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      align-items: end;
      margin-bottom: 8px;
    }
    label {
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
    }
    select {
      min-width: 112px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--control);
      color: var(--foreground);
      padding: 7px 28px 7px 9px;
      font: inherit;
    }
    .checkbox {
      display: flex;
      gap: 7px;
      align-items: center;
      min-height: 34px;
      color: var(--foreground);
      font-size: 13px;
      font-weight: 500;
    }
    .checkbox input { margin: 0; }
    #chart {
      width: 100%;
      height: clamp(480px, 72vh, 760px);
      min-height: 480px;
    }
    #status {
      min-height: 20px;
      margin-top: 2px;
      font-size: 12px;
    }
    @media (max-width: 520px) {
      main { padding-top: 16px; }
      #chart { height: 520px; min-height: 520px; }
    }
  </style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p id="subtitle"></p>
  <div class="controls" aria-label="Chart controls">
    <label>Clock
      <select id="clock">
        <option value="wall">Wall time</option>
        <option value="cpu">CPU time</option>
      </select>
    </label>
    <label>Statistic
      <select id="statistic">
        <option value="median">Median</option>
        <option value="mean">Mean</option>
      </select>
    </label>
    <label>Runtime scale
      <select id="scale">
        <option value="linear">Linear</option>
        <option value="log">Logarithmic</option>
      </select>
    </label>
    <label class="checkbox">
      <input id="points" type="checkbox" checked>
      Show measured cells
    </label>
  </div>
  <div id="chart" role="img" aria-label="Interactive three-dimensional runtime surface"></div>
  <div id="status" aria-live="polite"></div>
</main>
<script>
"use strict";
const chartData = __CHART_DATA__;
const chart = document.getElementById("chart");
const clockControl = document.getElementById("clock");
const statisticControl = document.getElementById("statistic");
const scaleControl = document.getElementById("scale");
const pointsControl = document.getElementById("points");
const subtitle = document.getElementById("subtitle");
const status = document.getElementById("status");
const axisTitle = chartData.axis[0].toUpperCase() + chartData.axis.slice(1);

subtitle.textContent = `${axisTitle} × effort · ${chartData.imageCount} images · ` +
  `${chartData.expectedRepetitions} complete dataset passes per cell`;

function theme() {
  const style = getComputedStyle(document.documentElement);
  return {
    background: style.getPropertyValue("--page").trim(),
    foreground: style.getPropertyValue("--foreground").trim(),
    grid: style.getPropertyValue("--grid").trim()
  };
}

function logarithmicTicks(values) {
  const positive = values.filter(value => value > 0);
  if (positive.length === 0) {
    return {values: [], labels: []};
  }
  const minimum = Math.min(...positive);
  const maximum = Math.max(...positive);
  const tickValues = [];
  const tickLabels = [];
  const firstExponent = Math.floor(Math.log10(minimum));
  const lastExponent = Math.ceil(Math.log10(maximum));
  for (let exponent = firstExponent; exponent <= lastExponent; exponent++) {
    for (const multiplier of [1, 2, 5]) {
      const value = multiplier * Math.pow(10, exponent);
      if (value < minimum * 0.999 || value > maximum * 1.001) {
        continue;
      }
      tickValues.push(Math.log10(value));
      tickLabels.push(
        value >= 1 ? Number(value.toPrecision(3)).toString() : value.toPrecision(2)
      );
    }
  }
  return {values: tickValues, labels: tickLabels};
}

function buildPlot() {
  const selectedClock = clockControl.value;
  const selectedStatistic = statisticControl.value;
  const field = `dataset_${selectedClock}_ms_${selectedStatistic}`;
  const p10Field = `dataset_${selectedClock}_ms_p10`;
  const p90Field = `dataset_${selectedClock}_ms_p90`;
  const statisticLabel = selectedStatistic === "median" ? "Median" : "Mean";
  const clockLabel = selectedClock === "wall" ? "wall" : "CPU";
  const z = [];
  const surfaceColor = [];
  const custom = [];
  const pointX = [];
  const pointY = [];
  const pointZ = [];
  const pointColor = [];
  const pointCustom = [];

  for (let effortIndex = 0; effortIndex < chartData.efforts.length; effortIndex++) {
    const zRow = [];
    const colorRow = [];
    const customRow = [];
    for (let parameterIndex = 0; parameterIndex < chartData.parameters.length; parameterIndex++) {
      const cell = chartData.cells[effortIndex][parameterIndex];
      const value = cell && cell[field] !== null ? cell[field] / 1000 : null;
      const displayValue = value !== null && scaleControl.value === "log" ?
        (value > 0 ? Math.log10(value) : null) : value;
      const details = cell ? [
        cell.parameterText,
        cell[p10Field] === null ? null : cell[p10Field] / 1000,
        cell[p90Field] === null ? null : cell[p90Field] / 1000,
        cell.sample_count,
        cell.complete_repetitions,
        cell.expected_repetitions,
        value
      ] : [chartData.parameterLabels[parameterIndex], null, null, 0, 0, chartData.expectedRepetitions, null];
      zRow.push(displayValue);
      colorRow.push(value);
      customRow.push(details);
      if (displayValue !== null) {
        pointX.push(chartData.parameters[parameterIndex]);
        pointY.push(chartData.efforts[effortIndex]);
        pointZ.push(displayValue);
        pointColor.push(value);
        pointCustom.push(details);
      }
    }
    z.push(zRow);
    surfaceColor.push(colorRow);
    custom.push(customRow);
  }

  const hoverTemplate = `<b>${axisTitle}: %{customdata[0]}</b><br>` +
    "Effort: %{y}<br>" +
    `${statisticLabel} ${clockLabel} time: %{customdata[6]:.3f} s<br>` +
    "p10–p90: %{customdata[1]:.3f}–%{customdata[2]:.3f} s<br>" +
    "Samples: %{customdata[3]}<br>" +
    "Complete passes: %{customdata[4]}/%{customdata[5]}<extra></extra>";
  const colorMinimum = pointColor.length ? Math.min(...pointColor) : 0;
  const colorMaximum = pointColor.length ? Math.max(...pointColor) : 1;
  const traces = [
    {
      type: "surface",
      x: chartData.parameters,
      y: chartData.efforts,
      z: z,
      surfacecolor: surfaceColor,
      customdata: custom,
      colorscale: "Viridis",
      cmin: colorMinimum,
      cmax: colorMaximum,
      colorbar: {title: {text: "Seconds"}, thickness: 16, len: 0.72},
      contours: {
        x: {show: true, color: "rgba(255,255,255,0.26)", width: 1},
        y: {show: true, color: "rgba(255,255,255,0.26)", width: 1}
      },
      hovertemplate: hoverTemplate,
      connectgaps: false,
      name: "Runtime surface"
    },
    {
      type: "scatter3d",
      mode: "markers",
      x: pointX,
      y: pointY,
      z: pointZ,
      customdata: pointCustom,
      marker: {
        size: 3.5,
        color: pointColor,
        colorscale: "Viridis",
        cmin: colorMinimum,
        cmax: colorMaximum,
        line: {color: "rgba(255,255,255,0.85)", width: 0.7},
        showscale: false
      },
      hovertemplate: hoverTemplate,
      visible: pointsControl.checked,
      name: "Measured cells"
    }
  ];
  const colors = theme();
  const logTicks = logarithmicTicks(pointColor);
  const zAxis = {
    title: {text: `${statisticLabel} ${clockLabel} time (seconds)`},
    gridcolor: colors.grid,
    zerolinecolor: colors.grid
  };
  if (scaleControl.value === "log") {
    zAxis.tickmode = "array";
    zAxis.tickvals = logTicks.values;
    zAxis.ticktext = logTicks.labels;
  }
  const layout = {
    autosize: true,
    margin: {l: 0, r: 0, t: 10, b: 0},
    paper_bgcolor: colors.background,
    plot_bgcolor: colors.background,
    font: {color: colors.foreground, family: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"},
    showlegend: false,
    scene: {
      bgcolor: colors.background,
      aspectmode: "manual",
      aspectratio: {x: 1.25, y: 1.4, z: 0.9},
      camera: {eye: {x: 1.55, y: -1.75, z: 1.15}},
      xaxis: {
        title: {text: axisTitle},
        tickmode: "array",
        tickvals: chartData.parameters,
        ticktext: chartData.parameterLabels,
        gridcolor: colors.grid,
        zerolinecolor: colors.grid
      },
      yaxis: {
        title: {text: "Effort"},
        tickmode: "array",
        tickvals: chartData.efforts,
        gridcolor: colors.grid,
        zerolinecolor: colors.grid
      },
      zaxis: zAxis
    },
    uirevision: "cjxl-runtime-surface"
  };
  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["sendDataToCloud"]
  };
  Plotly.react(chart, traces, layout, config);
  status.textContent = `${chartData.completeCells}/${chartData.totalCells} complete cells · ` +
    `${statisticLabel} ${clockLabel} time · ${scaleControl.value} scale`;
}

for (const control of [clockControl, statisticControl, scaleControl, pointsControl]) {
  control.addEventListener("change", buildPlot);
}
const colorScheme = window.matchMedia("(prefers-color-scheme: dark)");
if (colorScheme.addEventListener) {
  colorScheme.addEventListener("change", buildPlot);
}
buildPlot();
</script>
</body>
</html>
"""
    replacements = {
        "TITLE": escaped_title,
        "PLOTLY_SCRIPT": script_tag,
        "CHART_DATA": data_json,
    }
    return re.sub(
        r"__(TITLE|PLOTLY_SCRIPT|CHART_DATA)__",
        lambda match: replacements[match.group(1)],
        template,
    )


def write_text_atomic(path, content):
    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = pathlib.Path(output.name)
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PlotError("Unable to write chart %s: %s" % (path, error)) from error
    return path


def create_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Sweep directory or path to its summary.csv.",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path (default: runtime-surface.html beside summary.csv).",
    )
    parser.add_argument("--title", help="Visible chart title.")
    parser.add_argument(
        "--plotly-js",
        default=PLOTLY_CDN,
        help="Plotly.js URL or local file for offline embedding.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Render missing or incomplete cells as gaps instead of failing.",
    )
    return parser


def main(argv=None):
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    try:
        summary_path = resolve_summary_path(args.input)
        chart_data = load_chart_data(summary_path, args.allow_incomplete)
        output_path = (
            pathlib.Path(args.output).expanduser().resolve()
            if args.output
            else summary_path.with_name(DEFAULT_OUTPUT_FILENAME)
        )
        content = generate_html(chart_data, args.title, args.plotly_js)
        write_text_atomic(output_path, content)
        print(
            "Wrote %s (%d/%d complete cells)"
            % (
                output_path,
                chart_data.complete_cells,
                len(chart_data.parameters) * len(chart_data.efforts),
            )
        )
        return 0
    except PlotError as error:
        print("error: %s" % error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
