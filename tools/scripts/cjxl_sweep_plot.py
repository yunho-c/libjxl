#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Generate interactive charts from a cjxl sweep summary.

The generated HTML embeds the sweep data and has no Python dependencies beyond
the standard library. By default it loads a pinned Plotly.js release from a
CDN. Pass a local Plotly.js file to --plotly-js for a fully offline chart. The
default runtime-surface chart preserves the original interface; duration-bpp
plots encoding time against compressed bits per pixel.

Example:

    python3 tools/scripts/cjxl_sweep_plot.py \
        --input build-samply/sweeps/kodak-distance-effort \
        --output build-samply/sweeps/kodak-distance-effort/runtime-surface.html

    python3 tools/scripts/cjxl_sweep_plot.py \
        --chart duration-bpp \
        --input build-samply/sweeps/kodak-distance-effort
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
CHART_TYPES = ("runtime-surface", "duration-bpp")
DEFAULT_OUTPUT_FILENAMES = {
    "runtime-surface": "runtime-surface.html",
    "duration-bpp": "duration-vs-bpp.html",
}
TIMING_NUMERIC_FIELDS = (
    "dataset_wall_ms_median",
    "dataset_wall_ms_mean",
    "dataset_wall_ms_p10",
    "dataset_wall_ms_p90",
    "dataset_cpu_ms_median",
    "dataset_cpu_ms_mean",
    "dataset_cpu_ms_p10",
    "dataset_cpu_ms_p90",
)
SIZE_NUMERIC_FIELDS = (
    "dataset_input_bytes",
    "dataset_encoded_bytes",
    "bits_per_pixel",
    "png_to_jxl_ratio",
)
TIMING_INTEGER_FIELDS = (
    "sample_count",
    "expected_sample_count",
    "complete_repetitions",
    "expected_repetitions",
    "image_count",
)
SIZE_INTEGER_FIELDS = (
    "size_sample_count",
    "expected_size_sample_count",
)
BASE_REQUIRED_FIELDS = (
    "axis",
    "parameter_text",
    "parameter_value",
    "effort",
    "complete",
    *TIMING_NUMERIC_FIELDS,
    *TIMING_INTEGER_FIELDS,
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


def chart_fields(chart_type):
    if chart_type not in CHART_TYPES:
        raise PlotError("Unsupported chart type: %s" % chart_type)
    numeric_fields = list(TIMING_NUMERIC_FIELDS)
    integer_fields = list(TIMING_INTEGER_FIELDS)
    required_fields = list(BASE_REQUIRED_FIELDS)
    if chart_type == "duration-bpp":
        numeric_fields.extend(SIZE_NUMERIC_FIELDS)
        integer_fields.extend(SIZE_INTEGER_FIELDS)
        required_fields.extend(("size_complete", *SIZE_NUMERIC_FIELDS))
        required_fields.extend(SIZE_INTEGER_FIELDS)
    return numeric_fields, integer_fields, required_fields


def load_chart_data(summary_path, allow_incomplete=False, chart_type="runtime-surface"):
    path = resolve_summary_path(summary_path)
    numeric_fields, integer_fields, required_fields = chart_fields(chart_type)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing_fields = [
            field for field in required_fields if field not in reader.fieldnames
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
            timing_complete = parse_complete(row["complete"], row_number)
            size_complete = (
                parse_complete(row["size_complete"], row_number)
                if chart_type == "duration-bpp"
                else True
            )
            complete = timing_complete and size_complete
            values = {}
            for field in numeric_fields:
                field_complete = (
                    size_complete if field in SIZE_NUMERIC_FIELDS else timing_complete
                )
                values[field] = parse_float(
                    row, field, row_number, allow_empty=not field_complete
                )
            values.update(
                {field: parse_int(row, field, row_number) for field in integer_fields}
            )
            values["parameterText"] = parameter_text
            values["complete"] = complete
            values["timingComplete"] = timing_complete
            values["sizeComplete"] = size_complete
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


def replace_template_markers(template, replacements):
    return re.sub(
        r"__(TITLE|PLOTLY_SCRIPT|CHART_DATA|DYNAMIC_TITLE)__",
        lambda match: replacements[match.group(1)],
        template,
    )


def generate_runtime_surface_html(chart_data, title, plotly_js=PLOTLY_CDN):
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
    return replace_template_markers(template, replacements)


def generate_duration_bpp_html(chart_data, title, plotly_js=PLOTLY_CDN):
    title_text = title or "libjxl Encoding Time vs. BPP"
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
    .duration-title { text-align: center; }
    .duration-controls { margin-top: 18px; }
    #chart {
      width: 100%;
      height: clamp(500px, 70vh, 720px);
      min-height: 500px;
    }
    #status {
      min-height: 20px;
      margin-top: 2px;
      font-size: 12px;
    }
    @media (max-width: 520px) {
      main { padding-top: 16px; }
      #chart { height: 600px; min-height: 600px; }
    }
  </style>
</head>
<body>
<main>
  <h1 class="duration-title">__TITLE__</h1>
  <div id="chart" role="img" aria-label="Encoding Time versus bits per pixel"></div>
  <div class="controls duration-controls" aria-label="Chart controls">
    <label><span id="parameter-label"></span>
      <select id="parameter-filter"></select>
    </label>
    <label>Clock
      <select id="clock">
        <option value="wall">Wall time</option>
        <option value="cpu">CPU time</option>
      </select>
    </label>
    <label>Metric
      <select id="size-metric">
        <option value="bpp">Bits per pixel</option>
        <option value="png-ratio">Compression ratio</option>
      </select>
    </label>
    <label>Duration scale
      <select id="scale">
        <option value="linear">Linear</option>
        <option value="log">Logarithmic</option>
      </select>
    </label>
    <label class="checkbox">
      <input id="uncertainty" type="checkbox">
      Show p10–p90
    </label>
    <label class="checkbox">
      <input id="frontier" type="checkbox">
      Show Pareto frontier
    </label>
  </div>
  <div id="status" aria-live="polite"></div>
</main>
<script>
"use strict";
const chartData = __CHART_DATA__;
const chart = document.getElementById("chart");
const parameterControl = document.getElementById("parameter-filter");
const parameterLabel = document.getElementById("parameter-label");
const clockControl = document.getElementById("clock");
const sizeMetricControl = document.getElementById("size-metric");
const scaleControl = document.getElementById("scale");
const uncertaintyControl = document.getElementById("uncertainty");
const frontierControl = document.getElementById("frontier");
const status = document.getElementById("status");
const titleElement = document.querySelector(".duration-title");
const dynamicTitle = __DYNAMIC_TITLE__;
const axisTitle = chartData.axis[0].toUpperCase() + chartData.axis.slice(1);
const colors = [
  "#440154", "#482878", "#3e4989", "#31688e", "#26828e",
  "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"
];
const symbols = [
  "circle", "square", "diamond", "cross", "x", "triangle-up",
  "triangle-down", "triangle-left", "triangle-right", "star"
];
const allParametersLabel = chartData.axis === "distance" ?
  "All distances" : "All quality values";

parameterLabel.textContent = axisTitle;
parameterControl.add(new Option(allParametersLabel, "all"));
parameterControl.add(new Option("Average", "average"));
for (let index = 0; index < chartData.parameterLabels.length; index++) {
  parameterControl.add(new Option(chartData.parameterLabels[index], String(index)));
}

function theme() {
  const style = getComputedStyle(document.documentElement);
  return {
    background: style.getPropertyValue("--page").trim(),
    foreground: style.getPropertyValue("--foreground").trim(),
    muted: style.getPropertyValue("--muted").trim(),
    grid: style.getPropertyValue("--grid").trim()
  };
}

function paretoFrontier(points, minimizeSize) {
  return points.filter((candidate, candidateIndex) => !points.some(
    (other, otherIndex) => {
      if (otherIndex === candidateIndex) {
        return false;
      }
      const noWorseSize = minimizeSize ?
        other.size <= candidate.size : other.size >= candidate.size;
      const betterSize = minimizeSize ?
        other.size < candidate.size : other.size > candidate.size;
      return other.duration <= candidate.duration && noWorseSize &&
        (other.duration < candidate.duration || betterSize);
    }
  )).sort((first, second) => first.duration - second.duration);
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function aggregateCells(cells, field, p10Field, p90Field) {
  const totalInputBytes = cells.reduce(
    (sum, cell) => sum + cell.dataset_input_bytes, 0
  );
  const totalEncodedBytes = cells.reduce(
    (sum, cell) => sum + cell.dataset_encoded_bytes, 0
  );
  const parameterNoun = chartData.axis === "distance" ?
    "distances" : "quality values";
  return {
    complete: true,
    parameterText: `Average across ${cells.length} ${parameterNoun}`,
    [field]: mean(cells.map(cell => cell[field])),
    [p10Field]: mean(cells.map(cell => cell[p10Field])),
    [p90Field]: mean(cells.map(cell => cell[p90Field])),
    bits_per_pixel: mean(cells.map(cell => cell.bits_per_pixel)),
    dataset_input_bytes: totalInputBytes,
    dataset_encoded_bytes: mean(
      cells.map(cell => cell.dataset_encoded_bytes)
    ),
    png_to_jxl_ratio: totalInputBytes / totalEncodedBytes,
    sample_count: cells.reduce((sum, cell) => sum + cell.sample_count, 0),
    size_sample_count: cells.reduce(
      (sum, cell) => sum + cell.size_sample_count, 0
    )
  };
}

function buildPlot() {
  const selectedClock = clockControl.value;
  const useBpp = sizeMetricControl.value === "bpp";
  const sizeField = useBpp ? "bits_per_pixel" : "png_to_jxl_ratio";
  const sizeAxisTitle = useBpp ?
    "Bits per pixel (lower is better)" :
    "Compression ratio (PNG/JXL, higher is better)";
  const sizeHoverLine = useBpp ?
    "Bits per pixel: %{y:.4f}" : "PNG/JXL ratio: %{y:.3f}×";
  const field = `dataset_${selectedClock}_ms_median`;
  const p10Field = `dataset_${selectedClock}_ms_p10`;
  const p90Field = `dataset_${selectedClock}_ms_p90`;
  const averageParameters = parameterControl.value === "average";
  const statisticLabel = averageParameters ? "Average median" : "Median";
  const uncertaintyLabel = averageParameters ? "Average p10–p90" : "p10–p90";
  const scopeLabel = averageParameters ? "Scope" : axisTitle;
  const clockLabel = selectedClock === "wall" ? "wall" : "CPU";
  const selectedParameterIndex =
    parameterControl.value === "all" || averageParameters ?
      null : Number(parameterControl.value);
  if (dynamicTitle) {
    const titleText = useBpp ?
      "libjxl Encoding Time vs. BPP" :
      "libjxl Encoding Time vs. Compression Ratio";
    titleElement.textContent = titleText;
    document.title = titleText;
  }
  chart.setAttribute(
    "aria-label",
    `Encoding time versus ${useBpp ? "bits per pixel" : "PNG/JXL ratio"}`
  );
  const traces = [];
  const allPoints = [];
  const paletteDenominator = Math.max(1, chartData.efforts.length - 1);

  for (let effortIndex = 0; effortIndex < chartData.efforts.length; effortIndex++) {
    const effort = chartData.efforts[effortIndex];
    const x = [];
    const y = [];
    const custom = [];
    const errorPlus = [];
    const errorMinus = [];
    const cells = [];
    for (
      let parameterIndex = 0;
      parameterIndex < chartData.parameters.length;
      parameterIndex++
    ) {
      if (selectedParameterIndex !== null && parameterIndex !== selectedParameterIndex) {
        continue;
      }
      const cell = chartData.cells[effortIndex][parameterIndex];
      if (!cell || !cell.complete || cell[field] === null || cell[sizeField] === null) {
        continue;
      }
      cells.push(cell);
    }
    const plotCells = averageParameters ?
      (cells.length === chartData.parameters.length ?
        [aggregateCells(cells, field, p10Field, p90Field)] : []) :
      cells;
    for (const cell of plotCells) {
      const duration = cell[field] / 1000;
      const p10 = cell[p10Field] / 1000;
      const p90 = cell[p90Field] / 1000;
      const details = [
        cell.parameterText,
        effort,
        duration,
        p10,
        p90,
        cell.bits_per_pixel,
        cell.dataset_encoded_bytes,
        cell.png_to_jxl_ratio,
        cell.sample_count,
        cell.size_sample_count
      ];
      x.push(duration);
      y.push(cell[sizeField]);
      custom.push(details);
      errorPlus.push(Math.max(0, p90 - duration));
      errorMinus.push(Math.max(0, duration - p10));
      allPoints.push({
        duration: duration,
        size: cell[sizeField],
        effort: effort,
        custom: details
      });
    }
    const colorIndex = Math.round(
      effortIndex * (colors.length - 1) / paletteDenominator
    );
    const color = colors[colorIndex];
    traces.push({
      type: "scatter",
      mode: selectedParameterIndex === null && !averageParameters ?
        "lines+markers" : "markers",
      name: `Effort ${effort}`,
      x: x,
      y: y,
      customdata: custom,
      line: {color: color, width: 1.6},
      marker: {
        color: color,
        size: 8,
        symbol: symbols[effortIndex % symbols.length],
        line: {color: theme().background, width: 1}
      },
      error_x: {
        type: "data",
        symmetric: false,
        array: errorPlus,
        arrayminus: errorMinus,
        visible: uncertaintyControl.checked,
        color: color,
        thickness: 1,
        width: 3
      },
      hovertemplate: `<b>Effort ${effort}</b><br>` +
        `${scopeLabel}: %{customdata[0]}<br>` +
        `${statisticLabel} ${clockLabel} time: %{customdata[2]:.3f} s<br>` +
        `${uncertaintyLabel}: ` +
        "%{customdata[3]:.3f}–%{customdata[4]:.3f} s<br>" +
        "Bits per pixel: %{customdata[5]:.4f}<br>" +
        "Encoded dataset: %{customdata[6]:,.0f} bytes<br>" +
        "PNG-to-JXL ratio: %{customdata[7]:.3f}×<br>" +
        "Timing samples: %{customdata[8]}<br>" +
        "Size samples: %{customdata[9]}<extra></extra>"
    });
  }

  if ((selectedParameterIndex !== null || averageParameters) && allPoints.length) {
    const effortCurve = [...allPoints].sort(
      (first, second) => first.effort - second.effort
    );
    traces.unshift({
      type: "scatter",
      mode: "lines",
      name: "Effort order",
      showlegend: false,
      hoverinfo: "skip",
      x: effortCurve.map(point => point.duration),
      y: effortCurve.map(point => point.size),
      line: {color: theme().muted, width: 1.5}
    });
  }

  const frontier = paretoFrontier(allPoints, useBpp);
  if (frontierControl.checked && frontier.length) {
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      name: "Pareto frontier",
      x: frontier.map(point => point.duration),
      y: frontier.map(point => point.size),
      customdata: frontier.map(point => point.custom),
      line: {color: theme().foreground, width: 2.5, dash: "dot"},
      marker: {
        color: theme().background,
        size: 10,
        symbol: "diamond",
        line: {color: theme().foreground, width: 2}
      },
      hovertemplate: "<b>Pareto frontier</b><br>" +
        "Effort: %{customdata[1]}<br>" +
        `${scopeLabel}: %{customdata[0]}<br>` +
        `${statisticLabel} ${clockLabel} time: %{customdata[2]:.3f} s<br>` +
        `${sizeHoverLine}<extra></extra>`
    });
  }

  const resolvedTheme = theme();
  const compact = window.innerWidth < 650;
  const layout = {
    autosize: true,
    margin: compact ? {l: 58, r: 12, t: 18, b: 118} : {l: 72, r: 118, t: 18, b: 66},
    paper_bgcolor: resolvedTheme.background,
    plot_bgcolor: resolvedTheme.background,
    font: {
      color: resolvedTheme.foreground,
      family: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    },
    hovermode: "closest",
    xaxis: {
      title: {text: `Encoding time (seconds)`},
      type: scaleControl.value,
      gridcolor: resolvedTheme.grid,
      zerolinecolor: resolvedTheme.grid,
      automargin: true
    },
    yaxis: {
      title: {text: sizeAxisTitle},
      gridcolor: resolvedTheme.grid,
      zerolinecolor: resolvedTheme.grid,
      automargin: true
    },
    legend: compact ? {
      orientation: "h",
      x: 0,
      y: -0.52,
      xanchor: "left",
      yanchor: "top",
      font: {size: 11}
    } : {
      orientation: "v",
      x: 1.01,
      y: 1,
      xanchor: "left",
      yanchor: "top"
    },
    uirevision: `cjxl-duration-bpp-${field}-${sizeMetricControl.value}-` +
      `${scaleControl.value}-${parameterControl.value}`
  };
  const config = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"]
  };
  Plotly.react(chart, traces, layout, config);
  const statusParts = [
    `${chartData.imageCount} images`,
    `${chartData.expectedRepetitions} dataset passes`
  ];
  status.textContent = statusParts.join(" · ");
}

for (const control of [
  parameterControl,
  clockControl,
  sizeMetricControl,
  scaleControl,
  uncertaintyControl,
  frontierControl
]) {
  control.addEventListener("change", buildPlot);
}
const colorScheme = window.matchMedia("(prefers-color-scheme: dark)");
if (colorScheme.addEventListener) {
  colorScheme.addEventListener("change", buildPlot);
}
let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(buildPlot, 120);
});
buildPlot();
</script>
</body>
</html>
"""
    replacements = {
        "TITLE": escaped_title,
        "PLOTLY_SCRIPT": script_tag,
        "CHART_DATA": data_json,
        "DYNAMIC_TITLE": "true" if title is None else "false",
    }
    return replace_template_markers(template, replacements)


def generate_html(chart_data, title, plotly_js=PLOTLY_CDN):
    """Backward-compatible alias for the original runtime surface renderer."""
    return generate_runtime_surface_html(chart_data, title, plotly_js)


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
        "--chart",
        choices=CHART_TYPES,
        default="runtime-surface",
        help="Chart to generate (default: runtime-surface).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Sweep directory or path to its summary.csv.",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path (default depends on --chart, beside summary.csv).",
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
        chart_data = load_chart_data(
            summary_path, args.allow_incomplete, chart_type=args.chart
        )
        output_path = (
            pathlib.Path(args.output).expanduser().resolve()
            if args.output
            else summary_path.with_name(DEFAULT_OUTPUT_FILENAMES[args.chart])
        )
        if args.chart == "duration-bpp":
            content = generate_duration_bpp_html(chart_data, args.title, args.plotly_js)
        else:
            content = generate_runtime_surface_html(
                chart_data, args.title, args.plotly_js
            )
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
