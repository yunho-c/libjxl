#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Validate and load raw libjxl encoder debug-data dumps."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any


SCHEMA_MAJOR = 1
DTYPES = {
    "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64",
    "int64", "float32", "float64",
}
DTYPE_SIZES = {
    "uint8": 1, "int8": 1, "uint16": 2, "int16": 2, "uint32": 4,
    "int32": 4, "uint64": 8, "int64": 8, "float32": 4, "float64": 8,
}
GRID_KINDS = {
    "pixel", "block", "color_tile", "group", "variable_block", "other",
}


class ManifestError(ValueError):
  """Raised when a debug-data manifest is malformed or unsupported."""


class VisualizationError(ValueError):
  """Raised when an artifact cannot be rendered with the requested options."""


def _expect_type(value: object, expected: type | tuple[type, ...],
                 where: str) -> None:
  if not isinstance(value, expected):
    raise ManifestError(f"{where}: expected {expected}, got {type(value)}")


def _expect_keys(value: dict[str, Any], required: set[str],
                 where: str, optional: set[str] | None = None) -> None:
  missing = required - value.keys()
  if missing:
    raise ManifestError(f"{where}: missing {', '.join(sorted(missing))}")
  unexpected = value.keys() - required - (optional or set())
  if unexpected:
    raise ManifestError(f"{where}: unexpected {', '.join(sorted(unexpected))}")


def _validate_int_list(value: object, length: int | None, minimum: int | None,
                       where: str) -> list[int]:
  _expect_type(value, list, where)
  values = value
  if length is not None and len(values) != length:
    raise ManifestError(f"{where}: expected {length} entries")
  for index, item in enumerate(values):
    if not isinstance(item, int) or isinstance(item, bool):
      raise ManifestError(f"{where}[{index}]: expected integer")
    if minimum is not None and item < minimum:
      raise ManifestError(f"{where}[{index}]: must be >= {minimum}")
  return values


def _validate_rect(value: object, where: str) -> list[int]:
  rect = _validate_int_list(value, 4, None, where)
  if rect[2] < 0 or rect[3] < 0:
    raise ManifestError(f"{where}: width and height must be nonnegative")
  return rect


def _expect_nonnegative_int(value: object, where: str) -> None:
  if not isinstance(value, int) or isinstance(value, bool) or value < 0:
    raise ManifestError(f"{where}: expected nonnegative integer")


def _expect_number(value: object, where: str) -> None:
  if not isinstance(value, (int, float)) or isinstance(value, bool):
    raise ManifestError(f"{where}: expected number")


def _validate_artifact(artifact: object, index: int) -> dict[str, Any]:
  where = f"artifacts[{index}]"
  _expect_type(artifact, dict, where)
  value = artifact
  required = {
      "name", "path", "stage", "dtype", "shape", "axes", "grid", "units",
      "semantic", "frame_index", "bytes", "file_bytes",
  }
  optional = {
      "iteration", "channel_names", "categories", "derived_from", "formula",
  }
  _expect_keys(value, required, where, optional)

  for key in ("name", "path", "stage", "dtype", "units", "semantic"):
    _expect_type(value[key], str, f"{where}.{key}")
  if not value["name"]:
    raise ManifestError(f"{where}.name: must not be empty")
  if value["dtype"] not in DTYPES:
    raise ManifestError(f"{where}.dtype: unsupported {value['dtype']!r}")

  rel_path = PurePosixPath(value["path"])
  if ("\\" in value["path"] or rel_path.is_absolute() or
      ".." in rel_path.parts or
      rel_path.suffix != ".npy"):
    raise ManifestError(f"{where}.path: unsafe or non-npy path")
  if value["path"] != value["name"] + ".npy":
    raise ManifestError(f"{where}.path: must be name plus .npy")

  shape = _validate_int_list(value["shape"], None, 0, f"{where}.shape")
  if len(shape) > 8:
    raise ManifestError(f"{where}.shape: rank greater than 8")
  _expect_type(value["axes"], list, f"{where}.axes")
  if len(value["axes"]) != len(shape):
    raise ManifestError(f"{where}: axes and shape ranks differ")
  for axis_index, axis in enumerate(value["axes"]):
    if not isinstance(axis, str) or not axis:
      raise ManifestError(f"{where}.axes[{axis_index}]: expected name")

  grid_where = f"{where}.grid"
  _expect_type(value["grid"], dict, grid_where)
  grid = value["grid"]
  _expect_keys(grid, {
      "kind", "origin_px", "spacing_px", "footprint_px", "valid_rect_px",
      "padded_rect_px", "value_is_block_anchor",
  }, grid_where)
  if grid["kind"] not in GRID_KINDS:
    raise ManifestError(f"{grid_where}.kind: unknown grid")
  _validate_int_list(grid["origin_px"], 2, None, f"{grid_where}.origin_px")
  _validate_int_list(grid["spacing_px"], 2, 1, f"{grid_where}.spacing_px")
  _validate_int_list(grid["footprint_px"], 2, 1,
                     f"{grid_where}.footprint_px")
  valid_rect = _validate_rect(grid["valid_rect_px"],
                              f"{grid_where}.valid_rect_px")
  padded_rect = _validate_rect(grid["padded_rect_px"],
                               f"{grid_where}.padded_rect_px")
  _expect_type(grid["value_is_block_anchor"], bool,
               f"{grid_where}.value_is_block_anchor")
  if (valid_rect[0] < padded_rect[0] or valid_rect[1] < padded_rect[1] or
      valid_rect[0] + valid_rect[2] > padded_rect[0] + padded_rect[2] or
      valid_rect[1] + valid_rect[3] > padded_rect[1] + padded_rect[3]):
    raise ManifestError(f"{grid_where}: valid rectangle is outside padding")

  for key in ("frame_index", "bytes", "file_bytes"):
    _expect_nonnegative_int(value[key], f"{where}.{key}")
  num_elements = 1
  for dimension in shape:
    num_elements *= dimension
  expected_bytes = num_elements * DTYPE_SIZES[value["dtype"]]
  if value["bytes"] != expected_bytes:
    raise ManifestError(
        f"{where}.bytes: expected {expected_bytes}, got {value['bytes']}")
  if value["file_bytes"] < value["bytes"] + 10:
    raise ManifestError(f"{where}.file_bytes: too small for .npy payload")
  if "iteration" in value:
    _expect_nonnegative_int(value["iteration"], f"{where}.iteration")
  if "channel_names" in value:
    _expect_type(value["channel_names"], list, f"{where}.channel_names")
    if not shape or len(value["channel_names"]) != shape[0]:
      raise ManifestError(f"{where}.channel_names: size does not match axis 0")
    for channel in value["channel_names"]:
      _expect_type(channel, str, f"{where}.channel_names entry")
  if "categories" in value:
    _expect_type(value["categories"], list, f"{where}.categories")
    for category_index, category in enumerate(value["categories"]):
      category_where = f"{where}.categories[{category_index}]"
      _expect_type(category, dict, category_where)
      _expect_keys(category, {"value", "name", "covered_blocks"},
                   category_where)
      if not isinstance(category["value"], int) or isinstance(
          category["value"], bool):
        raise ManifestError(f"{category_where}.value: expected integer")
      _expect_type(category["name"], str, f"{category_where}.name")
      _validate_int_list(category["covered_blocks"], 2, 0,
                         f"{category_where}.covered_blocks")
  if ("derived_from" in value) != ("formula" in value):
    raise ManifestError(
        f"{where}: derived_from and formula must occur together")
  if "derived_from" in value:
    _expect_type(value["derived_from"], list, f"{where}.derived_from")
    for source in value["derived_from"]:
      if not isinstance(source, str) or not source:
        raise ManifestError(f"{where}.derived_from: expected artifact names")
    _expect_type(value["formula"], str, f"{where}.formula")
  return value


def validate_manifest(manifest: object) -> dict[str, Any]:
  """Validates schema 1.x and returns the typed manifest dictionary."""
  _expect_type(manifest, dict, "manifest")
  value = manifest
  _expect_keys(value, {
      "schema_version", "libjxl_revision", "profile", "frame", "encoder",
      "artifacts",
  }, "manifest")
  _expect_type(value["schema_version"], dict, "schema_version")
  version = value["schema_version"]
  _expect_keys(version, {"major", "minor"}, "schema_version")
  if version["major"] != SCHEMA_MAJOR:
    raise ManifestError(
        f"unsupported schema major {version['major']}; expected {SCHEMA_MAJOR}")
  if (not isinstance(version["minor"], int) or
      isinstance(version["minor"], bool) or version["minor"] < 0):
    raise ManifestError("schema_version.minor: expected nonnegative integer")

  _expect_type(value["libjxl_revision"], str, "libjxl_revision")
  _expect_type(value["profile"], str, "profile")
  _expect_type(value["frame"], dict, "frame")
  _expect_keys(value["frame"], {"xsize", "ysize"}, "frame")
  for key in ("xsize", "ysize"):
    dimension = value["frame"][key]
    if (not isinstance(dimension, int) or isinstance(dimension, bool) or
        dimension <= 0):
      raise ManifestError(f"frame.{key}: expected positive integer")
  _expect_type(value["encoder"], dict, "encoder")
  _expect_keys(value["encoder"], {
      "distance", "effort", "decoding_speed_tier", "resampling",
      "streaming_mode", "color_transform", "input_color_space",
      "intensity_target", "gaborish", "epf_iterations",
  }, "encoder")
  encoder = value["encoder"]
  _expect_number(encoder["distance"], "encoder.distance")
  if not isinstance(encoder["effort"], int) or isinstance(
      encoder["effort"], bool):
    raise ManifestError("encoder.effort: expected integer")
  for key in ("decoding_speed_tier", "color_transform", "input_color_space",
              "epf_iterations"):
    _expect_nonnegative_int(encoder[key], f"encoder.{key}")
  if (not isinstance(encoder["resampling"], int) or
      isinstance(encoder["resampling"], bool) or encoder["resampling"] < 1):
    raise ManifestError("encoder.resampling: expected positive integer")
  for key in ("streaming_mode", "gaborish"):
    _expect_type(encoder[key], bool, f"encoder.{key}")
  _expect_number(encoder["intensity_target"], "encoder.intensity_target")

  _expect_type(value["artifacts"], list, "artifacts")
  artifacts = [
      _validate_artifact(artifact, index)
      for index, artifact in enumerate(value["artifacts"])
  ]
  names = [artifact["name"] for artifact in artifacts]
  if len(names) != len(set(names)):
    raise ManifestError("artifact names are not unique")
  if names != sorted(names):
    raise ManifestError("artifacts are not sorted by name")
  return value


def load_manifest(dump_dir: str | Path) -> dict[str, Any]:
  """Loads and validates manifest.json from a dump directory."""
  path = Path(dump_dir) / "manifest.json"
  try:
    manifest = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ManifestError(f"failed to read {path}: {error}") from error
  return validate_manifest(manifest)


class DebugDump:
  """Validated debug dump with lazy, name-based NumPy array loading."""

  def __init__(self, dump_dir: str | Path):
    self.directory = Path(dump_dir)
    self.manifest = load_manifest(self.directory)
    self.artifacts = {
        artifact["name"]: artifact
        for artifact in self.manifest["artifacts"]
    }

  def load(self, name: str, mmap_mode: str | None = None):
    """Loads an artifact without applying normalization or color mapping."""
    try:
      import numpy as np
    except ImportError as error:
      raise RuntimeError("NumPy is required to load debug artifacts") from error
    if name not in self.artifacts:
      raise KeyError(name)
    artifact = self.artifacts[name]
    path = self.directory / PurePosixPath(artifact["path"])
    array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    expected_shape = tuple(artifact["shape"])
    if array.shape != expected_shape:
      raise ManifestError(
          f"{name}: expected shape {expected_shape}, got {array.shape}")
    if array.dtype.name != artifact["dtype"]:
      raise ManifestError(
          f"{name}: expected dtype {artifact['dtype']}, got {array.dtype.name}")
    if array.nbytes != artifact["bytes"]:
      raise ManifestError(
          f"{name}: expected {artifact['bytes']} bytes, got {array.nbytes}")
    return array

  def pixel_coordinates(self, name: str):
    """Returns 1D sample-center x/y coordinates in full-frame pixels."""
    try:
      import numpy as np
    except ImportError as error:
      raise RuntimeError("NumPy is required for coordinates") from error
    artifact = self.artifacts[name]
    if len(artifact["shape"]) < 2:
      raise ManifestError(f"{name}: artifact has no two-dimensional grid")
    grid = artifact["grid"]
    origin_x, origin_y = grid["origin_px"]
    spacing_x, spacing_y = grid["spacing_px"]
    footprint_x, footprint_y = grid["footprint_px"]
    ysize, xsize = artifact["shape"][-2:]
    x = origin_x + 0.5 * footprint_x + np.arange(xsize) * spacing_x
    y = origin_y + 0.5 * footprint_y + np.arange(ysize) * spacing_y
    return x, y


def parse_selections(values: list[str]) -> dict[str, int]:
  """Parses repeated AXIS=INDEX selections used for higher-rank tensors."""
  selections: dict[str, int] = {}
  for value in values:
    if "=" not in value:
      raise VisualizationError(
          f"invalid selection {value!r}; expected AXIS=INDEX")
    axis, index_string = value.split("=", 1)
    if not axis or axis in selections:
      raise VisualizationError(f"invalid or duplicate selection axis {axis!r}")
    try:
      selections[axis] = int(index_string)
    except ValueError as error:
      raise VisualizationError(
          f"selection index for {axis!r} must be an integer") from error
  return selections


def select_artifact(array: Any, artifact: dict[str, Any],
                    selections: dict[str, int],
                    channel: str | None = None) -> tuple[Any, list[str],
                                                         dict[str, int]]:
  """Applies named integer indexing without changing the stored array."""
  axes = list(artifact["axes"])
  unknown = selections.keys() - set(axes)
  if unknown:
    raise VisualizationError(
        f"unknown selection axes: {', '.join(sorted(unknown))}")
  chosen = dict(selections)
  if channel is not None:
    if "channel" not in axes:
      raise VisualizationError("--channel requires an artifact channel axis")
    if "channel" in chosen:
      raise VisualizationError("select channel with either --channel or "
                               "--select, not both")
    channel_axis = axes.index("channel")
    names = artifact.get("channel_names", [])
    try:
      channel_index = int(channel)
    except ValueError:
      if channel not in names:
        raise VisualizationError(
            f"unknown channel {channel!r}; choices are {names}")
      channel_index = names.index(channel)
    if not 0 <= channel_index < array.shape[channel_axis]:
      raise VisualizationError(
          f"channel index {channel_index} is outside axis size "
          f"{array.shape[channel_axis]}")
    chosen["channel"] = channel_index

  index: list[int | slice] = []
  remaining_axes: list[str] = []
  normalized: dict[str, int] = {}
  for axis, size in zip(axes, array.shape):
    if axis not in chosen:
      index.append(slice(None))
      remaining_axes.append(axis)
      continue
    coordinate = chosen[axis]
    if coordinate < 0:
      coordinate += size
    if not 0 <= coordinate < size:
      raise VisualizationError(
          f"selection {axis}={chosen[axis]} is outside axis size {size}")
    index.append(coordinate)
    normalized[axis] = coordinate
  return array[tuple(index)], remaining_axes, normalized


def _spatial_extent(
    artifact: dict[str, Any],
    shape: tuple[int, ...]) -> tuple[float, float, float, float]:
  grid = artifact["grid"]
  origin_x, origin_y = grid["origin_px"]
  spacing_x, spacing_y = grid["spacing_px"]
  footprint_x, footprint_y = grid["footprint_px"]
  ysize, xsize = shape[-2:]
  right = origin_x + (xsize - 1) * spacing_x + footprint_x
  bottom = origin_y + (ysize - 1) * spacing_y + footprint_y
  return float(origin_x), float(right), float(bottom), float(origin_y)


def _validate_spatial_axes(artifact: dict[str, Any], axes: list[str]) -> None:
  expected = artifact["axes"][-2:]
  if len(expected) != 2 or len(axes) < 2 or axes[-2:] != expected:
    raise VisualizationError(
        f"remaining axes {axes} do not preserve spatial axes {expected}; "
        "select only non-spatial tensor axes")


def _continuous_limits(array: Any, mapping: str, vmin: float | None,
                       vmax: float | None,
                       percentiles: tuple[float, float] | None
                       ) -> tuple[float, float]:
  import numpy as np
  usable = np.asarray(array)[np.isfinite(array)]
  if mapping == "log":
    usable = usable[usable > 0]
  if usable.size == 0:
    qualifier = "positive " if mapping == "log" else "finite "
    raise VisualizationError(f"artifact has no {qualifier}values to render")
  if percentiles is not None:
    low, high = percentiles
    if not 0 <= low < high <= 100:
      raise VisualizationError("percentiles must satisfy 0 <= LOW < HIGH <= 100")
    default_min, default_max = np.percentile(usable, [low, high])
  else:
    default_min = np.min(usable)
    default_max = np.max(usable)
  lower = float(default_min if vmin is None else vmin)
  upper = float(default_max if vmax is None else vmax)
  if not math.isfinite(lower) or not math.isfinite(upper):
    raise VisualizationError("render range must be finite")
  if mapping == "log" and lower <= 0:
    raise VisualizationError("log mapping requires a positive lower bound")
  if lower > upper:
    raise VisualizationError("render range lower bound exceeds upper bound")
  if lower == upper:
    delta = max(abs(lower), 1.0) * 1e-6
    lower -= delta
    upper += delta
    if mapping == "log" and lower <= 0:
      lower = max(upper * 1e-6, float(np.finfo(np.float32).tiny))
  return lower, upper


def _write_sidecar(output: Path, metadata: dict[str, Any]) -> Path:
  sidecar = output.with_suffix(output.suffix + ".json")
  sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
  return sidecar


def render_artifact(dump: DebugDump, name: str, output: str | Path,
                    selections: dict[str, int] | None = None,
                    channel: str | None = None, rgb: bool = False,
                    mapping: str = "auto", vmin: float | None = None,
                    vmax: float | None = None,
                    percentiles: tuple[float, float] | None = None,
                    linthresh: float = 1e-3, cmap: str | None = None,
                    colorbar: bool = True, legend: bool = True,
                    overlay: str | Path | None = None,
                    overlay_alpha: float = 0.45,
                    title: str | None = None) -> Path:
  """Renders one selected artifact to PNG and records the mapping sidecar."""
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as colors
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
  except ImportError as error:
    raise RuntimeError(
        "Matplotlib and NumPy are required for PNG rendering") from error

  if name not in dump.artifacts:
    raise KeyError(name)
  artifact = dump.artifacts[name]
  selected, axes, normalized_selections = select_artifact(
      dump.load(name), artifact, selections or {}, channel)
  _validate_spatial_axes(artifact, axes)
  output_path = Path(output)
  if output_path.suffix.lower() != ".png":
    raise VisualizationError("render output must use a .png extension")
  if not 0 <= overlay_alpha <= 1:
    raise VisualizationError("overlay alpha must be between zero and one")

  channel_axis = axes.index("channel") if "channel" in axes else None
  if rgb:
    if selected.ndim != 3 or channel_axis is None or \
        selected.shape[channel_axis] != 3:
      raise VisualizationError(
          "--rgb requires exactly three remaining values on a channel axis")
    display = np.moveaxis(np.asarray(selected, dtype=np.float64),
                          channel_axis, -1)
    spatial_shape = display.shape[:2]
  else:
    if selected.ndim != 2:
      raise VisualizationError(
          f"rendering requires a 2D field after selection; remaining axes are "
          f"{axes}. Use --select or --channel, or --rgb for three channels")
    display = np.asarray(selected)
    spatial_shape = display.shape

  categories = artifact.get("categories", [])
  selected_mapping = mapping
  if mapping == "auto":
    selected_mapping = "categorical" if categories and not rgb else "linear"
  if selected_mapping not in {"linear", "log", "symlog", "categorical"}:
    raise VisualizationError(f"unknown mapping {selected_mapping!r}")
  if rgb and selected_mapping == "categorical":
    raise VisualizationError("categorical mapping cannot render an RGB image")
  if selected_mapping == "symlog" and linthresh <= 0:
    raise VisualizationError("symmetric-log linthresh must be positive")
  if (selected_mapping == "categorical" and
      (vmin is not None or vmax is not None or percentiles is not None)):
    raise VisualizationError(
        "range and percentile options do not apply to categorical mapping")

  source = None
  if overlay is not None:
    try:
      from PIL import Image, ImageOps
    except ImportError as error:
      raise RuntimeError(
          "Pillow is required for source-image overlays") from error
    with Image.open(overlay) as image:
      source = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    frame = dump.manifest["frame"]
    if source.shape[:2] != (frame["ysize"], frame["xsize"]):
      raise VisualizationError(
          f"overlay is {source.shape[1]}x{source.shape[0]}, expected "
          f"{frame['xsize']}x{frame['ysize']}")

  figure, axis = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
  frame = dump.manifest["frame"]
  if source is not None:
    axis.imshow(source, extent=(0, frame["xsize"], frame["ysize"], 0),
                interpolation="nearest")
  extent = _spatial_extent(artifact, tuple(spatial_shape))
  alpha = overlay_alpha if source is not None else 1.0
  render_metadata: dict[str, Any] = {
      "artifact": name,
      "source_dtype": artifact["dtype"],
      "source_shape": artifact["shape"],
      "selected_axes": axes,
      "selections": normalized_selections,
      "mapping": selected_mapping,
      "grid_extent_px": list(extent),
      "colormap": cmap,
      "colorbar": colorbar,
      "legend": legend,
      "overlay": str(overlay) if overlay is not None else None,
      "overlay_alpha": alpha,
  }

  if selected_mapping == "categorical":
    finite_values = np.asarray(display)[np.isfinite(display)]
    unique_values = sorted(int(value) for value in np.unique(finite_values))
    category_by_value = {int(item["value"]): item for item in categories}
    labels = [
        category_by_value.get(value, {"name": f"unknown ({value})"})["name"]
        for value in unique_values
    ]
    indexed = np.full(display.shape, np.nan, dtype=np.float64)
    for index, value in enumerate(unique_values):
      indexed[display == value] = index
    palette_name = cmap or "tab20"
    palette = plt.colormaps[palette_name].resampled(max(len(unique_values), 1))
    axis.imshow(indexed, origin="upper", extent=extent,
                interpolation="nearest", cmap=palette,
                vmin=-0.5, vmax=max(len(unique_values) - 0.5, 0.5),
                alpha=alpha)
    if legend and unique_values:
      handles = [
          patches.Patch(color=palette(index / max(len(unique_values) - 1, 1)),
                        label=f"{value}: {label}")
          for index, (value, label) in enumerate(zip(unique_values, labels))
      ]
      axis.legend(handles=handles, title=artifact["units"], loc="upper left",
                  bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    render_metadata["categories"] = [
        {"value": value, "label": label}
        for value, label in zip(unique_values, labels)
    ]
    render_metadata["colormap"] = palette_name
  else:
    lower, upper = _continuous_limits(display, selected_mapping, vmin, vmax,
                                      percentiles)
    if selected_mapping == "linear":
      norm = colors.Normalize(vmin=lower, vmax=upper, clip=False)
    elif selected_mapping == "log":
      norm = colors.LogNorm(vmin=lower, vmax=upper, clip=False)
    else:
      norm = colors.SymLogNorm(linthresh=linthresh, vmin=lower, vmax=upper,
                               clip=False)
    masked = np.ma.masked_invalid(display)
    if selected_mapping == "log":
      masked = np.ma.masked_less_equal(masked, 0)
    if rgb:
      axis.imshow(norm(masked).filled(0), origin="upper", extent=extent,
                  interpolation="nearest", alpha=alpha)
    else:
      palette_name = cmap or "viridis"
      image = axis.imshow(masked, origin="upper", extent=extent,
                          interpolation="nearest", cmap=palette_name,
                          norm=norm, alpha=alpha)
      if colorbar:
        bar = figure.colorbar(image, ax=axis)
        bar.set_label(artifact["units"])
      render_metadata["colormap"] = palette_name
    render_metadata["range"] = [lower, upper]
    render_metadata["percentiles"] = (list(percentiles)
                                        if percentiles is not None else None)
    if selected_mapping == "symlog":
      render_metadata["linthresh"] = linthresh

  axis.set_title(title or name)
  axis.set_xlabel("full-frame x (pixels)")
  axis.set_ylabel("full-frame y (pixels)")
  axis.set_xlim(0, frame["xsize"])
  axis.set_ylim(frame["ysize"], 0)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=150)
  plt.close(figure)
  _write_sidecar(output_path, render_metadata)
  return output_path


def export_exr(dump: DebugDump, name: str, output: str | Path,
               selections: dict[str, int] | None = None,
               channel: str | None = None) -> Path:
  """Exports a 2D field or channel-first image to float32 OpenEXR."""
  import numpy as np
  if name not in dump.artifacts:
    raise KeyError(name)
  artifact = dump.artifacts[name]
  selected, axes, normalized_selections = select_artifact(
      dump.load(name), artifact, selections or {}, channel)
  _validate_spatial_axes(artifact, axes)
  output_path = Path(output)
  if output_path.suffix.lower() != ".exr":
    raise VisualizationError("EXR output must use a .exr extension")

  if selected.ndim == 2:
    channels = np.asarray(selected, dtype="<f4")[None, :, :]
    if "channel" in normalized_selections and artifact.get("channel_names"):
      channel_names = [artifact["channel_names"][
          normalized_selections["channel"]]]
    else:
      channel_names = ["Y"]
  elif selected.ndim == 3 and "channel" in axes:
    channel_axis = axes.index("channel")
    channels = np.asarray(np.moveaxis(selected, channel_axis, 0), dtype="<f4")
    original_names = artifact.get("channel_names", [])
    channel_names = (list(original_names) if len(original_names) == len(channels)
                     else [f"channel_{index}" for index in range(len(channels))])
  else:
    raise VisualizationError(
        f"EXR export requires a 2D field or one channel axis; remaining axes "
        f"are {axes}. Use --select to choose tensor indices")

  output_path.parent.mkdir(parents=True, exist_ok=True)
  backend = ""
  try:
    import Imath
    import OpenEXR
    header = OpenEXR.Header(channels.shape[2], channels.shape[1])
    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
    header["channels"] = {
        str(channel_name): Imath.Channel(pixel_type)
        for channel_name in channel_names
    }
    writer = OpenEXR.OutputFile(str(output_path), header)
    try:
      writer.writePixels({
          str(channel_name): np.ascontiguousarray(channel_values).tobytes()
          for channel_name, channel_values in zip(channel_names, channels)
      })
    finally:
      writer.close()
    backend = "OpenEXR"
  except ImportError:
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
      import cv2
    except ImportError as error:
      raise RuntimeError(
          "EXR export requires the OpenEXR Python bindings or OpenCV with "
          "OpenEXR support") from error
    if len(channels) == 1:
      cv_image = channels[0]
      file_channels = ["Y"]
    elif len(channels) == 3:
      cv_image = np.moveaxis(channels[[2, 1, 0]], 0, -1)
      file_channels = ["R", "G", "B"]
    elif len(channels) == 4:
      cv_image = np.moveaxis(channels[[2, 1, 0, 3]], 0, -1)
      file_channels = ["R", "G", "B", "A"]
    else:
      raise RuntimeError(
          "OpenCV EXR fallback supports one, three, or four channels; install "
          "the OpenEXR bindings for arbitrary channel counts")
    try:
      written = cv2.imwrite(str(output_path), cv_image)
    except cv2.error as error:
      raise RuntimeError(f"OpenCV failed to write {output_path}: {error}") \
          from error
    if not written:
      raise RuntimeError(f"OpenCV failed to write {output_path}")
    backend = "OpenCV"
    # OpenCV names one-channel files Y and three-channel files RGB while
    # accepting BGR memory order.
    channel_names = [
        f"{file_channel}={source_channel}"
        for file_channel, source_channel in zip(file_channels, channel_names)
    ]

  _write_sidecar(output_path, {
      "artifact": name,
      "backend": backend,
      "channels": channel_names,
      "selections": normalized_selections,
      "source_dtype": artifact["dtype"],
      "source_shape": artifact["shape"],
      "stored_dtype": "float32",
  })
  return output_path


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dump_dir", type=Path)
  parser.add_argument("--list", action="store_true",
                      help="list validated artifact names")
  parser.add_argument("--render", metavar="ARTIFACT",
                      help="render or export one named artifact")
  parser.add_argument("--output", type=Path,
                      help="output .png preview or raw float .exr")
  parser.add_argument("--select", action="append", default=[],
                      metavar="AXIS=INDEX",
                      help="select a tensor index; may be repeated")
  parser.add_argument("--channel",
                      help="select a channel by name or integer index")
  parser.add_argument("--rgb", action="store_true",
                      help="render three remaining channels as RGB")
  parser.add_argument("--mapping",
                      choices=("auto", "linear", "log", "symlog",
                               "categorical"),
                      help="PNG value mapping (default: metadata-aware auto)")
  parser.add_argument("--range", nargs=2, type=float,
                      metavar=("MIN", "MAX"), dest="value_range",
                      help="explicit PNG mapping range")
  parser.add_argument("--percentile", nargs=2, type=float,
                      metavar=("LOW", "HIGH"),
                      help="derive missing PNG range bounds from percentiles")
  parser.add_argument("--linthresh", type=float, default=1e-3,
                      help="linear threshold for symlog mapping")
  parser.add_argument("--cmap", help="Matplotlib colormap name")
  parser.add_argument("--no-colorbar", action="store_true",
                      help="omit the continuous-value color bar")
  parser.add_argument("--no-legend", action="store_true",
                      help="omit the categorical legend")
  parser.add_argument("--overlay", type=Path,
                      help="source image to place beneath the artifact")
  parser.add_argument("--overlay-alpha", type=float, default=0.45,
                      help="artifact opacity for an overlay (default: 0.45)")
  parser.add_argument("--title", help="override the PNG title")
  args = parser.parse_args(argv)
  try:
    dump = DebugDump(args.dump_dir)
    print(f"valid encoder debug dump: {len(dump.artifacts)} artifacts")
    if args.list:
      for name in dump.artifacts:
        print(name)
    if (args.render is None) != (args.output is None):
      parser.error("--render and --output must be provided together")
    if args.render is not None:
      selections = parse_selections(args.select)
      if args.output.suffix.lower() == ".png":
        lower = args.value_range[0] if args.value_range else None
        upper = args.value_range[1] if args.value_range else None
        render_artifact(
            dump, args.render, args.output, selections=selections,
            channel=args.channel, rgb=args.rgb,
            mapping=args.mapping or "auto", vmin=lower, vmax=upper,
            percentiles=(tuple(args.percentile)
                         if args.percentile is not None else None),
            linthresh=args.linthresh, cmap=args.cmap,
            colorbar=not args.no_colorbar, legend=not args.no_legend,
            overlay=args.overlay, overlay_alpha=args.overlay_alpha,
            title=args.title)
      elif args.output.suffix.lower() == ".exr":
        if (args.mapping is not None or args.value_range is not None or
            args.percentile is not None or args.cmap is not None or
            args.overlay is not None or args.rgb or args.title is not None):
          raise VisualizationError(
              "EXR is a raw float export; PNG mapping, RGB, title, and overlay "
              "options do not apply")
        export_exr(dump, args.render, args.output, selections=selections,
                   channel=args.channel)
      else:
        raise VisualizationError("--output must end in .png or .exr")
      print(f"wrote {args.output} and {args.output}.json")
  except (KeyError, OSError, RuntimeError, ValueError) as error:
    print(f"encoder debug-data operation failed: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
