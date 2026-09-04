#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Validate and load raw libjxl encoder debug-data dumps."""

from __future__ import annotations

import argparse
import json
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


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("dump_dir", type=Path)
  parser.add_argument("--list", action="store_true",
                      help="list validated artifact names")
  args = parser.parse_args(argv)
  try:
    dump = DebugDump(args.dump_dir)
    print(f"valid encoder debug dump: {len(dump.artifacts)} artifacts")
    if args.list:
      for name in dump.artifacts:
        print(name)
  except (ManifestError, OSError) as error:
    print(f"encoder debug dump validation failed: {error}", file=sys.stderr)
    return 1
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv[1:]))
