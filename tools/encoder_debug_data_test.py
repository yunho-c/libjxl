#!/usr/bin/env python3

# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Tests for encoder_debug_data.py and its manifest schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import encoder_debug_data


def make_manifest(file_bytes: int) -> dict[str, object]:
  return {
      "schema_version": {"major": 1, "minor": 0},
      "libjxl_revision": "test",
      "profile": "phase1-test",
      "frame": {"xsize": 3, "ysize": 2},
      "encoder": {
          "distance": 1.0,
          "effort": 7,
          "decoding_speed_tier": 0,
          "resampling": 1,
          "streaming_mode": False,
          "color_transform": 0,
          "input_color_space": 0,
          "intensity_target": 255.0,
          "gaborish": True,
          "epf_iterations": 2,
      },
      "artifacts": [{
          "name": "aq/initial/quant_field",
          "path": "aq/initial/quant_field.npy",
          "stage": "adaptive_quantization",
          "dtype": "float32",
          "shape": [2, 3],
          "axes": ["block_y", "block_x"],
          "grid": {
              "kind": "block",
              "origin_px": [0, 0],
              "spacing_px": [8, 8],
              "footprint_px": [8, 8],
              "valid_rect_px": [0, 0, 3, 2],
              "padded_rect_px": [0, 0, 8, 8],
              "value_is_block_anchor": False,
          },
          "units": "relative_inverse_quantization_step",
          "semantic": "test field",
          "frame_index": 0,
          "bytes": 24,
          "file_bytes": file_bytes,
      }],
  }


class EncoderDebugDataTest(unittest.TestCase):

  def test_loads_raw_array_and_coordinates(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_string:
      temp_dir = Path(temp_dir_string)
      array_path = temp_dir / "aq" / "initial" / "quant_field.npy"
      array_path.parent.mkdir(parents=True)
      expected = np.arange(6, dtype=np.float32).reshape(2, 3)
      np.save(array_path, expected)
      manifest = make_manifest(array_path.stat().st_size)
      (temp_dir / "manifest.json").write_text(
          json.dumps(manifest), encoding="utf-8")

      dump = encoder_debug_data.DebugDump(temp_dir)
      np.testing.assert_array_equal(
          expected, dump.load("aq/initial/quant_field"))
      x, y = dump.pixel_coordinates("aq/initial/quant_field")
      np.testing.assert_array_equal([4.0, 12.0, 20.0], x)
      np.testing.assert_array_equal([4.0, 12.0], y)

  def test_rejects_unknown_major_and_unsorted_artifacts(self) -> None:
    manifest = make_manifest(128)
    unsupported = copy.deepcopy(manifest)
    unsupported["schema_version"]["major"] = 2
    with self.assertRaisesRegex(encoder_debug_data.ManifestError,
                                "unsupported schema major"):
      encoder_debug_data.validate_manifest(unsupported)

    unsorted = copy.deepcopy(manifest)
    second = copy.deepcopy(unsorted["artifacts"][0])
    second["name"] = "ac/final/strategy_id"
    second["path"] = "ac/final/strategy_id.npy"
    unsorted["artifacts"].append(second)
    with self.assertRaisesRegex(encoder_debug_data.ManifestError,
                                "not sorted"):
      encoder_debug_data.validate_manifest(unsorted)

  def test_manifest_matches_json_schema(self) -> None:
    try:
      import jsonschema
    except ImportError:
      self.skipTest("jsonschema package is not installed")
    schema_path = Path(__file__).with_name("encoder_debug_data_schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(make_manifest(128))


if __name__ == "__main__":
  unittest.main()
