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

  def test_renders_mapped_overlay_and_records_sidecar(self) -> None:
    try:
      import matplotlib  # pylint: disable=unused-import,import-outside-toplevel
      from PIL import Image  # pylint: disable=import-outside-toplevel
    except ImportError:
      self.skipTest("Matplotlib and Pillow are not installed")
    with tempfile.TemporaryDirectory() as temp_dir_string:
      temp_dir = Path(temp_dir_string)
      array_path = temp_dir / "aq" / "initial" / "quant_field.npy"
      array_path.parent.mkdir(parents=True)
      values = np.asarray([[-2.0, -0.5, 0.0], [0.25, 1.0, 3.0]],
                          dtype=np.float32)
      np.save(array_path, values)
      manifest = make_manifest(array_path.stat().st_size)
      manifest["artifacts"][0]["grid"] = {
          "kind": "pixel",
          "origin_px": [0, 0],
          "spacing_px": [1, 1],
          "footprint_px": [1, 1],
          "valid_rect_px": [0, 0, 3, 2],
          "padded_rect_px": [0, 0, 3, 2],
          "value_is_block_anchor": False,
      }
      (temp_dir / "manifest.json").write_text(
          json.dumps(manifest), encoding="utf-8")
      source_path = temp_dir / "source.png"
      Image.new("RGB", (3, 2), color=(64, 96, 128)).save(source_path)

      dump = encoder_debug_data.DebugDump(temp_dir)
      output = temp_dir / "preview.png"
      encoder_debug_data.render_artifact(
          dump, "aq/initial/quant_field", output, mapping="symlog",
          vmin=-2.0, vmax=3.0, linthresh=0.1, overlay=source_path,
          overlay_alpha=0.4)
      self.assertEqual(b"\x89PNG\r\n\x1a\n", output.read_bytes()[:8])
      sidecar = json.loads(
          output.with_suffix(".png.json").read_text(encoding="utf-8"))
      self.assertEqual("symlog", sidecar["mapping"])
      self.assertEqual([-2.0, 3.0], sidecar["range"])
      self.assertEqual(0.1, sidecar["linthresh"])
      self.assertEqual(str(source_path), sidecar["overlay"])
      np.testing.assert_array_equal(values, dump.load("aq/initial/quant_field"))

  def test_renders_categorical_legend_and_named_selection(self) -> None:
    try:
      import matplotlib  # pylint: disable=unused-import,import-outside-toplevel
    except ImportError:
      self.skipTest("Matplotlib is not installed")
    with tempfile.TemporaryDirectory() as temp_dir_string:
      temp_dir = Path(temp_dir_string)
      array_path = temp_dir / "ac" / "final" / "strategy_id.npy"
      array_path.parent.mkdir(parents=True)
      values = np.asarray([[0, 1, 1], [0, 0, 1]], dtype=np.uint8)
      np.save(array_path, values)
      manifest = make_manifest(array_path.stat().st_size)
      artifact = manifest["artifacts"][0]
      artifact.update({
          "name": "ac/final/strategy_id",
          "path": "ac/final/strategy_id.npy",
          "stage": "ac_strategy_selection",
          "dtype": "uint8",
          "units": "enum",
          "bytes": 6,
          "categories": [
              {"value": 0, "name": "DCT", "covered_blocks": [1, 1]},
              {"value": 1, "name": "IDENTITY", "covered_blocks": [1, 1]},
          ],
      })
      (temp_dir / "manifest.json").write_text(
          json.dumps(manifest), encoding="utf-8")

      dump = encoder_debug_data.DebugDump(temp_dir)
      output = temp_dir / "strategies.png"
      encoder_debug_data.render_artifact(
          dump, "ac/final/strategy_id", output, mapping="auto")
      sidecar = json.loads(
          output.with_suffix(".png.json").read_text(encoding="utf-8"))
      self.assertEqual("categorical", sidecar["mapping"])
      self.assertEqual(
          [{"label": "DCT", "value": 0},
           {"label": "IDENTITY", "value": 1}], sidecar["categories"])

      tensor = np.arange(12).reshape(2, 2, 3)
      selected, axes, choices = encoder_debug_data.select_artifact(
          tensor, {"axes": ["phase", "block_y", "block_x"]},
          {"phase": 1})
      np.testing.assert_array_equal(tensor[1], selected)
      self.assertEqual(["block_y", "block_x"], axes)
      self.assertEqual({"phase": 1}, choices)

  def test_exports_float_exr_when_backend_is_available(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir_string:
      temp_dir = Path(temp_dir_string)
      array_path = temp_dir / "aq" / "initial" / "quant_field.npy"
      array_path.parent.mkdir(parents=True)
      values = np.arange(6, dtype=np.float32).reshape(2, 3)
      np.save(array_path, values)
      manifest = make_manifest(array_path.stat().st_size)
      (temp_dir / "manifest.json").write_text(
          json.dumps(manifest), encoding="utf-8")
      dump = encoder_debug_data.DebugDump(temp_dir)
      output = temp_dir / "field.exr"
      try:
        encoder_debug_data.export_exr(
            dump, "aq/initial/quant_field", output)
      except RuntimeError as error:
        self.skipTest(str(error))
      self.assertEqual(b"v/1\x01", output.read_bytes()[:4])
      sidecar = json.loads(
          output.with_suffix(".exr.json").read_text(encoding="utf-8"))
      self.assertEqual("float32", sidecar["stored_dtype"])
      self.assertIn(sidecar["backend"], ("OpenEXR", "OpenCV"))
      if sidecar["backend"] == "OpenCV":
        import cv2  # pylint: disable=import-outside-toplevel
        decoded = cv2.imread(str(output), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(values, decoded)


if __name__ == "__main__":
  unittest.main()
