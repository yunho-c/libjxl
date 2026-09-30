#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Common-cohort throughput regressions using synthetic saved timings."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cjxl_quality_characterization as study
import cjxl_throughput_table as throughput


class ThroughputTableTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        images = [dict(image_id=name, corpus="fixture", resolution_class=resolution,
                       width=1000, height=mp * 1000, pixels=mp * 1000000,
                       pfm_sha256=name + "-reference", color_encoding="linear-sRGB")
                  for name, mp, resolution in (("a", 1, "small"), ("b", 48, "48mp"),
                                                ("c", 2, "small"), ("d", 3, "small"))]
        self.studies = []
        for codec in ("libjxl", "gjxl"):
            config = dict(images=copy.deepcopy(images), efforts=[1, 2], num_threads=8,
                          repetitions=2, warmups=1, configuration_id=codec + "-config",
                          encoder_revision=codec + "-revision", tool_hashes={})
            if codec == "gjxl":
                config.update(backend="cuda", gpu_aq_mode="fully-resident")
            rows = []
            for image in images:
                for effort in config["efforts"]:
                    for quality in (30, 50):
                        job = study.fixed_key(image["image_id"], quality, effort)
                        for repetition in range(config["repetitions"]):
                            # CUDA has an absent setting; CPU has a partial one.
                            if (codec, image["image_id"], effort, quality) == ("gjxl", "b", 2, 50):
                                continue
                            if (codec, image["image_id"], effort, quality, repetition) == ("libjxl", "c", 2, 30, 1):
                                continue
                            seconds = {"a": 1, "b": 100, "c": 10, "d": 3}[image["image_id"]]
                            if codec == "gjxl":
                                seconds /= 2
                            rows.append({
                                **{k: image[k] for k in (*study.SOURCE_FIELDS, "pixels")},
                                "effort": effort, "quality": quality, "repetition": repetition,
                                "job_id": job, "sample_id": f"{job}|repetition={repetition}",
                                "harness_revision": config["encoder_revision"], "thread_count": 8,
                                "distance": {30: 4., 50: 2.}[quality], "encoder": codec,
                                "configuration_id": config["configuration_id"],
                                "reference_sha256": image["pfm_sha256"], "resampling": 1,
                                "backend": "cuda", "gpu_aq_mode": "fully-resident",
                                "elapsed_nanoseconds": seconds * 1e9,
                                "output_sha256": job, "encoded_bytes": 100,
                            })
            run = self.root / codec
            self.studies.append(dict(codec=codec, config=config, records=rows,
                                     qualities=[30, 50], distances={"30": 4., "50": 2.},
                                     run=str(run), metadata_sha256="metadata",
                                     ledger=dict(path=str(run / "timings.jsonl"))))
        self.original = copy.deepcopy(self.studies)

    def build(self, **kwargs):
        with mock.patch.object(throughput, "_load_runs", return_value=(self.studies, {})):
            return throughput.build_tables("cpu", "gpu", qualities=(30, 50), **kwargs)

    def test_strict_default_retains_missing_cells_and_full_cohort(self):
        result = self.build()
        table = result["table"].set_index("effort")
        self.assertTrue(table.loc[1].notna().all())
        self.assertTrue(table.loc[2].isna().all())
        self.assertEqual(len(result["methodology"]["images"]), 4)
        self.assertFalse(result["methodology"]["missing_ok"])
        self.assertEqual(result["coverage"].query("status == 'incomplete'").shape[0], 2)

    def test_common_cohort_is_fixed_across_encoders_efforts_and_qualities(self):
        result = self.build(missing_ok=True)
        self.assertEqual(set(result["tuples"]["image_id"]), {"a", "d"})
        self.assertEqual(len(result["tuples"]), 2 * 2 * 2 * 2)
        self.assertTrue(result["tuples"]["timing_complete"].all())
        self.assertTrue(result["coverage"]["expected_images"].eq(2).all())
        self.assertTrue(result["coverage"]["missing_samples"].eq(0).all())
        self.assertEqual(set(result["by_resolution"]["resolution_class"]), {"small"})
        for row in result["table"].to_dict("records"):
            self.assertAlmostEqual(row["libjxl_mp_s"], 1.)
            self.assertAlmostEqual(row["gjxl_mp_s"], 2.)
            self.assertAlmostEqual(row["gjxl_over_libjxl"], 2.)
        cohort = result["methodology"]["cohort_selection"]
        self.assertEqual(cohort["requested_image_count"], 4)
        self.assertEqual(cohort["included_image_count"], 2)
        exclusions = {row["image_id"]: row for row in cohort["excluded_images"]}
        self.assertEqual(exclusions["b"]["incomplete_settings"], [dict(
            encoder="gjxl", effort=2, quality=50, timing_sample_count=0, required_samples=2)])
        self.assertEqual(exclusions["c"]["incomplete_settings"], [dict(
            encoder="libjxl", effort=2, quality=30, timing_sample_count=1, required_samples=2)])
        self.assertEqual(self.studies, self.original)
        paths = throughput.write_tables(result, self.root / "analysis")
        saved = json.loads(Path(paths["methodology"]).read_text())
        self.assertEqual(saved["cohort_selection"], cohort)
        self.assertIn("2/4 eligible images", saved["caption"])

    def test_only_selected_settings_determine_intersection(self):
        result = self.build(missing_ok=True, efforts=[1])
        self.assertEqual(len(result["methodology"]["images"]), 4)
        self.assertEqual(result["methodology"]["cohort_selection"]["excluded_images"], [])
        result = self.build(missing_ok=True, image_ids=["a", "b"])
        self.assertEqual(result["methodology"]["cohort_selection"]["included_image_ids"], ["a"])

    def test_empty_intersection_raises(self):
        with self.assertRaisesRegex(ValueError, "no fully timed images"):
            self.build(missing_ok=True, image_ids=["b", "c"])

    def test_invalid_record_in_excluded_image_still_raises(self):
        row = next(row for row in self.studies[1]["records"] if row["image_id"] == "b")
        row["elapsed_nanoseconds"] = -1
        with self.assertRaisesRegex(ValueError, "Invalid complete-encode duration"):
            self.build(missing_ok=True)

    def test_reference_mismatch_still_raises(self):
        self.studies[1]["config"]["images"][1]["pfm_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Reference or geometry mismatch"):
            self.build(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
