#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Saved-data CUDA analysis regressions; no benchmarks or GPU calls."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

import cjxl_cuda_analysis as analysis
import cjxl_quality_characterization as quality
import cjxl_runtime_characterization_notebook as notebook


class CudaAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def frame(self, images=("a", "b")):
        frame = pd.DataFrame([dict(image_id=image, resolution_class="small", megapixels=1.,
                                   effort=1, quality=q) for image in images for q in (30, 50)])
        frame.attrs["expected_images"] = [dict(image_id="a", resolution_class="small", width=1000, height=1000),
                                         dict(image_id="b", resolution_class="small", width=2000, height=1000),
                                         dict(image_id="c", resolution_class="48mp", width=8000, height=6000)]
        cells = pd.DataFrame([dict(resolution_class="small", effort=1, quality=q,
                                   runtime_ms_per_mp=value, timing_complete=True)
                              for q, value in ((30, 10.), (50, 20.))])
        return frame, cells

    def test_entirely_missing_resolution_is_retained(self):
        frame, cells = self.frame()
        rows = notebook.resolution_throughput_rows(frame, cells, effort=1, qualities=(30, 50)).set_index("resolution_class")
        self.assertAlmostEqual(rows.loc["small", "throughput_mp_s"], 75.)
        self.assertEqual(rows.loc["small", "mean_megapixels"], 1.5)
        self.assertEqual(rows.loc["48mp", "image_count"], 1)
        self.assertFalse(rows.loc["48mp", "timing_complete"])
        self.assertTrue(math.isnan(rows.loc["48mp", "throughput_mp_s"]))

    def test_partial_cohort_cannot_become_a_complete_throughput(self):
        frame, cells = self.frame(images=("a",))
        rows = notebook.resolution_throughput_rows(frame, cells, effort=1, qualities=(30, 50)).set_index("resolution_class")
        self.assertFalse(rows.loc["small", "timing_complete"])
        self.assertTrue(math.isnan(rows.loc["small", "throughput_mp_s"]))
        self.assertEqual(rows.loc["small", "image_count"], 2)
        self.assertEqual(rows.loc["small", "mean_megapixels"], 1.5)

    def test_missing_quality_is_not_averaged_away(self):
        frame, cells = self.frame()
        rows = notebook.resolution_throughput_rows(frame, cells.iloc[:1], effort=1, qualities=(30, 50))
        self.assertFalse(rows["timing_complete"].any())

    def test_backend_labels_and_mixed_backend_rejection(self):
        self.assertEqual(notebook.encoder_label(pd.DataFrame([dict(encoder="gjxl", backend="cuda")])),
                         "gjxl (fully-resident CUDA)")
        self.assertEqual(notebook.encoder_label(pd.DataFrame([dict(encoder="gjxl")])),
                         "gjxl (fully-resident Metal)")
        with self.assertRaises(ValueError):
            notebook.encoder_label(pd.DataFrame([dict(encoder="gjxl", backend=b) for b in ("cuda", "metal")]))
        with self.assertRaises(ValueError):
            notebook.encoder_label(pd.DataFrame([dict(encoder=e) for e in ("libjxl", "gjxl")]))

    def test_fixed_coverage_includes_failures_and_not_yet_attempted_settings(self):
        config = dict(collection_mode="fixed", images=[dict(image_id="a", resolution_class="48mp", width=8000, height=6000)],
                      efforts=[1, 5, 7], qualities=[80])
        key = lambda e: quality.fixed_key("a", 80, e)
        timings = [dict(job_id=key(1), timing_complete=True, timing_sample_count=5)]
        records = {"scores":[dict(source_job_id=key(1))],
                   "failures":[dict(job_id=key(5), phase="fixed-timing", status="cuda-out-of-memory")]}
        with mock.patch.object(notebook, "load_quality_helpers", return_value=quality), \
             mock.patch.object(quality, "fixed_rows", return_value=timings), \
             mock.patch.object(quality, "read_records", side_effect=lambda run, name, config: records[name]):
            rows = notebook.cuda_coverage_rows(self.root, config)
        self.assertEqual([row["status"] for row in rows], ["complete", "cuda-out-of-memory", "incomplete"])
        self.assertEqual([row["timing_samples"] for row in rows], [5, 0, 0])

    def test_discovery_uses_initialized_studies_not_directory_names(self):
        older = self.root / "gjxl-cuda-study-old"
        newer = self.root / "gjxl-cuda-integration-study-new"
        for root, stamp in ((older, 100), (newer, 200)):
            (root / "fixed").mkdir(parents=True)
            path = root / "fixed/metadata.json"
            path.write_text('{}')
            os.utime(path, (stamp, stamp))
        (self.root / "gjxl-cuda-study-uninitialized").mkdir()
        self.assertEqual(notebook.discover_cuda_study(self.root), newer)
        (newer / "ARCHIVED.md").write_text("Wrong corpus; retained as historical evidence.")
        self.assertEqual(notebook.discover_cuda_study(self.root), older)
        (older / "ARCHIVED.md").write_text("Archived")
        self.assertEqual(notebook.discover_cuda_study(self.root), self.root / "gjxl-cuda-study")

    def test_percent_conversion_retains_upstream_cells(self):
        cells = analysis.notebook_cells("# %%\nx = 1\n# %% [markdown]\n# Text\n# %%\ny = 2\n")
        self.assertEqual([cell.cell_type for cell in cells], ["code", "code", "markdown", "code"])
        self.assertEqual(cells[-1].source, "y = 2")

    def test_run_all_requires_completed_matching_audit(self):
        records = {mode: dict(configuration_id=mode, coverage=dict(collection_finished=True))
                   for mode in ("fixed", "calibrated")}
        verified = {mode: dict(**row, accepted_results_valid=True) for mode, row in records.items()}
        path = self.root / "verification.json"
        path.write_text(json.dumps(verified))
        self.assertEqual(analysis.verified_studies(self.root, records), verified)
        records["fixed"]["configuration_id"] = "another-run"
        with self.assertRaises(ValueError):
            analysis.verified_studies(self.root, records)

    def test_render_comparison_ignores_svg_metadata_but_checks_png_bytes(self):
        before = dict(coverage_rows=2, figure_count=1, artifact_sha256={"plot.png":"pixels", "plot.svg":"date-one"})
        after = dict(coverage_rows=2, figure_count=1, artifact_sha256={"plot.png":"pixels", "plot.svg":"date-two"})
        analysis.verify_same_pixels(before, after)
        after["artifact_sha256"]["plot.png"] = "different-pixels"
        with self.assertRaises(ValueError):
            analysis.verify_same_pixels(before, after)


if __name__ == "__main__":
    unittest.main()
