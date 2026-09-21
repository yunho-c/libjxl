#!/usr/bin/env python3
"""Regression checks for paper regrouping and equal-image attribution."""
import unittest

import pandas as pd

import cjxl_gjxl_paper_breakdown as paper


def fixture():
    flat, gpu = [], []
    for resolution in paper.PANELS:
        for image, repetitions, host_ms, gpu_ms in (("a", 1, 10, 10), ("b", 3, 30, 0)):
            for sample in range(repetitions):
                key = dict(resolution_class=resolution, image_id=image, setting="Q80",
                           effort=5, sample_index=sample)
                flat.extend([dict(key, kind="flat", stage="Input color transform", ms=host_ms),
                             dict(key, kind="flat", stage="GPU: Other GPU stages", ms=gpu_ms)])
                if gpu_ms:
                    gpu.append(dict(key, stage="aq.gaborish", ms=gpu_ms))
    return dict(manifest={"efforts": [5]}, samples=pd.DataFrame(flat),
                gpu_stages=pd.DataFrame(gpu), coverage=pd.DataFrame([
                    dict(kind="flat", resolution_class=r, effort=5, status="complete")
                    for r in paper.PANELS]))


class PaperBreakdownTest(unittest.TestCase):
    def test_absent_stage_is_zero_before_equal_image_average(self):
        samples, means, _ = paper.paper_data(fixture())
        self.assertTrue(samples.groupby(paper.KEYS).size().eq(9).all())
        for resolution in paper.PANELS:
            part = means[means.resolution_class == resolution].set_index("group")
            self.assertAlmostEqual(part.loc["reconstruct", "ms"], 5)
            self.assertAlmostEqual(part.loc["reconstruct", "percent"], 20)
            self.assertAlmostEqual(part.ms.sum(), 25)

    def test_changed_complete_call_total_is_rejected(self):
        report = fixture()
        report["gpu_stages"].loc[0, "ms"] += 1
        with self.assertRaisesRegex(ValueError, "complete-call partition"):
            paper.paper_data(report)

    def test_incomplete_panel_is_rejected(self):
        report = fixture()
        report["coverage"].loc[0, "status"] = "incomplete"
        with self.assertRaisesRegex(ValueError, "complete image/effort coverage"):
            paper.paper_data(report)

    def test_unknown_stage_is_rejected(self):
        for classifier in (paper.gpu_group, paper.host_group):
            with self.assertRaisesRegex(ValueError, "Unmapped"):
                classifier("unknown.future_stage")


if __name__ == "__main__":
    unittest.main()
