"""Coverage and partial-timing safeguards for the calibrated trend preview."""
import copy
import unittest

import cjxl_bd_rate as bd


class CalibratedPreviewTest(unittest.TestCase):
    def studies(self):
        images = [dict(image_id=name, corpus="test", resolution_class="small",
                       width=100, height=100, pfm_sha256=name) for name in ("a", "b")]
        result = []
        for encoder in ("libjxl", "gjxl"):
            config = dict(encoder=encoder, images=images, efforts=[1, 2],
                          metric="fast-ssim2", metric_version={"version": "test"},
                          intensity_target=80, num_threads=8, repetitions=5, warmups=1,
                          djxl="decoder", tool_hashes={"decoder": "same-binary"},
                          configuration_id=encoder)
            scores = []
            for image in images:
                for effort in config["efforts"]:
                    if encoder == "gjxl" and image["image_id"] == "b" and effort == 2:
                        continue
                    count = 3 if encoder == "libjxl" and image["image_id"] == "a" and effort == 2 else 5
                    for distance, score, size in ((3., 60., 1000), (1., 85., 4000)):
                        scores.append({**image, "encoder": encoder, "metric": "fast-ssim2",
                                       "reference_sha256": image["pfm_sha256"], "pixels": 10000,
                                       "effort": effort, "distance": distance, "score": score,
                                       "encoded_bytes": size * (0.8 if encoder == "gjxl" else 1),
                                       "resampling": 1, "elapsed_ms": 10., "timing_sample_count": count})
            outcomes = ([dict(image_id="b", effort=2, status="cuda-out-of-memory")]
                        if encoder == "gjxl" else [])
            result.append(dict(run=encoder, config=config, scores=scores,
                               observation_source="calibrated", calibration_outcomes=outcomes))
        return result

    def test_same_intersection_and_partial_flags_without_mutation(self):
        studies = self.studies()
        before = copy.deepcopy(studies)
        report = bd.analyze_common_calibrated_cohort(studies, baseline_effort=1)
        self.assertEqual(studies, before)
        self.assertEqual(report["cohort_selection"]["included_image_ids"], ["a"])
        self.assertTrue(all(p["cohort"] == ["a"] for p in report["points"]))
        self.assertTrue(all(p["status"] == "ready" for p in report["points"]))
        partial = [p for p in report["points"] if p["scope"] == "all" and p["timing_provisional"]]
        self.assertEqual([(p["encoder"], p["effort"]) for p in partial], [("libjxl", 2)])
        self.assertEqual(partial[0]["timing_sample_min"], 3)
        self.assertAlmostEqual(partial[0]["mean_encode_ms"], 10.)
        excluded = report["cohort_selection"]["excluded_images"]
        self.assertEqual(excluded[0]["reasons"][0]["collection_failures"], ["cuda-out-of-memory"])
        for p in report["points"]:
            if p["encoder"] == "gjxl":
                self.assertAlmostEqual(p["bd_rate_pchip"], -20.)

    def test_strict_default_keeps_full_cohort_and_rejects_partial_timings(self):
        report = bd.analyze(self.studies(), baseline_effort=1,
                            quality_range=(75, 84.5), minimum_points=2)
        p = next(p for p in report["points"] if p["scope"] == "all"
                 and p["encoder"] == "libjxl" and p["effort"] == 2)
        self.assertEqual(p["cohort"], ["a", "b"])
        self.assertEqual(p["missing_reasons"], {"incomplete-timing": 1})
        self.assertNotIn("mean_encode_ms", p)

    def test_insufficient_repetitions_cannot_create_a_trend_point(self):
        studies = self.studies()
        for row in studies[0]["scores"]:
            if row["image_id"] == "a" and row["effort"] == 2:
                row["timing_sample_count"] = 2
        with self.assertRaisesRegex(ValueError, "No common image cohort"):
            bd.analyze_common_calibrated_cohort(studies, baseline_effort=1)

    def test_common_preview_refuses_fixed_observations(self):
        studies = self.studies()
        studies[0]["observation_source"] = "fixed"
        with self.assertRaisesRegex(ValueError, "requires calibrated"):
            bd.analyze_common_calibrated_cohort(studies, baseline_effort=1)


if __name__ == "__main__":
    unittest.main()
