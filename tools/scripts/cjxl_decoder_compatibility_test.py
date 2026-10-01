"""Scope an accepted decoder assumption without weakening other BD-rate checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cjxl_bd_rate as bd


class DecoderCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.evidence = self.root / "evidence.json"
        self.evidence.write_text('{"sampled": true}')
        self.studies = []
        image = dict(image_id="image", corpus="test", resolution_class="small",
                     width=100, height=100, pfm_sha256="reference")
        for encoder in ("libjxl", "gjxl"):
            config = dict(encoder=encoder, images=[copy.deepcopy(image)], efforts=[1],
                          metric="fast-ssim2", metric_version={"version": "test"},
                          intensity_target=80, num_threads=8, repetitions=5, warmups=1,
                          djxl="decoder", scorer="metric", configuration_id=encoder,
                          tool_hashes={"decoder": encoder + "-decoder", "metric": "same-scorer"})
            rows = [{**image, "encoder": encoder, "metric": "fast-ssim2",
                     "reference_sha256": "reference", "pixels": 10000, "effort": 1,
                     "distance": distance, "score": score,
                     "encoded_bytes": size * (0.8 if encoder == "gjxl" else 1.),
                     "resampling": 1, "elapsed_ms": 1., "timing_sample_count": 5}
                    for distance, score, size in ((4., 70., 1000), (3., 75., 2000),
                                                  (2., 85., 3000), (1., 90., 4000))]
            self.studies.append(dict(run=encoder, config=config, scores=rows,
                                     scores_sha256=encoder + "-scores"))
        self.record = dict(
            decision="assume-equivalent", reason="Same-source builds and sampled pixel identity",
            decoder_source_revision="e8ff09762481785938d8e4e01333ed3917571161",
            evidence=[dict(path=self.evidence.name, sha256=bd.study.digest(self.evidence))],
            studies=sorted([dict(encoder=item["config"]["encoder"],
                                 configuration_id=item["config"]["configuration_id"],
                                 scores_sha256=item["scores_sha256"],
                                 decoder_sha256=item["config"]["tool_hashes"]["decoder"],
                                 scorer_sha256="same-scorer") for item in self.studies],
                           key=lambda row: row["encoder"]))
        self.path = self.root / "compatibility.json"
        self.path.write_text(json.dumps(self.record))

    def analyze(self, **kwargs):
        return bd.analyze_same_effort(self.studies, efforts=[1], **kwargs)

    def test_different_decoders_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, "same pinned decoder"):
            self.analyze()

    def test_accepted_record_retains_original_identities_and_results(self):
        before = copy.deepcopy(self.studies)
        report = self.analyze(decoder_compatibility=self.path)
        self.assertEqual(before, self.studies)
        self.assertEqual(report["decoder_compatibility"]["decision"], "assume-equivalent")
        self.assertEqual(report["decoder_compatibility"]["record_sha256"], bd.study.digest(self.path))
        self.assertEqual({r["decoder_sha256"] for r in report["sources"]},
                         {"libjxl-decoder", "gjxl-decoder"})
        for point in report["points"]:
            self.assertAlmostEqual(point["bd_rate_pchip"], -20.)

    def test_changed_study_score_or_tool_is_rejected(self):
        before = copy.deepcopy(self.studies)
        for key in ("configuration_id", "scores_sha256", "decoder", "metric"):
            with self.subTest(key=key):
                self.studies = copy.deepcopy(before)
                if key == "scores_sha256":
                    self.studies[0][key] = "changed"
                elif key == "configuration_id":
                    self.studies[0]["config"][key] = "changed"
                else:
                    self.studies[0]["config"]["tool_hashes"][key] = "changed"
                with self.assertRaisesRegex(ValueError, "does not match"):
                    self.analyze(decoder_compatibility=self.path)

    def test_metric_and_reference_checks_still_apply(self):
        before = copy.deepcopy(self.studies)
        self.studies[1]["config"]["metric_version"] = {"version": "other"}
        with self.assertRaisesRegex(ValueError, "metric_version"):
            self.analyze(decoder_compatibility=self.path)
        self.studies = before
        self.studies[1]["config"]["images"][0]["pfm_sha256"] = "other-reference"
        with self.assertRaisesRegex(ValueError, "Score reference"):
            self.analyze(decoder_compatibility=self.path)

    def test_changed_evidence_is_rejected(self):
        self.evidence.write_text("changed")
        with self.assertRaises(bd.study.StudyError):
            self.analyze(decoder_compatibility=self.path)


if __name__ == "__main__":
    unittest.main()
