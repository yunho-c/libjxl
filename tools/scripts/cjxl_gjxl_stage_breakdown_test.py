#!/usr/bin/env python3
"""Checks for scientific accounting, missing cohorts, and saved-data plotting."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cjxl_gjxl_stage_breakdown as profiles


class StageBreakdownTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = dict(
            schema_version=2, label="Test capture", source_revision="test",
            source_description="Fixture", binary_sha256="build", samples_per_capture=2,
            warmups=1, cpu_threads=8, efforts=list(range(1, 11)), settings=["Q80"],
            settings_label="Q80", images=[dict(image_id=i, resolution_class="test",
                                             width=10, height=10) for i in ("a", "b")],
            captures=[],
        )
        for image in self.manifest["images"]:
            source = self.root / (image["image_id"] + ".pfm")
            source.write_bytes(image["image_id"].encode())
            image.update(input_path=source.name,
                         input_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        self.host = dict(schema_version=17, scope="metal-public-workflow", gpu_aq="fully-resident",
                         collect_final_score=False, effort=4, cpu_threads=8, density="default",
                         compression="automatic", validation="metal-only", warmups=1,
                         sample_count=2, distance=1,
                         workloads=[dict(source_width=10, source_height=10, samples=[])])
        p = dict(total=100, input_preparation=10, quantization_pipeline=40, codestream_encoding=45,
                 input_geometry_and_storage=1, input_color_transform=1, input_matrix_scale_stats=1,
                 input_resident_preparation=4, input_quantization_preparation=2,
                 codestream_validation=1, codestream_dc_tokenization=4, codestream_ac_tokenization=10,
                 codestream_entropy_optimization=10, codestream_section_writing=10, codestream_assembly=5)
        self.host["workloads"][0]["samples"] = [dict(sample_index=i, backend="metal",
                                                   phase_nanoseconds=copy.deepcopy(p)) for i in (0, 1)]
        self.gpu = {k: copy.deepcopy(v) for k, v in self.host.items()
                    if k in ("scope", "gpu_aq", "collect_final_score", "warmups", "sample_count", "distance")}
        self.gpu.update(schema_version=4, mode="stage", workloads=[dict(source_width=10, source_height=10,
            samples=[dict(sample_index=i, capabilities=dict(timestamp_counter=True, stage_boundary=True),
                          submissions=[dict(stages=[dict(stage_id="frontend.initial_quantization",
                          begin_timestamp=100, end_timestamp=120, gpu_nanoseconds=20)])]) for i in (0, 1)])])
        for image in ("a", "b"):
            self.manifest["captures"].append(dict(image_id=image, setting="Q80", effort=4, distance=1,
                                                  host=self.write_raw(image+"-host", self.host, image),
                                                  gpu=self.write_raw(image+"-gpu", self.gpu, image)))

    def tearDown(self):
        plt.close("all")

    def write_raw(self, name, data, image="a"):
        path = self.root / (name + ".json")
        path.write_text(json.dumps(data))
        return dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    binary_sha256="build", argv=["benchmark", "--effort", "4", "--distance", "1",
                                                  "--cpu-threads", "8", "--input", image + ".pfm"])

    def load(self):
        p = self.root / "manifest.json"
        p.write_text(json.dumps(self.manifest))
        return profiles.load_profiles(p)

    def test_complete_effort_and_additive_partitions(self):
        r = self.load()
        self.assertEqual(set(r["means"]["effort"]), {4})
        self.assertEqual(len(r["coverage"]), 40)
        self.assertEqual(sum(r["coverage"]["status"] == "complete"), 4)
        totals = r["means"].groupby("kind")["percent"].sum()
        for total in totals:
            self.assertAlmostEqual(total, 100)
        raw = profiles.host_partitions(self.host["workloads"][0]["samples"][0])
        self.assertEqual(sum(raw["workflow"].values()), 100)
        self.assertEqual(sum(raw["serializer"].values()), 45)
        self.assertEqual(sum(raw["flat"].values()), 100)

    def test_flat_replaces_parents_and_uses_workflow_denominator(self):
        report = self.load()
        flat = report["means"].query("kind == 'flat'").set_index("stage")
        self.assertNotIn("CPU serializer", flat.index)
        self.assertNotIn("Input preparation", flat.index)
        self.assertAlmostEqual(flat["ms"].sum(), 100 / 1e6)
        self.assertAlmostEqual(flat.loc["Entropy optimization", "percent"], 10)
        self.assertAlmostEqual(flat.loc["Resident input preparation", "percent"], 4)
        self.assertAlmostEqual(flat.loc["Other serializer", "percent"], 5)

    def test_flat_is_independent_of_gpu_counters_and_worker_timers(self):
        before = self.load()["means"].query("kind == 'flat'").to_dict("records")
        g = copy.deepcopy(self.gpu)
        for sample in g["workloads"][0]["samples"]:
            stage = sample["submissions"][0]["stages"][0]
            stage.update(end_timestamp=20100, gpu_nanoseconds=20000)
        h = copy.deepcopy(self.host)
        for sample in h["workloads"][0]["samples"]:
            sample["phase_nanoseconds"]["codestream_entropy_prefix_histogram_build_work"] = 10**9
        for capture in self.manifest["captures"]:
            image = capture["image_id"]
            capture["gpu"] = self.write_raw(image + "-long-gpu", g, image)
            capture["host"] = self.write_raw(image + "-long-workers", h, image)
        after = self.load()["means"].query("kind == 'flat'").to_dict("records")
        self.assertEqual(before, after)

    def test_incomplete_image_does_not_silently_shrink_cohort(self):
        self.manifest["captures"].pop()
        self.assertTrue(self.load()["means"].empty)

    def test_missing_gpu_does_not_remove_host_bars(self):
        del self.manifest["captures"][0]["gpu"]
        self.assertEqual(set(self.load()["means"]["kind"]), {"flat", "workflow", "serializer"})

    def test_missing_setting_prevents_pooled_effort(self):
        self.manifest["settings"].append("Q90")
        self.assertTrue(self.load()["means"].empty)

    def test_all_ten_efforts_are_supported(self):
        self.manifest["captures"] = []
        for effort in range(1, 11):
            for image in ("a", "b"):
                h = copy.deepcopy(self.host)
                h["effort"] = effort
                host = self.write_raw(f"{image}-e{effort}-host", h, image)
                gpu = self.write_raw(f"{image}-e{effort}-gpu", self.gpu, image)
                host["argv"][2] = gpu["argv"][2] = str(effort)
                self.manifest["captures"].append(dict(image_id=image, effort=effort,
                    distance=1, setting="Q80", host=host, gpu=gpu))
        r = self.load()
        self.assertEqual(set(r["means"]["effort"]), set(range(1, 11)))
        self.assertTrue((r["coverage"]["status"] == "complete").all())
        figure = profiles.plot_breakdown(r, "test", normalize=False)
        self.assertEqual(len(figure.axes), 1)
        for effort in range(1, 11):
            total = sum(p.get_height() for p in figure.axes[0].patches
                        if abs(p.get_x() + p.get_width() / 2 - effort) < 1e-9)
            self.assertAlmostEqual(total, 100 / 1e6)

    def test_invalid_counter_repetition_prevents_gpu_bar(self):
        bad = copy.deepcopy(self.gpu)
        bad["workloads"][0]["samples"][1]["submissions"][0]["stages"][0]["begin_timestamp"] = 0
        self.manifest["captures"][0]["gpu"] = self.write_raw("bad", bad)
        report = self.load()
        self.assertNotIn("gpu", set(report["means"]["kind"]))
        self.assertEqual(len(report["rejected"]), 1)

    def test_cross_submission_overlaps_are_rejected(self):
        sample = copy.deepcopy(self.gpu["workloads"][0]["samples"][0])
        sample["submissions"] *= 2
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            profiles.gpu_stages(sample)

    def test_hash_and_effort_provenance_fail_closed(self):
        c = self.manifest["captures"][0]
        c["gpu"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()
        c["gpu"] = self.write_raw("gpu", self.gpu)
        c["gpu"]["argv"][2] = "5"
        with self.assertRaisesRegex(ValueError, "effort/distance"):
            self.load()

    def test_percentages_use_duration_ratios(self):
        h = copy.deepcopy(self.host)
        p = h["workloads"][0]["samples"][1]["phase_nanoseconds"]
        p["total"] = 300
        p["quantization_pipeline"] = 240
        for c in self.manifest["captures"]:
            c["host"] = self.write_raw(c["image_id"]+"-ratio", h, c["image_id"])
        r = self.load()["means"]
        v = r[(r["kind"] == "workflow") & r["stage"].str.startswith("Quantization")]
        self.assertAlmostEqual(v["percent"].iloc[0], 70)  # (40+240)/(100+300); not (40%+80%)/2.

    def test_negative_residual_rejected(self):
        s = copy.deepcopy(self.host["workloads"][0]["samples"][0])
        s["phase_nanoseconds"]["total"] = 1
        with self.assertRaisesRegex(ValueError, "exceeds"):
            profiles.host_partitions(s)
        s["phase_nanoseconds"]["total"] = 100
        s["phase_nanoseconds"]["input_resident_preparation"] = 100
        with self.assertRaisesRegex(ValueError, "Input partition exceeds"):
            profiles.host_partitions(s)

    def test_same_size_wrong_input_is_rejected(self):
        self.manifest["captures"][1]["host"]["argv"][-1] = "a.pfm"
        with self.assertRaisesRegex(ValueError, "Capture input"):
            self.load()

    def test_input_bytes_and_declared_identity_are_required(self):
        (self.root / "a.pfm").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Input hash mismatch"):
            self.load()
        del self.manifest["images"][0]["input_sha256"]
        with self.assertRaisesRegex(ValueError, "Missing input identity"):
            self.load()

    def test_reused_profile_cannot_fill_another_image(self):
        self.manifest["captures"][1]["host"] = copy.deepcopy(self.manifest["captures"][0]["host"])
        self.manifest["captures"][1]["host"]["argv"][-1] = "b.pfm"
        with self.assertRaisesRegex(ValueError, "Profile file reused"):
            self.load()

    def test_ambiguous_input_arguments_and_duplicate_input_paths_rejected(self):
        ref = self.manifest["captures"][0]["host"]
        original = list(ref["argv"])
        for argv in (original[:-2], original[:-1], original + ["--input", "a.pfm"]):
            ref["argv"] = argv
            with self.subTest(argv=argv), self.assertRaisesRegex(ValueError, "exactly one --input"):
                self.load()
        ref["argv"] = original
        self.manifest["images"][1]["input_path"] = "a.pfm"
        with self.assertRaisesRegex(ValueError, "multiple image IDs"):
            self.load()

    def test_old_manifest_requires_explicit_migration(self):
        self.manifest["schema_version"] = 1
        with self.assertRaisesRegex(ValueError, "schema 2"):
            self.load()

    def test_saved_data_render_never_launches_process(self):
        self.load()
        with mock.patch("subprocess.Popen", side_effect=AssertionError("No collection")):
            report = profiles.generate_breakdowns(self.root / "manifest.json", self.root / "plots",
                                                   formats=("png",))
        self.assertTrue((self.root / "plots/gjxl-stage-breakdown-test.png").is_file())
        self.assertEqual(len(report["figures"]["test"].axes), 1)
        ax = report["figures"]["test"].axes[0]
        self.assertEqual(len(ax.get_xticks()), 10)
        self.assertEqual(sum(t.get_text() == "missing" for t in ax.texts), 9)
        self.assertEqual(report["gpu_figures"], {})
        self.assertFalse(list((self.root / "plots").glob("gjxl-gpu-stage-diagnostic-*")))

    def test_optional_gpu_diagnostic_is_a_separate_figure(self):
        self.load()
        report = profiles.generate_breakdowns(self.root / "manifest.json", self.root / "plots",
                                              formats=("png",), gpu_diagnostics=True)
        self.assertEqual(len(report["figures"]["test"].axes), 1)
        self.assertEqual(len(report["gpu_figures"]["test"].axes), 1)
        self.assertTrue((self.root / "plots/gjxl-gpu-stage-diagnostic-test.png").is_file())


if __name__ == "__main__":
    unittest.main()
