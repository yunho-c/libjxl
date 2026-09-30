#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""CPU collector regressions using a synthetic harness; no benchmark collection."""
import contextlib
import dataclasses
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cjxl_runtime_characterization as runtime


class RuntimeTimingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.args = runtime.parse_args([
            "run", "--phase", "timing", "--corpus", str(self.root / "corpus.json"),
            "--output", str(self.root / "run"), "--benchmark", "fake-harness",
            "--efforts", "7", "--qualities", "80",
        ])
        self.image = runtime.CorpusImage("kodak/01", "kodak", "small",
                                        self.root / "in.png", self.root / "in.pfm",
                                        10, 10, "pfm-hash", "source-hash")
        self.calls = []
        self.payload = lambda n: b"same"
        self.duration = lambda request: 1000
        self.raw_edit = lambda doc: None
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(runtime, "run_command", side_effect=self.harness).start()
        quiet = contextlib.ExitStack()
        quiet.enter_context(contextlib.redirect_stdout(io.StringIO()))
        quiet.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.addCleanup(quiet.close)

    def harness(self, command, **kwargs):
        if "--version" in command:
            raise runtime.StudyError("Harness has no --version option")
        options = dict(zip(command[1::2], command[2::2]))
        self.calls.append(options)
        raw = Path(options["--raw-samples"])
        request = json.loads((raw.parent / "request.json").read_text())
        payload = self.payload(len(self.calls))
        Path(options["--output"]).write_bytes(payload)
        doc = dict(schema_version=1, encoder="libjxl", stage_profile_enabled=False,
                   timing_semantics="complete-encode-wall-time",
                   input_layout="interleaved-linear-srgb-f32", revision="pinned-revision",
                   input_width=self.image.width, input_height=self.image.height,
                   thread_count=int(options["--num-threads"]), effort=int(options["--effort"]),
                   requested_distance=float(options["--distance"]), validation_encodes=1,
                   warmups=int(options["--warmups"]), sample_count=1,
                   samples=[dict(sample_index=0, encoded_bytes=len(payload),
                                 elapsed_nanoseconds=self.duration(request))])
        self.raw_edit(doc)
        raw.write_text(json.dumps(doc), encoding="utf-8")

    def collect(self, limit=None):
        return runtime.command_timing(self.args, [self.image], "fake-harness",
                                      self.args.efforts, runtime.JobBudget(limit))

    def warmup(self, limit=None):
        return runtime.command_warmup_check(self.args, [self.image], "fake-harness",
                                            runtime.JobBudget(limit))

    def rows(self):
        return runtime.read_jsonl(self.args.output / "timings.jsonl")

    def summary(self):
        return json.loads((self.args.output / "warmup-check/summary.json").read_text())

    def prepare_run(self):
        self.image.pfm_path.write_bytes(b"PF\n10 10\n-1.0\n" + bytes(10 * 10 * 12))
        self.image = dataclasses.replace(self.image, pfm_sha256=runtime.sha256_file(self.image.pfm_path))
        self.args.corpus.write_text(json.dumps({
            "input_layout": "interleaved-linear-srgb-f32", "color_encoding": runtime.PFM_COLOR_ENCODING,
            "image_count": 1, "images": [dataclasses.asdict(self.image)],
        }, default=str))
        binary = self.root / "harness.exe"
        binary.write_bytes(b"synthetic-harness")
        self.args.benchmark = str(binary)
        mock.patch.object(runtime, "environment_snapshot", return_value={}).start()
        mock.patch.object(runtime, "optional_command_output", return_value="test host").start()

    def test_timing_only_run_and_verify_need_no_stage_tools(self):
        self.prepare_run()
        runtime.command_run(self.args)
        self.assertEqual(len(self.rows()), 5)
        self.assertEqual(len(self.calls), 15)  # 10 diagnostic + 5 measured processes.
        args = runtime.parse_args(["verify", "--run", str(self.args.output),
                                   "--djxl", self.args.benchmark])
        with mock.patch.object(runtime, "run_command") as decode:
            runtime.command_verify(args)
        decode.assert_called_once()
        report = json.loads((self.args.output / "verification.json").read_text())
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["stage_profile_count"], 0)
        runtime.command_run(self.args)
        self.assertEqual(len(self.calls), 15)

    def test_warmup_policy_is_frozen_and_error_prevents_sweep(self):
        self.prepare_run()
        self.args.warmup_policy = "error"
        self.duration = lambda r: 1000 if r["warmups"] == 1 else 1200
        with self.assertRaisesRegex(runtime.StudyError, "warmup sensitivity"):
            runtime.command_run(self.args)
        self.assertFalse(self.rows())
        self.args.warmup_policy = "continue"
        with self.assertRaisesRegex(runtime.StudyError, "metadata does not match"):
            runtime.command_run(self.args)
        self.assertEqual(len(self.calls), 10)

    def test_each_repetition_has_raw_evidence_and_actual_output(self):
        self.assertTrue(self.collect())
        rows = self.rows()
        self.assertEqual([r["repetition"] for r in rows], list(range(5)))
        self.assertEqual(len({r["raw_path"] for r in rows}), 5)
        self.assertEqual(len(list((self.args.output / "outputs").rglob("*.jxl"))), 1)
        self.assertTrue(all("--output" in call for call in self.calls))
        runtime.verify_timing_records(rows)
        self.assertTrue(self.collect())
        self.assertEqual(len(self.calls), 5)

    def test_same_size_different_output_is_rejected_and_preserved(self):
        self.payload = lambda n: b"same" if n == 1 else b"diff"
        with self.assertRaisesRegex(runtime.StudyError, "Codestream changed"):
            self.collect()
        self.assertEqual(len(self.rows()), 1)
        failures = list((self.args.output / "raw").rglob("failure.json"))
        self.assertEqual(len(failures), 1)
        self.assertEqual((failures[0].parent / "output.jxl").read_bytes(), b"diff")
        self.assertTrue((failures[0].parent / "raw.json").is_file())
        self.assertEqual(Path(self.rows()[0]["output_path"]).read_bytes(), b"same")

    def test_interruption_resume_retains_raw_without_duplicate_samples(self):
        self.assertFalse(self.collect(limit=2))
        self.assertEqual(len(self.rows()), 2)
        self.assertTrue(self.collect())
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(len(self.rows()), 5)

    def test_crash_before_ledger_append_preserves_orphan_and_resumes(self):
        with mock.patch.object(runtime, "append_jsonl", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.collect()
        self.assertTrue(self.collect())
        self.assertEqual(len(self.rows()), 5)
        self.assertEqual(len(list((self.args.output / "raw").rglob("raw.json"))), 6)

    def test_duplicate_record_is_rejected_before_launch(self):
        self.collect(limit=1)
        runtime.append_jsonl(self.args.output / "timings.jsonl", self.rows()[0])
        with self.assertRaisesRegex(runtime.StudyError, "Duplicate"):
            self.collect()
        self.assertEqual(len(self.calls), 1)

    def test_modified_retained_output_is_rejected_on_resume(self):
        self.collect(limit=1)
        Path(self.rows()[0]["output_path"]).write_bytes(b"diff")
        with self.assertRaisesRegex(runtime.StudyError, "hash mismatch"):
            self.collect()
        self.assertEqual(len(self.calls), 1)

    def test_modified_raw_is_rejected_on_resume(self):
        self.collect(limit=1)
        Path(self.rows()[0]["raw_path"]).write_text("{}")
        with self.assertRaisesRegex(runtime.StudyError, "hash mismatch"):
            self.collect()

    def test_ledger_duration_must_match_raw(self):
        self.collect(limit=1)
        rows = self.rows()
        rows[0]["elapsed_nanoseconds"] += 1
        with self.assertRaisesRegex(runtime.StudyError, "Raw timing/ledger mismatch"):
            runtime.verify_timing_records(rows)

    def test_incorrect_harness_protocol_is_rejected(self):
        self.raw_edit = lambda doc: doc.update(warmups=99)
        with self.assertRaisesRegex(runtime.StudyError, "configuration"):
            self.collect()
        self.assertFalse(self.rows())

    def test_legacy_records_are_not_relabelled_verified(self):
        with self.assertRaisesRegex(runtime.StudyError, "legacy"):
            runtime.verify_timing_records([dict(sample_id="old")])

    def test_warmup_default_flags_and_continues_without_sweep_records(self):
        self.duration = lambda r: 1000 if r["warmups"] == 1 else 1200
        self.assertTrue(self.warmup())
        self.assertFalse(self.rows())
        self.assertEqual(len(self.calls), 10)
        self.assertEqual([int(c["--warmups"]) for c in self.calls], [1, 3, 3, 1, 1, 3, 3, 1, 1, 3])
        self.assertTrue(self.summary()["needs_review"])
        self.assertTrue(self.summary()["proceed"])
        self.assertTrue(self.warmup())
        self.assertEqual(len(self.calls), 10)

    def test_warmup_error_does_not_retry_on_resume(self):
        self.args.warmup_policy = "error"
        self.duration = lambda r: 1000 if r["warmups"] == 1 else 1200
        for _ in range(2):
            with self.assertRaisesRegex(runtime.StudyError, "warmup sensitivity"):
                self.warmup()
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(self.summary()["action"], "error")

    def test_warmup_retry_preserves_initial_flag_even_if_retry_passes(self):
        self.args.warmup_policy = "retry"
        self.duration = lambda r: 1200 if r["warmups"] == 3 and "attempt=0|" in r["sample_id"] else 1000
        self.assertTrue(self.warmup())
        self.assertEqual(len(self.calls), 20)
        self.assertEqual(len(self.summary()["attempts"]), 2)
        self.assertTrue(self.summary()["needs_review"])
        self.assertFalse(self.summary()["latest_needs_review"])
        self.assertTrue(self.summary()["proceed"])

    def test_warmup_retry_budget_survives_resume(self):
        self.args.warmup_policy = "retry"
        self.duration = lambda r: 1000 if r["warmups"] == 1 else 1200
        self.assertFalse(self.warmup(limit=13))
        for _ in range(2):
            with self.assertRaisesRegex(runtime.StudyError, "warmup sensitivity"):
                self.warmup()
        self.assertEqual(len(self.calls), 20)
        self.assertEqual(self.summary()["action"], "error")

    def test_warmup_pass_does_not_retry(self):
        self.args.warmup_policy = "retry"
        self.assertTrue(self.warmup())
        self.assertEqual(len(self.calls), 10)
        self.assertFalse(self.summary()["needs_review"])

    def test_cli_rejects_invalid_warmup_threshold_or_retry_budget(self):
        base = ["run", "--phase", "timing", "--corpus", "c", "--output", "r", "--benchmark", "b"]
        for option, value in [("--warmup-threshold-percent", "nan"),
                              ("--warmup-threshold-percent", "0"), ("--warmup-max-retries", "-1")]:
            with self.assertRaises(SystemExit):
                runtime.parse_args(base + [option, value])


if __name__ == "__main__":
    unittest.main()
