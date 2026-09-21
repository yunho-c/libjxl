#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""CPU-only regressions for durable CUDA collection, pause and provenance."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import cjxl_cuda_study as pipeline
import cjxl_cuda_validate as audit
import cjxl_quality_characterization as quality


class CudaStudyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)
        image = dict(image_id="small", corpus="test", resolution_class="small",
                     width=8, height=8, pfm_sha256="reference", color_encoding=quality.COLOR)
        self.config = dict(encoder="gjxl", backend="cuda", gpu_aq_mode="fully-resident",
                           collection_mode="fixed", images=[image], efforts=[1],
                           qualities=[80], quality_to_distance={"80":1.9}, repetitions=5,
                           warmups=1, seed=7, num_threads=8, encoder_revision="revision",
                           thread_semantics="maximum-participating-cpu-threads",
                           configuration_id="fixture", continue_cuda_oom=True)
        helper = self.root / "cjxl_sweep_common.py"
        helper.write_bytes((pipeline.SCRIPTS / helper.name).read_bytes())
        self.config["sweep_helpers_sha256"] = quality.digest(helper)

    def test_pause_and_resume_keep_exact_committed_repetitions(self):
        pause = self.root / "pause.request"
        calls = []

        def harness(config, image, effort, distance, directory, run, budget, warmups):
            calls.append(effort)
            output, raw = directory / "out.jxl", directory / "raw.json"
            output.write_bytes(b"deterministic-output")
            raw.write_text('{}')
            if len(calls) == 1:
                pause.touch()  # Request arrives during an operation; finish committing it.
            return output, raw, {"elapsed_nanoseconds": 123, "encoded_bytes": output.stat().st_size}

        with mock.patch.object(quality, "harness", side_effect=harness):
            with self.assertRaises(quality.PauseRequested):
                quality.collect_fixed_timings(self.root, self.config, quality.Budget(pause_file=pause))
            first = (self.root / "timings.jsonl").read_bytes()
            self.assertEqual(len(quality.ledger(self.root / "timings.jsonl")), 1)
            pause.unlink()
            quality.collect_fixed_timings(self.root, self.config, quality.Budget(pause_file=pause))
            # A second resume must neither rerun work nor rewrite earlier records.
            quality.collect_fixed_timings(self.root, self.config, quality.Budget(pause_file=pause))
        rows = quality.ledger(self.root / "timings.jsonl")
        self.assertEqual(len(calls), 5)
        self.assertEqual({row["repetition"] for row in rows}, set(range(5)))
        self.assertTrue((self.root / "timings.jsonl").read_bytes().startswith(first))
        self.assertEqual(len({row["output_sha256"] for row in rows}), 1)

    def test_oom_is_terminal_and_not_replaced_by_samples(self):
        with mock.patch.object(quality, "harness", side_effect=quality.CudaOutOfMemory(
                "cudaErrorMemoryAllocation (out of memory)")) as harness:
            quality.collect_fixed_timings(self.root, self.config, quality.Budget())
            quality.collect_fixed_timings(self.root, self.config, quality.Budget())
        self.assertEqual(harness.call_count, 1)
        failure, = quality.ledger(self.root / "failures.jsonl")
        self.assertEqual(failure["status"], "cuda-out-of-memory")
        progress = quality.fixed_completion(self.root, self.config)
        self.assertTrue(progress["collection_finished"])
        self.assertFalse(progress["complete"])
        self.assertEqual(progress["timing_samples"], 0)

    def test_other_errors_do_not_become_oom(self):
        with mock.patch.object(quality, "harness", side_effect=quality.StudyError("invalid output")):
            with self.assertRaisesRegex(quality.StudyError, "invalid output"):
                quality.collect_fixed_timings(self.root, self.config, quality.Budget())
        self.assertFalse((self.root / "failures.jsonl").exists())

    def test_changed_output_on_resume_is_rejected(self):
        count = 0

        def harness(config, image, effort, distance, directory, run, budget, warmups):
            nonlocal count
            count += 1
            output, raw = directory / "out.jxl", directory / "raw.json"
            output.write_bytes(b"first" if count == 1 else b"changed")
            raw.write_text('{}')
            return output, raw, {"elapsed_nanoseconds":123, "encoded_bytes":output.stat().st_size}

        with mock.patch.object(quality, "harness", side_effect=harness):
            with self.assertRaisesRegex(quality.StudyError, "changed artifact"):
                quality.collect_fixed_timings(self.root, self.config, quality.Budget())
        self.assertEqual(len(quality.ledger(self.root / "timings.jsonl")), 1)

    def test_exclusive_lock_releases_after_process_exit(self):
        script = self.root / "lock.py"
        script.write_text(
            "import sys\nfrom pathlib import Path\n"
            f"sys.path.insert(0, {str(pipeline.SCRIPTS)!r})\n"
            "from cjxl_quality_characterization import acquire_run_lock\n"
            "with Path(sys.argv[1]).open('a') as f:\n"
            "    acquire_run_lock(f)\n")
        path = self.root / ".lock"
        with path.open("a") as lock:
            quality.acquire_run_lock(lock)
            result = subprocess.run([sys.executable, script, path], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run([sys.executable, script, path], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_monitored_child_observes_pause_and_no_next_phase(self):
        script = self.root / "child.py"
        script.write_text(
            "from pathlib import Path\nimport sys,time\n"
            "root=Path(sys.argv[1])\n(root/'started').touch()\n"
            "while not (root/'pause.request').exists(): time.sleep(.01)\n"
            "(root/'checkpoint').write_text('preserved')\nsys.exit(130)\n")
        errors = []

        def request():
            deadline = time.monotonic() + 5
            while not (self.root / "started").exists():
                if time.monotonic() > deadline:
                    errors.append("child did not start")
                    return
                time.sleep(.01)
            pipeline.request_pause(self.root)

        thread = threading.Thread(target=request)
        thread.start()
        device = mock.Mock()
        device.sample.return_value = {"ac_connected":1}
        try:
            with self.assertRaises(quality.PauseRequested):
                pipeline.monitored_run(self.root, {"require_ac":True}, "fake",
                                       [sys.executable, script, self.root], device)
        finally:
            thread.join(timeout=6)
        self.assertFalse(errors)
        self.assertEqual((self.root / "checkpoint").read_text(), "preserved")
        self.assertTrue((self.root / "pause.request").exists())

    def test_plan_and_frozen_inputs_cannot_change(self):
        frozen = self.root / "collector.py"
        frozen.write_text("# snapshot\n")
        plan = {"schema_version":1, "frozen_files":{"collector.py":quality.digest(frozen)}}
        quality.write_json(self.root / "study-plan.json", plan)
        checksum = self.root / "study-plan.sha256"
        checksum.write_text(quality.digest(self.root / "study-plan.json"))
        self.assertEqual(pipeline.load_plan(self.root), plan)
        frozen.write_text("# changed\n")
        with self.assertRaises(quality.StudyError):
            pipeline.load_plan(self.root)
        quality.write_json(self.root / "study-plan.json", {**plan, "revision":"changed"})
        with self.assertRaises(quality.StudyError):
            pipeline.load_plan(self.root)

    def test_calibration_seeds_only_from_the_new_fixed_run(self):
        quality.write_json(self.root / "build-record.json", {"benchmark":"benchmark"})
        plan = dict(protocol=pipeline.PROTOCOL, decoder="decoder", scorer="scorer",
                    source="source", pilot_images=["small", "large"])
        args = pipeline.init_command(self.root, plan, "calibrated")
        self.assertEqual(args[args.index("--seed-run") + 1], self.root / "fixed")
        args = pipeline.init_command(self.root, plan, "calibrated", pilot=True)
        self.assertEqual(args[args.index("--seed-run") + 1], self.root / "pilot-fixed")

    def test_false_unresolved_reason_is_rejected(self):
        record = dict(image_id="a", effort=1, output_sha256="output", distance=1.0,
                      score=70.1, recorded_at="2026-01-01T00:00:00")
        outcome = dict(image_id="a", effort=1, target=70.0, match_id="a|effort=1|target=70",
                       status="unresolved-discontinuity", recorded_at="2026-01-01T00:00:01")
        records = dict(scores=[record], probes=[], calibration=[outcome])
        config = dict(max_evaluations=24, tolerance=.5, minimum_distance=.01, maximum_distance=25)
        with self.assertRaisesRegex(AssertionError, "accepted score"):
            audit.validate_terminal_calibrations(config, records)

    def test_warmup_command_does_not_dispatch_full_collection(self):
        config = {**self.config, "pilot":False, "measurement_efforts":[1]}
        with mock.patch.object(quality, "load_config", return_value=config), \
             mock.patch.object(quality, "process_commands", return_value=""), \
             mock.patch.object(quality, "require_encoding_allowed"), \
             mock.patch.object(quality, "collect_warmup_check") as warmup, \
             mock.patch.object(quality, "collect_full_run") as full, \
             mock.patch.object(quality, "run_completion", return_value={"complete":False}), \
             mock.patch.object(quality, "summarize"):
            self.assertEqual(quality.main(["warmup", "--run", str(self.root)]), 0)
        warmup.assert_called_once()
        full.assert_not_called()


if __name__ == "__main__":
    unittest.main()
