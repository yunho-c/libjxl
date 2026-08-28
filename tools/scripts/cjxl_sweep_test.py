#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Tests for cjxl_sweep.py."""

import contextlib
import csv
import io
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cjxl_sweep


def write_minimal_png_header(path, width=8, height=4):
    header = (
        b"\x89PNG\r\n\x1a\n"
        + (13).to_bytes(4, "big")
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header)


def make_image(path, relative_path, width=8, height=4):
    return cjxl_sweep.InputImage(
        path=path,
        relative_path=relative_path,
        size_bytes=24,
        sha256="0" * 64,
        width=width,
        height=height,
    )


class ParsingTest(unittest.TestCase):
    def test_parse_values_normalizes_and_deduplicates(self):
        values = cjxl_sweep.parse_values("0.50, 1, 1.500, 1e0", "distance")
        self.assertEqual(["0.5", "1", "1.5"], [value.text for value in values])
        self.assertEqual([0.5, 1.0, 1.5], [value.number for value in values])

    def test_parse_values_validates_axis_range(self):
        with self.assertRaises(cjxl_sweep.SweepError):
            cjxl_sweep.parse_values("25.1", "distance")
        with self.assertRaises(cjxl_sweep.SweepError):
            cjxl_sweep.parse_values("nan", "quality")

    def test_parse_efforts_expands_ranges(self):
        self.assertEqual([1, 2, 3, 5], cjxl_sweep.parse_efforts("1-3,5,3"))
        with self.assertRaises(cjxl_sweep.SweepError):
            cjxl_sweep.parse_efforts("10-11")

    def test_reserved_arguments_are_rejected(self):
        with self.assertRaises(cjxl_sweep.SweepError):
            cjxl_sweep.validate_extra_args(["--distance=3"])
        cjxl_sweep.validate_extra_args(["--brotli_effort=4"])


class DatasetAndScheduleTest(unittest.TestCase):
    def test_recursive_dataset_discovery_and_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            write_minimal_png_header(root / "nested" / "b.png", 10, 20)
            write_minimal_png_header(root / "a.png", 30, 40)
            dataset_root, images = cjxl_sweep.discover_inputs(root, "**/*.png")
            self.assertEqual(root.resolve(), dataset_root)
            self.assertEqual(
                ["a.png", "nested/b.png"], [image.relative_path for image in images]
            )
            self.assertEqual([1200, 200], [image.pixels for image in images])

    def test_schedule_is_deterministic_complete_and_unique(self):
        root = pathlib.Path("/tmp/cjxl-sweep-test")
        images = [
            make_image(root / "a.png", "a.png"),
            make_image(root / "b.png", "b.png"),
        ]
        values = [cjxl_sweep.SweepValue("0.5", 0.5), cjxl_sweep.SweepValue("1", 1.0)]
        first = cjxl_sweep.build_schedule("distance", values, [1, 2], 2, images, 123)
        second = cjxl_sweep.build_schedule("distance", values, [1, 2], 2, images, 123)
        self.assertEqual(first, second)
        self.assertEqual(16, len(first))
        self.assertEqual(16, len({job.sample_id for job in first}))
        for repetition in range(2):
            jobs = [job for job in first if job.repetition == repetition]
            self.assertEqual(
                {"a.png", "b.png"}, {job.image.relative_path for job in jobs}
            )
            self.assertEqual(4, len({job.cell_id for job in jobs}))


class SummaryTest(unittest.TestCase):
    def test_resume_rejects_a_different_host(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / cjxl_sweep.METADATA_FILENAME
            existing = {
                "schema_version": cjxl_sweep.SCHEMA_VERSION,
                "run_config": {"axis": "distance"},
                "host": {"machine": "first"},
            }
            candidate = {
                "schema_version": cjxl_sweep.SCHEMA_VERSION,
                "run_config": {"axis": "distance"},
                "host": {"machine": "second"},
            }
            cjxl_sweep.write_json_atomic(path, existing)
            with self.assertRaises(cjxl_sweep.SweepError):
                cjxl_sweep.ensure_compatible_metadata(path, candidate)

    def test_summary_uses_complete_dataset_passes(self):
        metadata = {
            "run_config": {
                "axis": "distance",
                "values": [
                    {"text": "1", "number": 1.0},
                    {"text": "2", "number": 2.0},
                ],
                "efforts": [1],
                "repetitions": 2,
                "num_threads": 0,
                "images": [
                    {"path": "a.png", "pixels": 100000},
                    {"path": "b.png", "pixels": 100000},
                ],
            },
        }

        def record(value, repetition, image, wall_ms, cpu_ms):
            sample_id = "distance=%s|effort=1|repetition=%d|image=%s" % (
                value,
                repetition,
                image,
            )
            return {
                "sample_id": sample_id,
                "status": "ok",
                "axis": "distance",
                "parameter_text": value,
                "effort": 1,
                "repetition": repetition,
                "image": image,
                "pixels": 100000,
                "wall_time_ns": wall_ms * 1_000_000,
                "user_cpu_time_ns": cpu_ms * 1_000_000,
                "system_cpu_time_ns": 0,
            }

        records = [
            record("1", 0, "a.png", 10, 8),
            record("1", 0, "b.png", 20, 16),
            record("1", 1, "a.png", 30, 24),
            record("1", 1, "b.png", 40, 32),
            record("2", 0, "a.png", 50, 40),
        ]
        rows = cjxl_sweep.build_summary_rows(metadata, records)
        complete, partial = rows
        self.assertTrue(complete["complete"])
        self.assertEqual(2, complete["complete_repetitions"])
        self.assertEqual(50.0, complete["dataset_wall_ms_median"])
        self.assertEqual(40.0, complete["dataset_cpu_ms_median"])
        self.assertEqual(250.0, complete["wall_ms_per_megapixel_median"])
        self.assertFalse(partial["complete"])
        self.assertEqual(0, partial["complete_repetitions"])
        self.assertIsNone(partial["dataset_wall_ms_median"])

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(1.3, cjxl_sweep.percentile([1, 2, 3, 4], 0.1))
        self.assertEqual(3.7, cjxl_sweep.percentile([1, 2, 3, 4], 0.9))


class IntegrationTest(unittest.TestCase):
    def test_run_resumes_and_writes_chart_ready_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            dataset = root / "dataset"
            image = dataset / "image.png"
            write_minimal_png_header(image)
            calls = root / "calls.txt"
            fake_cjxl = root / "fake_cjxl.py"
            fake_cjxl.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib\n"
                "import sys\n"
                'if "--version" in sys.argv:\n'
                '  print("fake cjxl 1.0")\n'
                "  raise SystemExit(0)\n"
                'with pathlib.Path(%r).open("a") as output:\n'
                '  output.write("run\\n")\n'
                'if "--disable_output" not in sys.argv:\n'
                "  raise SystemExit(3)\n" % str(calls),
                encoding="utf-8",
            )
            fake_cjxl.chmod(0o755)
            output = root / "output"
            base_args = [
                "run",
                "--cjxl",
                str(fake_cjxl),
                "--dataset",
                str(dataset),
                "--axis",
                "distance",
                "--values",
                "1,2",
                "--efforts",
                "1-2",
                "--repetitions",
                "1",
                "--warmups",
                "0",
                "--num-threads",
                "0",
                "--progress-every",
                "0",
                "--output",
                str(output),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, cjxl_sweep.main(base_args + ["--max-jobs", "2"]))
            self.assertEqual(
                2, len(cjxl_sweep.read_jsonl(output / cjxl_sweep.RAW_FILENAME))
            )

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(0, cjxl_sweep.main(base_args))
                self.assertEqual(0, cjxl_sweep.main(base_args))
            records = cjxl_sweep.read_jsonl(output / cjxl_sweep.RAW_FILENAME)
            self.assertEqual(4, len(records))
            self.assertEqual(4, len(calls.read_text(encoding="utf-8").splitlines()))
            with (output / cjxl_sweep.SUMMARY_FILENAME).open(
                newline="", encoding="utf-8"
            ) as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(4, len(rows))
            self.assertEqual({"true"}, {row["complete"] for row in rows})
            self.assertTrue(all(row["dataset_wall_ms_median"] for row in rows))


if __name__ == "__main__":
    unittest.main()
