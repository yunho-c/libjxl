#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
"""End-to-end checks of batch measurements and the shared GJXL CSV contract."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import statistics
import struct
import subprocess
import tempfile
import unittest


RAW_FIELDS = (
    "codec,workload,source,width,height,batch_size,sample,order,"
    "requested_backend,backend,aq_mode,distance,effort,thread_policy,"
    "timing_boundary,serial_ns,batch_ns,encoded_bytes_per_image"
).split(",")


class ImageBatchBenchmarkTest(unittest.TestCase):
    binary: Path
    cjxl: Path | None = None

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.raw = self.root / "samples.csv"

    def image(self, name="image.pfm", width=17, height=13, endian="<"):
        path = self.root / name
        values = [(i * 13 % 97) / 100 for i in range(width * height * 3)]
        scale = "-1" if endian == "<" else "1"
        path.write_bytes(
            f"PF\n{width} {height}\n{scale}\n".encode()
            + struct.pack(f"{endian}{len(values)}f", *values)
        )
        return path

    def run_benchmark(self, *arguments):
        return subprocess.run(
            [
                str(self.binary),
                "--batch-sizes", "1,4",
                "--samples", "2",
                "--warmups", "1",
                "--threads-per-image", "2",
                *map(str, arguments),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

    def raw_rows(self):
        with self.raw.open(newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, RAW_FIELDS)
            return list(reader)

    def test_raw_pairs_and_summary_agree(self):
        source = self.image()
        result = self.run_benchmark("--input", source, "--raw-samples", self.raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.raw_rows()
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({row["encoded_bytes_per_image"] for row in rows}), 1)
        self.assertGreater(int(rows[0]["encoded_bytes_per_image"]), 0)
        for row in rows:
            self.assertEqual(
                (row["codec"], row["backend"], row["requested_backend"]),
                ("libjxl", "cpu", "cpu"),
            )
            self.assertEqual(row["aq_mode"], "n/a")
            self.assertEqual(row["effort"], "7")
            self.assertAlmostEqual(float(row["distance"]), 1.2, places=6)
            self.assertEqual(row["source"], str(source))
            self.assertEqual((row["width"], row["height"]), ("17", "13"))
            self.assertEqual(row["thread_policy"], "fixed_per_image:2")
            self.assertEqual(row["timing_boundary"], "linear_rgb_to_in_memory_codestream")
            self.assertEqual(
                row["order"], "serial-first" if row["sample"] == "0" else "batch-first"
            )
            self.assertGreater(int(row["serial_ns"]), 0)
            self.assertGreater(int(row["batch_ns"]), 0)
        summaries = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(len(summaries), 2)
        for summary in summaries:
            samples = [row for row in rows if row["batch_size"] == summary["batch_size"]]
            serial_ms = statistics.median(int(row["serial_ns"]) for row in samples) / 1e6
            batch_ms = statistics.median(int(row["batch_ns"]) for row in samples) / 1e6
            ratios = [int(row["serial_ns"]) / int(row["batch_ns"]) for row in samples]
            count = int(summary["batch_size"])
            for column, expected in (
                ("serial_median_ms", serial_ms),
                ("batch_median_ms", batch_ms),
                ("batch_ms_per_image", batch_ms / count),
                ("batch_images_per_second", 1000 * count / batch_ms),
                ("paired_speedup_median", statistics.median(ratios)),
                ("paired_speedup_min", min(ratios)),
                ("paired_speedup_max", max(ratios)),
            ):
                self.assertAlmostEqual(float(summary[column]), expected, delta=0.00051)

    def test_directory_selection_and_csv_escaping(self):
        second = self.image('b,"line\n2.pfm')
        first = self.image("a.PFM", 16, 16)
        alias = self.root / "alias.pfm"
        alias.symlink_to(second)
        nested = self.root / "nested"
        nested.mkdir()
        (nested / "invalid.pfm").write_text("must not be visited")
        (self.root / "ignored.png").write_text("not a PFM")
        result = self.run_benchmark(
            "--input", self.root, "--input", first, "--input", alias,
            "--raw-samples", self.raw,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            [row["source"] for row in self.raw_rows()], [str(first)] * 4 + [str(second)] * 4
        )
        summaries = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(
            [(row["width"], row["height"]) for row in summaries],
            [("16", "16")] * 2 + [("17", "13")] * 2,
        )

    def test_endianness_and_single_thread_policy(self):
        self.image("little.pfm")
        self.image("big.pfm", endian=">")
        result = self.run_benchmark(
            "--input", self.root, "--threads-per-image", "1",
            "--effort", "5", "--distance", "2", "--raw-samples", self.raw,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.raw_rows()
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["encoded_bytes_per_image"] for row in rows}), 1)
        for row in rows:
            self.assertEqual(row["thread_policy"], "fixed_per_image:1")
            self.assertEqual((row["effort"], float(row["distance"])), ("5", 2.0))

    def test_rejects_bad_inputs_options_and_output_collisions(self):
        source = self.image()
        before = source.read_bytes()
        result = self.run_benchmark("--input", source, "--raw-samples", source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already exists", result.stderr)
        self.assertEqual(source.read_bytes(), before)
        empty = self.root / "empty"
        empty.mkdir()
        bad_files = {
            "truncated.pfm": b"PF\n17 13\n-1\n",
            "nonfinite.pfm": b"PF\n1 1\n-1\n" + struct.pack("<fff", float("nan"), 0, 0),
            "scaled.pfm": before.replace(b"\n-1\n", b"\n-2\n", 1),
            "gray.pfm": b"Pf\n1 1\n-1\n" + struct.pack("<f", 0.5),
            "disguised.pfm": b"P6\n1 1\n255\n\x00\x00\x00",
        }
        for name, data in bad_files.items():
            (self.root / name).write_bytes(data)
        for args in (
            (),
            ("--input", self.root / "missing.pfm"),
            ("--input", empty),
            *(("--input", self.root / name) for name in bad_files),
            *(("--input", source, flag, value) for flag, value in (
                ("--samples", "0"), ("--samples", "-1"), ("--warmups", "0"),
                ("--batch-sizes", "1,1"), ("--batch-sizes", "2,1"),
                ("--batch-sizes", "1,"), ("--threads-per-image", "0"),
                ("--effort", "11"), ("--distance", "nan"), ("--distance", "-1"),
                ("--distance", "garbage"), ("--samples", "9" * 100),
                ("--raw-samples", self.root / "missing-parent/out.csv"),
            )),
        ):
            with self.subTest(args=args):
                result = self.run_benchmark(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Benchmark error:", result.stderr)
        dangling = self.root / "dangling.csv"
        dangling.symlink_to(self.root / "nonexistent.csv")
        result = self.run_benchmark("--input", source, "--raw-samples", dangling)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "nonexistent.csv").exists())

    def test_linear_input_matches_cjxl(self):
        if self.cjxl is None:
            self.skipTest("Pass --cjxl for the independent CLI encoding check")
        source = self.image(width=63, height=49)
        encoded = self.root / "reference.jxl"
        reference = subprocess.run(
            [str(self.cjxl), str(source), str(encoded), "-d", "1.2", "-e", "7",
             "--num_threads=2", "-x", "color_space=RGB_D65_SRG_Rel_Lin"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(reference.returncode, 0, reference.stderr)
        result = self.run_benchmark("--input", source, "--raw-samples", self.raw)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            {int(row["encoded_bytes_per_image"]) for row in self.raw_rows()},
            {encoded.stat().st_size},
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--cjxl", type=Path)
    args, remaining = parser.parse_known_args()
    ImageBatchBenchmarkTest.binary = args.benchmark.resolve()
    ImageBatchBenchmarkTest.cjxl = args.cjxl.resolve() if args.cjxl else None
    unittest.main(argv=[__file__, *remaining])
