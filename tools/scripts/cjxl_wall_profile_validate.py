#!/usr/bin/env python3
"""Validate expanded libjxl wall timers against the pinned ordinary encoder.

Retains every raw measurement, output hash, and command. Identity runs use the
38-input comparison corpus plus sweep edge cases; perturbation runs alternate
ordinary, sink-off, and sink-on processes. Run without concurrent benchmarks.
"""

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import time

import cjxl_runtime_characterization as study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinary", type=pathlib.Path, required=True)
    parser.add_argument("--instrumented", type=pathlib.Path, required=True)
    parser.add_argument("--decoder", type=pathlib.Path, required=True)
    parser.add_argument("--pilot-corpus", type=pathlib.Path, required=True)
    parser.add_argument("--sweep-corpus", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    pilot = json.loads(args.pilot_corpus.read_text())
    sweep = json.loads(args.sweep_corpus.read_text())["images"]
    cases = [
        (row["name"], args.pilot_corpus.parent / row["canonical_path"], 7, 90, 8)
        for row in pilot["inputs"]
    ]
    kodak = next(row for row in sweep if row["image_id"] == "kodak/01")
    for effort in (1, 4, 5, 6, 8, 9, 10):
        for quality in (10, 90):
            cases.append(
                ("kodak-edge", pathlib.Path(kodak["pfm_path"]), effort, quality, 8)
            )
    cases.append(("serial-edge", pathlib.Path(kodak["pfm_path"]), 8, 90, 0))
    for resolution in ("12mp", "24mp", "48mp"):
        row = next(row for row in sweep if row["resolution_class"] == resolution)
        cases.append((resolution, pathlib.Path(row["pfm_path"]), 5, 90, 8))
    results = []

    def run(case, variant, label, warmups=0, samples=2):
        name, path, effort, quality, threads = case
        raw = args.output / (label + ".json")
        encoded = args.output / (label + ".jxl")
        binary = args.ordinary if variant == "ordinary" else args.instrumented
        command = [
            str(binary),
            "--input",
            str(path),
            "--raw-samples",
            str(raw),
            "--output",
            str(encoded),
            "--distance",
            str(study.quality_to_distance(quality)),
            "--effort",
            str(effort),
            "--num-threads",
            str(threads),
            "--warmups",
            str(warmups),
            "--samples",
            str(samples),
        ]
        if variant == "on":
            command.append("--stage-profile")
        start = time.monotonic()
        process = subprocess.run(command, capture_output=True, text=True)
        study.atomic_json(
            args.output / (label + ".command.json"),
            {
                "argv": command,
                "returncode": process.returncode,
                "stderr": process.stderr,
                "stdout": process.stdout,
                "process_seconds": time.monotonic() - start,
            },
        )
        if process.returncode:
            raise RuntimeError(process.stderr)
        document = json.loads(raw.read_text())
        if variant == "on":
            for sample in document["samples"]:
                study.validate_wall_sample(sample)
        return {
            "variant": variant,
            "raw": str(raw),
            "encoded": str(encoded),
            "sha256": hashlib.sha256(encoded.read_bytes()).hexdigest(),
            "median_ns": statistics.median(
                sample["elapsed_nanoseconds"] for sample in document["samples"]
            ),
        }

    for index, case in enumerate(cases):
        variants = [
            run(case, variant, "identity-%03d-%s" % (index, variant))
            for variant in ("ordinary", "off", "on")
        ]
        if len({v["sha256"] for v in variants}) != 1:
            raise RuntimeError("Codestream identity failed: %s" % (case,))
        decode = subprocess.run(
            [str(args.decoder), variants[-1]["encoded"], "--disable_output"],
            capture_output=True,
            text=True,
        )
        if decode.returncode:
            raise RuntimeError("Decode failed: " + decode.stderr)
        results.append(
            {"case": [str(v) for v in case], "variants": variants, "decode_ok": True}
        )
        study.atomic_json(args.output / "identity-progress.json", results)
        print("identity %d/%d %s" % (index + 1, len(cases), case[0]), flush=True)

    perturbations = []
    for effort, quality, threads in (
        (3, 90, 8),
        (7, 90, 8),
        (8, 90, 8),
        (10, 10, 8),
        (7, 90, 0),
    ):
        case = (
            "perturbation",
            pathlib.Path(kodak["pfm_path"]),
            effort,
            quality,
            threads,
        )
        pairs = []
        for pair in range(6):
            order = (
                ("ordinary", "off", "on")
                if pair % 2 == 0
                else ("on", "off", "ordinary")
            )
            values = {
                variant: run(
                    case,
                    variant,
                    "perturb-e%d-q%d-t%d-p%d-%s"
                    % (effort, quality, threads, pair, variant),
                    warmups=1,
                    samples=3,
                )
                for variant in order
            }
            if len({v["sha256"] for v in values.values()}) != 1:
                raise RuntimeError("Perturbation codestream mismatch")
            pairs.append(values)
        overhead = [
            100 * (p["on"]["median_ns"] / p["off"]["median_ns"] - 1) for p in pairs
        ]
        total = [
            100 * (p["on"]["median_ns"] / p["ordinary"]["median_ns"] - 1) for p in pairs
        ]
        perturbations.append(
            {
                "effort": effort,
                "quality": quality,
                "threads": threads,
                "pairs": pairs,
                "sink_overhead_percent_median": statistics.median(overhead),
                "sink_overhead_percent_pairs": overhead,
                "vs_ordinary_percent_median": statistics.median(total),
            }
        )
        study.atomic_json(args.output / "perturbation-progress.json", perturbations)
        print(
            "perturbation",
            effort,
            quality,
            threads,
            "sink overhead",
            statistics.median(overhead),
            flush=True,
        )
    summary = {
        "identity_case_count": len(results),
        "all_byte_identical": True,
        "all_decoded": True,
        "wall_profile_version": 2,
        "identity": results,
        "perturbation": perturbations,
        "overhead_review_required": any(
            p["sink_overhead_percent_median"] > 5 for p in perturbations
        ),
    }
    study.atomic_json(args.output / "summary.json", summary)
    print("validation summary:", args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
