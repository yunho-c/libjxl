#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Audit saved CUDA grids, raw reports, hashes and calibrated matches; no encoding."""
import argparse
import collections
import json
import math
from pathlib import Path
import sys

import cjxl_quality_characterization as study


def validate_requested_grid(config, plan, corpus_path, pilot=False):
    """Compare the full run to the requested study, not just its own metadata."""
    corpus = json.loads(corpus_path.read_text())
    all_images = {image["image_id"]: image for image in corpus["images"]}
    assert len(corpus["images"]) == len(all_images) > 0
    expected_images = ({name: all_images[name] for name in plan["pilot_images"]}
                       if pilot else all_images)
    actual_images = {image["image_id"]: image for image in config["images"]}
    assert len(config["images"]) == len(actual_images) == len(expected_images)
    assert set(actual_images) == set(expected_images)
    assert config["corpus_sha256"] == study.digest(corpus_path)
    for image_id, image in actual_images.items():
        for field in ("width", "height", "pfm_sha256", "source_sha256", "resolution_class"):
            assert image[field] == expected_images[image_id][field], (image_id, field)
    protocol = plan["protocol"]
    efforts = [1, 4, 10] if pilot else protocol["efforts"]
    assert config["efforts"] == efforts
    if study.is_fixed(config):
        assert config["qualities"] == ([30, 80] if pilot else protocol["qualities"])
    else:
        assert config["measurement_efforts"] == efforts
        assert config["targets"] == ([60, 85] if pilot else protocol["targets"])
        for key in ("tolerance", "minimum_distance", "maximum_distance", "max_evaluations"):
            assert config[key] == protocol[key]


def validate_calibration_links(records):
    """Require accepted scores and timed bytes to trace to a decoded observation."""
    observations = collections.defaultdict(list)
    for row in records["scores"] + records["probes"]:
        observations[(row["image_id"], row["effort"], row["output_sha256"])].append(row)
    outcomes = {row["match_id"]: row for row in records["calibration"]}
    linked = 0
    score_fields = ("distance", "score", "reference_sha256", "decoded_sha256",
                    "metric", "encoded_bytes", "scoring_mode")
    for key, outcome in outcomes.items():
        assert key == study.match_key(outcome["image_id"], outcome["effort"], outcome["target"])
        if outcome["status"] != "matched":
            continue
        candidates = observations[(outcome["image_id"], outcome["effort"], outcome["output_sha256"])]
        assert any(all(candidate[field] == outcome[field] for field in score_fields)
                   for candidate in candidates), (key, "No matching decoded score observation")
        assert math.isclose(outcome["score_error"], outcome["score"] - outcome["target"], abs_tol=1e-12)
        linked += 1
    measurement_fields = ("image_id", "effort", "target", "distance", "score",
                          "output_sha256", "reference_sha256", "decoded_sha256",
                          "metric", "encoded_bytes", "scoring_mode")
    for row in records["measurements"]:
        outcome = outcomes[row["match_id"]]
        assert outcome["status"] == "matched"
        assert all(row[field] == outcome[field] for field in measurement_fields), row["match_id"]
    return linked


def validate_terminal_calibrations(config, records):
    """Require unresolved reasons to agree with observations available at the time."""
    observations = collections.defaultdict(list)
    probes = collections.defaultdict(list)
    for row in records["scores"] + records["probes"]:
        observations[(row["image_id"], row["effort"])].append(row)
    for row in records["probes"]:
        probes[row["match_id"]].append(row)
    assert all(len(rows) <= config["max_evaluations"] for rows in probes.values())
    checked = collections.Counter()
    for outcome in records["calibration"]:
        status = outcome["status"]
        if status == "matched":
            continue
        key = outcome["match_id"]
        if status == "cuda-out-of-memory":
            assert "cudaErrorMemoryAllocation" in outcome.get("error", ""), key
        else:
            assert status in ("evaluation-budget-exhausted", "no-resolved-bracket",
                              "unresolved-discontinuity"), (key, status)
            available = [row for row in observations[(outcome["image_id"], outcome["effort"])]
                         if row["recorded_at"] <= outcome["recorded_at"]
                         and config["minimum_distance"] <= row["distance"] <= config["maximum_distance"]]
            assert available, key
            assert all(abs(row["score"] - outcome["target"]) > config["tolerance"]
                       for row in available), (key, "An accepted score was already available")
            own_probes = probes[key]
            assert all(row["recorded_at"] <= outcome["recorded_at"] for row in own_probes), key
            remaining = config["max_evaluations"] - len(own_probes)
            if status == "evaluation-budget-exhausted":
                assert remaining == 0, (key, len(own_probes))
            else:
                assert remaining > 0, key

            def unexpected_probe(distance):
                raise AssertionError((key, "Unresolved search could continue", distance))

            best, replayed_status, calls = study.search(
                available, outcome["target"], config["tolerance"], unexpected_probe,
                config["minimum_distance"], config["maximum_distance"], remaining)
            assert best is None and replayed_status == status and calls == 0, key
        checked[status] += 1
    return dict(checked)


def validate(run, plan, corpus_path, pilot=False):
    config = study.load_config(run, verify=True)
    validate_requested_grid(config, plan, corpus_path, pilot)
    assert config["encoder_revision"] == plan["revision"]
    assert config["backend"] == "cuda" and config["gpu_aq_mode"] == "fully-resident"
    assert config["density"] == "default" and config["compression"] == "automatic"
    assert config["collect_final_score"] is False
    assert (config["num_threads"], config["repetitions"], config["warmups"]) == (8, 5, 1)
    assert config["metric_version"]["fast_ssim2_revision"] == study.METRIC_REVISION
    progress = study.run_completion(run, config)
    assert progress["collection_finished"], progress
    images = {i["image_id"]: i for i in config["images"]}
    records = {name: study.read_records(run, name, config, verify=True)
               for name in ("timings", "scores", "probes", "calibration", "measurements", "failures")}
    for name, key in (("timings", "sample_id"), ("scores", "source_job_id"),
                      ("calibration", "match_id"), ("failures", "job_id")):
        assert len(records[name]) == len({row[key] for row in records[name]}), (name, "duplicate IDs")
    assert len(records["measurements"]) == len({(row["match_id"], row["repetition"])
                                              for row in records["measurements"]})
    linked_matches = 0 if study.is_fixed(config) else validate_calibration_links(records)
    terminal_checks = {} if study.is_fixed(config) else validate_terminal_calibrations(config, records)
    samples = records["timings"] if study.is_fixed(config) else records["measurements"]
    for row in samples:
        raw = json.loads(Path(row["raw_path"]).read_text())
        sample = raw["samples"][0]
        assert raw["final_score_evaluated"] is False
        if row["effort"] <= 4:
            assert raw["score_history_count"] == 0
            assert raw["strategy_counts"] and raw["strategy_counts"][0] > 0
            assert sum(raw["strategy_counts"][1:]) == 0
        assert raw["backend"] == "cuda" and raw["gpu_aq_mode"] == "fully-resident"
        assert raw["revision"] == config["encoder_revision"]
        assert raw["timing_semantics"] == "complete-encode-wall-time"
        assert raw["stage_profile_enabled"] is False
        assert raw["validation_encodes"] == 1 and raw["warmups"] == 1
        assert raw["sample_count"] == 1 and raw["thread_count"] == 8
        assert raw["effort"] == row["effort"] and raw["resampling"] == 1
        assert (raw["input_width"], raw["input_height"]) == (images[row["image_id"]]["width"], images[row["image_id"]]["height"])
        assert math.isclose(raw["requested_distance"], row["distance"], rel_tol=1e-6, abs_tol=1e-7)
        assert type(sample["elapsed_nanoseconds"]) is int and sample["elapsed_nanoseconds"] > 0
        assert sample["elapsed_nanoseconds"] == row["elapsed_nanoseconds"]
        assert sample["encoded_bytes"] == row["encoded_bytes"] == Path(row["output_path"]).stat().st_size
    for row in records["scores"] + records["probes"]:
        assert math.isfinite(row["score"])
        assert len(row["decoded_sha256"]) == 64
        assert row["reference_sha256"] == images[row["image_id"]]["pfm_sha256"]
        assert row["metric"] == "fast-ssim2"
    if study.is_fixed(config):
        expected = {study.fixed_key(i, q, e) for i in images for q in config["qualities"] for e in config["efforts"]}
        timings = collections.defaultdict(list)
        for row in records["timings"]:
            timings[row["job_id"]].append(row)
        failures = {r["job_id"]: r for r in records["failures"] if r["phase"] == "fixed-timing"}
        scores = {r["source_job_id"]: r for r in records["scores"]}
        assert expected == set(timings) | set(failures)
        for key, rows in timings.items():
            if key in failures:
                continue
            assert len(rows) == 5
            assert {r["repetition"] for r in rows} == set(range(5))
            assert len({r["output_sha256"] for r in rows}) == 1
            assert rows[0]["output_sha256"] == scores[key]["output_sha256"]
        for failure in failures.values():
            assert failure["status"] == "cuda-out-of-memory" and "cudaErrorMemoryAllocation" in failure["error"]
        unresolved = collections.Counter(r["status"] for r in failures.values())
    else:
        expected = {study.match_key(i, e, t) for i in images for e in config["measurement_efforts"] for t in config["targets"]}
        outcomes = {r["match_id"]: r for r in records["calibration"]}
        assert expected == set(outcomes)
        measurements = collections.defaultdict(list)
        for row in records["measurements"]:
            measurements[row["match_id"]].append(row)
        for key, outcome in outcomes.items():
            if outcome["status"] == "matched":
                assert abs(outcome["score"] - outcome["target"]) <= config["tolerance"]
                assert len(measurements[key]) == 5
                assert {r["repetition"] for r in measurements[key]} == set(range(5))
                assert all(r["output_sha256"] == outcome["output_sha256"] for r in measurements[key])
            else:
                assert key not in measurements
        unresolved = collections.Counter(r["status"] for r in outcomes.values() if r["status"] != "matched")
        for key, count in collections.Counter(r["match_id"] for r in records["probes"]).items():
            assert count <= config["max_evaluations"]
    return {"configuration_id": config["configuration_id"], "coverage": progress,
            "requested_full_grid_verified": not pilot,
            "record_counts": {name: len(rows) for name, rows in records.items()},
            "unresolved_status_counts": dict(unresolved),
            "raw_timing_reports_checked": len(samples), "hashes_verified": True,
            "calibrated_matches_linked_to_decoded_scores": linked_matches,
            "unresolved_calibrations_verified": terminal_checks,
            "accepted_results_valid": True}


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("Run validation without Python -O; assertions are required")
    from cjxl_cuda_study import load_plan
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.study)
    prefix = "pilot-" if args.pilot else ""
    report = {name: validate(args.study / (prefix + name), plan, args.study / "corpus.json", args.pilot)
              for name in ("fixed", "calibrated")}
    study.write_json(args.study / (prefix + "verification.json"), report)
    print(json.dumps(report, indent=2))
