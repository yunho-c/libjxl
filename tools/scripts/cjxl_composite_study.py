#!/usr/bin/env python3
"""Materialize an explicitly mixed-revision GJXL analysis dataset by effort.

No encoders, decoders or scorers are launched. Source ledgers are read only;
codestreams and raw reports stay at their original, hashed artifact paths.
"""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import cjxl_quality_characterization as study


LEDGERS = ("timings", "scores", "calibration", "measurements", "probes")
PROTOCOL = (
    "images", "metric", "metric_version", "intensity_target", "targets",
    "tolerance", "num_threads", "thread_semantics", "repetitions", "warmups",
    "minimum_distance", "maximum_distance", "max_evaluations", "seed",
    "backend", "metal_aq_mode", "density", "compression", "collect_final_score",
)


def selected_records(run, config, efforts):
    records = {
        name: [row for row in study.read_records(run, name, config, verify=True)
               if row["effort"] in efforts]
        for name in LEDGERS
    }
    # Never hide unresolved targets, but require a finished pass and every
    # accepted setting's complete timing observations.
    images = {image["image_id"] for image in config["images"]}
    if study.is_fixed(config):
        expected = {study.fixed_key(image, q, effort)
                    for image in images for q in config["qualities"] for effort in efforts}
        actual = {row["source_job_id"] for row in records["scores"]}
        timed = {(row["job_id"], row["repetition"]) for row in records["timings"]}
        wanted = {(key, r) for key in expected for r in range(config["repetitions"])}
        if actual != expected or timed != wanted:
            raise study.StudyError("Selected fixed efforts have incomplete scores/timings")
        study.fixed_rows(run, config)  # Includes cross-repetition output checks.
    else:
        expected = {study.match_key(image, effort, target)
                    for image in images for target in config["targets"] for effort in efforts}
        outcomes = {row["match_id"]: row for row in records["calibration"]}
        if set(outcomes) != expected:
            raise study.StudyError("Selected calibrated efforts have pending targets")
        accepted = {key for key, row in outcomes.items() if row["status"] == "matched"}
        wanted = {(key, r) for key in accepted for r in range(config["repetitions"])}
        actual = {(row["match_id"], row["repetition"]) for row in records["measurements"]}
        if actual != wanted:
            raise study.StudyError("Selected accepted matches have incomplete timings")
        for row in records["measurements"]:
            match = outcomes[row["match_id"]]
            if (row["output_sha256"] != match["output_sha256"]
                    or row["distance"] != match["distance"]
                    or abs(match["score"] - match["target"]) > config["tolerance"]):
                raise study.StudyError("Matched measurement identity or tolerance differs")
    return records


def build_composite(old_run, new_run, output, replace_efforts=(1, 2, 3, 4)):
    old_run, new_run, output = [Path(p).expanduser().resolve()
                                for p in (old_run, new_run, output)]
    configs = [study.load_config(p, verify=True) for p in (old_run, new_run)]
    old, new = configs
    if any(study.encoder_name(c) != "gjxl" or c.get("analysis_only") for c in configs):
        raise study.StudyError("Use two original GJXL source runs")
    if study.is_fixed(old) != study.is_fixed(new):
        raise study.StudyError("Cannot mix fixed and calibrated runs")
    for key in (*PROTOCOL, *( ("qualities", "quality_to_distance", "sweep_helpers_sha256")
                             if study.is_fixed(old) else ())):
        if old.get(key) != new.get(key):
            raise study.StudyError("Incompatible composite protocol: " + key)
    for key in ("djxl", "scorer"):
        if old["tool_hashes"][old[key]] != new["tool_hashes"][new[key]]:
            raise study.StudyError("Composite requires the same pinned " + key)
    replace_efforts = set(replace_efforts)
    if (not replace_efforts or not replace_efforts <= set(old["efforts"])
            or not replace_efforts <= set(new["efforts"])):
        raise study.StudyError("Replacement efforts missing from a source")
    selections = (set(old["efforts"]) - replace_efforts, replace_efforts)
    sources, records = [], {name: [] for name in LEDGERS}
    for run, config, efforts in zip((old_run, new_run), configs, selections):
        if not efforts:
            continue
        hashes = {name: study.digest(run / (name + ".jsonl"))
                  for name in LEDGERS if (run / (name + ".jsonl")).exists()}
        selected = selected_records(run, config, efforts)
        if any(study.digest(run / (name + ".jsonl")) != sha for name, sha in hashes.items()):
            raise study.StudyError("Source ledgers changed while building composite")
        origin = {"run": str(run), "configuration_id": config["configuration_id"],
                  "encoder_revision": config["encoder_revision"],
                  "efforts": sorted(efforts), "ledger_sha256": hashes,
                  "metadata_sha256": study.digest(run / "metadata.json"),
                  "encoder_build_sha256": config["encoder_build_sha256"]}
        sources.append(origin)
        for name, rows in selected.items():
            for row in rows:
                records[name].append({**row, "composite_origin": {
                    "run": str(run), "configuration_id": row["configuration_id"],
                    "encoder_revision": config["encoder_revision"],
                }})

    if output.exists():
        existing = study.load_config(output)
        if existing.get("composite_sources") != sources:
            raise study.StudyError("Destination exists with different composite sources")
        for name, rows in records.items():
            expected = [{**r, "configuration_id": existing["configuration_id"]} for r in rows]
            if study.read_records(output, name, existing) != expected:
                raise study.StudyError("Existing composite ledger differs: " + name)
        return output

    config = dict(old)
    for key in ("configuration_id", "benchmark", "encoder_build_sha256",
                "collector_snapshot_sha256", "tuple_snapshot_sha256", "seed_snapshot_sha256"):
        config.pop(key, None)
    label = "GJXL composite: " + "; ".join(
        f"e{min(s['efforts'])}-{max(s['efforts'])} {s['encoder_revision'][:7]}"
        for s in sorted(sources, key=lambda s: min(s["efforts"])))
    config.update(analysis_only=True, composite_sources=sources, display_label=label,
                  encoder_revision="mixed-revision-composite",
                  preview_available=study.is_fixed(old))
    config["tool_hashes"] = {}
    for source_config in configs:
        for path, sha in source_config["tool_hashes"].items():
            if path in config["tool_hashes"] and config["tool_hashes"][path] != sha:
                raise study.StudyError("A shared tool path changed between sources")
            config["tool_hashes"][path] = sha
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + ".prepare-", dir=output.parent) as tmp:
        temporary = Path(tmp) / "dataset"
        temporary.mkdir()
        study.write_json(temporary / "encoder-build.json", {
            "kind": "analysis-only-composite", "label": label, "sources": sources,
            "note": "Different builds and collection sessions by effort; not a single-build measurement",
        })
        config["encoder_build_sha256"] = study.digest(temporary / "encoder-build.json")
        if study.is_fixed(config):
            shutil.copy2(old_run / "cjxl_sweep_common.py", temporary / "cjxl_sweep_common.py")
        config["configuration_id"] = study.identity(config)
        study.write_json(temporary / "metadata.json", config)
        for name, rows in records.items():
            if rows:
                with (temporary / (name + ".jsonl")).open("w") as stream:
                    for row in rows:
                        stream.write(json.dumps({**row, "configuration_id": config["configuration_id"]},
                                                sort_keys=True) + "\n")
                study.read_records(temporary, name, config)
        study.summarize(temporary, config)
        study.write_json(temporary / "progress.json", study.run_completion(temporary, config))
        (temporary / "README.md").write_text(
            "# " + label + "\n\nAnalysis only; collection is disabled.\n\n"
            "Each ledger observation retains its original configuration and revision in "
            "`composite_origin`. Raw reports and codestreams remain at their source paths. "
            "Keep the source runs. `metadata.json` records source ledger hashes and effort routing.\n")
        temporary.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace-efforts", default="1,2,3,4")
    args = parser.parse_args()
    print(build_composite(args.old_run, args.new_run, args.output,
                          [int(e) for e in args.replace_efforts.split(",")]))


if __name__ == "__main__":
    main()
