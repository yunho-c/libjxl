#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Prepare, qualify, run and cooperatively pause independent GJXL CUDA studies.

No collection starts on import, prepare, build, status or pause. Existing studies
keep their own immutable collector snapshots; this is not a migration command.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import cjxl_quality_characterization as quality

SCRIPTS = Path(__file__).resolve().parent
FROZEN_SCRIPTS = ("cjxl_quality_characterization.py", "cjxl_cuda_characterization.py",
                  "cjxl_sweep_common.py", "cjxl_cuda_study.py", "cjxl_cuda_validate.py")
PROTOCOL = {"efforts": list(range(1, 11)), "qualities": [10, 30, 50, 70, 80, 90, 95],
            "targets": [60, 70, 85], "tolerance": 0.5, "minimum_distance": 0.01,
            "maximum_distance": 25, "max_evaluations": 24, "timeout": 1800}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(source, *arguments):
    return subprocess.check_output(["git", "-C", str(source), *arguments], text=True).strip()


def check_source(plan):
    source = Path(plan["source"])
    if git(source, "rev-parse", "HEAD") != plan["revision"]:
        raise quality.StudyError("Source HEAD changed; use the pinned checkout")
    if git(source, "status", "--porcelain"):
        raise quality.StudyError("Use a clean GJXL checkout, preferably a detached worktree")


def prepare(args):
    if args.study.exists():
        raise quality.StudyError("Study directory exists; do not overwrite or reinitialize it")
    source = args.source.resolve()
    revision = git(source, "rev-parse", args.revision + "^{commit}")
    plan = {"schema_version": 1, "source": str(source), "revision": revision,
            "protocol": PROTOCOL, "cuda_arch": args.cuda_arch, "generator": args.generator,
            "cmake_args": args.cmake_arg, "require_ac": not args.allow_battery,
            "policy": "dct8-e1-e4", "created_at": quality.utc(),
            "pilot_images": args.pilot_images.split(","),
            "warmup_images": args.warmup_images.split(",")}
    check_source(plan)
    corpus = read(args.corpus)
    images = corpus["images"]
    ids = {image["image_id"] for image in images}
    if not images or len(ids) != len(images):
        raise quality.StudyError("Corpus must contain unique image IDs")
    if not set(plan["pilot_images"] + plan["warmup_images"]) <= ids:
        raise quality.StudyError("Pilot and warmup IDs must occur in the corpus")
    for image in images:
        if image["color_encoding"] != quality.COLOR:
            raise quality.StudyError("Corpus must contain canonical linear-sRGB PFMs")
        if not Path(image["pfm_path"]).is_absolute():
            raise quality.StudyError("Corpus PFM paths must be absolute")
        quality.verify_file(image["pfm_path"], image["pfm_sha256"])
    for binary in (args.decoder, args.scorer):
        if not binary.is_file():
            raise quality.StudyError(f"Missing tool: {binary}")
    args.study.mkdir(parents=True)
    shutil.copy2(args.corpus, args.study / "corpus.json")
    for name, binary in (("decoder", args.decoder), ("scorer", args.scorer)):
        destination = args.study / "tools" / name
        destination.mkdir(parents=True)
        for path in {binary.resolve(), *binary.resolve().parent.glob("*.dll")}:
            shutil.copy2(path, destination / path.name)
        plan[name] = str(destination / binary.name)
    frozen = args.study / "collection-code"
    frozen.mkdir()
    for name in FROZEN_SCRIPTS:
        shutil.copy2(SCRIPTS / name, frozen / name)
    shutil.copytree(SCRIPTS / "cuda_quality_harness", args.study / "harness")
    plan["frozen_files"] = {
        str(path.relative_to(args.study)): quality.digest(path)
        for directory in (frozen, args.study / "harness", args.study / "tools")
        for path in directory.rglob("*") if path.is_file()
    }
    plan["frozen_files"]["corpus.json"] = quality.digest(args.study / "corpus.json")
    quality.write_json(args.study / "study-plan.json", plan)
    (args.study / "study-plan.sha256").write_text(quality.digest(args.study / "study-plan.json") + "\n")
    print(f"Prepared {args.study}; no build or collection started")


def load_plan(root):
    quality.verify_file(root / "study-plan.json", (root / "study-plan.sha256").read_text().strip())
    plan = read(root / "study-plan.json")
    if plan.get("schema_version") != 1 or "frozen_files" not in plan:
        raise quality.StudyError("This command requires a newly prepared study; keep legacy frozen runners")
    for name, expected in plan["frozen_files"].items():
        quality.verify_file(root / name, expected)
    return plan


def command(root, label, arguments, env=None):
    """Synchronous build/init/analysis command; collection uses monitored_run."""
    arguments = list(map(str, arguments))
    quality.append(root / "pipeline-events.jsonl", {"event": "started", "phase": label,
                                                   "command": arguments})
    with (root / (label + ".log")).open("a", encoding="utf-8") as log:
        result = subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT,
                                env=env, **quality.process_group_options())
    quality.append(root / "pipeline-events.jsonl", {"event": "stopped", "phase": label,
                                                   "returncode": result.returncode})
    if result.returncode:
        raise quality.StudyError(f"{label} failed; see {root / (label + '.log')}")


def build(root, plan, jobs):
    if (root / "build-record.json").exists():
        raise quality.StudyError("Build is already frozen; prepare a new study to rebuild")
    check_source(plan)
    source = Path(plan["source"])
    native = root / "native"
    commands = [
        ["cmake", "-S", source, "-B", root / "build", "-G", plan["generator"],
         "-DCMAKE_BUILD_TYPE=Release", "-DGJXL_ENABLE_CUDA=ON", "-DGJXL_ENABLE_METAL=OFF",
         "-DGJXL_BUILD_TESTS=OFF", "-DGJXL_BUILD_BENCHMARKS=OFF", "-DGJXL_CUDA_COMPACT_AC=OFF",
         f"-DCMAKE_CUDA_ARCHITECTURES={plan['cuda_arch']}",
         f"-DCMAKE_INSTALL_PREFIX={native}", *plan["cmake_args"]],
        ["cmake", "--build", root / "build", "--config", "Release", "--parallel", str(jobs)],
        ["cmake", "--install", root / "build", "--config", "Release"],
        ["cmake", "-S", root / "harness", "-B", root / "harness-build", "-G", plan["generator"],
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={native}",
         f"-DGJXL_SOURCE_DIR={source}", f"-DGJXL_QUALITY_REVISION={plan['revision']}",
         f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={root / 'bin'}",
         f"-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE={root / 'bin'}", *plan["cmake_args"]],
        ["cmake", "--build", root / "harness-build", "--config", "Release", "--parallel", str(jobs)],
    ]
    quality.write_json(root / "build-commands.json", [[str(arg) for arg in row] for row in commands])
    for index, arguments in enumerate(commands):
        command(root, f"build-{index}", arguments)
    check_source(plan)
    benchmark = root / "bin" / executable("gjxl_cuda_quality_benchmark")
    version = json.loads(subprocess.check_output([str(benchmark), "--version"], text=True))
    if version.get("revision") != plan["revision"]:
        raise quality.StudyError("Built harness reports the wrong source revision")
    tracked = git(source, "ls-files", "-z").split("\0")
    artifacts = [root / "build/CMakeCache.txt", root / "harness-build/CMakeCache.txt",
                 root / "build-commands.json"]
    for directory in (root / "bin", native, root / "tools", root / "harness"):
        artifacts.extend(path for path in directory.rglob("*") if path.is_file())
    record = {
        "revision": plan["revision"], "source": str(source), "benchmark": str(benchmark),
        "benchmark_sha256": quality.digest(benchmark), "version": version,
        "source_hashes": {name: quality.digest(source / name) for name in tracked
                          if name and (source / name).is_file()},
        "source_diff": git(source, "diff", "HEAD", "--binary"),
        "submodules": git(source, "submodule", "status"),
        "runtime_dlls": [str(path) for path in (root / "bin").glob("*.dll")],
        "artifacts": {str(path): quality.digest(path) for path in artifacts},
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                                        "--format=csv,noheader"], text=True).strip(),
        "platform": platform.platform(), "python": sys.version,
        "low_effort_policy": plan["policy"], "timing_semantics": "complete-encode-wall-time",
    }
    quality.write_json(root / "build-record.json", record)
    print("Build frozen. Run qualification separately before collection.")


def executable(name):
    return name + (".exe" if os.name == "nt" else "")


class Power(ctypes.Structure):
    _fields_ = [("ac", ctypes.c_ubyte), ("flags", ctypes.c_ubyte),
                ("percent", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte),
                ("seconds", ctypes.c_ulong), ("full_seconds", ctypes.c_ulong)]


class Device:
    """Low-overhead observations; no encoder or nvidia-smi subprocess in timing."""
    def __init__(self):
        self.error = None
        try:
            self.lib = (ctypes.WinDLL("nvml.dll") if os.name == "nt"
                        else ctypes.CDLL("libnvidia-ml.so.1"))
            if self.lib.nvmlInit_v2() != 0:
                raise RuntimeError("nvmlInit failed")
            self.handle = ctypes.c_void_p()
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
                raise RuntimeError("Cannot observe CUDA device 0")
        except (OSError, RuntimeError) as error:
            self.error = str(error)

    def sample(self):
        result = {}
        if os.name == "nt":
            power = Power()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)):
                result["ac_connected"] = int(power.ac)
        else:
            supplies = list(Path("/sys/class/power_supply").glob("*/online"))
            if supplies:
                result["ac_connected"] = int(any(p.read_text().strip() == "1" for p in supplies))
        if self.error:
            return {**result, "nvml_error": self.error}
        for name, function, arguments in (
            ("temperature_c", "nvmlDeviceGetTemperature", [0]),
            ("power_mw", "nvmlDeviceGetPowerUsage", []),
            ("sm_clock_mhz", "nvmlDeviceGetClockInfo", [1]),
            ("memory_clock_mhz", "nvmlDeviceGetClockInfo", [2]),
        ):
            value = ctypes.c_uint()
            if getattr(self.lib, function)(self.handle, *arguments, ctypes.byref(value)) == 0:
                result[name] = value.value
        return result


def request_pause(root, reason="user"):
    quality.write_json(root / "pause.request", {"reason": reason, "requested_at": quality.utc()})
    print("Pause requested; the active operation finishes before the next collector boundary.")


def check_pause(root):
    if (root / "pause.request").exists():
        raise quality.PauseRequested()


def monitored_run(root, plan, label, arguments, device):
    check_pause(root)
    observed = device.sample()
    if plan["require_ac"] and observed.get("ac_connected") != 1:
        raise quality.StudyError("AC power is not confirmed; collection was not started")
    quality.append(root / "telemetry.jsonl", {"phase": label, **observed})
    with (root / (label + ".log")).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(list(map(str, arguments)), stdout=log, stderr=subprocess.STDOUT,
                                   **quality.process_group_options())
        quality.append(root / "pipeline-events.jsonl", {"event": "started", "phase": label,
                                                       "pid": process.pid})
        try:
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    observed = device.sample()
                    quality.append(root / "telemetry.jsonl", {"phase": label, **observed})
                    if plan["require_ac"] and observed.get("ac_connected") != 1:
                        # Keep telemetry for reviewing records around the transition.
                        quality.stop_process(process)
                        request_pause(root, "AC power lost; inspect the interrupted setting before resuming")
                        raise quality.StudyError("AC power lost; active subprocess tree stopped")
        except KeyboardInterrupt:
            request_pause(root, "keyboard interrupt")
            process.wait()  # Collector sees the marker outside its timed call.
        except BaseException:
            quality.stop_process(process)
            raise
    quality.append(root / "pipeline-events.jsonl", {"event": "stopped", "phase": label,
                                                   "returncode": process.returncode})
    check_pause(root)
    return process.returncode


def init_command(root, plan, mode, *, pilot=False):
    location = root / (("pilot-" if pilot else "") + mode)
    p = plan["protocol"]
    arguments = [sys.executable, root / "collection-code/cjxl_quality_characterization.py", "init",
                 "--encoder", "gjxl", "--backend", "cuda", "--run", location,
                 "--corpus", root / "corpus.json", "--decoder", plan["decoder"],
                 "--scorer", plan["scorer"], "--build-record", root / "build-record.json",
                 "--benchmark", read(root / "build-record.json")["benchmark"],
                 "--gjxl-source", plan["source"], "--mode", "fixed" if mode == "fixed" else "matched",
                 "--efforts", "1,4,10" if pilot else ",".join(map(str, p["efforts"])),
                 "--qualities", "30,80" if pilot else ",".join(map(str, p["qualities"])),
                 "--targets", "60,85" if pilot else ",".join(map(str, p["targets"]))]
    arguments += ["--images", ",".join(plan["pilot_images"])] if pilot else ["--all-images"]
    for key in ("tolerance", "minimum_distance", "maximum_distance", "max_evaluations", "timeout"):
        arguments += ["--" + key.replace("_", "-"), str(p[key])]
    if mode == "calibrated":
        arguments += ["--seed-run", root / ("pilot-fixed" if pilot else "fixed")]
    return arguments


def collect_pair(root, plan, device, *, pilot=False):
    for mode in ("fixed", "calibrated"):
        check_pause(root)
        name = ("pilot-" if pilot else "") + mode
        location = root / name
        if not (location / "metadata.json").exists():
            command(root, name + "-init", init_command(root, plan, mode, pilot=pilot))
        result = monitored_run(root, plan, name, [
            sys.executable, location / "collector.py", "run", "--run", location,
            "--pause-file", root / "pause.request"], device)
        if result not in (0, 130) or not read(location / "progress.json")["collection_finished"]:
            raise quality.StudyError(f"{name} incomplete; inspect its log and saved progress")


def qualify(root, plan, device):
    for name in ("low_effort_strategy_policy", "cuda_low_effort"):
        check_pause(root)
        result = monitored_run(root, plan, name, [root / "bin" / executable(f"gjxl_{name}_test")], device)
        if result:
            raise quality.StudyError(f"Native test failed: {name}")
    collect_pair(root, plan, device, pilot=True)
    command(root, "pilot-validation", [sys.executable, root / "collection-code/cjxl_cuda_validate.py",
                                      "--study", root, "--pilot"])
    reports = []
    for effort in (1, 10):
        check_pause(root)
        location = root / f"warmup-e{effort}"
        arguments = init_command(root, plan, "fixed", pilot=True)
        for option, value in (("--run", location), ("--efforts", str(effort)),
                              ("--images", ",".join(plan["warmup_images"]))):
            arguments[arguments.index(option) + 1] = value
        if not (location / "metadata.json").exists():
            command(root, location.name + "-init", arguments)
        result = monitored_run(root, plan, location.name, [
            sys.executable, location / "collector.py", "warmup", "--run", location,
            "--pause-file", root / "pause.request"], device)
        if result not in (0, 130):
            raise quality.StudyError("Warmup diagnostic failed")
        report = read(location / "warmup-check-summary.json")
        if len(report["results"]) != 2:
            raise quality.StudyError("Expected both warmup sentinels to finish")
        reports.extend(report["results"])
    passed = not any(row["needs_review"] for row in reports)
    quality.write_json(root / "qualification.json", {
        "passed": passed, "revision": plan["revision"],
        "benchmark_sha256": read(root / "build-record.json")["benchmark_sha256"],
        "native_policy_tests": ["low_effort_strategy_policy", "cuda_low_effort"],
        "pilot_validation": "pilot-verification.json", "warmup": reports,
    })
    if not passed:
        raise quality.StudyError("Warmup sensitivity needs a documented review; samples are retained")


def status(root):
    result = {"pause_requested": (root / "pause.request").exists()}
    if (root / "pipeline-state.json").exists():
        result["pipeline"] = read(root / "pipeline-state.json")
    for name in ("pilot-fixed", "pilot-calibrated", "fixed", "calibrated"):
        if (root / name / "metadata.json").exists():
            config = quality.load_config(root / name)
            result[name] = quality.run_completion(root / name, config)
    print(json.dumps(result, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "build", "qualify", "run", "pause", "status", "validate"):
        child = sub.add_parser(name)
        child.add_argument("--study", type=lambda p: Path(p).resolve(), required=True)
        if name in ("run", "qualify"):
            child.add_argument("--resume", action="store_true", help="explicitly clear an existing pause request")
        if name == "build":
            child.add_argument("--jobs", type=int, default=4)
        if name == "prepare":
            for option in ("source", "corpus", "decoder", "scorer"):
                child.add_argument("--" + option, type=Path, required=True)
            child.add_argument("--revision", required=True)
            child.add_argument("--cuda-arch", required=True, help="CMake CUDA architecture, e.g. 86")
            child.add_argument("--generator", default="Ninja")
            child.add_argument("--cmake-arg", action="append", default=[], help="pass as --cmake-arg=-DNAME=value")
            child.add_argument("--allow-battery", action="store_true", help="also use for hosts without AC telemetry; recorded in plan")
            child.add_argument("--pilot-images", required=True, help="comma-separated representative image IDs")
            child.add_argument("--warmup-images", required=True, help="exactly two IDs: small and large")
    args = parser.parse_args(argv)
    root = args.study
    if args.command == "prepare":
        if len(args.warmup_images.split(",")) != 2:
            parser.error("--warmup-images requires two IDs")
        prepare(args)
        return 0
    if args.command == "status":
        status(root)
        return 0
    plan = load_plan(root)
    if args.command == "pause":
        request_pause(root)
        return 0
    with (root / ".pipeline.lock").open("a") as lock:
        quality.acquire_run_lock(lock)
        if args.command == "build":
            if args.jobs < 1:
                parser.error("--jobs must be positive")
            build(root, plan, args.jobs)
            return 0
        if args.command == "validate":
            command(root, "validation", [sys.executable, root / "collection-code/cjxl_cuda_validate.py", "--study", root])
            return 0
        if args.resume:
            (root / "pause.request").unlink(missing_ok=True)
        state = "running"
        quality.write_json(root / "pipeline-state.json", {"state": state, "pid": os.getpid(), "at": quality.utc()})
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            check_pause(root)
            device = Device()
            if args.command == "qualify":
                qualify(root, plan, device)
            else:
                qualified = read(root / "qualification.json")
                record = read(root / "build-record.json")
                if not qualified.get("passed") or qualified.get("revision") != plan["revision"] or qualified.get("benchmark_sha256") != record["benchmark_sha256"]:
                    raise quality.StudyError("Qualification must pass for this frozen revision and binary")
                collect_pair(root, plan, device)
                command(root, "validation", [sys.executable, root / "collection-code/cjxl_cuda_validate.py", "--study", root])
            state = "complete"
            return 0
        except quality.PauseRequested:
            state = "paused"
            print("Paused; completed records are preserved. Resume explicitly with --resume.")
            return 130
        except BaseException:
            state = "error"
            raise
        finally:
            quality.write_json(root / "pipeline-state.json", {"state": state, "pid": os.getpid(), "at": quality.utc()})
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (quality.StudyError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
