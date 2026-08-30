#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Aggregate symbolized Samply captures of cjxl.

The parser consumes Samply's Firefox-profiler JSON and the presymbolicated
``.json.syms.json`` sidecar written next to each capture. It reports sampled
thread-CPU attribution at three levels:

* mutually exclusive cjxl operation buckets, selected from the full stack;
* flat leaf functions;
* inclusive functions.

Example:

  python3 tools/scripts/cjxl_samply_profile.py \
      'build-samply/profiles/kodak-??-d1-e7-t1.json.gz' \
      --output build-samply/profiles/analysis.md

The default ``current`` delta attribution reproduces the method used for the
existing Kodak REPORT.md: sample ``i`` receives ``threadCPUDelta[i]``. For
threaded captures, ``--delta-attribution previous`` can instead charge the CPU
delta ending at sample ``i`` to sample ``i - 1``. Neither mode turns sampled
thread CPU into wall-clock stage timing.
"""

import argparse
import collections
import dataclasses
import glob
import gzip
import json
import pathlib
import statistics
import sys


PROFILE_SUFFIX = ".json.gz"
SYMBOL_SUFFIX = ".json.syms.json"
SCHEMA_VERSION = 1

STARTUP_OPERATION = "Process startup and static initializers"
PNG_OPERATION = "PNG input decode"
UNRESOLVED_OPERATION = "Unresolved or missing stack"
NORMALIZED_EXCLUSIONS = frozenset((STARTUP_OPERATION, PNG_OPERATION))

# Rules are intentionally ordered. A reconstruction frame below ProcessRectACS,
# for example, belongs to AC strategy search rather than the later AR rule.
OPERATION_RULES = (
    (
        PNG_OPERATION,
        ("DecodeImageAPNG", "Context::FeedChunks"),
    ),
    (
        "Packed input to float planes",
        ("ConvertFromExternalPlaneNoSizeCheck",),
    ),
    (
        "sRGB to XYB color transform",
        ("SRGBToXYB", "ToXYB("),
    ),
    (
        "AC strategy search and candidate DCT/IDCT",
        ("ProcessRectACS",),
    ),
    (
        "Adaptive quantization map",
        ("AdaptiveQuantizationMap",),
    ),
    (
        "Chroma-from-luma heuristics",
        ("CfLHeuristics",),
    ),
    (
        "Patch dictionary search",
        ("FindBestPatchDictionary",),
    ),
    (
        "AR heuristics and roundtrip reconstruction/filtering",
        (
            "ComputeARHeuristics",
            "ReconstructImage",
            "DecodeGroupForRoundtrip",
            "LowMemoryRenderPipeline",
        ),
    ),
    (
        "Entropy modeling, tokenization, and bit writing",
        (
            "BuildAndEncodeHistograms",
            "BuildAndStoreEntropy",
            "ChooseUintConfigs",
            "ClusterHistograms",
            "WriteTokens",
            "EncodeGroupTokenizedCoefficients",
            "TokenizeCoefficients",
            "ComputeCoeffOrder",
        ),
    ),
    (
        "Final coefficient DCT and quantization",
        (
            "EncodeGroups",
            "ComputeCoefficients",
            "QuantizeRoundtripYBlockAC",
            "AdjustQuantBlockAC",
        ),
    ),
    (
        "Modular/DC side data",
        (
            "ModularFrameEncoder",
            "ModularCompress",
            "AddVarDCTDC",
            "EncodeModularChannel",
        ),
    ),
    (
        "Other lossy heuristics",
        ("LossyFrameHeuristics",),
    ),
)


class ProfileError(Exception):
    """An expected profile input or output error."""


@dataclasses.dataclass(frozen=True)
class FrameSymbol:
    name: str
    library: str


@dataclasses.dataclass(frozen=True)
class CaptureSummary:
    path: str
    sample_count: int
    cpu_delta_us: int


@dataclasses.dataclass
class Analysis:
    delta_attribution: str
    captures: list
    sample_count: int = 0
    cpu_delta_us: int = 0
    resolved_leaf_cpu_us: int = 0
    flat: collections.Counter = dataclasses.field(default_factory=collections.Counter)
    inclusive: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )
    operations: collections.Counter = dataclasses.field(
        default_factory=collections.Counter
    )


def _address_key(address):
    if isinstance(address, str):
        try:
            return int(address, 0)
        except ValueError:
            return address
    return address


class SymbolResolver:
    """Resolves Samply frame addresses using a presymbolication sidecar."""

    def __init__(self, profile, symbols):
        self.profile = profile
        self.by_code_id = {}
        strings = symbols.get("string_table", [])
        for module in symbols.get("data", []):
            code_id = module.get("code_id") or module.get("debug_id")
            if not code_id:
                continue
            symbol_table = module.get("symbol_table", [])
            address_to_name = {}
            for known_address in module.get("known_addresses", []):
                if not isinstance(known_address, list) or len(known_address) != 2:
                    continue
                address, symbol_index = known_address
                try:
                    string_index = symbol_table[symbol_index]["symbol"]
                    name = strings[string_index]
                except (IndexError, KeyError, TypeError):
                    continue
                address_to_name[_address_key(address)] = name
            self.by_code_id[str(code_id).upper()] = address_to_name

    def frame(self, thread, frame_index):
        frame_table = thread["frameTable"]
        function_table = thread["funcTable"]
        resource_table = thread["resourceTable"]
        strings = thread["stringArray"]

        function_index = frame_table["func"][frame_index]
        name_index = function_table["name"][function_index]
        fallback = strings[name_index]
        resource_index = function_table["resource"][function_index]
        if resource_index is None or resource_index < 0:
            return FrameSymbol(fallback, "<unknown>")

        library_index = resource_table["lib"][resource_index]
        library = self.profile["libs"][library_index]
        library_name = library.get("name", "<unknown>")
        code_id = str(library.get("codeId", "")).upper()
        address = _address_key(frame_table["address"][frame_index])
        name = self.by_code_id.get(code_id, {}).get(address, fallback)
        return FrameSymbol(name, library_name)


def _load_json(path, compressed=False):
    try:
        if compressed:
            with gzip.open(path, "rt", encoding="utf-8") as source:
                return json.load(source)
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError("Failed to read %s: %s" % (path, error)) from error


def symbols_path(profile_path):
    text = str(profile_path)
    if not text.endswith(PROFILE_SUFFIX):
        raise ProfileError("Profile must end in %s: %s" % (PROFILE_SUFFIX, text))
    return pathlib.Path(text[: -len(PROFILE_SUFFIX)] + SYMBOL_SUFFIX)


def expand_profile_paths(arguments):
    """Expands files, directories, and quoted glob expressions."""
    paths = {}
    missing = []
    for argument in arguments:
        candidate = pathlib.Path(argument).expanduser()
        if candidate.is_dir():
            matches = candidate.glob("*" + PROFILE_SUFFIX)
        else:
            matches = (pathlib.Path(path) for path in glob.glob(str(candidate)))
        found = False
        for match in matches:
            if match.is_file() and str(match).endswith(PROFILE_SUFFIX):
                found = True
                paths[str(match.resolve())] = match
        if not found:
            missing.append(argument)
    if missing:
        raise ProfileError("No profiles matched: %s" % ", ".join(missing))
    return [paths[key] for key in sorted(paths)]


def _unwind_stack(thread, resolver, stack_index):
    stack_table = thread["stackTable"]
    frames = []
    visited = set()
    current = stack_index
    while current is not None:
        if current in visited:
            raise ProfileError("Cycle in Samply stack table at index %s" % current)
        visited.add(current)
        frame_index = stack_table["frame"][current]
        frames.append(resolver.frame(thread, frame_index))
        current = stack_table["prefix"][current]
    return frames


def sample_weights(cpu_deltas, attribution):
    """Returns the CPU delta charged to each sampled stack."""
    if attribution == "current":
        return list(cpu_deltas)
    if attribution != "previous":
        raise ProfileError("Unknown delta attribution: %s" % attribution)
    if not cpu_deltas:
        return []
    weights = [0] * len(cpu_deltas)
    weights[0] = cpu_deltas[0]
    for index in range(len(cpu_deltas) - 1):
        weights[index] += cpu_deltas[index + 1]
    return weights


def classify_operation(frames, is_main_thread):
    if not frames:
        return UNRESOLVED_OPERATION
    names = tuple(frame.name for frame in frames)
    for operation, patterns in OPERATION_RULES:
        if any(pattern in name for pattern in patterns for name in names):
            return operation
    if is_main_thread and "main" not in names:
        return STARTUP_OPERATION
    return "Other CLI, encoder, and runtime"


def _is_resolved(name):
    return bool(name) and not name.startswith("0x")


def analyze_capture(path, attribution, analysis):
    sidecar_path = symbols_path(path)
    if not sidecar_path.is_file():
        raise ProfileError("Missing symbol sidecar for %s: %s" % (path, sidecar_path))

    profile = _load_json(path, compressed=True)
    symbols = _load_json(sidecar_path)
    resolver = SymbolResolver(profile, symbols)
    capture_samples = 0
    capture_cpu = 0

    try:
        threads = profile["threads"]
        for thread_index, thread in enumerate(threads):
            samples = thread["samples"]
            stacks = samples["stack"]
            cpu_deltas = samples["threadCPUDelta"]
            if len(stacks) != len(cpu_deltas):
                raise ProfileError(
                    "%s has %d stacks but %d thread CPU deltas"
                    % (path, len(stacks), len(cpu_deltas))
                )
            declared_length = samples.get("length")
            if declared_length is not None and declared_length != len(stacks):
                raise ProfileError(
                    "%s declares %d samples but contains %d"
                    % (path, declared_length, len(stacks))
                )
            if any(delta < 0 for delta in cpu_deltas):
                raise ProfileError("%s contains a negative thread CPU delta" % path)

            weights = sample_weights(cpu_deltas, attribution)
            capture_samples += len(stacks)
            capture_cpu += sum(cpu_deltas)
            is_main_thread = thread_index == 0

            for stack_index, weight in zip(stacks, weights):
                if weight <= 0:
                    continue
                if stack_index is None:
                    analysis.operations[UNRESOLVED_OPERATION] += weight
                    continue
                frames = _unwind_stack(thread, resolver, stack_index)
                operation = classify_operation(frames, is_main_thread)
                analysis.operations[operation] += weight

                leaf = frames[0]
                analysis.flat[leaf] += weight
                if _is_resolved(leaf.name):
                    analysis.resolved_leaf_cpu_us += weight

                seen = set()
                for frame in frames:
                    if frame not in seen:
                        analysis.inclusive[frame] += weight
                        seen.add(frame)
    except (IndexError, KeyError, TypeError) as error:
        raise ProfileError("Malformed Samply profile %s: %s" % (path, error)) from error

    analysis.sample_count += capture_samples
    analysis.cpu_delta_us += capture_cpu
    analysis.captures.append(CaptureSummary(str(path), capture_samples, capture_cpu))


def analyze_profiles(paths, attribution="current"):
    analysis = Analysis(delta_attribution=attribution, captures=[])
    for path in paths:
        analyze_capture(path, attribution, analysis)
    if analysis.cpu_delta_us <= 0:
        raise ProfileError("Profiles contain no positive sampled thread CPU")
    return analysis


def _percent(value, total):
    if total <= 0:
        return None
    return 100.0 * value / total


def _sorted_counter(counter):
    return sorted(
        counter.items(),
        key=lambda item: (
            -item[1],
            getattr(item[0], "name", str(item[0])),
            getattr(item[0], "library", ""),
        ),
    )


def _analysis_rows(analysis, top_functions):
    normalized_cpu = analysis.cpu_delta_us - sum(
        analysis.operations.get(operation, 0) for operation in NORMALIZED_EXCLUSIONS
    )
    operations = []
    for operation, cpu_delta in _sorted_counter(analysis.operations):
        operations.append(
            {
                "operation": operation,
                "cpu_delta_us": cpu_delta,
                "whole_command_percent": _percent(cpu_delta, analysis.cpu_delta_us),
                "encoder_plus_input_percent": (
                    None
                    if operation in NORMALIZED_EXCLUSIONS
                    else _percent(cpu_delta, normalized_cpu)
                ),
            }
        )

    def function_rows(counter):
        rows = []
        for frame, cpu_delta in _sorted_counter(counter)[:top_functions]:
            rows.append(
                {
                    "function": frame.name,
                    "library": frame.library,
                    "cpu_delta_us": cpu_delta,
                    "whole_command_percent": _percent(cpu_delta, analysis.cpu_delta_us),
                }
            )
        return rows

    return (
        normalized_cpu,
        operations,
        function_rows(analysis.flat),
        function_rows(analysis.inclusive),
    )


def analysis_as_json_value(analysis, top_functions):
    normalized_cpu, operations, flat, inclusive = _analysis_rows(
        analysis, top_functions
    )
    capture_cpu = [capture.cpu_delta_us for capture in analysis.captures]
    return {
        "schema_version": SCHEMA_VERSION,
        "delta_attribution": analysis.delta_attribution,
        "summary": {
            "capture_count": len(analysis.captures),
            "sample_count": analysis.sample_count,
            "cpu_delta_us": analysis.cpu_delta_us,
            "normalized_cpu_delta_us": normalized_cpu,
            "resolved_leaf_cpu_percent": _percent(
                analysis.resolved_leaf_cpu_us, analysis.cpu_delta_us
            ),
            "per_capture_cpu_delta_us": {
                "minimum": min(capture_cpu),
                "maximum": max(capture_cpu),
                "mean": statistics.fmean(capture_cpu),
            },
        },
        "captures": [dataclasses.asdict(capture) for capture in analysis.captures],
        "operations": operations,
        "flat_functions": flat,
        "inclusive_functions": inclusive,
    }


def _markdown_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def _format_percent(value):
    return "excluded" if value is None else "%.2f%%" % value


def render_markdown(analysis, top_functions):
    normalized_cpu, operations, flat, inclusive = _analysis_rows(
        analysis, top_functions
    )
    capture_cpu = [capture.cpu_delta_us for capture in analysis.captures]
    resolved_percent = _percent(analysis.resolved_leaf_cpu_us, analysis.cpu_delta_us)
    lines = [
        "# cjxl Samply profile analysis",
        "",
        "- Captures: %d" % len(analysis.captures),
        "- Samples: %d" % analysis.sample_count,
        "- Sampled thread CPU: %.3f ms" % (analysis.cpu_delta_us / 1000.0),
        "- Delta attribution: `%s`" % analysis.delta_attribution,
        "- Weighted leaf-symbol resolution: %.2f%%" % resolved_percent,
        "- Per-capture sampled CPU: %.3f–%.3f ms; mean %.3f ms"
        % (
            min(capture_cpu) / 1000.0,
            max(capture_cpu) / 1000.0,
            statistics.fmean(capture_cpu) / 1000.0,
        ),
        "",
        "## Operation-level CPU attribution",
        "",
        "| Operation | CPU delta | Whole command | Encoder plus input conversion* |",
        "|---|---:|---:|---:|",
    ]
    for row in operations:
        lines.append(
            "| %s | %.3f ms | %s | %s |"
            % (
                _markdown_cell(row["operation"]),
                row["cpu_delta_us"] / 1000.0,
                _format_percent(row["whole_command_percent"]),
                _format_percent(row["encoder_plus_input_percent"]),
            )
        )
    lines.extend(
        (
            "",
            "\\* The normalized column excludes process startup/static "
            "initialization and PNG decoding. Its denominator is %.3f ms."
            % (normalized_cpu / 1000.0),
            "",
            "## Hottest leaf functions",
            "",
            "| Flat CPU | CPU delta | Function | Library |",
            "|---:|---:|---|---|",
        )
    )
    for row in flat:
        lines.append(
            "| %s | %.3f ms | `%s` | `%s` |"
            % (
                _format_percent(row["whole_command_percent"]),
                row["cpu_delta_us"] / 1000.0,
                _markdown_cell(row["function"]),
                _markdown_cell(row["library"]),
            )
        )
    lines.extend(
        (
            "",
            "## Hottest inclusive functions",
            "",
            "| Inclusive CPU | CPU delta | Function | Library |",
            "|---:|---:|---|---|",
        )
    )
    for row in inclusive:
        lines.append(
            "| %s | %.3f ms | `%s` | `%s` |"
            % (
                _format_percent(row["whole_command_percent"]),
                row["cpu_delta_us"] / 1000.0,
                _markdown_cell(row["function"]),
                _markdown_cell(row["library"]),
            )
        )
    lines.extend(
        (
            "",
            "Percentages are sampled thread-CPU attribution, not wall-clock "
            "stage timings.",
            "",
        )
    )
    return "\n".join(lines)


def write_output(content, output_path):
    if output_path == "-":
        sys.stdout.write(content)
        if not content.endswith("\n"):
            sys.stdout.write("\n")
        return
    path = pathlib.Path(output_path).expanduser()
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise ProfileError("Failed to write %s: %s" % (path, error)) from error


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profiles",
        nargs="+",
        help="Samply .json.gz files, directories, or quoted glob expressions",
    )
    parser.add_argument(
        "--delta-attribution",
        choices=("current", "previous"),
        default="current",
        help="stack receiving each threadCPUDelta (default: current)",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="output representation (default: markdown)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="output file, or - for standard output (default: -)",
    )
    parser.add_argument(
        "--top-functions",
        type=int,
        default=20,
        help="number of flat and inclusive functions to emit (default: 20)",
    )
    args = parser.parse_args(argv)
    if args.top_functions < 0:
        parser.error("--top-functions must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        paths = expand_profile_paths(args.profiles)
        analysis = analyze_profiles(paths, args.delta_attribution)
        if args.format == "json":
            content = json.dumps(
                analysis_as_json_value(analysis, args.top_functions),
                indent=2,
                sort_keys=True,
            )
        else:
            content = render_markdown(analysis, args.top_functions)
        write_output(content, args.output)
    except ProfileError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
