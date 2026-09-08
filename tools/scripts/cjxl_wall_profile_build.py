#!/usr/bin/env python3
"""Build and fingerprint the expanded profiler without committing its worktree.

The source must already contain the wall-v2 patch. Uses a separate build tree
so old stage runs retain their original binaries. Rebuilding a tree already
used by a measurement run is intentionally refused.
"""

import argparse
import hashlib
import json
import pathlib
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--comparison-repo", type=pathlib.Path, required=True)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    comparison = args.comparison_repo.resolve()
    root = args.build_root.resolve()
    if (root / "build-manifest.json").exists():
        parser.error("Build manifest already exists; choose a new build root")
    root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    patch = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--binary", "HEAD"]
    )
    (root / "instrumentation.patch").write_bytes(patch)
    harness_source = (
        comparison / "benchmarks/libjxl_comparison/libjxl_comparison_benchmark.cpp"
    )
    (root / "libjxl_comparison_benchmark.cpp").write_bytes(harness_source.read_bytes())
    commands = [
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(root / "libjxl"),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_SHARED_LIBS=ON",
            "-DJPEGXL_ENABLE_TOOLS=ON",
            "-DJPEGXL_ENABLE_DEVTOOLS=ON",
            "-DJPEGXL_ENABLE_BENCHMARK=OFF",
            "-DJPEGXL_ENABLE_STAGE_PROFILER=ON",
            "-DJPEGXL_ENABLE_EXAMPLES=OFF",
            "-DJPEGXL_ENABLE_JNI=OFF",
            "-DJPEGXL_ENABLE_MANPAGES=OFF",
            "-DJPEGXL_ENABLE_DOXYGEN=OFF",
            "-DJPEGXL_ENABLE_SJPEG=OFF",
            "-DJPEGXL_ENABLE_OPENEXR=OFF",
            "-DJPEGXL_ENABLE_VIEWERS=OFF",
            "-DJPEGXL_ENABLE_TRANSCODE_JPEG=OFF",
            "-DHWY_ENABLE_TESTS=OFF",
            "-DHWY_ENABLE_EXAMPLES=OFF",
        ],
        [
            "cmake",
            "--build",
            str(root / "libjxl"),
            "--target",
            "jxl",
            "jxl_threads",
            "djxl",
            "-j",
            "8",
        ],
        [
            "cmake",
            "-S",
            str(comparison / "benchmarks/libjxl_comparison"),
            "-B",
            str(root / "harness"),
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGJXL_LIBJXL_SOURCE=" + str(source),
            "-DGJXL_LIBJXL_BUILD=" + str(root / "libjxl"),
            "-DGJXL_LIBJXL_REVISION=" + revision,
            "-DGJXL_LIBJXL_STAGE_PROFILE=ON",
        ],
        ["cmake", "--build", str(root / "harness"), "-j", "8"],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    files = {
        "benchmark": root / "harness/gjxl_libjxl_comparison_benchmark",
        "library": root / "libjxl/lib/libjxl.dylib",
        "threads": root / "libjxl/lib/libjxl_threads.dylib",
        "decoder": root / "libjxl/tools/djxl",
        "instrumentation_patch": root / "instrumentation.patch",
        "harness_source": root / "libjxl_comparison_benchmark.cpp",
        "cmake_cache": root / "libjxl/CMakeCache.txt",
    }
    manifest = {
        "schema_version": 3,
        "wall_profile_version": 2,
        "libjxl_revision": revision,
        "source": str(source),
        "commands": commands,
        "files": {
            name: {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in files.items()
        },
    }
    (root / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Frozen build manifest:", root / "build-manifest.json")


if __name__ == "__main__":
    main()
