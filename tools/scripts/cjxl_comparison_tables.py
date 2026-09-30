#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8,<4", "numpy>=1.26,<3", "pandas>=2.2,<3",
#                 "scipy>=1.13,<2", "jinja2>=3,<4"]
# ///
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Export effort-wise throughput and BD-rate tables from explicit local fixed sweeps.

Reads saved measurements only. Never builds, encodes, scores, or calibrates.
"""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libjxl-run", type=Path, required=True,
                        help="libjxl quality-study directory containing fixed-sweep scores")
    parser.add_argument("--gjxl-run", type=Path, required=True,
                        help="Metal or CUDA fixed-study directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="analysis directory outside both source studies and CPU timing run")
    parser.add_argument("--efforts", type=int, nargs="+", help="default: all libjxl efforts")
    parser.add_argument("--min-megapixels", type=float, default=1.0,
                        help="throughput cohort only; default: 1 MP")
    parser.add_argument("--quality-range", type=float, nargs=2, default=(75, 85),
                        metavar=("LOW", "HIGH"), help="BD-rate interval only; default: 75 85")
    args = parser.parse_args(argv)

    import cjxl_throughput_table as throughput
    from cjxl_runtime_characterization_notebook import generate_same_effort_bd_rate_table

    tables = throughput.build_tables(
        args.libjxl_run, args.gjxl_run, min_megapixels=args.min_megapixels,
        efforts=args.efforts,
    )
    # This writer rejects output paths inside any source study before writing.
    paths = throughput.write_tables(tables, args.output_dir)
    generate_same_effort_bd_rate_table(
        args.libjxl_run, args.gjxl_run, args.output_dir,
        quality_range=args.quality_range, efforts=args.efforts,
    )
    print("Saved encoding-throughput and bd-rate-same-effort tables in",
          args.output_dir.expanduser().resolve())
    print("Throughput methodology and coverage:", paths["methodology"])
    print("Throughput uses matched nominal settings; BD-rate uses measured quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
