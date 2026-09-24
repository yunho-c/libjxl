#!/usr/bin/env python3
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license in LICENSE.
"""Render saved CUDA data and optionally execute the analysis notebook; never encode."""
import argparse
import csv
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def notebook_cells(source):
    """Convert all percent-format cells; retain upstream cells without fixed counts."""
    import nbformat
    cells = [nbformat.v4.new_code_cell("%matplotlib inline")]
    block, markdown = [], False

    def flush():
        if any(line.strip() for line in block):
            body = "\n".join(re.sub(r"^# ?", "", line) for line in block) if markdown else "\n".join(block)
            factory = nbformat.v4.new_markdown_cell if markdown else nbformat.v4.new_code_cell
            cells.append(factory(body.strip()))

    for line in source.splitlines():
        if line.startswith("# %%"):
            flush()
            block = []
            markdown = "[markdown]" in line
        else:
            block.append(line)
    flush()
    return cells


def verified_studies(root, records):
    verified = read(root / "verification.json")
    for mode in ("fixed", "calibrated"):
        expected = verified[mode]
        actual = records[mode]
        if (not expected.get("accepted_results_valid")
                or not expected["coverage"].get("collection_finished")
                or expected["configuration_id"] != actual["configuration_id"]
                or expected["coverage"] != actual["coverage"]):
            raise ValueError(f"{mode} does not match a completed, audited study")
    return verified


def verify_figures(output, figures, records):
    from PIL import Image
    with (output / "coverage.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len({(row["run"], row["setting_id"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate coverage settings")
    for mode, record in records.items():
        coverage = record["coverage"]
        expected = coverage["expected_tuples" if mode == "fixed" else "expected_matches"]
        selected = [row for row in rows if row["run"] == mode]
        complete = coverage["scored_tuples" if mode == "fixed" else "timed_matches"]
        if len(selected) != expected or sum(row["status"] == "complete" for row in selected) != complete:
            raise ValueError(f"Incomplete coverage export for {mode}")
    hashes = {}
    for name in figures:
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Missing figure: {path}")
            if extension == "svg":
                ET.parse(path)
            else:
                with Image.open(path) as picture:
                    picture.verify()
            hashes[str(path.relative_to(output))] = sha(path)
    return {"coverage_rows": len(rows), "figure_count": len(figures), "artifact_sha256": hashes}


def execute_notebook(root, source, output, records):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import AsyncKernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    verified_studies(root, records)
    cells = notebook_cells(source.read_text(encoding="utf-8"))
    cells.append(nbformat.v4.new_code_cell(
        "import json as _json\nfrom pathlib import Path as _Path\n"
        f"_expected = _json.loads({json.dumps(records)!r})\n"
        "for _mode in ('fixed', 'calibrated'):\n"
        "    assert cuda_studies[_mode]['configuration_id'] == _expected[_mode]['configuration_id']\n"
        "    assert cuda_studies[_mode]['coverage'] == _expected[_mode]['coverage']\n"
        f"assert CUDA_FIXED_RUN.resolve() == _Path({str(root / 'fixed')!r})\n"
        f"assert CUDA_RUN.resolve() == _Path({str(root / 'calibrated')!r})\n"
        f"assert INPUT_CSV.resolve() == _Path({str(root / 'fixed/summary/image-tuples.csv')!r})\n"
        f"assert OUTPUT_DIR.resolve() == _Path({str(output)!r})\n"
        "assert 'coverage' in cuda_figures\nprint('CUDA_RUN_ALL_VERIFIED')\n"))
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata.update(source_file=str(source), source_sha256=sha(source))
    notebook.metadata["kernelspec"] = dict(name="cuda-analysis", display_name="CUDA analysis", language="python")
    kernel_root = output / "jupyter/kernels"
    kernel_dir = kernel_root / "cuda-analysis"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    environment = {"MPLBACKEND": "module://matplotlib_inline.backend_inline", "PYTHONUTF8": "1",
                   "CJXL_CUDA_STUDY_ROOT": str(root), "CJXL_CUDA_FIXED_RUN": str(root / "fixed"),
                   "CJXL_CUDA_RUN": str(root / "calibrated"), "CJXL_CHARACTERIZATION_PLOT_DIR": str(output),
                   "CJXL_IMAGE_TUPLES_CSV": str(root / "fixed/summary/image-tuples.csv")}
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "CUDA analysis", "language": "python", "env": environment,
    }, indent=2), encoding="utf-8")
    manager = AsyncKernelManager(kernel_name="cuda-analysis",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
    client = NotebookClient(notebook, km=manager, timeout=900, allow_errors=False,
                            resources={"metadata": {"path": str(source.parent)}})
    client.owns_km = True
    client.on_cell_start = lambda **kw: print(f"cell {kw['cell_index']}: {kw['cell']['cell_type']}", flush=True)
    destination = output / "cuda-analysis.executed.ipynb"
    try:
        client.execute()
    finally:
        nbformat.write(notebook, destination)
    outputs = [out for cell in notebook.cells if cell.cell_type == "code" for out in cell.get("outputs", [])]
    if any(out.output_type == "error" for out in outputs):
        raise ValueError("Notebook execution contains errors")
    if not any("CUDA_RUN_ALL_VERIFIED" in out.get("text", "") for out in outputs):
        raise ValueError("Notebook did not verify study identities")
    return {"source_sha256": notebook.metadata["source_sha256"], "cell_count": len(cells),
            "executed_code_cells": sum(cell.cell_type == "code" and cell.execution_count is not None for cell in cells),
            "inline_image_outputs": sum("image/png" in out.get("data", {}) or "image/svg+xml" in out.get("data", {}) for out in outputs),
            "errors": 0, "executed_notebook": str(destination), "collection_started": False}


def verify_same_pixels(before, after):
    # SVG includes generation timestamps and document IDs. Compare rendered PNG
    # bytes; still parse, hash and retain every final SVG for provenance.
    for key in ("coverage_rows", "figure_count"):
        if before[key] != after[key]:
            raise ValueError("Notebook figure coverage differs from direct rendering")
    pngs = lambda report: {name: digest for name, digest in report["artifact_sha256"].items()
                           if name.endswith(".png")}
    if pngs(before) != pngs(after):
        raise ValueError("Notebook PNG figures differ from direct saved-data rendering")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--output-dir", type=lambda p: Path(p).resolve())
    parser.add_argument("--notebook", type=lambda p: Path(p).resolve(),
                        default=Path(__file__).with_name("cjxl_runtime_characterization_notebook.py"))
    parser.add_argument("--execute", action="store_true", help="Run All after rendering; requires completed raw-data audit")
    args = parser.parse_args(argv)
    output = args.output_dir or args.study / "plots"
    if not (args.study / "fixed/metadata.json").is_file():
        parser.error("No initialized fixed study found")
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("cuda_runtime_figures", args.notebook)
    notebook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(notebook)
    destination = output / "gjxl-cuda"
    figures, records = notebook.generate_cuda_figures(args.study / "fixed", args.study / "calibrated",
                                                     destination, ("png", "svg"))
    report = verify_figures(destination, figures, records)
    report.update(source=str(args.notebook), source_sha256=sha(args.notebook), studies=records,
                  collection_started=False, visual_review="Inspect figures separately; not implied by structural checks")
    if args.execute:
        report["notebook_execution"] = execute_notebook(args.study, args.notebook, output, records)
        after = verify_figures(destination, figures, records)
        verify_same_pixels(report, after)
        report.update(after)
    items = []
    for name in figures:
        escaped = html.escape(name)
        items.append(f'<section><h2>{escaped}</h2><a href="gjxl-cuda/{escaped}.svg">SVG</a>'
                     f'<img loading="lazy" src="gjxl-cuda/{escaped}.png" alt="{escaped}"></section>')
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>GJXL CUDA saved-data analysis</title>'
        '<style>body{max-width:1200px;margin:2rem auto;font-family:system-ui}img{width:100%}section{margin:3rem 0}</style>'
        '<h1>GJXL CUDA saved-data analysis</h1><p>Missing settings remain explicit. '
        'See <a href="gjxl-cuda/coverage.csv">coverage</a>, '
        '<a href="gjxl-cuda/studies.json">study identities</a>, and '
        '<a href="analysis-verification.json">verification</a>.</p>' + ''.join(items), encoding="utf-8")
    (output / "analysis-verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("figure_count", "coverage_rows", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
