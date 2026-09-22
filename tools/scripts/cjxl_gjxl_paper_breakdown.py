#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8,<4", "numpy>=1.26,<3", "pandas>=2.2,<3"]
# ///
"""Publication figure from saved, paired GJXL profiles; never run encodes."""

import argparse
import json
from pathlib import Path
import shlex
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import cjxl_gjxl_paired_breakdown as paired


DEFAULT_PANELS = ("12mp",)
PANEL_ALIASES = {
    "kodak": "kodak_0_4mp", "clic": "clic_1_8_to_3_4mp",
    "unsplash12mp": "12mp", "unsplash24mp": "24mp", "unsplash48mp": "48mp",
}
GROUPS = {
    "input": ("Input preparation", "#DDCC77"),
    "search": ("GPU: AC search", "#4477AA"),
    "reconstruct": ("GPU: Transform / reconstruction", "#88CCEE"),
    "perceptual": ("GPU: Perceptual evaluation", "#117733"),
    "finalize": ("GPU: Quantization / finalization", "#44AA99"),
    "tokens": ("CPU: Tokenization", "#CC6677"),
    "entropy": ("CPU: Entropy optimization", "#AA4499"),
    "serialize": ("CPU: Other serialization", "#882255"),
    "remaining": ("Remaining elapsed time", "#D5D7DA"),
}
KEYS = ["resolution_class", "image_id", "setting", "effort", "sample_index"]


def resolve_panels(manifest, panels=None):
    """Resolve aliases to declared resolution classes, preserving caller order."""
    available = {i["resolution_class"].lower(): i["resolution_class"]
                 for i in manifest["images"]}
    requested = DEFAULT_PANELS if panels is None else panels
    if isinstance(requested, str):
        requested = (requested,)
    resolved = []
    for panel in requested:
        key = panel.strip().lower()
        key = PANEL_ALIASES.get(key, key)
        if key not in available:
            raise ValueError(f"Unknown or unavailable panel {panel!r}; available classes: "
                             + ", ".join(available.values()))
        resolution = available[key]
        if resolution in resolved:
            raise ValueError("Duplicate panel: " + resolution)
        resolved.append(resolution)
    if not resolved:
        raise ValueError("Select at least one panel")
    return tuple(resolved)


def panel_description(manifest, resolution):
    images = [i for i in manifest["images"] if i["resolution_class"] == resolution]
    title = paired.legacy.RESOLUTION_LABELS.get(resolution, resolution)
    if title.startswith("Unsplash "):
        title = title.replace("Unsplash ", "Unsplash, ", 1)
    count = f"{len(images)} image" + ("s" if len(images) != 1 else "")
    mps = sorted({i["width"] * i["height"] / 1e6 for i in images})
    sizes = " / ".join(f"{mp:.2f}" for mp in mps) + " MP"
    subtitle = f"{count}, {sizes}"
    if len(images) == 1 and images[0]["image_id"].startswith("unsplash/"):
        content = images[0]["image_id"].split("/")[1].replace("_", " ")
        subtitle = f"{count}, {content}"
    return title, subtitle, f"{title} ({count}, {sizes})"


def host_group(stage):
    if stage in (*paired.legacy.INPUT_GROUPS, "Other input preparation"):
        return "input"
    if stage in ("DC tokenization", "AC tokenization"):
        return "tokens"
    if stage == "Entropy optimization":
        return "entropy"
    if stage in ("Validation", "Section writing", "Assembly", "Other serializer"):
        return "serialize"
    if stage in (paired.PIPELINE_REMAINDER, "Other workflow", paired.OUTER):
        return "remaining"
    raise ValueError("Unmapped host stage: " + stage)


def gpu_group(stage):
    # Classify exact IDs so Gaborish is a filter and indirect dispatch setup
    # belongs to AC search, instead of inheriting the diagnostic 'Other GPU'.
    group = paired.legacy.gpu_group(stage)
    if group in ("Reference features", "Butteraugli comparison"):
        return "perceptual"
    if group == "AC strategy search" or stage == "aq.strategy_dispatch":
        return "search"
    if stage == "aq.gaborish" or group in (
            "Forward transform", "Trial coefficients", "Trial reconstruction", "Loop filtering"):
        return "reconstruct"
    if group in ("Initial quantization", "Quantizer adjustment", "AQ policy",
                 "Final CfL", "DC quantization", "Final coefficients"):
        return "finalize"
    raise ValueError("Unmapped GPU stage: " + stage)


def paper_data(report, panels=None):
    panels = resolve_panels(report["manifest"], panels)
    coverage = report["coverage"].query("kind == 'flat' and resolution_class in @panels")
    expected = {(res, e) for res in panels for e in report["manifest"]["efforts"]}
    actual = set(zip(coverage.resolution_class, coverage.effort))
    if actual != expected or not coverage.status.eq("complete").all():
        raise ValueError("Paper panels require complete image/effort coverage")
    flat = report["samples"].query("kind == 'flat' and resolution_class in @panels").copy()
    host = flat[~flat.stage.str.startswith("GPU: ")].copy()
    gpu = report["gpu_stages"].query("resolution_class in @panels").copy()
    host["group"] = host.stage.map(host_group)
    gpu["group"] = gpu.stage.map(gpu_group)
    mapped = pd.concat([host, gpu], ignore_index=True)
    # A GPU stage can be absent when it does no work. Materialize zero groups
    # for every call before averaging, including images with no such stage.
    index = pd.MultiIndex.from_frame(flat[KEYS].drop_duplicates().merge(
        pd.DataFrame({"group": list(GROUPS)}), how="cross"))
    samples = mapped.groupby(KEYS + ["group"]).ms.sum().reindex(
        index, fill_value=0).rename("ms").reset_index()
    original = flat.groupby(KEYS).ms.sum().sort_index()
    regrouped = samples.groupby(KEYS).ms.sum().sort_index()
    if not original.index.equals(regrouped.index) or not np.allclose(
            original, regrouped, rtol=1e-12, atol=1e-9):
        raise ValueError("Paper regrouping changed the measured complete-call partition")
    tuples = samples.groupby(KEYS[:-1] + ["group"], as_index=False).ms.mean()
    means = tuples.groupby(["resolution_class", "effort", "group"], as_index=False).ms.mean()
    means["percent"] = 100 * means.ms / means.groupby(
        ["resolution_class", "effort"]).ms.transform("sum")
    mapping = mapped[["stage", "group"]].drop_duplicates().sort_values(["group", "stage"])
    return samples, means, mapping


def make_figure(means, manifest, panels=None, *, show_panel_titles=False,
                legend_position="bottom"):
    if legend_position not in ("bottom", "right"):
        raise ValueError("legend_position must be 'bottom' or 'right'")
    panels = resolve_panels(manifest, panels)
    style = {
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
        "axes.labelsize": 9, "axes.linewidth": .6, "axes.edgecolor": ".25",
        "text.color": ".12", "axes.labelcolor": ".12", "xtick.color": ".2",
        "ytick.color": ".2", "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.6, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "gjxl-paper-breakdown",
        "hatch.linewidth": .32, "figure.facecolor": "white", "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        # Keep the 7-inch publication width. With a side legend, stack panels
        # vertically so the ten timing annotations in each panel stay legible.
        side_legend = legend_position == "right"
        columns = 1 if side_legend else min(2, len(panels))
        rows = (len(panels) + columns - 1) // columns
        plot_height = 2.03
        bottom_margin = .46 if side_legend else 1.04
        top_margin = .58 if show_panel_titles else .22
        row_gap = .88 if show_panel_titles else .5
        height = rows * plot_height + (rows - 1) * row_gap + top_margin + bottom_margin
        fig, axes = plt.subplots(rows, columns, figsize=(7.0, height),
                                 sharey=True, squeeze=False)
        fig.subplots_adjust(left=.078, right=.625 if side_legend else .993,
                            top=1-top_margin/height,
                            bottom=bottom_margin/height, wspace=.10,
                            hspace=row_gap/plot_height)
        for index, (ax, res) in enumerate(zip(axes.flat, panels)):
            ax.set_label(res)
            part = means[means.resolution_class == res]
            efforts = manifest["efforts"]
            values = part.pivot(index="effort", columns="group", values="percent").reindex(
                index=efforts, columns=GROUPS).fillna(0)
            if not np.allclose(values.sum(axis=1), 100, atol=1e-9):
                raise ValueError("A paper bar does not sum to 100 percent")
            bottom = np.zeros(len(efforts))
            for group, (_, color) in GROUPS.items():
                y = values[group].to_numpy()
                ax.bar(efforts, y, bottom=bottom, width=.78, color=color,
                       edgecolor="white", linewidth=.28, zorder=3)
                if group == "remaining":
                    ax.bar(efforts, y, bottom=bottom, width=.78, facecolor="none",
                           edgecolor="#8A8D91", linewidth=0, hatch="///", zorder=4)
                bottom += y
            totals = part.groupby("effort").ms.sum()
            for effort, ms in totals.items():
                decimals = 1 if ms < 100 else 0
                label = f"{ms:.{decimals}f}"
                ax.text(effort, 103.8, label, ha="center", va="bottom", fontsize=7.0)
            ax.set(xlim=(.35, 10.65), ylim=(0, 113), xticks=efforts,
                   yticks=[0, 25, 50, 75, 100], xlabel="Effort")
            ax.tick_params(axis="both", length=2.8, width=.6, pad=3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_bounds(0, 100)
            ax.grid(axis="y", color=".90", linewidth=.45, zorder=0)
            ax.text(0, 1.0, "Mean profiled time (ms)", transform=ax.transAxes,
                    fontsize=7.2, color=".35")
            if index % columns == 0:
                ax.set_ylabel("Encode time (%)", labelpad=6)
            else:
                ax.tick_params(axis="y", left=False)
                ax.spines["left"].set_visible(False)
            if show_panel_titles:
                bounds = ax.get_position()
                title, subtitle, _ = panel_description(manifest, res)
                fig.text(bounds.x0, bounds.y1 + .44/height,
                         f"({chr(ord('a') + index)})  {title}",
                         fontsize=10.2, fontweight="bold", va="top")
                fig.text(bounds.x0, bounds.y1 + .25/height, subtitle,
                         fontsize=8.0, color=".35", va="top")
        for ax in list(axes.flat)[len(panels):]:
            fig.delaxes(ax)
        # Legend order follows the bottom-to-top stack: by row for the bottom
        # legend, top-to-bottom for the single-column right legend.
        handles = [Patch(facecolor=color, edgecolor="#8A8D91" if k == "remaining" else "white",
                         linewidth=.35, hatch="///" if k == "remaining" else None, label=label)
                   for k, (label, color) in GROUPS.items()]
        if side_legend:
            grid_center = (bottom_margin + height - top_margin) / (2 * height)
            fig.legend(handles=handles, ncols=1, loc="center left",
                       bbox_to_anchor=(.65, grid_center), frameon=False,
                       handlelength=1.65, handleheight=.85, handletextpad=.55,
                       labelspacing=.9, borderaxespad=0)
        else:
            order = [0, 3, 6, 1, 4, 7, 2, 5, 8]
            fig.legend(handles=[handles[i] for i in order], ncols=3, loc="lower center",
                       bbox_to_anchor=(.52, .065/height), frameon=False, handlelength=1.65,
                       handleheight=.85, columnspacing=1.55, handletextpad=.55, labelspacing=.7)
        return fig


def make_caption(manifest, panels=None):
    c = manifest
    panels = resolve_panels(c, panels)
    selection = "; ".join(panel_description(c, res)[2] for res in panels)
    return (
        "Runtime composition of the fully resident Metal GJXL encoder at nominal Q80 "
        f"(distance 1.9), for {selection}. Each bar partitions the complete profiled "
        "encode-call time; numbers above bars give mean milliseconds per encode. "
        f"{c['samples']} repetitions are averaged per image, then images receive equal "
        "weight. GPU intervals and host phases come from the same encode, with "
        "nonoverlapping attribution and explicit elapsed-time residuals. Perceptual "
        "evaluation includes Butteraugli reference features and comparisons. Hatched "
        "remaining time includes pipeline orchestration, gaps and outer workflow work; "
        "it is not attributed wholly to CPU or GPU execution. Measurements use Apple "
        f"M4 Pro (20 GPU cores), {c['cpu_threads']} CPU participants and GJXL revision "
        f"{c['source_revision'][:7]}. Profiling perturbs execution; no scaling to "
        "ordinary-run timings is applied. Input loading and backend creation are excluded."
    )


def export(config_path, output_dir, panels=None, *, show_panel_titles=False,
           legend_position="bottom"):
    report = paired.load_profiles(config_path)
    c = report["manifest"]
    panels = resolve_panels(c, panels)
    samples, means, mapping = paper_data(report, panels)
    if c["quality"] != 80 or c["distance"] != 1.9 or c["efforts"] != list(range(1, 11)):
        raise ValueError("This paper caption/layout expects nominal Q80 and efforts 1-10")
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    figure = make_figure(means, c, panels, show_panel_titles=show_panel_titles,
                         legend_position=legend_position)
    size_inches = figure.get_size_inches().tolist()
    stem = "gjxl-runtime-breakdown-paper"
    # The PDF/SVG backends read these settings when saving, after make_figure's
    # styling context has ended. Keep vector text editable and fonts embedded.
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "gjxl-paper-breakdown",
                         "hatch.linewidth": .32}):
        for ext in ("pdf", "svg", "png"):
            figure.savefig(out/f"{stem}.{ext}", dpi=600)
    plt.close(figure)
    samples.to_csv(out/f"{stem}-samples.csv", index=False)
    means.to_csv(out/f"{stem}-means.csv", index=False)
    mapping.to_csv(out/f"{stem}-stage-map.csv", index=False)
    caption = make_caption(c, panels)
    (out/f"{stem}-caption.md").write_text(caption + "\n")
    (out/f"{stem}.tex").write_text(
        "\\begin{figure*}[t]\n  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n"
        f"  \\caption{{{caption}}}\n"
        "  \\label{fig:gjxl-runtime-breakdown}\n\\end{figure*}\n")
    sources = [Path(__file__), Path(paired.__file__), Path(paired.legacy.__file__)]
    (out/f"{stem}-methodology.json").write_text(json.dumps({
        "config": str(Path(config_path).resolve()), "config_sha256": paired.digest(config_path),
        "encoder_revision": c["source_revision"], "nominal_quality": c["quality"],
        "distance": c["distance"], "panels": panels, "groups": GROUPS,
        "show_panel_titles": show_panel_titles,
        "legend_position": legend_position,
        "selected_images": [i for i in c["images"] if i["resolution_class"] in panels],
        "boundary": paired.SEMANTICS["flat"], "aggregation": paired.SEMANTICS["aggregation"],
        "size_inches": size_inches, "dpi": 600, "scaling": "none",
        "grouping": "Exact GPU stages remapped; sample complete-call totals unchanged.",
        "sample_partitions_checked": samples[KEYS].drop_duplicates().shape[0],
        "bar_count": means[["resolution_class", "effort"]].drop_duplicates().shape[0],
        "data_scope": "Selected images; content and resolution vary together; not matched quality.",
        "generator_sha256": paired.digest(__file__),
        "source_sha256": {p.name: paired.digest(p) for p in sources},
        "software": {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2) + "\n")
    for source in sources:
        target = out/source.name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
    (out/"README.md").write_text(
        "# GJXL paper runtime figure\n\n"
        "Use the vector PDF at its native 7-inch width (a two-column figure). "
        "The SVG retains editable text; the PNG is a 600-dpi export. "
        "The `.tex` file supplies a figure environment and caption.\n\n"
        "## Reproduce\n\n"
        "Run from this directory, using the archived scripts and original saved captures:\n\n"
        "```sh\nuv run cjxl_gjxl_paper_breakdown.py \\\n"
        f"  --config {shlex.quote(str(Path(config_path).resolve()))} \\\n"
        f"  --panels {shlex.join(panels)} \\\n"
        f"  --legend-position {legend_position} \\\n"
        + ("  --show-panel-titles \\\n" if show_panel_titles else "") +
        "  --output-dir .\n```\n\n"
        "This command only reads saved measurements; it never runs the encoder. "
        "The loader verifies the input and capture hashes, settings, complete cohort "
        "coverage, and nonoverlapping timing intervals. Regrouping must preserve "
        "each complete-call total.\n\n"
        "Panel titles and subtitles are hidden by default; enable both with "
        "`--show-panel-titles`. Use `--panels 12mp` (the default), or choose "
        "multiple panels in display order, e.g. `--panels kodak 48mp`. "
        "Available aliases include `kodak`, `clic`, `12mp`, `24mp`, and `48mp`; "
        "canonical resolution-class names from the capture config also work. "
        "Use `--legend-position bottom` (the default) or `--legend-position right`. "
        "With a bottom legend, the layout uses at most two panels per row. "
        "With a right legend, panels stack vertically to preserve readable "
        "timing labels within the 7-inch figure width.\n\n"
        "## Reading the figure\n\n"
        "Bars show raw profiled runtime shares, without ordinary-run scaling. "
        "Repetitions are averaged within each image, then images are weighted equally. "
        "Percentages are ratios of those mean stage times to mean total time, "
        "rather than averages of per-image percentages. Numbers above the bars are "
        "profiled milliseconds per encode, rounded to 0.1 ms below 100 ms and "
        "1 ms otherwise. The CSVs retain full precision.\n\n"
        "The transform/reconstruction group includes forward transforms, trial "
        "coefficients, trial reconstruction, Gaborish and loop filtering. Perceptual "
        "evaluation includes reference features and Butteraugli comparisons. Exact "
        "GPU stage IDs and host phases are listed in `*-stage-map.csv`. Input "
        "preparation is a host wall-time interval that can include GPU work. "
        "The hatched residual is remaining elapsed time, including orchestration, "
        "gaps and outer workflow work; it is not a CPU-only category.\n\n"
        "These are selected images, not a whole-corpus mean or a controlled "
        "resolution-scaling experiment. Profiling can change stage times, and the "
        "collection was not a device-isolated idle experiment. Use the ordinary-run "
        "measurements for production latency claims. Nominal Q80 is not a "
        "measured-quality match. The capture revision, hashes, aggregation, "
        "software versions and output dimensions are in `*-methodology.json`.\n")
    return out/f"{stem}.pdf"


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--panels", nargs="+", default=DEFAULT_PANELS, metavar="CLASS",
                        help="Panels in display order (default: 12mp). Use kodak, clic, "
                             "12mp, 24mp, 48mp, or resolution classes from the config.")
    parser.add_argument("--show-panel-titles", action="store_true",
                        help="Show subplot titles and subtitles (hidden by default).")
    parser.add_argument("--legend-position", choices=("bottom", "right"), default="bottom",
                        help="Stage legend placement (default: bottom). A right legend "
                             "stacks multiple panels vertically at the same figure width.")
    return parser


if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    print(export(args.config, args.output_dir, args.panels,
                 show_panel_titles=args.show_panel_titles, legend_position=args.legend_position))
