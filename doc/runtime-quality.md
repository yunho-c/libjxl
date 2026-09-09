# Matched-quality speed–compression study

This workflow scores retained encodes, produces an interpolated preview,
calibrates encoder distance to measured quality, and times accepted settings.
It is separate from the original runtime and wall-v2 ledgers. Importing its
module or running the notebook never starts collection.

`cjxl_quality_characterization.py` owns collection, calibration, and CSV
summaries. All drawing functions—including Pareto, rate–quality, metric-comparison,
runtime, and stage plots—live directly in
`cjxl_runtime_characterization_notebook.ipynb`. The quality
CLI exports data only and never imports the notebook or plotting dependencies.
The notebook imports the CLI's saved-data helpers, not the other way around.

## Figure semantics

Default targets are **fast-ssim2 60, 70 and 85, with ±0.5-point tolerance**.
The adapter pins [imazen/fast-ssim2 c3867954](https://github.com/imazen/fast-ssim2/tree/c3867954c7bec8a761951df9256354b305fd0cff).
Do not mix scores from different implementations or versions. Matching
happens per image and effort, before aggregation.

- X: total compressed bits / total **original** pixels; left is better.
- Y: total original megapixels / sum of per-image median complete-encode
  seconds; logarithmic, up is better.
- Open markers: **interpolated preview**. Size and runtime are estimates.
- Filled markers: **measured**, requiring an actual in-tolerance score and
  five independent uninstrumented timing samples.
- Blue lines connect efforts in order; missing efforts break the line.
  Dashed black lines connect the nondominated points (Pareto frontier).

Every point must cover the full selected image cohort. Images are not silently
removed when a target cannot be matched. Original partial timing sets can
contribute to previews and are flagged in the CSV. Incomplete new timing sets
cannot contribute to measured points. Pilot subsets are examples, not
estimates of the full Kodak/CLIC/Unsplash distributions.

Preview interpolation is linear in measured score and logarithmic in size and
elapsed time. It requires an unambiguous locally monotone adjacent interval.
No extrapolation, sorting by score to hide reversals, or interpolation across
the baseline's internal downsampling transition at requested distance 10 is
allowed. Diagnostic rate–quality curves show all scored points, including
intervals rejected by preview interpolation.

## Build and initialize

Run commands from the libjxl checkout. The collector uses the Python standard
library; the notebook needs matplotlib >=3.8, NumPy >=1.26, and pandas >=2.2.
Python 3.13 on the study machine already includes the notebook dependencies.
Use the installed stable
Rust toolchain; the machine's older default nightly does not compile the
locked SIMD dependency. This does not change the global default toolchain.

    cargo +stable build --release --locked --manifest-path tools/scripts/quality_metric/Cargo.toml
    cargo +stable test --release --locked --manifest-path tools/scripts/quality_metric/Cargo.toml
    python3.13 -m unittest discover -s tools/scripts -p cjxl_quality_characterization_test.py

Initialize the bounded pilot (one shell command):

    python3.13 tools/scripts/cjxl_quality_characterization.py init --run ../libjxl-runtime-study-2026-09-03/quality-pilot-20260908 --corpus ../libjxl-runtime-study-2026-09-03/corpus/corpus.json --tuples ../libjxl-runtime-study-2026-09-03/run-e8ff0976-wall-v2/summary/image-tuples.partial.csv --source-run ../libjxl-runtime-study-2026-09-03/run-e8ff0976 --scorer tools/scripts/quality_metric/target/release/cjxl-quality-metric --pilot

Initialization freezes selected IDs, targets, tolerance, the source CSV,
source metadata identity, reference hashes, baseline harness/decoder and local
shared-library hashes, and the metric binary identity. Existing directories
cannot be reinitialized. Original results, codestreams and frozen builds are
never modified.

The adapter reads RGB float PFM, respecting scale, byte order and bottom-up
rows. It rejects malformed/nonfinite samples and mismatched dimensions.
Decoding explicitly requests RGB_D65_SRG_Rel_Lin, matching the original linear
sRGB references. There is no PNG conversion or 8-bit quantization.
The adapter caches the reference at <=8 MP and uses 256-row strips above 8 MP
to bound metric working memory. Original/decoded pixel buffers still consume
memory. Only one scorer/reference is resident at once, and scoring processes
are closed before timing.

## Bounded pilot and resume

The pilot uses the first three sorted Kodak IDs (01, 02, 03). It scores their
existing seven-quality outputs at efforts 1–10: 210 comparisons. New
calibration/timing is limited to efforts 3, 5, 7: 27 image/effort/target jobs.

    python3.13 tools/scripts/cjxl_quality_characterization.py pilot --run ../libjxl-runtime-study-2026-09-03/quality-pilot-20260908 --dry-run
    caffeinate -i -m -s python3.13 tools/scripts/cjxl_quality_characterization.py pilot --run ../libjxl-runtime-study-2026-09-03/quality-pilot-20260908 --budget-seconds 900

The shared budget covers scoring, calibration and measurement. Plotting is
a separate notebook operation; no collection command renders figures.
SIGINT and deadline/timeout handling terminate child process groups and retain
completed records. Repeat a collection command to resume. Budgets are per
invocation and never automatically renewed. Collection locks the quality run
and refuses to overlap the original runtime-sweep runner. Avoid running other
benchmarks, builds or CPU-heavy jobs concurrently with measurement.

Calibration searches requested distance within [0.55, 15.266666666667],
reusing existing scores and probes. It reconsiders adjacent measured brackets,
uses midpoint refinement, and tries endpoints/coarse probes when necessary.
Unresolved searches are explicitly unavailable, not proof that the target
is mathematically unattainable. At most 12 new probes per target are allowed,
including previous attempts after resume. Accept only an actual score within
tolerance, selecting by smallest absolute error, then encoded size, then
distance. Accepted selections remain fixed for subsequent measurement.

Measurement uses eight workers, one warmup and one measured sample in each
of five independent processes. The harness also performs its own untimed
validation encode. Every measured codestream must match the scored hash.
Probes never count as timing samples. I/O, process startup, decoding, scoring
and calibration are outside the complete-encode API timing boundary.
Repetitions use deterministic effort-major shuffled image/target order.

## Commands and outputs

Replace RUN below with an initialized quality-study directory:

    python3.13 tools/scripts/cjxl_quality_characterization.py score --run RUN --max-jobs 10
    python3.13 tools/scripts/cjxl_quality_characterization.py calibrate --run RUN --max-jobs 3
    python3.13 tools/scripts/cjxl_quality_characterization.py measure --run RUN --max-jobs 5
    python3.13 tools/scripts/cjxl_quality_characterization.py preview --run RUN
    python3.13 tools/scripts/cjxl_quality_characterization.py summarize --run RUN

Collection commands accept --budget-seconds, --max-jobs and --dry-run.
A job is one scored original output, one calibration outcome, or one timing
sample, respectively. A subprocess has a default 300-second timeout, bounded
by the remaining invocation budget. Initialize future large/high-effort runs
with a larger --timeout rather than editing frozen configuration.

Append-only JSONL ledgers: scores, probes, calibration and measurements.
Separate command/event logs record provenance and collection durations.
Each row carries a configuration ID. Probe codestreams and raw harness reports
are retained; decoded PFMs are temporary. Changed references, tools or outputs
fail resume checks. An invalid/torn ledger tail requires inspection and is
never silently discarded.

The summary directory contains:

- quality-scores.csv: scored original sweep outputs.
- preview-matches.csv and measured-matches.csv: per-image target results,
  status, score/error and timing coverage.
- preview-points.csv and measured-points.csv: fixed-cohort aggregate points.
- coverage.json: counts of complete matches and aggregate points.

Both `preview` and `summarize` export data only; they do not create, update, or
delete figures. Any figures left in an older run's summary directory are historical.
The notebook draws preview/measured Pareto and per-image rate–quality diagnostics,
saving PNG/SVG files in `OUTPUT_DIR` (or `--output-dir` in its script form).

The notebook accepts QUALITY_RUN or the CJXL_QUALITY_RUN environment variable.
Its script form accepts --quality-run RUN alongside the normal timing CSV.
Edit the notebook Python source first, then derive/execute its Jupyter form:

    jupytext --to ipynb --execute tools/scripts/cjxl_runtime_characterization_notebook.py

For future full collection, initialize a new directory using an explicit
comma-separated --images list and --measurement-efforts 1,2,3,4,5,6,7,8,9,10.
Targets, tolerance, evaluation cap and timeout are configurable at init.
Full-corpus collection and new effort-8–10 encodes are not part of this pilot.

## Full per-image SSIMU2 run

The separate `quality-full-20260908` run includes all 65 images: 24 Kodak,
32 CLIC test, and three images each at approximately 12, 24 and 48 MP. It
calibrates efforts 1–10 to fast-ssim2 60, 70 and 85 (±0.5), producing up to
1,950 matched settings and 9,750 independent timing samples. This is per-image
calibration, not matching the corpus-average score.

The `run` command processes efforts from low to high. At each effort it
scores existing outputs, calibrates every image/target, then performs the five
timing repetitions before advancing. Thus the slowest efforts do not delay
matched outputs and timing results for lower efforts. Partial summary CSVs and
`progress.json` refresh after every phase. Every accepted calibration retains
a codestream in `matched/`, with image/effort/target and path recorded in
`calibration.jsonl`; repeated timing encodes must be byte-identical to it.

For this full run, distance bounds are explicitly widened to [0.01, 25] so
the old seven-quality grid does not limit the attainable target range. Up to
24 new probes per image/effort/target are allowed, with a two-hour subprocess
timeout for the largest/highest-effort jobs. Failed target searches are
reported as incomplete, never silently counted as matched.

Measured quality is not necessarily continuous or monotone in requested
distance: a codec decision can jump across the entire accepted score band.
An unresolved search is not proof that the target is unattainable, and does
not authorize widening the tolerance. Keep its probes for a separate,
provenance-recorded follow-up; the remaining settings continue normally.
During a phase, the append-only ledgers are more current than `progress.json`.

Initialization retains `collector.py` and its hash alongside the immutable
configuration and source CSV. New runs do not copy or hash the notebook;
the standalone collector needs only the standard library to export summaries.
Existing run snapshots are not rewritten, so older frozen collectors retain
their original behavior. Later checkout edits do not change the overnight
process's code. Resume the full pipeline with:

    caffeinate -i -m -s python3.13 -u ../libjxl-runtime-study-2026-09-03/quality-full-20260908/collector.py run --run ../libjxl-runtime-study-2026-09-03/quality-full-20260908

The initial launch is detached and logs to `runner.log`. `launcher.json`
records the launched process-group leader; `runner.json` identifies the Python
runner. On macOS, caffeinate may exec the runner in that original PID and keep
a separate helper process holding the sleep-prevention assertions.
To pause, confirm that runner PID still belongs to this exact command, then
send it SIGINT. In-flight subprocesses are terminated and completed records
remain resumable. Never launch a second instance while its process is alive.
Process liveness, not the presence of a PID file alone, determines whether it
is running. The per-run lock also rejects overlapping collectors.

The run has no automatic elapsed-time cutoff. Caffeinate prevents idle sleep
while it runs. Existing pilot results and original timing/stage ledgers are
not modified. Efforts 8–10 may require substantially more than one night;
low-effort results become available first.

## Butteraugli alternative (preview only)

Use a **separate run directory** to rescore the same retained codestreams and
reuse exactly the original timing samples. No Rust changes or new encodes are
needed. Butteraugli configurations reject `calibrate`, `measure`, and `pilot`,
including through the Python collection entrypoints. Only `score` collects
new observations. It uses the same budgets, interruption handling, hash checks,
run lock, and resumable JSONL ledger as fast-ssim2.

    python3.13 tools/scripts/cjxl_quality_characterization.py init --metric butteraugli --run ../libjxl-runtime-study-2026-09-03/quality-butteraugli-pilot-20260908 --corpus ../libjxl-runtime-study-2026-09-03/corpus/corpus.json --tuples ../libjxl-runtime-study-2026-09-03/quality-pilot-20260908/source-tuples.csv --source-run ../libjxl-runtime-study-2026-09-03/run-e8ff0976 --scorer ../gjxl-libjxl-comparison/build/libjxl-comparison/libjxl/tools/butteraugli_main --pilot
    python3.13 tools/scripts/cjxl_quality_characterization.py score --run ../libjxl-runtime-study-2026-09-03/quality-butteraugli-pilot-20260908 --dry-run
    caffeinate -i -m -s python3.13 tools/scripts/cjxl_quality_characterization.py score --run ../libjxl-runtime-study-2026-09-03/quality-butteraugli-pilot-20260908 --budget-seconds 900
    python3.13 tools/scripts/cjxl_quality_characterization.py preview --run ../libjxl-runtime-study-2026-09-03/quality-butteraugli-pilot-20260908 --compare-run ../libjxl-runtime-study-2026-09-03/quality-pilot-20260908

`--pilot` at initialization selects the same three Kodak images, with 210
scoring jobs at efforts 1–10 and zero calibration/measurement jobs. Repeating
`score` resumes without repeating completed observations. Reusing the frozen
fast-ssim2 `source-tuples.csv` ensures both metrics refer to identical inputs,
codestreams, and timing samples, even if the main sweep summary later changes.

The scorer must be `butteraugli_main` alongside the frozen `djxl`. Metadata
records the binary hash, source-build revision, shared-library hashes, color
encoding, and metric settings. Both input PFMs are explicitly interpreted as
`RGB_D65_SRG_Rel_Lin`, with `--intensity_target 80` (SDR viewing nits),
`hf_asymmetry=1`, and `xmul=1`. Change viewing intensity at initialization with
`--intensity-target`, not by modifying frozen metadata.

The primary score is the tool's conventional **Butteraugli distance** (its first
output line), not the encoder's requested distance. Lower is better. The
auxiliary `3-norm` is retained as `butteraugli_pnorm3`, along with raw metric
stdout, but is not used for matching. Every retained encode is decoded with
the pinned decoder to linear float PFM before scoring. Scoring is outside the
previously recorded encode timing boundary.

Butteraugli's default preview targets are **1, 2, 3**. Use `--targets` at
initialization to choose other distances. These numbers have no direct
equivalence to the fast-ssim2 targets 60, 70, 85. The interpolation direction
is reversed, but retains the same log-size/log-time interpolation, monotonicity,
no-extrapolation, fixed-cohort, and resampling-boundary safeguards. A target
outside even one image's supported range makes that effort's pooled point
unavailable; the plot and CSV report the missing coverage explicitly.

`preview --compare-run` additionally writes `effort4-vs5-preview.csv`: per-image
and (for a single corpus/resolution) pooled e5-versus-e4 size percentage changes
and encode-time ratios, at each metric's own targets. Positive size change means
e5 produces larger files. Missing comparisons remain unavailable rather than
becoming zeros.

The notebook saves `butteraugli-effort4-vs5-metrics.png` and `.svg`: one column
per image, one row per metric, showing e4/e5 scores on identical retained
encodes. These are raw observations joined by diagnostic lines, not interpolations.
Lines break at the encoder's internal resampling boundary.

The comparison validates identical reference cohorts and frozen source CSVs,
and checks matching observations' codestream hashes, decoded-PFM hashes,
sizes, and timing fields. It never overwrites the alternative metric's run.
Metric identifiers also appear in the aggregate CSVs and prevent accidental
cross-metric pooling.

The notebook's `BUTTERAUGLI_RUN`/`CJXL_BUTTERAUGLI_RUN` points to the new run.
Section 8 loads its preview and uses `BUTTERAUGLI_COMPARE_RUN` (overridable via
`CJXL_BUTTERAUGLI_COMPARE_RUN`) for the paired e4/e5 diagnostics. This keeps
the comparison on the matching three-image pilot even when `QUALITY_RUN`
points to the full SSIMU2 study. The notebook accepts `--butteraugli-run RUN` alongside
`--quality-run SSIM_RUN`. Butteraugli figures use a filename prefix to avoid
overwriting the existing plots. Neither notebook path starts collection.

Use the comparison to test **metric sensitivity**, not to declare a universal
effort winner. Agreement across two metrics still depends on the image cohort
and interpolation accuracy. Investigate small size differences using tighter
calibration only after the preview identifies a worthwhile case.

### Recorded three-image pilot, 2026-09-08

All 210 retained encodes were scored in approximately 30 seconds of collection
time. No new encoding, calibration, or timing samples were produced. A resume
check performed zero new scoring jobs. All 39 existing fast-ssim2 aggregate
points remained numerically unchanged after the metric-aware refactor.

The Butteraugli preview has 27/30 complete pooled points. At target 1, e1–e3
cannot bracket Kodak02: its best retained score is about 1.038. Those points
are unavailable, not extrapolated. The e4/e5 comparison covers all targets:

| Butteraugli target | e5 file size relative to e4 | e5 encode-time multiplier |
| --- | ---: | ---: |
| 1 | 3.0% smaller | 2.03× |
| 2 | 7.6% larger | 2.18× |
| 3 | 1.7% larger | 2.20× |

These are pooled interpolation estimates for Kodak01–03, not a full-Kodak
result. The e4 advantage also appears under Butteraugli at targets 2 and 3,
but reverses at target 1. The metric scales are not interchangeable, and this
does not establish equal-quality rankings across metrics or a universal winner.
