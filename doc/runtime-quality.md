# Matched-quality speed–compression study

This workflow scores retained encodes, produces an interpolated preview,
calibrates encoder distance to measured quality, and times accepted settings.
It is separate from the original runtime and wall-v2 ledgers. Importing its
module or running the notebook never starts collection.

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
library; plotting additionally needs matplotlib >=3.8. Python 3.13 on the study
machine already includes the notebook dependencies. Use the installed stable
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

The shared budget covers scoring, calibration and measurement, not plotting.
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
- coverage.json and preview/measured Pareto and diagnostic rate–quality
  figures in both PNG and SVG.

The notebook accepts QUALITY_RUN or the CJXL_QUALITY_RUN environment variable.
Its script form accepts --quality-run RUN alongside the normal timing CSV.
Edit the notebook Python source first, then derive/execute its Jupyter form:

    jupytext --to ipynb --execute tools/scripts/cjxl_runtime_characterization_notebook.py

For future full collection, initialize a new directory using an explicit
comma-separated --images list and --measurement-efforts 1,2,3,4,5,6,7,8,9,10.
Targets, tolerance, evaluation cap and timeout are configurable at init.
Full-corpus collection and new effort-8–10 encodes are not part of this pilot.
