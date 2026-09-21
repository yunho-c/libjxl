# GJXL runtime breakdowns in the analysis notebook

The **GJXL runtime breakdown by effort** section in
`tools/scripts/cjxl_runtime_characterization_notebook.{py,ipynb}` reads saved
profiles with `cjxl_gjxl_stage_breakdown.py`. Running the notebook does not
collect measurements. Configure:

- `GJXL_STAGE_MANIFEST` (or environment variable `CJXL_GJXL_STAGE_MANIFEST`):
  profile manifest path; `None` skips the section.
- `GJXL_STAGE_RESOLUTION`: a manifest resolution class; `None` renders all.
- `NORMALIZE_STAGE_BARS`: percentages when true, mean milliseconds when false.
- `GJXL_STAGE_GPU_DIAGNOSTICS`: false by default; true adds a separate GPU
  diagnostic figure. GPU counters never enter the flat wall-time chart.

The default bundle is
`/Users/yunhocho/GitHub/gjxl/reports/runtime-breakdown-preview-20260920/notebook-manifest-v2.json`.
It contains six historical E4 zero-AQ candidate captures (base `b1fbfc1` plus
a retained patch), three repetitions after one warmup, an eight-thread CPU
cap, and image-specific distances near fast-ssim2 85. It is not a current-main
measurement or a full-corpus sweep. The default CLIC view pools its two images.
All efforts 1–10 are shown; only E4 has data. A missing manifest prints a
collection-needed message instead of attempting collection.

## Measurement boundaries

The default is a **single flat stacked bar per effort**, with one denominator:
internal workflow wall time. Input preparation is replaced by geometry/storage,
color transform, matrix-scale statistics, resident preparation, quantization
setup, and its residual. The CPU serializer is replaced by validation, DC/AC
tokenization, entropy optimization, section writing, assembly, and its residual.
Quantization pipeline time and other workflow time complete the partition.

Parent timers are replaced, never added to their children. Flattening is done
within each raw sample before aggregation. The sum of every flat sample must
equal its original workflow total; all three residuals must be nonnegative.
Serializer percentages now use the whole workflow denominator. For example,
a 20 ms entropy phase within a 50 ms serializer in a 100 ms workflow is 20%
of the flat bar, not 40%. Overlapping aggregate worker counters are not used.
A large residual remains unresolved attribution, not a known GPU stage.

**Quantization remains unresolved** in the saved host data; its segment is
hatched and explicitly labeled. This is a genuine missing-measurement boundary,
not another collapsed plot panel. It includes host orchestration, GPU execution
and waits. Splitting it accurately requires new instrumentation and captures.
The workflow total excludes outer teardown/publication, so do not substitute
it for uninstrumented complete public-call latency in throughput figures.

The optional GPU figure groups **nonoverlapping timestamp intervals from a
separate instrumented invocation**. It retains initial quantization, quantizer adjustment,
forward transforms, final CfL, DC quantization and final coefficients, plus
reference features, AC search, trial reconstruction, loop filtering,
Butteraugli and AQ policy when those intervals are recorded. Exact stage IDs
are also exported. Stages repeated across iterations are summed per sample.
Unrecognized IDs are retained in Other GPU stages, not discarded.

These GPU durations are normalized to the sum of measured stage intervals.
They omit resident input preparation and any uninstrumented gaps. They are
not a complete-GPU-time denominator. Command-buffer duration, nested host
intervals and dispatch entries are not added to stage counters. Missing
dispatch timestamps are not measured zero-duration kernels.

In current GJXL source, `GpuAdaptiveQuantizationProvider::SupportsDeferredSearch`
in `src/gpu/ops/quantization_pipeline.cpp` requires `!profiling_session_`.
Consequently, enabling stage profiling disables the combined deferred ACS/AQ
path; stage boundaries also change compute-encoder structure. These counters
are diagnostics of that instrumented path. They cannot be used to split the
uninstrumented quantization bar by scaling GPU percentages into host time.

## What still needs measuring

The full-effort fixed-Q and calibrated GJXL quality studies save
uninstrumented complete-call timings, with stage profiling disabled. They
cannot supply a finer breakdown retrospectively. Populating E1–E10 at a new
revision requires a separate profiling study with a frozen source/build,
explicit image/quality cohort, CPU policy, warmup and repetition counts.
Collect host and GPU profiles separately and sequentially, retaining raw
files, exact commands, hashes, provenance and failed/rejected attempts.

For a paper figure describing the **production combined ACS/AQ path**, first
add instrumentation that observes that path without disabling it. Validate
its execution/output equivalence and measure profiling overhead against
uninstrumented complete-call controls. Existing stage profiling can support
a separately labeled diagnostic study now, but new captures alone do not
remove its path-change limitation.

For scale, 65 images × seven fixed qualities × ten efforts × two profile
modes is 9,100 process invocations, each with validation, warmup and repeated
encodes. A six-image, one-setting pilot across ten efforts is 120 invocations.
No such collection was started when this notebook section was added.

The existing benchmark interfaces, for a deliberately chosen setting, are:

```sh
# Host workflow and serializer phases; GPU counter profiling remains disabled.
"$BENCH" --input "$INPUT" --scope metal-public-workflow --validation metal-only \
  --gpu-aq fully-resident --effort "$EFFORT" --distance "$DISTANCE" \
  --cpu-threads 8 --warmups 1 --samples 3 --raw-samples "$HOST_JSON"

# Separate instrumented GPU diagnostic, not the combined production path.
"$BENCH" --input "$INPUT" --scope metal-public-workflow --validation metal-only \
  --gpu-aq fully-resident --effort "$EFFORT" --distance "$DISTANCE" \
  --cpu-threads 8 --warmups 1 --samples 3 \
  --gpu-profile stage --gpu-profile-output "$GPU_JSON"
```

These are command templates, not a resumable collector. Never combine
`--raw-samples` with `--gpu-profile`. Retain effort and CPU settings in the
manifest and commands: the current GPU JSON schema does not record them all.

## Manifest contract and coverage

The JSON manifest has schema version 2. Its top-level fields are:

- `label`, `source_revision`, `source_description`, `binary_sha256`:
  one explicitly identified build/policy per bundle, including any patch.
- `efforts`: the full intended effort grid, normally `[1,2,3,4,5,6,7,8,9,10]`.
- `settings`: stable setting IDs, e.g. `["Q80"]` or `["target85"]`, and
  `settings_label`: a human-readable description of the quality policy.
- `samples_per_capture`, `warmups`, `cpu_threads`: collection protocol.
- `images`: objects with `image_id`, `resolution_class`, `width`, `height`,
  `input_path` (absolute or relative to the manifest), and `input_sha256`.
  Each image ID must identify a distinct input path. Retain the input files;
  their hashes are checked once per image when loading the bundle.
- `captures`: one object per collected `(image_id, setting, effort)`, with
  `distance` and optional `host`/`gpu` references. Each reference has `path`
  (relative to the manifest or absolute), `sha256`, `binary_sha256`, and `argv`
  (the actual command as a JSON array). Missing captures can simply be absent.

The default manifest is a complete example with real hashes and commands.
It preserves the original schema-1 preview as a separate file. To migrate an
older bundle, obtain each input path/hash from its retained corpus manifest,
verify the bytes and both capture commands, and save a new schema-2 manifest.
Do not infer input identity from image dimensions. Each capture command must
contain exactly one `--input` matching its declared image; relative input
paths resolve against the manifest directory. A raw profile path may appear
only once in a bundle. The profile JSON itself does not contain input hashes,
so this association relies on the retained commands and manifest provenance.
Do not add absent images/settings to the measured cohort opportunistically:
declare the intended grid before collection and keep it fixed. Do not merge
different build hashes or policies into one manifest.

Host schema 17 and GPU schema 4 in stage mode are supported. The loader
checks input/profile hashes, input paths, duplicate profile references and
tuples/sample indices, profile metadata, image dimensions, command
effort/distance and binary identity. It rejects negative
host residuals, unavailable GPU counters, malformed timestamps and overlaps
across stages/submissions. Rejected timing samples are exported with reasons;
their missing repetitions prevent that effort/kind from entering the chart.
Structural/provenance failures stop loading rather than producing a chart.

A resolution/effort/measurement kind is shown only when every image × setting tuple has
all requested valid repetitions. Host and GPU coverage are independent;
missing GPU captures never suppress valid host results. No missing effort
is treated as zero. Repetitions are averaged within each tuple, then tuples
are averaged with equal weight. Percentages divide summed stage means by
summed totals; no independent per-stage medians are used.

Exports under `OUTPUT_DIR` are `gjxl-stage-breakdown-<resolution>.png/svg`,
`gjxl-stage-means.csv`, `gjxl-stage-samples.csv`, `gjxl-stage-coverage.csv`,
`gjxl-stage-gpu-stages.csv`, `gjxl-stage-rejected.csv`, and
`gjxl-stage-methodology.json`. Exact GPU-stage rows are validated raw samples;
the grouped means table contains only complete cohorts. `kind=flat` in the
means/samples/coverage exports supplies the default plot; the workflow and
serializer kinds remain available for auditing. They are overlapping views of
the same measurements and must not be summed with `flat`. The optional GPU
figure is saved as `gjxl-gpu-stage-diagnostic-<resolution>.png/svg`. None of
these exports are joined to uninstrumented throughput measurements.
