# GJXL runtime breakdowns in the analysis notebook

The **GJXL runtime breakdown by effort** section in
`tools/scripts/cjxl_runtime_characterization_notebook.{py,ipynb}` reads saved
profiles. Running the notebook never builds, encodes, profiles or scores.

- `GJXL_STAGE_MANIFEST` / `CJXL_GJXL_STAGE_MANIFEST`: saved run configuration
  or legacy manifest; `None` skips the section.
- `GJXL_STAGE_RESOLUTION`: a resolution class; `None` renders all five groups.
- `NORMALIZE_STAGE_BARS`: percentages when true, mean milliseconds when false.
- `GJXL_STAGE_SCALE_TO_ORDINARY`: false by default. True requests an explicitly
  estimated allocation summing to the paired ordinary-call total.
- `GJXL_STAGE_GPU_DIAGNOSTICS`: false by default; true adds a GPU-only figure.

## Fresh six-image Q80 study

The default configuration is
`/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921/config.json`.
All 60 image/effort settings are complete: six images spanning 0.39–48 MP,
nominal Q80 (distance 1.9), efforts 1–10, two warmups per mode and six paired
repetitions. The default CLIC view pools its two images with equal weight.
The remaining groups contain one Kodak, 12 MP, 24 MP and 48 MP image each.
This selected-image study is not a full-corpus or matched-quality comparison.

The encoder is revision `4f3e414` on `perf/profile-production-alignment`
(worktree `gjxl-profile-production-alignment`), including the production-path
alignment in `3e1ef9e`. It is not a claim about current `main`. The run used
Apple M4 Pro / 20 GPU cores / 48 GiB, macOS 15.6, eight CPU participants,
forced fully resident Metal, SIMD-group matrix DCT, fused-tuned inverse,
production shaders (`GJXL_ENABLE_METAL_PROFILING=OFF`), and no final score.
The encoder source was unchanged; the standalone capture harness is frozen
and hashed separately in the run directory.

Each process loads one image and creates one backend outside timing. It
performs an ordinary reference encode, then alternates ordinary/profiled
order by round and effort. Both modes time the complete synchronous encode
API with that backend, including work before return such as publication and
teardown. Image loading, backend creation, returned-value destruction,
validation and JSON/file writes are excluded. The ordinary call requests
neither host nor GPU profiling. Each profiled call returns **both** the host
phase profile and GPU counters, so their attribution describes the same encode.
All warmup and measured results must match the reference bytes, full encoding
summary and committed-submission count. All 60 reference codestreams also
passed `djxl` decoding to PFM after timing collection.

The run retains commands, input/binary/source hashes, raw captures, reference
outputs, environment snapshots, completion ledgers and decoder validation.
Collection rejected competing codec/build/test processes; system services and
display activity were not eliminated. Paired offsets therefore include ordinary
run-to-run variation and are not a universal instrumentation correction.

## One additive complete-call partition

`cjxl_gjxl_paired_breakdown.py` replaces each parent with its children **within
each sample**, then averages. Input preparation becomes geometry/storage,
color transform, matrix-scale statistics, resident preparation, quantization
setup and a residual. The serializer becomes validation, DC/AC tokenization,
entropy optimization, section writing, assembly and a residual.

The quantization parent becomes its nonoverlapping measured GPU intervals:
reference features, initial quantization, quantizer adjustment, AC strategy
search, forward transform, trial coefficients and reconstruction, loop
filtering, Butteraugli comparison, AQ policy, final CfL, DC quantization,
final coefficients and any other GPU stages. Exact stage IDs are exported.
Repeated stages are summed per sample. **Pipeline orchestration / gaps** is
the parent wall time minus these GPU intervals; it can include CPU work,
waits, counter work and uninstrumented GPU activity. It is not CPU-only time.
Resident input preparation is included as host elapsed time and is not
invented as another GPU counter interval.

Other workflow time and **Outer publication / teardown** close the partition
to the externally timed complete profiled call. The latter is simply that
call minus the internal workflow timer. Parents, nested dispatch timings,
command-buffer duration and overlapping worker aggregates are never added
to their children. Every residual must be nonnegative and every partition
must equal its complete-call total.

GPU timestamp validity and interval nonoverlap are checked. An invalid
zero timestamp is accepted only for a verified empty stage: all its dispatches
are indirect and have a zero grid dimension. Missing timestamps otherwise
reject the case; they are not interpreted as zero-duration kernels.

Production alignment preserves combined deferred ACS/AQ and submission
boundaries. Profiling still splits compute encoders and records/processes
counters, so raw bars describe the instrumented path. Hollow diamonds show
ordinary complete-call means. The optional **estimated** view multiplies
each sample's segments by its paired ordinary/profiled complete-call ratio
before averaging. This preserves ordinary totals, but does not establish
stage-specific corrections or remove execution noise. Raw bars remain default.

## Saved-data contract, coverage and exports

Fresh bundles use `schema_version: 1`, `capture_kind: paired-same-call-v1`.
The run configuration fixes images and hashes, settings/efforts, the exact job
grid, source revision, frozen binary/code hashes and measurement protocol.
Each `cases/<case_id>/complete.json` identifies the accepted attempt and its
raw-profile, command and reference-output hashes. The attempt contains
`paired.json` (ordinary/profiled complete-call timing and host counters),
`gpu.json` (schema 4, `production-aligned-resident-v1`), `command.json`,
`reference.jxl`, stdout and stderr. The collector's frozen `code/validate.py`
and the saved-data loader independently validate the timing partitions.

A resolution/effort bar requires every declared image × setting tuple and
all requested valid repetitions. Missing cases stay missing. Metadata/hash
failures stop loading; rejected timing partitions are exported and suppress
their bars. Samples are averaged within an image/setting, then images/settings
receive equal weight. Percentages divide summed stage means by summed totals.

Exports include `gjxl-stage-{samples,means,coverage,gpu-stages,rejected,overhead}.csv`
and `gjxl-stage-methodology.json`. The sample/mean tables contain `kind=flat`,
`scaled` and `gpu` (distinct views, never to be summed together). Paired offsets
in `overhead.csv` are profiled minus ordinary milliseconds and percentages.
Figures use `gjxl-stage-breakdown-<resolution>.png/svg`, optional
`gjxl-stage-scaled-*` and `gjxl-gpu-stage-diagnostic-*`. The retained study also
contains absolute-time figures named `gjxl-stage-ms-*`.

To recollect, use the GJXL worktree's `tools/runtime_profile` capture harness
and resumable collector with a **new** frozen run directory. The current
six-image study needs no additional measurements. Further cohorts or changed
encoder revisions need fresh captures; uninstrumented historical quality
sweeps cannot supply stage attribution retrospectively.

## Paper figure

`tools/scripts/cjxl_gjxl_paper_breakdown.py` exports a publication figure
from the saved paired Q80 captures, at efforts 1-10. It defaults to a single
12 MP panel with subplot titles and subtitles hidden. It reads saved data
only and uses the same validated complete-call partition as the notebook.

```sh
uv run tools/scripts/cjxl_gjxl_paper_breakdown.py \
  --config /Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-stages-q80-20260921/config.json \
  --output-dir /Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/gjxl-runtime-paper-configurable-20260921/output/pdf/default-12mp
```

Use `--panels` to select one or more datasets/size classes in display order:

| Alias | Capture resolution class |
| --- | --- |
| `kodak` | `kodak_0_4mp` |
| `clic` | `clic_1_8_to_3_4mp` |
| `12mp` | `12mp` |
| `24mp` | `24mp` |
| `48mp` | `48mp` |

Canonical class names also work, and names are case-insensitive. For example,
append `--panels kodak 48mp --show-panel-titles` to show Kodak and Unsplash
48 MP with both title and subtitle lines. Use `--panels clic 48mp
--show-panel-titles` to reproduce the original panel selection and headings.
Use a separate output directory for each figure variant. The layout uses
one full-width panel for a single selection and at most two panels per row
for multiple selections. Hiding the headings also removes their reserved
vertical space; the timing annotations and axis labels remain visible.

Use `--legend-position right` for a single-column stage legend beside the
plot, or `--legend-position bottom` for the default three-column legend
underneath. The right-legend layout stacks multiple panels vertically to
keep timing labels readable within the same 7-inch figure width.

The Python entry points accept the same options, e.g.
`export(config_path, output_dir, panels=["kodak", "48mp"], show_panel_titles=True,
legend_position="right")`.
Captions, selected-image metadata, CSVs and reproduction commands follow the
chosen panels. An unknown, duplicate or unavailable class is an error;
incomplete selected cohorts cannot silently produce a bar.

The figure has nine stage groups, 100% stacked bars labeled "Encode time (%)",
and mean profiled milliseconds above each bar. Exact GPU stages are regrouped, including
Gaborish in transform/reconstruction and indirect dispatch setup in AC
search. Every sample must still sum to its original complete-call time.
Input preparation can include GPU work; the hatched remaining elapsed time
includes orchestration, gaps and outer workflow work and is not CPU-only.
There is no ordinary-run scaling. The panels contain selected images, so
their differences do not establish resolution scaling independently of content.

Outputs include a 7-inch-wide vector PDF with embedded fonts, editable SVG,
600-dpi PNG, a caption and LaTeX figure snippet, sample/mean/stage-map CSVs,
methodology metadata, and copies of the rendering and loading scripts.
The output README gives reproduction instructions and attribution details.

## Historical preview boundary

Legacy schema-2 manifests still dispatch to `cjxl_gjxl_stage_breakdown.py`.
The historical preview is
`/Users/yunhocho/GitHub/gjxl/reports/runtime-breakdown-preview-20260920/notebook-manifest-v2.json`:
E4 only, base `b1fbfc1` plus patch, six images, image-specific distances near
fast-ssim2 85, three repetitions. Its flat host bars use **internal workflow**
time, leave the quantization pipeline unresolved and exclude outer teardown.
Its GPU counters came from separate calls in which profiling disabled combined
deferred ACS/AQ. They remain separate diagnostics and cannot be substituted
into those host bars. Estimated ordinary scaling is unavailable for that data.

## Legacy schema-2 bundles

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

The historical preview manifest is a complete example with real hashes and commands.
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
figure is saved as `gjxl-gpu-stage-diagnostic-<resolution>.png/svg`. These legacy captures have no paired ordinary controls.
