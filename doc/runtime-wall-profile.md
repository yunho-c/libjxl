# Expanded encoder wall profiles

The wall-v3 profiler is implemented in the sibling `libjxl-gjxl-stage-profile`
checkout. Its comparison harness lives in `gjxl-libjxl-comparison`. Ordinary
encoding and existing stage/Samply artifacts are retained unchanged. No encoder
scheduling barriers are added.

## Measurements

Harness schema 3 preserves the legacy serializer `phase_nanoseconds` and
aggregate-worker `work_nanoseconds` fields. Each sample declares
`wall_profile_version`: 2 for the original expanded boundaries, 3 for the
additional remainder attribution described below. The wall fields are:

- `wall_exclusive_nanoseconds`: nonoverlapping caller-thread intervals, summed
  across invocations. In v2 their sum equals `wall_root_nanoseconds`; in v3
  it equals `wall_root_nanoseconds + wall_api_nanoseconds`, exactly.
- `wall_inclusive_nanoseconds`: each scope including its nested scopes; these
  values must not be summed as an encode-time partition.
- `wall_invocations`: number of measured calls, including cheap early returns.
- `frame_invocations`, `refinement_iterations`, `internal_width`,
  `internal_height`, and `resampling`: explanatory counters and main-frame
  logical encoding geometry. Internal dimensions exclude block padding.

The root encloses the outermost instrumented `EncodeFrame` dispatch. Version 3
also measures selected API work outside that root, recorded separately as
`wall_api_nanoseconds`. The API residual is complete encode elapsed time minus
both root and measured API time (v2 subtracts only the root). Nested internal frames are
accounted using a caller-thread scope stack, rather than counting their full
duration again. Worker callbacks do not enter that scope stack. Timers enclose
pool dispatch through completion, including worker imbalance and waits.

The exclusive stages cover input unpacking, XYB conversion, downsampling,
patch/spline processing, initial AQ (including fast constant-field setup),
inverse Gaborish, heuristic initialization, AC/CfL tile processing, heuristic
finalization, perceptual quantizer refinement, final coefficients, DC
preparation, AR filter selection, AC metadata, block contexts, coefficient
ordering, modular tree construction and the four serializer phases. Frame
setup/other is the remainder inside the root.

Refinement has nested reference preparation, trial roundtrip, trial coefficient
generation, reconstruction/filtering, Butteraugli comparison and field-update
scopes. DC preparation is reported across both trial and final invocations;
the inclusive refinement and roundtrip durations also contain trial DC work.
The notebook's refinement plot uses only disjoint immediate children, with an
explicit remainder. Final coefficient timers are selected by caller context so
trial coefficient work is never mislabeled final.

AC selection, CfL and quant-field adjustment remain interleaved within each
parallel tile. The additive wall stage is their combined enclosing pool pass.
Separating them into additive wall bars would require changing the schedule.

### Version 3: measured remainder attribution

Four additional exclusive timers feed the existing 15 expanded / 10 compact
groups; they do not add paper-table columns:

| Raw timer | Measured boundary | Expanded group | Compact group |
| --- | --- | --- | --- |
| `input_copy` | `JxlEncoderChunkedFrameAdapter::CopyBuffers` during nonstreaming input submission | Input / color preparation | Preprocessing |
| `reference_copy` | Allocation, copy and logical resize of pre-Gaborish `orig_opsin` for AR heuristics | Input / color preparation | Preprocessing |
| `chroma_quantization` | `ComputeChromacityAdjustments`, including pixel statistics when enabled | Initial quantization | AQ / AC / CfL heuristics |
| `output_handling` | Output-processor flushing and the harness output vector's initial allocation and growth | Assembly | Tokenization / output |

These scopes time only identified work. `output_handling` does not enclose all
of `JxlEncoderProcessOutput`, which also performs perceptual analysis and
encoding. A flush nested inside a frame subtracts from its enclosing wall
scope; only a scope outside the frame contributes to `wall_api_nanoseconds`.
Its exclusive duration appears once in the stage map in either case. Explicit
API scopes do not create additional frame invocations or propagate to workers.

The private benchmark hook attaches a sink to both frame settings and that
encoder's output wrapper. It is intended for this harness's single-frame,
buffer-output lifecycle; the sink must outlive output processing. General
multi-frame or externally replaced output processors are outside this study.

Unclassified frame/API setup and shared allocation/destruction remain **Other
encoder / API work**, formerly **Frame / API remainder**. This mixed residual
must not be classified wholesale as entropy/codestream work. No sampled CPU
percentages are redistributed into wall-time columns. The expanded labels
**Input unpack / color** and **Initial AQ** become **Input / color preparation**
and **Initial quantization** to accommodate the added scopes.

Version-2 CSVs retain their original values and attribution; only display labels
change. The notebook rejects mixed versions and inconsistent version/stage
schemas, and records the version in the Plotly title and methodology JSON.
The default paper table still uses the saved v2 cohort. Producing v3 paper
numbers requires a separately collected, complete v3 stage ledger.

### Stage hierarchy and overlap

The notebook's 15 plotted categories do not overlap: they sum exclusive wall
times and partition the representative instrumented encode. The underlying
inclusive timers do nest. This simplified tree combines the reporting groups
with the important timer containment; it is not an exact call graph or a set
of inclusive values to add together. Internal frame encodes can repeat stages
recursively and are omitted here.

```text
Complete encode
├── Input / color preparation
├── Downsampling
├── Initial quantization
├── Inverse Gaborish
├── Feature search
├── AC/CfL tile heuristics
│   ├── Setup
│   ├── Combined parallel tile pass
│   └── Finalization
├── Perceptual refinement [inclusive]
│   ├── Reference preparation
│   ├── Candidate roundtrip [inclusive; repeated]
│   │   ├── Trial coefficient generation
│   │   ├── Trial DC preparation *
│   │   ├── Reconstruction / filtering
│   │   └── Other roundtrip work
│   ├── Butteraugli comparison
│   ├── Quantization-field update
│   └── Other refinement work
├── Final coefficients
├── AR filter selection
├── DC / metadata / ordering *
│   ├── DC preparation
│   ├── AC metadata
│   ├── Block-context selection
│   ├── Coefficient ordering
│   └── Modular-tree construction
├── Tokenization
├── Entropy model
├── Model/token emission
├── Assembly
└── Other encoder / API work
```

The starred trial DC work is inside the inclusive refinement and roundtrip
timers, but its exclusive time is assigned only to the plotted
"DC / metadata / ordering" category. It is excluded from the plotted
"Perceptual refinement" category. Likewise, trial coefficient generation
belongs to refinement, not to the "Final coefficients" category.

For each timer invocation, exclusive time equals inclusive time minus measured
direct-child intervals; invocations are then accumulated. The plotted
categories group these exclusive values, including the outside-frame API
remainder, so their sum equals `profiled_complete_wall_ms`. Use
`wall_exclusive_*` for additive breakdowns and `wall_inclusive_*` for the
duration of a whole enclosing operation. The latter must not be summed across
parents and children.

### Interpreting Samply alongside wall timers

Samply operation percentages use sampled thread-CPU time, not elapsed encode
time. A worker reconstruction stack may not contain its coordinating caller,
so the sample label need not identify the enclosing AR or quantizer-refinement
wall stage. Use the expanded wall fields for those latency boundaries.

The Samply parser now matches `ModularFrameEncoder::` methods rather than the
bare type name. The bare name also appears in unrelated function arguments
and previously caused false Modular/DC attribution. Regenerate older Samply
summaries with the current parser; the captures themselves are unchanged.

## Build and validate

Run from the main libjxl checkout. Use a new build root for every instrumentation
change. The builder saves the exact source patch, harness source, commands,
binary/library hashes and CMake cache fingerprint; it refuses to replace a
frozen manifest. Uncommitted source changes therefore remain reproducible.

```sh
python3 tools/scripts/cjxl_wall_profile_build.py \
  --source ../libjxl-gjxl-stage-profile \
  --comparison-repo ../gjxl-libjxl-comparison \
  --build-root ../gjxl-libjxl-comparison/build/libjxl-wall-v3-NEW
```

`cjxl_wall_profile_validate.py --help` describes the ordinary/instrumented
binary, decoder and two corpus arguments. It checks byte identity for ordinary,
instrumented sink-off and sink-on encodes, decodes retained outputs, validates
every wall partition, and retains balanced six-pair perturbation results. Its
5% sink-overhead threshold triggers review, not a claim of statistical
equivalence. Do not run other benchmarks or builds during perturbation checks.

Use `--wall-profile-version 3` for v3 validation. `--case-set remainder-pilot`
selects 12 identity cases: Kodak 01 and one CLIC image at E1/3/7/8 Q80, Kodak
E7 Q10 and E8 Q95, Kodak E7 Q80 without workers, and one 12 MP E5 Q90 input.
It screens overhead at Kodak E1/7/8 Q80 with eight workers and E7 Q80 without
workers, retaining six alternating process triplets per configuration. This
bounded pilot excludes E9–10 and does not require `--pilot-corpus`; the default
`full` case set retains the original wider qualification and requires that
manifest. Neither validation mode runs implicitly from building or plotting.

The standalone `tools/encoder_wall_timer_test.cc` in the profiling checkout
tests nested frames, exclusive/inclusive accounting, API roots and nested API
scopes, idempotent early stop, disabled sinks and worker isolation. Main-checkout
Python tests exercise both schema versions and table grouping invariants.

### Version-3 validation status (2026-09-11)

The Release build and standalone timer checks passed, along with 37 focused
Python tests. The first 12-case pilot completed before the user's instruction
to defer it arrived. All outputs were byte-identical and decoded, and measured
partitions passed, but a concurrent performance-sensitive measurement was
reported. Its timing results are excluded from overhead qualification; no
overhead conclusion or broad remeasurement is authorized by that run.
The retained run contains a `QUALIFICATION_STATUS.json` marking this exclusion.

The fresh [pilot summary](../../libjxl-runtime-study-2026-09-03/wall-v3-validation-20260911T211101Z/summary.json)
passes the bounded overhead screen: all 12 identity cases are byte-identical
across ordinary, sink-off and sink-on variants and decode successfully. All
96 instrumented samples (24 identity plus 72 perturbation) satisfy the v3
frame/API partition. Frozen build hashes match before and after the run, and
the live profiling source and harness match the frozen build inputs.

Each Q80 configuration uses Kodak 01, six alternating process triplets, one
warmup and three timed samples per process. Values are medians of paired
percentage differences:

| Effort | Workers | Sink-on vs sink-off | Sink-on vs ordinary |
| --- | --- | --- | --- |
| 1 | 8 | +2.06% | +0.38% |
| 7 | 8 | -0.91% | -0.08% |
| 8 | 8 | -0.17% | -0.63% |
| 7 | 0 | -0.02% | +0.10% |

No configuration crosses the 5% median sink-overhead review threshold. The
largest median increase is approximately 0.10 ms at E1. Individual pairs
still fluctuate (E1: -5.56% to +7.57%; E7 with workers: -2.35% to +10.97%),
and Spotlight indexing was visible before the run. This is a successful
bounded perturbation screen, not proof of zero overhead or full-sweep
qualification; negative differences do not indicate an encoding speedup.
Raw results, commands, build verification and process snapshots are retained
in the new directory. The earlier excluded pilot remains unchanged. Complete
v3 stage remeasurement and regeneration of paper values remain separate work.

### Recorded qualification (2026-09-07)

The retained [validation summary](../../libjxl-runtime-study-2026-09-03/wall-v2-validation-20260907b/summary.json)
reports 56/56 byte-identical ordinary/sink-off/sink-on cases, all decoded.
Coverage includes the 38-input pilot corpus, Kodak effort/quality edge cases,
no-worker execution, and 12/24/48 MP inputs. The five perturbation configurations
use Kodak 01 with six alternating process triplets, one warmup and three timed
samples per process. Values below are medians of paired percentage differences:

| Effort | Quality | Workers | Sink-on vs sink-off | Sink-on vs ordinary |
| --- | --- | --- | --- | --- |
| 3 | 90 | 8 | -0.03% | -0.65% |
| 7 | 90 | 8 | +0.04% | +0.37% |
| 8 | 90 | 8 | -1.22% | +0.19% |
| 10 | 10 | 8 | +0.75% | +0.62% |
| 7 | 90 | 0 | +0.40% | +0.35% |

No configuration crossed the 5% median sink-overhead review threshold. These
are a perturbation screen, not statistical proof of zero overhead across the
sweep: individual pairs fluctuate, especially for short encodes. Negative
differences are not evidence that instrumentation accelerates encoding. Raw
samples, commands, outputs and individual pair differences remain beside the
summary. Full-corpus collection and final sweep verification are separate
from this qualification result.

## Stage-only remeasurement and resume

Use `cjxl_runtime_characterization.py run --phase stages`, supply the new
`--stage-benchmark` and `--stage-build-manifest`, and add:

```text
--reference-run ../libjxl-runtime-study-2026-09-03/run-e8ff0976
--output ../libjxl-runtime-study-2026-09-03/run-e8ff0976-wall-v3
--wall-profile-version 3
```

Supply the ordinary benchmark, cjxl and corpus arguments as in the original
run. Reference corpus hash, quality/effort grid and thread count must match.
The reference codestream is hash-compared against every new instrumented
encode. Existing outputs and timing records are read, not copied or replaced.
New profiles and execution events use the new output directory. Resume with
the identical arguments; persisted jobs of the requested version are skipped, while an
interrupted unrecorded job is retried. Build and configuration changes reject
resume instead of mixing measurements.

`summarize --run NEW_RUN` joins its new stage ledger with reference timings.
While collection is in progress, add `--partial` to write explicitly labeled
`image-tuples.partial.csv`, `aggregated.partial.csv`, and `REPORT.partial.md`
without replacing the unsuffixed exports. These are fixed snapshots; rerun the
command to refresh them. Reference-run Samply captures remain in the original
run and are not included in the new run's Samply summary.
CSV `wall_exclusive_*_ms` fields, including `encode_api_other`, form an additive
complete-encode partition. `wall_inclusive_*_ms` fields are nested diagnostics.
Each tuple exports one sample selected by complete elapsed-time median, rather
than independently selecting each stage median. During collection, missing
stage columns remain empty; the notebook excludes incomplete image groups.
The pooled expanded-stage and refinement figures require every input image
and quality for a resolution/effort before showing its bar. A partially
collected effort therefore cannot bias the pooled quality mix.

`verify --stage-only --run NEW_RUN --djxl PATH` verifies all requested tuples,
new wall accounting, codestream hashes and decodability without claiming that
the reference timing or Samply sweep is finished. Full original sweep
completion still requires its outstanding timing and sampling jobs.
Decode verification resumes only when the saved decode hash matches the
current codestream; a stale or hashless decode record causes a fresh decode.

After the expanded stage pass finishes, resume outstanding ordinary timing
and Samply jobs in the reference run using its original arguments with
`--phase timing` or `--phase profiles`. The new stage ledger already contains
the legacy serializer fields, so filling the old serializer-only ledger again
is not required. Keep benchmark phases sequential to avoid CPU contention.
Once reference timing coverage is complete, run `verify` on the new run
without `--stage-only` to verify reference timings plus expanded stages and
codestreams. Audit the reference run's requested Samply captures and symbol
sidecars separately: the timing/stage verifier does not verify Samply coverage.

Edit the Jupytext `.py` notebook and regenerate the paired `.ipynb`. The notebook
supports legacy, wall-v2 and wall-v3 CSVs and chooses the expanded wall figure when
the new fields are present.

## Effort-by-stage tables for the paper

The notebook's **Paper table: encoding wall time by effort and stage** cell
reads the same saved CSV independently of the other figure cells. It shares
the expanded plot's 15 stage groups, including its trial-DC accounting. These
are useful boundaries to retain: AC/CfL is one enclosing parallel pass, while
tokenization, entropy-model construction and emission are distinct serializer
costs. They are accounting categories, not a strictly sequential pipeline.
The wide table is intended for a full-width layout or supplementary material.

**Compact stages** folds the 15 stage columns into 10 using this fixed mapping:

| Compact column | Expanded columns summed |
| --- | --- |
| Preprocessing | Input / color preparation + Inverse Gaborish |
| AQ / AC / CfL heuristics | Initial quantization + AC/CfL tile heuristics |
| Coefficients / metadata | Final coefficients + DC / metadata / ordering |
| Tokenization / output | Tokenization + Model/token emission + Assembly |

Downsampling, Feature search, Perceptual refinement, AR filter selection,
Entropy model, and Other encoder / API work remain separate. Grouping sums the
unrounded exclusive durations and preserves totals and percentage denominators.
Coefficients / metadata includes trial DC preparation; Perceptual refinement
continues to exclude it. `STAGE_TABLE_COMPACT` defaults to `False` and sets the
initial grouping and the selected DataFrame/CSV. The Plotly button affects only
the figure and works independently of the percentage toggle, the small-stage
filter, and the corpus/quality dropdown, including in the offline HTML.

Keep quality and corpus/resolution explicit rather than treating the
all-quality average as an encoder property. In this study, Q10 enables
encoder-side downsampling. At effort 7, Q50 (distance 4.6) and Q70 (2.8) straddle
the distance-3 streaming eligibility condition in `CanDoStreamingEncoding`;
streaming skips the full-frame patch/spline searches. Q90 (1.0) and Q95 (0.55)
straddle the 0.7 threshold below which the default EPF iteration count becomes
zero; AR selection then returns after setting the default field. These source
conditions explain the observed changes, but other effort/image conditions
also affect which work executes. Do not present them as universal Q cutoffs.

`STAGE_TABLE_QUALITY`, `STAGE_TABLE_RESOLUTION`, and `STAGE_TABLE_UNIT` select
the initial view. CLIC/Q80 is only the initial selection. The Plotly dropdown
also provides every collected quality and an explicitly labeled pooled view
for each corpus/resolution. For a paper, choose the quality used in the
corresponding experiment and give quality-specific variants in supplementary
tables. The dropdown changes only the figure; rerun the cell with changed
configuration to change the selected DataFrame/CSV.

Milliseconds are the default for the selected DataFrame, CSV and Plotly table.
The **Show percentages** toggle adds the stage's percentage below its duration
in the same cell. It works independently of the corpus/quality selector and
also works in the offline HTML. `STAGE_TABLE_SHOW_PERCENT` controls its initial
state. The toggle affects presentation only; CSV values remain numeric.

**Hide stages <1 ms** hides columns whose unrounded maximum across the
displayed efforts is strictly below 1 ms for the selected quality/corpus.
The threshold always uses milliseconds, even in a percentage view. Stages
with a maximum of exactly 1 ms remain visible. `STAGE_TABLE_HIDE_SMALL` sets
the initial state. In compact mode, the threshold applies to the combined
column, so small constituent stages can together exceed 1 ms and remain visible.
The filter and percentage toggle remain independent when
switching quality/corpus. Effort and Total stay visible; totals and percentage
denominators still include hidden stages. DataFrames and CSVs retain every
column in their respective grouping.

- `stage_wall_table`: selected DataFrame with effort rows and stage columns,
  in mean ms/encode by default.
- `stage_wall_percent`: all views indexed by resolution, quality and effort;
  stage columns are percentages and `Total (ms)` is mean profiled ms/encode.
- `stage_wall_ms`: the same views with every duration in mean ms/encode.
- `stage_wall_compact_ms` and `stage_wall_compact_percent`: the same views
  with the 10 compact stage columns and unchanged `Total (ms)`.
- `stage_wall_coverage`: expected, present and valid tuple counts, with reasons
  for excluding incomplete resolution/effort groups.

Stage milliseconds are arithmetic means of the saved representative
instrumented samples. Percentages are 100 times summed stage time divided by
summed complete profiled time, matching the plot. This is a time-weighted
composition, not an equal-image mean of per-image percentages. The pooled view
uses one representative sample per image/quality tuple; it does not pool
corpora/resolutions. Uninstrumented speed measurements remain separate.

The full image-by-quality grid observed in the input CSV is required at an
effort before any quality-specific table is generated. A missing stage or
missing tuple excludes that effort for that resolution from every view.
The current 2026-09-08 partial snapshot supports E1–E9; E10 is incomplete.
Absent groups are never replaced with zero. Stage sums are checked against
complete profiled time; percentage stage columns sum to 100 before rounding.
The display distinguishes exact zero from positive values below 0.1.

The cell writes `stage-wall-selected.csv`, `stage-wall-percent.csv`,
`stage-wall-ms.csv`, `stage-wall-compact-ms.csv`, `stage-wall-compact-percent.csv`,
`stage-wall-coverage.csv`, `stage-wall-methodology.json`
and an offline `stage-wall-table.html` under `OUTPUT_DIR`. CSVs retain numeric
precision. The methodology file records the input path/hash, units, mappings
for both groupings, selected view, and run-metadata identity when available.
No benchmark collection is invoked. Plotly is a notebook dependency; CSV construction uses
pandas, and the collection CLI keeps its standard-library-only boundary.
