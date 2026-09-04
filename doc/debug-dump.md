# Encoder frontend debug data dumps

## Status

This document proposes development-only infrastructure for dumping
image-shaped intermediate state from the VarDCT encoder frontend. It is a
design and implementation plan, not a description of an existing stable API.

The primary recommendation is:

* store encoder-native values in `.npy` files;
* describe their meaning and spatial coordinates in `manifest.json`;
* generate PNG, OpenEXR, overlays, and other visualizations afterward;
* keep the raw-data path separate from `JxlDebugImageCallback`.

The intended data flow is:

```text
encoder-native arrays
        |
        v
EncoderDebugDataSink ---> .npy + manifest.json   (canonical data)
                                 |
                                 v
                          offline renderer
                          PNG / EXR / overlays    (visualization)
```

## Motivation

The encoder already has useful image-oriented debugging hooks. In particular,
`JxlEncoderSetDebugImageCallback` can receive labeled 16-bit RGB debug images,
and compile-time debug paths can produce visualizations for adaptive
quantization, Butteraugli error, and AC strategy selection.

Those images are helpful for visual inspection, but they are not clean
numerical observations of the encoder:

* scalar maps are converted through display-oriented color maps;
* values can be clamped or normalized;
* the callback receives 16-bit RGB pixels rather than the original dtype;
* lower-resolution fields may be presented without enough coordinate metadata;
* categorical values such as AC strategy IDs become colors;
* the transformation from internal value to image is not generally invertible.

For example, the current `quant_heatmapNNNNN` path visualizes
`1.0 / quant_field` using a nonlinear heat map. It does not expose the
continuous quantization field directly. Similarly, the AC strategy dump is a
color-coded rendering rather than the raw per-block strategy representation.

The goal of this proposal is to make the encoder frontend observable in its
native numerical representation. This should support:

* analysis of individual encoder decisions;
* comparisons across encoder versions and settings;
* numerical experiments in Python, NumPy, or other analysis tools;
* validation of alternate implementations;
* visualization with user-selected mappings instead of mappings fixed by the
  encoder.

## Scope

The initial scope is image-shaped state distributed over pixel, block, tile,
group, or variable-block space. Important areas include:

* input and color transformation;
* XYB and feature-subtracted XYB;
* inverse Gaborish precompensation;
* adaptive quantization and masking;
* chroma-from-luma fields;
* AC strategy search and selection;
* Butteraugli refinement iterations;
* EPF/AR candidate evaluation and selection;
* optional reconstruction snapshots around decoder-side loop filters.

Sequential encoder data is explicitly out of scope for this proposal. There is
no requirement to dump tokens, histograms, entropy codes, bitstreams, sections,
or the final codestream.

The first implementation should support:

* a single still frame;
* one-shot, non-streaming encoding;
* the VarDCT path;
* non-JPEG input rather than JPEG coefficient transcoding.

Animations, streaming/chunked encoding, JPEG transcoding, and Modular can be
added after the spatial model and artifact schema are proven.

## Design principles

### Preserve native values

The canonical dump must contain the value exactly as the algorithm consumes or
produces it. The dump stage must not:

* normalize by the observed minimum and maximum;
* clamp to a display range;
* apply a transfer function;
* colorize scalar or categorical values;
* upsample a block or tile field to pixel resolution;
* replace a native value with a display-oriented reciprocal or logarithm.

Internal units are not always physical units. AC search scores, for example,
are heuristic objectives. "Native" means that the stored value is the value
used by the algorithm, with its semantics documented.

Useful derived arrays may also be dumped, but they must be separate artifacts
with explicit provenance. For example:

```json
{
  "name": "aq/iter_001/error_ratio",
  "derived_from": ["aq/iter_001/tile_distmap"],
  "formula": "tile_distmap / butteraugli_target",
  "units": "ratio"
}
```

### Keep dumping observational

Enabling a dump must not change encoder decisions or the output codestream.
In particular:

* a diagnostic output must not select a different computational path;
* existing arithmetic used by a decision must not be reassociated or
  refactored solely for instrumentation;
* debug allocations must not be made when the corresponding artifact was not
  requested;
* the sink receives const views and cannot modify encoder state.

If a value is not normally computed at a given effort setting, the dump should
record that it is unavailable. It should not silently enable a slower encoder
path. A separately computed diagnostic value is acceptable if it is clearly
marked as diagnostic and cannot affect encoding.

### Separate data from presentation

The encoder should emit raw values and semantic metadata. An offline tool
should decide how to display them.

Presentation choices can then include:

* fixed natural ranges;
* linear, logarithmic, or symmetric-log scaling;
* a diverging map around zero;
* percentile-based scaling;
* categorical palettes;
* explicit clipping indicators;
* color bars and units;
* optional overlays on the source or reconstructed image.

The renderer should record the selected visualization transform alongside each
preview it produces.

## Raw data sink

Add an internal `EncoderDebugDataSink` rather than changing the existing
`JxlDebugImageCallback` signature. The existing callback is image-oriented and
part of the public C API. Changing it would create unnecessary API and ABI
concerns.

A conceptual C++ interface is:

```cpp
class EncoderDebugDataSink {
 public:
  virtual ~EncoderDebugDataSink() = default;

  virtual bool Wants(const DebugArtifactInfo& info) const = 0;

  // Emit consumes the view synchronously. The encoder retains ownership.
  virtual Status Emit(const DebugArtifactInfo& info,
                      const DebugTensorView& tensor) = 0;

  virtual Status EmitScalar(const DebugArtifactInfo& info,
                            double value) = 0;
};
```

`DebugTensorView` should contain:

* an explicit dtype enum;
* a data pointer;
* shape;
* byte strides;
* axis names;
* the logical rectangle to serialize.

Byte strides are important because libjxl image rows can contain allocator
padding. The filesystem sink should serialize only the logical array, row by
row where necessary.

`DebugArtifactInfo` should contain:

* a stable hierarchical name;
* stage/category;
* frame and iteration context;
* spatial grid information;
* units and a short semantic description;
* optional channel names;
* optional categorical legend;
* optional derivation metadata.

The `Wants` query must be cheap and must happen before allocating diagnostic
arrays or computing derived values.

### Integration

Add an internal sink pointer to `CompressParams` adjacent to the existing
`debug_image` fields. It will then be copied into `PassesEncoderState` by the
normal encoder setup.

Avoid adding a stable public C API in the first implementation. A non-exported
helper in `lib/jxl/encode_internal.h` can attach a sink to
`JxlEncoderFrameSettings` for a developer tool.

Add a build option such as:

```text
JPEGXL_ENABLE_ENCODER_DEBUG_DATA=OFF
```

When this option is disabled, detailed instrumentation in hot paths should
compile away. When it is enabled but no sink is attached, the remaining
runtime overhead should be limited to coarse null or `Wants` checks.

Filesystem and NumPy serialization should live under `tools` or `lib/extras`,
not in the core codec library.

## Artifact format

### Recommendation

Use one `.npy` file per numeric artifact and one run-level `manifest.json`.

| Format | Role | Rationale |
|---|---|---|
| `.npy` | Canonical | Exact dtype, arbitrary dimensions, signed values, NaN/Inf, categorical tensors, direct NumPy loading, and memory mapping. |
| OpenEXR | Optional preview/export | Preserves float image channels and is convenient in image tools, but adds dependencies and is restricted to a 2D collection of channels. |
| PFM | Not recommended as canonical | Preserves float values, but supports only one or three channels and carries little semantic metadata. |
| PNG | Preview only | Widely viewable, but requires normalization, quantization, or color mapping for most internal fields. |

Prefer individual `.npy` files over `.npz` initially. Individual files can be
written independently, memory-mapped, and recovered from an interrupted run.
If storage size becomes problematic, compress the completed dump directory or
consider a chunked container such as Zarr in a later iteration.

Use:

* C-contiguous output;
* channel-first image arrays, such as `[channel, y, x]`;
* `float32` when the encoder value is `float`;
* `float64` only when the source value is double or the distinction is useful;
* explicit-width integer types;
* a fixed documented byte order.

### Spatial metadata

Image-shaped fields live on several grids:

| Grid | Typical spacing |
|---|---|
| Pixel | 1 x 1 pixels |
| VarDCT block | 8 x 8 pixels |
| Encoder/CfL tile | 64 x 64 pixels |
| Group | Usually 256 x 256 pixels |
| Variable block | Anchored on the 8 x 8 grid with variable coverage |

Every artifact must describe how array coordinates map to the full frame.
Required spatial metadata includes:

* grid kind;
* full-frame pixel origin;
* pixel spacing or footprint;
* valid rectangle;
* padded rectangle where relevant;
* axis names;
* whether a value is repeated over a variable-block footprint or valid only at
  its first block.

Candidate arrays add non-spatial axes such as `candidate`,
`strategy`, or `search_phase`.

### Manifest

The run-level manifest should contain:

* trace schema version;
* libjxl Git revision;
* frame dimensions;
* encoder distance and effort/speed tier;
* decoding speed tier;
* resampling and streaming mode;
* input color encoding and intensity target;
* Gaborish and EPF configuration;
* Highway target, architecture, and thread count;
* selected dump profile and filters;
* a list of artifacts;
* artifact byte counts and checksums.

An artifact entry should resemble:

```json
{
  "name": "aq/iter_001/butteraugli_diffmap",
  "path": "aq/iter_001/butteraugli_diffmap.npy",
  "dtype": "float32",
  "shape": [1080, 1920],
  "axes": ["y", "x"],
  "grid": {
    "kind": "pixel",
    "origin_px": [0, 0],
    "spacing_px": [1, 1],
    "valid_rect_px": [0, 0, 1920, 1080]
  },
  "units": "butteraugli_distance",
  "semantic": "Per-pixel comparator difference",
  "bytes": 8294400
}
```

Categorical fields require a legend. For example, the AC strategy artifact
should map every stored strategy ID to its enum name and block footprint.

Use stable artifact names, but version the schema from the beginning. Readers
must reject unsupported major schema versions rather than silently
misinterpreting artifacts.

## Proposed capture points

The main implementation path is:

* `ComputeEncodingData` in `lib/jxl/enc_frame.cc` for input preparation and
  color transformation;
* `LossyFrameHeuristics` in `lib/jxl/enc_heuristics.cc` for feature
  subtraction, initial AQ, inverse Gaborish, CfL, and AC selection;
* `FindBestQuantization` in
  `lib/jxl/enc_adaptive_quantization.cc` for Butteraugli AQ iterations;
* `ProcessRectACS` and `EstimateEntropy` in
  `lib/jxl/enc_ac_strategy.cc` for AC candidate analysis;
* `ComputeARHeuristics` in `lib/jxl/enc_heuristics.cc` for EPF selection.

### Input and color transform

Capture:

* `color/input_encoded`: source samples as represented after input copying,
  with their declared color encoding;
* `color/linear_srgb`: unclamped linear-light RGB when available;
* `color/xyb_after_transform`: native, unscaled encoder XYB;
* valid and padded rectangles.

`ToXYB` mutates its input image and only produces a separate linear RGB copy
for paths that request one. Instrument the transformation itself or compute a
strictly separate diagnostic copy. Do not request a linear copy through a
different encoder path if doing so could change arithmetic or dispatch.

Do not call `ScaleXYB` for the canonical dump. That affine scaling is useful
for color-profile conversion and presentation, not for observing the native
encoder representation.

### Feature subtraction and inverse Gaborish

Capture:

* `features/xyb_before_subtraction`;
* `features/xyb_after_splines` when splines are enabled;
* `features/xyb_after_patches` when patches are enabled;
* `gaborish/input_xyb`;
* `gaborish/output_xyb`;
* optional `gaborish/delta_xyb`.

The Gaborish snapshots above describe encoder-side inverse Gaborish
precompensation. Decoder-side Gaborish output during a round trip is a
different artifact and should be named accordingly.

### Initial adaptive quantization and masking

Capture:

* `aq/initial/quant_field`: continuous block-resolution field;
* `aq/initial/mask_block`;
* `aq/initial/mask_pixel`;
* `aq/post_ac_adjust/quant_field`;
* `aq/post_ac_adjust/raw_quant_field`.

An optional detailed AQ profile may also capture:

* the pixel-level Laplacian or activity input;
* pre-erosion fields;
* post-erosion fields;
* intermediate modulation fields.

Keep the continuous quantization field and the integer raw quantizer field as
distinct artifacts. Both are important and they have different units.

### Chroma from luma

There are preliminary and refined CfL passes. Capture both:

* `cfl/pass_0/ytox_code` and `ytob_code`;
* `cfl/pass_0/ytox_ratio` and `ytob_ratio`;
* `cfl/pass_1/ytox_code` and `ytob_code`;
* `cfl/pass_1/ytox_ratio` and `ytob_ratio`;
* base correlation and color-factor scalars.

The raw signed maps preserve the values that enter the codestream model. The
floating ratio maps show their effective numerical meaning. These fields live
on the 64 x 64-pixel color-tile grid.

### AC strategy search

At minimum, capture the final selection as:

* `ac/final/strategy_id`;
* `ac/final/is_first_block`;
* `ac/final/covered_blocks_x`;
* `ac/final/covered_blocks_y`.

For a comprehensive AC search profile, capture:

* `ac/search/candidate_total_cost`;
* `ac/search/candidate_valid`;
* `ac/search/candidate_coefficient_cost`;
* `ac/search/candidate_nonzero_cost`;
* `ac/search/candidate_information_loss`;
* `ac/search/candidate_quant_norm`;
* `ac/search/merge_current_cost`;
* `ac/search/merge_candidate_cost`;
* `ac/search/merge_accepted`;
* intermediate selected-strategy and priority fields after major merge phases.

Suggested candidate tensor axes are:

```text
[search_phase, strategy, block_y, block_x]
```

Not every candidate is legal at every anchor. Initialize unavailable float
entries to NaN and provide a separate `candidate_valid` mask. The mask makes
alignment and effort-dependent pruning explicit.

The AC search first selects among the 8 x 8 variants, then hierarchically
evaluates larger merges. The same strategy/anchor may be considered in
different phases, so a phase axis or separate phase artifacts are preferable
to overwriting earlier observations.

Do not restructure `EstimateEntropy` arithmetic to obtain component values.
Snapshot existing accumulators, or compute observational component totals in
parallel while leaving the original decision value untouched.

AC processing is tile-parallel. Use tile-local diagnostic buffers and merge or
emit them after `RunOnPool` completes. Do not perform filesystem writes from
the inner candidate loop.

### Butteraugli AQ iterations

For each iteration, capture:

* `aq/iter_NNN/quant_field_in`;
* `aq/iter_NNN/raw_quant_field`;
* `aq/iter_NNN/decoded_linear_rgb`;
* `aq/iter_NNN/butteraugli_diffmap`;
* `aq/iter_NNN/tile_distmap`;
* `aq/iter_NNN/error_ratio`;
* `aq/iter_NNN/quant_update_multiplier`;
* `aq/iter_NNN/clamped_low`;
* `aq/iter_NNN/clamped_high`;
* `aq/iter_NNN/quant_rounding_stall`;
* `aq/iter_NNN/quant_field_out`.

Store the scalar score, target, quant field bounds, DC quantizer, and update
exponent in the iteration metadata.

Capturing both `quant_field_in` and `quant_field_out` is preferable to
reconstructing the update afterward. The update contains clamping and a
special case that advances the raw quantizer by one step when ordinary
multiplication would round back to the same value.

`butteraugli_diffmap` should be the comparator's raw float field.
`tile_distmap` should be the native 16th-norm field used by the AQ update.
Neither should pass through `CreateHeatMapImage`.

### EPF/AR selection

Capture:

* `epf/candidate_values`;
* `epf/candidate_error` with axes
  `[candidate, block_y, block_x]`;
* `epf/selected_sharpness`;
* optional candidate reconstructed XYB images;
* optional final reconstruction after applying the selected field.

The candidate error fields already exist inside `ComputeARHeuristics`. Stack
them without normalization and store the actual candidate values, such as
`[0, 2, 7]`, as axis metadata or a small companion array.

Candidate reconstructions are large. Make them part of an explicit deep
filter profile rather than an ordinary frontend dump.

### Decoder-side loop-filter observations

The encoder-side inverse Gaborish snapshot does not isolate the effects of
decoder-side Gaborish and EPF during roundtrip evaluation. A later phase can
add observation points to the render pipeline for:

* reconstruction before loop filtering;
* reconstruction after Gaborish;
* reconstruction after each EPF stage;
* final roundtrip reconstruction in XYB and linear RGB.

These should be optional because AQ and EPF searches can reconstruct the image
many times.

## Runtime profiles and filtering

Comprehensive dumps can be very large. A 4K three-channel float image is close
to 100 MiB before considering multiple stages and iterations.

Support named profiles:

| Profile | Contents |
|---|---|
| `overview` | Color stages and finalized quant, CfL, AC, and EPF fields |
| `aq` | Initial AQ plus all Butteraugli iterations |
| `ac` | Final AC map and candidate search tensors |
| `filters` | Inverse Gaborish and EPF selection fields |
| `filters-deep` | Candidate and render-pipeline reconstructions |
| `all` | Every enabled image-shaped artifact |

Also support:

* include/exclude filters by artifact path;
* frame selection;
* pixel ROI, converted explicitly to each artifact grid;
* iteration selection;
* single-thread trace mode;
* maximum output-byte budget;
* optional checksums.

The manifest should list requested artifacts that were unavailable and explain
why, for example because an effort tier skipped a heuristic.

## Developer tool

Add a dedicated executable, tentatively `jxl_encoder_dump`. A separate tool:

* keeps development options out of normal `cjxl` use;
* avoids turning the raw sink into a supported public API;
* can use internal encoder headers;
* can enforce the initially supported one-shot VarDCT configuration;
* can provide trace-specific profiles, ROI selection, and output budgets.

After the design is stable, `cjxl` may gain a developer-build-only
`--debug_dump_dir` option that uses the same sink and serializer.

Suggested initial command:

```text
jxl_encoder_dump input.png output.jxl \
  --debug_dump_dir dump \
  --debug_dump_profile overview,aq \
  --debug_dump_roi 0,0,512,512
```

The encoded output is useful for confirming that an instrumented run is
identical to a normal run, even though codestream bytes are not themselves a
debug artifact.

## Threading and streaming

The sink must be safe to call from encoder code that may run concurrently, but
the preferred design is to avoid sink calls in hot parallel loops:

1. query `Wants` before allocating diagnostics;
2. allocate a full-frame or tile-local diagnostic tensor;
3. let each worker populate only its own region;
4. emit after the parallel phase completes;
5. sort manifest entries by their stable artifact key.

This produces deterministic names and manifest ordering independent of worker
completion order.

Streaming mode is intentionally deferred. `ComputeEncodingData` expands each
streaming patch around its group so inverse Gaborish and AQ see border pixels.
That creates overlapping local arrays and makes full-frame coordinates,
ownership, and deduplication more complicated. When streaming support is
added, every tile artifact must carry an absolute origin and valid rectangle;
the dump reader, not the encoder, should decide whether to mosaic overlapping
artifacts.

## Implementation phases

### Phase 1: sink, schema, and serializer

* Add the internal sink, tensor view, artifact metadata, and cheap `Wants`
  query.
* Adapt the `.npy` writer and manifest concepts already used by
  `libjxl-tiny`.
* Add the filesystem sink outside the core codec.
* Add schema validation and NumPy loading helpers.
* Dump one pixel field and one block field to validate strides and
  coordinates.

### Phase 2: developer frontend and overview profile

* Add `jxl_encoder_dump`.
* Support single-frame, one-shot VarDCT encoding.
* Capture input, linear RGB when available, XYB, inverse Gaborish input/output,
  initial AQ, final quant field, CfL, final AC strategy, and EPF selection.
* Verify the instrumented and non-instrumented codestreams are identical.

### Phase 3: AQ and Butteraugli iteration profile

* Add an explicit iteration context independent of `AuxOut` counters.
* Capture raw comparator and tile fields before visualization.
* Capture quant fields before and after each mutation.
* Add update multiplier and clamp/stall masks.

### Phase 4: AC and EPF search diagnostics

* Introduce tile-local AC candidate tensors.
* Record 8 x 8 candidates, hierarchical merge phases, validity, and decisions.
* Expose meaningful cost components without changing decision arithmetic.
* Stack EPF candidate error fields and optionally capture reconstructions.

### Phase 5: roundtrip filter taps and visualization

* Add optional render-pipeline observation points before and after Gaborish and
  EPF stages.
* Add a Python viewer that loads the manifest and `.npy` artifacts.
* Support explicit mappings, color bars, categorical legends, EXR export, and
  optional source-image overlays.

### Phase 6: broader encoder configurations

* Add streaming/chunked spatial records.
* Add multiple frames and animations.
* Evaluate useful Modular frontend fields.
* Add JPEG-transcoding-specific observations only where they have a clear
  image-space interpretation.

## Validation

### Behavioral invariance

* A build with debug dumping disabled must match an ordinary build.
* An enabled build with no sink must produce the same codestream.
* An enabled build with a sink and any profile must produce the same
  codestream.
* Instrumentation must not change selected strategies, quant fields, or
  iteration counts.

### Serializer tests

Test:

* every supported dtype;
* one-, two-, three-, and four-dimensional arrays;
* non-contiguous and padded row strides;
* cropped rectangles;
* NaN and infinity preservation;
* explicit byte order;
* artifact-name collision detection;
* manifest schema validation.

### Spatial tests

Use small deterministic images:

* constant gray and constant color;
* horizontal and vertical gradients;
* an impulse;
* a checkerboard;
* seeded noise;
* odd dimensions;
* dimensions crossing 8-, 64-, and 256-pixel boundaries.

Verify that pixel, block, tile, and group artifacts map back to the correct
full-frame coordinates.

### Determinism

For the same input, settings, build, and Highway target:

* artifact names and manifest ordering must be stable;
* integer and categorical fields must match exactly;
* floating fields should be byte-identical when the underlying encoder result
  is deterministic;
* otherwise, comparison tooling should use explicit per-artifact tolerances.

Run representative dumps with one and multiple worker threads to catch
uninitialized candidate entries, overlapping writes, and ordering bugs.

### Performance

Measure:

* ordinary builds with the feature disabled;
* enabled builds with no sink;
* the `overview` profile;
* candidate-heavy `ac` and `filters-deep` profiles.

The disabled configuration should have no meaningful runtime or memory
regression. Large costs in explicitly requested deep profiles are acceptable,
but the output budget and ROI controls must make them manageable.

## Acceptance criteria

The initial infrastructure is ready when:

* `jxl_encoder_dump` can dump a one-shot VarDCT encode;
* the output contains a valid manifest and loadable `.npy` arrays;
* at least one pixel, block, color-tile, and candidate-axis artifact is
  represented correctly;
* raw arrays contain encoder-native values with no hidden display mapping;
* tracing does not change the output codestream;
* the `overview` and `aq` profiles are implemented;
* a Python tool can render selected artifacts to PNG without modifying the
  canonical data;
* artifact names and coordinates are deterministic.

The important architectural decision is that raw data and visualization remain
separate. `.npy` plus a spatially explicit manifest is the ground truth;
image files are derived views that can be regenerated with whatever scale,
palette, transfer function, or overlay best serves the analysis.
