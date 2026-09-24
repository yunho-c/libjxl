# Cross-machine libjxl sweep handoff

Please reproduce our libjxl CPU fixed-quality sweep on the target CUDA machine, so we can produce a comparable “Speed vs. compression efficiency” plot. Run the CPU baseline in the same OS/runtime environment as the CUDA encoder. Record the exact CUDA implementation and revision being compared: a libjxl-versus-GJXL comparison measures different encoders and hardware together, not the acceleration of identical code.

Use the specification below. Inspect existing artifacts first, then build and validate a small pilot before starting the full resumable sweep. Preserve existing studies, and keep any currently paused Mac run paused. This request is for the libjxl baseline; do not start an additional CUDA sweep, calibrated study, or stage-profiling campaign unless that work is separately requested.

Published starting points, verified against the remotes on September 24, 2026:

- `https://github.com/yunho-c/libjxl.git`, branch `perf/quality-effort-sweep`, remote tip `6238450b9995f786faf38bb4ddae02a6cca5f3f3`. It includes the sweep/BD-rate helpers and the newer CUDA infrastructure (`a7c59acc`) and analysis (`6238450b`). Read `doc/runtime-cuda-quality.md` before adapting any collector: Windows locking and process handling already exist there. Its CUDA pipeline runs both fixed and calibrated phases, so do not invoke it as a libjxl-only sweep launcher.
- `https://github.com/yunho-c/gjxl.git`, branch `perf/libjxl-comparison`, remote tip `950357c304b794ce92ce2b7c2e2148cebd774cbd`. This contains the CPU harness, build helper, and pinned `third_party/libjxl` submodule. Initialize submodules recursively. The Mac-only additional commit `c235e30` concerns wall-v3 instrumentation and is not required for the ordinary CPU baseline.

Use fresh checkouts at explicitly recorded revisions. The Mac libjxl working branch has independent unpublished plot changes and should not be used as an implicit source of the remote branch state. Corpus PFMs and frozen run artifacts need a separate verified transfer; cloning these repositories does not supply them.

1. Freeze the encoder and build provenance.

   Our original libjxl revision is `e8ff09762481785938d8e4e01333ed3917571161`. Build that revision in Release with its pinned dependencies/submodules and normal optimized SIMD dispatch. Use a separate checkout/build/run directory. Do not silently replace it with latest main, a system libjxl, or an instrumented build.

   Use the complete-encode API harness from `gjxl-libjxl-comparison/benchmarks/libjxl_comparison/libjxl_comparison_benchmark.cpp`. The executable is `gjxl_libjxl_comparison_benchmark`. Freeze the exact harness source as well as libjxl; the harness protocol can change independently of the encoder revision. Inspect and record untimed validation encodes in addition to explicit warmups. The currently inspected harness performs one untimed validation encode before its requested warmup.

   Retain source revisions, dirty diffs if unavoidable, source hashes, CMake configuration, compiler/version/flags, executable and loaded-library hashes, and the decoder/scorer identities. Different platforms will have different binary hashes; record actual local identities rather than pretending they match the Mac executable.

2. Verify the exact corpus before collecting anything.

   Use the same 65 canonical linear-sRGB, three-channel float32 PFMs: 24 Kodak, 32 CLIC, and 9 controlled Unsplash variants at approximately 12/24/48 MP. Compare by `image_id`, dimensions, and each file's SHA-256. Preserve float values and color encoding; do not regenerate the resized PFMs from JPEG/PNG or substitute 8-bit inputs.

   Mac source manifest: `/Users/yunhocho/GitHub/libjxl-runtime-study-2026-09-03/corpus/corpus.json`.
   Its SHA-256 is `74aeb8b73e1e96325157c97a9c596d4cf0f4b8d4aa8259121866fe8de6eb3d6f`.

   Keep that original manifest as provenance and create a separate local path mapping. Rewriting absolute paths necessarily changes the local manifest hash; pixel identities must still match.

   If the target is G14W: a September 21, 2026 audit recorded the corrected manifest at `C:/Users/G14/GitHub/gjxl-cuda-study-2026-09-20/corpus/corpus-mac.json`. It selects canonical replacements for `unsplash/campus_interior/48mp` and `unsplash/forest_stream/48mp`, whose older local files differed. This is historical context, not a fresh remote verification: rehash all 65 inputs before using it. Preserve the older manifest/files for old-run provenance.

3. Reproduce the fixed grid and timing protocol.

   - Efforts: every integer from 1 through 10, including the expensive e9/e10 cases.
   - Nominal qualities: 10, 30, 50, 70, 80, 90, 95.
   - Exact quality-to-distance mapping: `10:15.266666666667`, `30:6.4`, `50:4.6`, `70:2.8`, `80:1.9`, `90:1.0`, `95:0.55`. Use the existing helper's command formatting.
   - Eight libjxl worker threads; one explicit warmup and one measured sample per fresh harness process; five independent measured processes per image/quality/effort. Five samples in one process are a different protocol.
   - Use the effort-major deterministic shuffled schedule from `cjxl_sweep_common.py`, seed `20260903`. Freeze the helper and collector. Log any separate scheduling selection.
   - Full completion means 4,550 image/quality/effort settings and 22,750 timing records. Retain one codestream, byte count, output hash, and measured quality per setting.

   Q30–Q95 supplies the standard BD-rate figure: 3,900 settings and 19,500 timing records. Prioritize this subset if scheduling requires it, while clearly tracking Q10 separately. The full historical sweep includes Q10, but its libjxl automatic downsampling excludes it from the normal BD-rate interval. Do not force resampling=1 and call that the same study.

4. Preserve complete-call timing semantics.

   Measure uninstrumented encoder wall time. The CPU harness times encoder creation/configuration, image submission, output production, and per-call cleanup. Input reading/conversion, process startup, worker-runner setup, warmups, output-file writes, decoding, and quality scoring are outside that timer. Retain raw per-repetition durations and use their median per setting.

   Audit the CUDA comparison timer to include the full equivalent encoding workflow: CPU work, host/device transfers required by the public API, synchronization, and production of the final host-accessible codestream. GPU work must be complete when the timer stops, with unrelated prior work drained before it starts. Kernel-only CUDA events or asynchronous launch duration cannot replace that end-to-end measurement. Keep batch throughput and cold-start timing separate from this single-image warm-call study. Record context/JIT initialization, buffer reuse, and warmup policies explicitly.

   [NVIDIA CUDA timing guidance](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#using-cpu-timers).

5. Keep resource and platform differences explicit.

   Run CPU and CUDA measurements sequentially on the same machine, with no concurrent builds, scoring, profiling, or other benchmark jobs. Use AC power and a documented stable power policy; record CPU/GPU models, RAM/VRAM, OS, native versus WSL execution, compiler, driver/toolkit, thread settings, and available temperature/throttling information. Do not mix native-Windows CUDA timing with a WSL CPU baseline without labeling and investigating that platform difference.

   Eight libjxl workers are not necessarily eight total CPU participants. Document the CUDA encoder's CPU-thread semantics too. This baseline reproduces our eight-worker study; any all-core CPU baseline is a separately labeled experiment. Cross-machine plots may be useful, but a CUDA/new-machine versus CPU/Mac ratio is not an isolated GPU speedup.

6. Decode and score the actual new outputs.

   Use our `tools/scripts/quality_metric` adapter 0.1.0, with fast-ssim2 pinned to `c3867954c7bec8a761951df9256354b305fd0cff`, the same linear-sRGB float decode/reference path, and intensity target 80. Preserve its large-image/strip-scoring behavior. Pin one decoder/scorer pipeline for both CPU and CUDA outputs, and check scorer agreement on common retained files when changing platforms.

   Do not copy Mac sizes or quality scores merely because effort and distance match. Native builds can produce different outputs; score and hash the codestreams actually being timed. Saved quality results are reusable only when output, reference, and decoder/scorer identities justify reuse. Compare sample outputs with the Mac to detect differences, but do not assume cross-architecture byte identity is guaranteed.

7. Reuse the collector carefully; do not blindly copy Mac launch commands.

   Relevant files in the libjxl checkout are:

   - `tools/scripts/cjxl_runtime_characterization.py`: original libjxl timing/output collector.
   - `tools/scripts/cjxl_sweep_common.py`: scheduling, Q-to-distance mapping, and summaries.
   - `tools/scripts/cjxl_quality_characterization.py`: scoring and separate calibrated studies.
   - `tools/scripts/cjxl_bd_rate.py` and the paired runtime notebook: saved-data analysis.
   - `doc/runtime-quality.md`: analysis and collection semantics.

   The current `--encoder gjxl --mode fixed` quality-collector path is not a ready-made libjxl fixed-sweep command. The original runtime CLI also requests stage/Samply tool paths even for timing-only work. Use the published quality collector's existing Windows locking/process support; do not duplicate that work from the older Mac checkout. Still audit the original libjxl timing/build path for Unix executable/library layout, local DLL provenance, and macOS host-tool assumptions. On Windows, implement and retain a small native adapter if needed; on Linux/WSL verify all paths and subprocess behavior. Preserve the measurement boundary, scheduling, identity checks, and append-only records. Do not fabricate stage tools or Mac binary identities to satisfy a manifest check.

   A single POSIX-shell harness invocation, after building and assigning local paths, is:

   ```sh
   "$LIBJXL_BENCHMARK" \
     --input "$INPUT_PFM" \
     --raw-samples "$RAW_JSON" \
     --output "$OUTPUT_JXL" \
     --distance 1.9 --effort 7 --num-threads 8 \
     --warmups 1 --samples 1
   ```

   This is one Q80/e7 repetition, not the sweep. Use unique raw-record paths and append repetition IDs 0–4. Freeze any portability changes and validate that resume skips completed samples without mixing machines/builds. Keep actual ledger schema versions; `wall_profile_version=2/3` describes separate instrumentation, not a requirement to relabel ordinary timing records. Stage profiling is unnecessary for this plot.

8. Validate, run resumably, and deliver the evidence.

   First dry-run the complete schedule and perform a short pilot containing Kodak, CLIC, and a large PFM, including a high-effort case. Check that outputs decode, hashes/geometry/configuration join correctly, quality scoring works, all five repetitions are retained, and interruption/resume and duplicate rejection work. Keep pilot data in its own run unless its final frozen configuration is identical to the full study.

   After the pilot passes, launch the full fixed sweep with a durable local scheduler, overlap protection, free-space checks, per-job timeout, live progress, retained logs, and a documented graceful pause/resume command. Estimate duration from the target-machine pilot, including untimed work; do not reuse the Mac ETA. Report missing settings and failures explicitly.

   Regenerate the speed-versus-BD-rate analysis from the new records: reference libjxl e7 on that machine, achieved SSIMULACRA2 interval [75,85], per-image PCHIP integration of log bytes, Akima sensitivity, equal image weights, and unchanged full cohorts. Interpolate log median time over the same 101-score grid used by the notebook. No extrapolation, silently shortened quality interval, or dropping missing images. Require complete timing at every retained unresampled point. If the CUDA curve does not bracket the interval, report that coverage gap before proposing extra collection.

   Return exact commands and revisions, hardware/software provenance, corpus verification, raw timing and score ledgers, retained JXL outputs, completeness by effort/quality/resolution, per-image and aggregate reports, and overall/resolution plots. Preserve failures and partial status. Keep different machines as separate runs; do not merge them into one timing population or overwrite the existing Mac notebook results by default.

   Calibrated target-60/70/85 runs are a separate follow-up. If requested later, reproduce tolerance ±0.5, distance bounds 0.01–25, up to 24 probes, and five timing repetitions per accepted match. They are not required to reproduce the standard fixed-sweep BD-rate plot.
