# GJXL CUDA fixed and calibrated quality studies

`tools/scripts/cjxl_cuda_study.py` packages the CUDA study setup independently
of any machine, corpus directory, study date, or source revision. It uses the
existing quality collector and a standalone CMake harness in
`tools/scripts/cuda_quality_harness/`. The encoder checkout is not modified.
The harness requires GJXL's updated efforts 1–4 policy: DCT8-only transforms
and no internal perceptual evaluation when the optional final score is off.

## Requirements and preparation

Use Python 3.10+, Git, CMake 3.25+, Ninja, a CUDA-capable compiler/toolkit, and
an NVIDIA GPU. On Windows, run the build from the matching Visual Studio
developer shell, with the desired CUDA toolkit on `PATH`. Compiler/toolkit
selection belongs to that shell or explicit CMake arguments; the scripts do
not assume a Visual Studio installation path or CUDA version.

Supply a clean GJXL checkout at the desired commit, a corpus manifest in the
existing `cjxl_quality_characterization.py` format, `djxl`, and the Rust
`tools/scripts/quality_metric` adapter at its pinned fast-ssim2 revision. The
manifest must contain absolute PFM paths, source/PFM hashes, dimensions, image
IDs, corpus/resolution labels, and canonical `RGB_D65_SRG_Rel_Lin` encoding.
Preparation verifies the PFMs and reuses their bytes without resizing them.
The original 65-image study used 24 Kodak images, 32 CLIC test images, and
three controlled photographs at approximately 12, 24 and 48 MP. Other corpus
sizes are supported; the full manifest remains the coverage denominator.

Example PowerShell setup (replace the paths, commit and image IDs):

```powershell
git -C ../gjxl worktree add --detach ../gjxl-study-source <commit>
python tools/scripts/cjxl_cuda_study.py prepare `
  --study ../cuda-study --source ../gjxl-study-source --revision <commit> `
  --corpus ../corpus/corpus.json --decoder ../tools/djxl.exe `
  --scorer ../tools/cjxl-quality-metric.exe --cuda-arch 86 `
  --pilot-images kodak/01,unsplash/campus_interior/24mp,unsplash/campus_interior/48mp `
  --warmup-images kodak/01,unsplash/campus_interior/24mp
python ../cuda-study/collection-code/cjxl_cuda_study.py build --study ../cuda-study
```

The example architecture 86 is appropriate for the original RTX 3060 laptop;
select the architecture for the actual device. `--cmake-arg=-DNAME=value` and
`--generator` customize configuration. Windows cudart DLLs are copied beside
the harness and native policy tests. The build record hashes source files,
libraries, executables, CMake caches and runtime DLLs, and records the revision,
submodules, platform, Python version, GPU and driver. Preparation snapshots the
collector, harness sources, decoder/scorer and their DLLs. A plan checksum and
file hashes prevent silently changing a prepared study. Never rebuild or edit
the frozen files to resume a run; prepare another study for a changed setup.

## Qualification, collection and pause

Only `qualify` and `run` start encodes. `prepare`, `build`, `pause`, `status`,
validation and notebook analysis do not start collection.

```powershell
python ../cuda-study/collection-code/cjxl_cuda_study.py qualify --study ../cuda-study
python ../cuda-study/collection-code/cjxl_cuda_study.py run --study ../cuda-study
```

Qualification runs the native low-effort strategy and CUDA tests, a separate
fixed/calibrated pilot (efforts 1/4/10, Q30/80, targets 60/85), and a raw/hash
audit. It also compares one versus three explicit warmups using five
counterbalanced pairs on two sentinels at efforts 1 and 10. All 40 diagnostic
samples stay outside the sweep. A change over 5% flags review and leaves
`qualification.json` unsuccessful. Retain that flag and all samples, investigate
it independently, and document the evidence and decision in the qualification
record before setting `passed` to true. Do not discard samples or repeatedly
rerun diagnostics until they happen to pass. The full runner requires a passed
qualification tied to the exact source revision and harness hash.

The runner completes fixed collection before seeding calibrated collection
from that study's decoded scores. Each effort's timing and external scoring
are sequential. Run only one study on a GPU at a time. The runner holds a
pipeline lock, and collectors hold run/device locks. On Windows it prevents
sleep while active. Ten-second AC/NVML observations record temperature, power
and clocks without launching a polling subprocess during timing. Clocks are
not locked. AC is required by default; `prepare --allow-battery` explicitly
records an override for hosts without AC telemetry. Loss of AC stops the active
process tree and requests a pause: inspect telemetry and nearby records before
resuming because the exact power transition may precede the polling sample.

From another terminal:

```powershell
python ../cuda-study/collection-code/cjxl_cuda_study.py pause --study ../cuda-study
python ../cuda-study/collection-code/cjxl_cuda_study.py status --study ../cuda-study
```

Pause writes `pause.request`. The active operation finishes and the collector
stops at its next boundary, outside the harness timing interval. It publishes
partial summaries and returns 130. The pipeline records `paused` and does not
advance to another phase. The request remains until an explicit resume:

```powershell
python ../cuda-study/collection-code/cjxl_cuda_study.py run --study ../cuda-study --resume
```

For an interrupted qualification, use `qualify --resume`. Completed timing
sample IDs and terminal failure outcomes are reused; unfinished work is
retried. Calibration retains its total per-target probe budget across resumes.
The collector also exposes `warmup --run RUN` as a diagnostic-only command and
`--pause-file PATH` on collection commands. Neither changes the saved timing
protocol. Older studies retain their own frozen collectors and original resume
commands: **these tools do not migrate or modify them**, and the new pause
option is not available in those older snapshots.

## Protocol and verification

The fixed protocol attempts efforts 1–10 at Q10/30/50/70/80/90/95 for every
manifest image. The calibrated protocol targets decoded fast-ssim2 scores
60/70/85 within ±0.5, distance bounds 0.01–25, and at most 24 new probes per
target. It uses fully-resident CUDA, default density, automatic compression,
eight maximum participating CPU threads, and optional final Butteraugli scoring
disabled. Fixed-Q timing comparisons are not matched-quality speedups.

Every successful setting has five fresh-process measurements. PFM loading,
one validation encode (including backend initialization), and one explicit
warmup precede a single timed public `EncodeLinearRgbVarDctCodestream` call.
Integer `steady_clock` nanoseconds include CPU/GPU work, preparation inside the
API, transfers, synchronization and serialization; file I/O, startup, validation,
warmup, decoding and external quality scoring are excluded. Validation and
warmup bytes must equal the timed output. Policy and determinism checks occur
outside the timed interval. No profiling path or CPU fallback is selected.

Actual CUDA allocation failures are saved with their error and setting, rather
than synthesized timings. All sizes are attempted anew for each revision.
Unresolved calibrated targets remain explicit. `collection_finished` means
every requested setting has a terminal outcome; `complete` requires numeric
coverage everywhere. The notebook retains missing settings in its denominator.

After both sweeps, the runner invokes the read-only validator. It verifies the
requested grids against the plan and corpus, artifact hashes, all raw timing
reports, five unique repetitions, deterministic outputs, low-effort policy,
decoded-score linkage and tolerance. Unresolved calibration reasons replay from
the observations available at their original timestamp, including the 24-probe
cap. Run it again independently with:

```powershell
python ../cuda-study/collection-code/cjxl_cuda_study.py validate --study ../cuda-study
```

Keep the corpus, binaries, environments, logs, ledgers, generated plots and
executed notebooks outside Git. Commit the reusable sources and documentation.
The notebooks read saved data only; analysis instructions accompany the CUDA
section of `cjxl_runtime_characterization_notebook.py`.

## Development checks (no GPU collection)

```powershell
python -m unittest discover -s tools/scripts -p cjxl_cuda_study_test.py
```
