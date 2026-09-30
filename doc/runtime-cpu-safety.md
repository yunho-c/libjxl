# CPU timing evidence and warmup qualification

The runtime collector hashes the actual codestream from **every fresh harness
process**, including repetitions 1–4. A size match is insufficient. All successful
repetitions must equal the retained per-setting codestream. Hashing and output
file writes remain outside the harness timer.

Each invocation retains its original harness JSON under `raw/timing/`, together
with a request record. The timing ledger links the raw file by path and SHA-256.
Only one successful codestream per setting is retained. A failed attempt keeps
its raw report, differing codestream, and failure reason for inspection. If the
process stops between saving artifacts and appending the ledger, those artifacts
remain as an uncommitted attempt; resume collects a new sample instead of
inventing a completed measurement. Resume verifies saved raw reports, outputs,
ledger linkage, and duplicate sample IDs before collecting pending timings.

Use a fresh run with the revised collector. Legacy timing rows lack independent
output verification and original raw reports; they cannot be upgraded by adding
the retained first output's hash. Keep historical runs with their frozen tools.
The new evidence protocol and warmup settings are frozen in run metadata.

## Warmup sensitivity

Before fixed timing, the collector runs a separate CPU warmup diagnostic by
default. It selects the smallest and largest images from the full manifest,
the lowest and highest requested efforts, and nominal Q80 (distance 1.9).
For each unique image/effort pair, five fresh-process pairs compare the configured
warmup count with two additional warmups: normally 1 versus 3. Pair order
alternates. Each process also performs the harness's untimed validation encode.

The absolute percentage change between the two median times flags sensitivity
above 5% by default. Paired changes are also reported as descriptive diagnostics.
All timing samples, hashes and original warnings live separately in
`warmup-check/` and `raw/warmup-check/`; none enter sweep medians or sample counts.
This check detects sensitivity, not proof of a steady thermal state.

Choose the response when creating the run:

- `--warmup-policy continue` (default): retain a warning and continue the sweep.
- `--warmup-policy retry`: repeat the complete diagnostic, at most
  `--warmup-max-retries` additional times (default 1). If the last attempt still
  flags sensitivity, stop with an error. Earlier attempts and warnings remain
  visible even if a later attempt passes. Resume never replenishes retries.
- `--warmup-policy error`: retain the diagnostic and stop before sweep timing.

`--warmup-threshold-percent` changes the threshold. `--no-check-warmups` explicitly
disables this diagnostic and is recorded in metadata. Configuration changes need
a new run directory; a failing diagnostic cannot silently change policy on resume.
Subprocess failures, invalid reports, or codestream mismatches always stop the
collector regardless of warmup policy. The policy applies only to sensitivity.

For a timing-only run, stage and Samply tools are unnecessary:

```powershell
python tools/scripts/cjxl_runtime_characterization.py run `
  --phase timing --corpus C:/study/corpus.json --output C:/study/cpu-fixed `
  --benchmark C:/tools/gjxl_libjxl_comparison_benchmark.exe `
  --ordinary-build-manifest C:/study/cpu-build.json `
  --num-threads 8 --warmups 1 --repetitions 5 --warmup-policy continue
```

These are collector options, not a complete cross-machine study launcher: build,
corpus/provenance checks, process supervision, external scoring and calibration
still follow the handoff. Diagnostics can be bounded with `--max-jobs`; their
budget is separate from each measurement phase. Rerun the identical command to
resume saved work. Full qualification uses 40 diagnostic processes with distinct
small/large images and low/high efforts, before any sweep processes.

`verify --run RUN --djxl DECODER` audits raw evidence for the new timing protocol
and decodes the retained outputs; timing-only runs do not require stage profiles.

## Archived CUDA studies

A study containing `ARCHIVED.md` is excluded from automatic CUDA notebook
discovery. Explicit paths remain available for historical inspection. Archive
in place when other artifacts depend on absolute corpus/tool paths; do not alter
frozen manifests, ledgers or encoded outputs to make an old study appear current.
