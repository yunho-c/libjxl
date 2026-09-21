# Paired B1/B4 throughput collection

`tools/scripts/cjxl_batch_characterization.py` collects matched nominal
distance/effort measurements for the paper's saved input cohort. Collection
is explicit, bounded and resumable. The runtime notebook reads the resulting
ledger through `PAPER_BATCH_RUN` / `CJXL_PAPER_BATCH_RUN`; it never launches
encoders. Pilot results appear in a separate table and do not fill the
full-cohort single-image paper table.

## Measurement protocol

Each independently launched process loads one linear-sRGB PFM, produces a
validation reference, warms one B1/B4 pair, and measures one B1/B4 pair.
Four copies of that image share the same input storage. Both encoders compare
every warm and measured codestream with their own single-image reference;
the collector also requires that reference's SHA-256 and byte count to match
the original fixed-study output. This verifies output equivalence within each
encoder. Equal requested distance and effort do not imply equal measured
quality between encoders.

B1 and B4 call order, and encoder order, alternate between rounds. The full
protocol uses five independent rounds; the pilot uses two. Complete-call wall
times include encoding, output allocation, CPU/GPU work and synchronization.
Loading, validation, file I/O, and warmups are outside measured intervals but
included in collection-time estimates. Encoder scheduling objects and thread
pools are prepared before timing. B1 here is a control using the batch harness,
so it is kept separate from the original single-image harness measurements.

GJXL uses fully-resident Metal and an explicit shared execution domain capped
at **8 CPU participants**, with a per-image cap of 8. libjxl uses **8 inner
workers for B1 and 2 per image for B4**. Its separate outer caller threads can
also execute serial work. This is an eight-inner-worker policy, not an identical
cap on active CPU threads. Raw records retain the policy and GJXL's observed
peak participant/backing/committed accounting. Domain accounting is distinct
from physical RSS and is not an OS memory measurement.

For an image with P megapixels and median call times t1 and t4, rates are
P/t1 and 4P/t4; batching gain is 4t1/t4. For an effort/quality, sum input pixels
and per-image median times before dividing. Average the quality-specific rates
equally. An aggregate is blank until every selected image, quality and round
is present. Retain resolution-specific tuples to inspect scaling. Batching
measures aggregate throughput, not reduced single-image latency or fused GPU
execution.

## Build and initialize

The local macOS build helper links only new benchmark drivers against existing
frozen Release libraries. It does not rebuild or modify codec sources. It
checks the revisions identified by both saved studies and retains the driver
sources, binary/library hashes and compiler commands in `build.json`.

```sh
python3.13 tools/scripts/cjxl_batch_build.py \
  --libjxl-run ../libjxl-runtime-study-2026-09-03/quality-full-20260908 \
  --gjxl-run ../libjxl-runtime-study-2026-09-03/fixed-gjxl-full-20260915 \
  --gjxl-harness ../gjxl/benchmarks/image_batch_benchmark.cpp \
  --output ../libjxl-runtime-study-2026-09-03/NEW-batch-build

python3.13 tools/scripts/cjxl_batch_characterization.py init \
  --libjxl-run ../libjxl-runtime-study-2026-09-03/quality-full-20260908 \
  --gjxl-run ../libjxl-runtime-study-2026-09-03/fixed-gjxl-full-20260915 \
  --build-manifest ../libjxl-runtime-study-2026-09-03/NEW-batch-build/build.json \
  --run ../libjxl-runtime-study-2026-09-03/NEW-batch-pilot --pilot
```

Initialization requires a new directory. It freezes input identities, revision
and build provenance, baseline output hashes and the collector/helper sources.
The default pilot selects the smallest image above 1 MP, alpine lake at 12/48 MP,
Q90, and efforts 3/7/9: 36 fresh-process pairs. `--image-ids ID [ID ...]` selects
an explicit subset at initialization. Without `--pilot`, initialization selects
the 41-image cohort, six qualities and ten efforts, with five rounds: 24,600
processes across both encoders. Initialization alone performs no encodes.

For the GJXL-only sweep excluding 48 MP, add `--encoders gjxl
--max-megapixels 24 --largest-first`. This keeps the 38 images from 1 through
24 MP and schedules 11,400 pairs (six qualities, ten efforts, five rounds).
The upper pixel bound is inclusive. Largest-first ordering exercises remaining
large-image quality settings early. The selected encoder list and image order
are frozen in metadata; older two-encoder studies retain their original behavior.

## Run, inspect and resume

Use the frozen collector inside the run directory. For example, from that
directory:

```sh
python3.13 cjxl_batch_characterization.py run --run . --dry-run
caffeinate -ims python3.13 -u cjxl_batch_characterization.py run --run . \
  --budget-seconds 1800 --job-timeout 1200 > runner.log 2>&1
python3.13 cjxl_batch_characterization.py summarize --run .
```

Collection verifies binaries, linked libraries, frozen collector sources and
inputs, and refuses a detected competing codec process or concurrent collection
under the same study parent. Each pair has its own command, logs and raw CSV.
The ledger is appended only after validation; retries never count interrupted
or unvalidated attempts. Repeating `run` resumes missing pairs and preserves
old attempts. Repeating it on a complete run performs zero encodes.

The default guards stop at 24 GiB sampled process-group RSS, 2 GiB system swap
growth, or less than 6 GiB free disk. They are checked about every two seconds;
they cannot prevent a rapid allocation spike. RSS does not capture all Metal
or system memory pressure, so the swap guard matters. A signal, timeout or
guard failure terminates the active process group and writes an incomplete
status and coverage summary. CPU/load, power, thermal and VM snapshots are
retained before and after collection. Later collector versions additionally
retain per-attempt `resources.jsonl` samples.

`summary/image-tuples.csv` retains resolution and completeness;
`summary/throughput.csv` contains complete-cohort rates;
`summary/coverage.json` records pair counts. `summary/estimate.json` scales
saved single-image times by observed process-wall multipliers to estimate a
five-round full grid. Quality/content/effort interpolation and incomplete
libjxl effort-10 baselines limit that estimate; it does not establish memory
feasibility. Inspect status and coverage before using any estimate.

For publication, collect in a quiet session with sufficient physical memory.
Do not remove a memory guard merely to fill the 48 MP cells. A memory limit
that changes effective batch concurrency is a different policy and requires
fresh, clearly labeled measurements. Do not replace missing large-image
measurements with small-image gains.

## Notebook output

The **Paired B1/B4 collection** section recomputes its table from validated raw
records. Pilot exports use `encoding-batch-pilot.*`; full-study exports use
`encoding-batch-throughput.*`. Each has CSV, booktabs LaTeX and HTML, plus
image tuples, coverage and a methodology JSON recording the exact analyzed
ledger hash. The table displays each encoder's B1/B4 MP/s and batching gain.
Set `PAPER_BATCH_RUN = None` to skip this independent section.
