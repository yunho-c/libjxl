# Image batch benchmark

`jxl_image_batch_benchmark` measures warm wall time for N sequential encodes
versus N concurrent encodes. It uses the existing libjxl image loader, complete
encoder helper, and persistent thread pools. Its CSV format is compatible with
GJXL's `gjxl_image_batch_benchmark`; the two tools build independently.

Build in a Release tree with developer tools enabled:

```sh
cmake -S . -B build-release-codex -DCMAKE_BUILD_TYPE=Release -DJPEGXL_ENABLE_DEVTOOLS=ON
cmake --build build-release-codex --target jxl_image_batch_benchmark -j 6
build-release-codex/tools/jxl_image_batch_benchmark \
  --input /path/to/pfms --batch-sizes 1,2,4 --raw-samples libjxl.csv
```

`--input` accepts an RGB PFM file or a directory and can be repeated. Directories
select `.pfm` files case-insensitively without recursion; canonical paths are
sorted and deduplicated. Each image is measured separately at its original
dimensions. Use the same prepared PFMs in both repositories, including the
desired small, 1080p, and 4K images. File conversion and resizing are separate
preparation steps.

Pixels are interpreted as **linear sRGB**, must be finite, and are not clamped.
The PFM scale must be `+1` or `-1` (either byte order): GJXL applies its magnitude
while libjxl's loader ignores it, so other scales are rejected. The on-disk PFM
layout is converted to libjxl's packed RGB representation before timing; GJXL
instead prepares planar RGB before timing.

Defaults are distance 1.2, effort 7, batches 1,2,4,8, three paired samples, and
one warmup pair. Override with `--distance`, `--effort` (1..10), `--batch-sizes`,
`--samples`, and `--warmups`. Counts must be positive; batch sizes must be unique
and increasing. `--help` lists the options.

## Thread policy and timing

`--threads-per-image N` fixes the intra-image thread setting for the entire run.
The default is libjxl's hardware worker count, resolved once and printed at
startup. With N=1, encoding runs on the outer image worker, with no inner worker
threads. With N>1, each outer worker has its own persistent pool of N inner
workers. The outer worker runs serial encoder sections and waits during
parallel sections; N describes the inner parallelism, not the total OS thread
count. Both serial and concurrent drivers remain alive while sampling.

Increasing batch size does **not** divide a fixed CPU budget. To examine
single-threaded encodes scaling across images, explicitly use
`--threads-per-image 1`. Keep this distinct from the default full per-image
parallelism experiment. GJXL currently uses its own automatic per-image CPU
policy, so these settings should be reported alongside cross-encoder results.

For batch size B, a one-worker driver encodes B copies sequentially and a
B-worker driver encodes B copies concurrently. Workers share read-only input.
Timing surrounds the whole batch call, including scheduling, result allocation,
encoder creation, encoding, and in-memory output growth. It excludes input
loading/conversion, pool construction, warmups, validation, and output file I/O.
It does not sum overlapping per-image timings. Fresh output buffers are used
for every call; encoder objects are created by the normal encode helper, while
thread pools persist.

A single-image reference is decoded and checked for dimensions and finite RGB
pixels. Every warmup and measured output must match its bytes exactly; failed
validation exits nonzero. All validation happens outside the timed intervals.
Order alternates serial-first and batch-first for warmups and measured pairs.

## Results

Stdout is summary CSV, with the same columns as GJXL's terminal summary:

```text
workload,width,height,batch_size,serial_median_ms,batch_median_ms,batch_ms_per_image,batch_images_per_second,paired_speedup_median,paired_speedup_min,paired_speedup_max
```

`--raw-samples NEW.csv` emits one row per validated pair, using GJXL's columns:

```text
codec,workload,source,width,height,batch_size,sample,order,requested_backend,backend,aq_mode,distance,effort,thread_policy,timing_boundary,serial_ns,batch_ns,encoded_bytes_per_image
```

Rows use `codec=libjxl`, `requested_backend=cpu`, `backend=cpu`, `aq_mode=n/a`,
`thread_policy=fixed_per_image:N`, and
`timing_boundary=linear_rgb_to_in_memory_codestream`. Workload and source are
canonical input paths; sample indices start at zero. Times are integer
nanoseconds for the **whole batch**, and byte counts are per image. The raw
destination must not exist and its parent must exist. Rows are flushed after
each pair; an unsuccessful run can leave partial output and must not be treated
as complete.

Within an encoder, batching speedup is `serial_ns / batch_ns`; throughput is
`batch_size * 1e9 / batch_ns`. Cross-encoder throughput comparisons use the same
inputs, batch size, and documented settings. Equal requested distance and
effort describe a configuration comparison, not matched decoded quality. Save
the command, source revisions/build configuration, and machine information
with each run. Use a new file for each independent process run, and alternate
encoder run order when collecting comparisons. Sample minima/maxima are ranges
within one process, not confidence bounds.

Run the focused CLI checks with:

```sh
python3 tools/scripts/image_batch_benchmark_test.py \
  --benchmark build-release-codex/tools/jxl_image_batch_benchmark \
  --cjxl build-release-codex/tools/cjxl
```

With `BUILD_TESTING=ON` and Python available, the core checks are also registered
as the `image_batch_benchmark_cli` CTest. `--cjxl` adds an optional independent
CLI check of the linear-input encoding size.
