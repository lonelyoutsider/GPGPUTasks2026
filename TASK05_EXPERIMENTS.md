# Task05: experiments from 0f1e618

## Baseline analysis

- `radix_sort_01_local_counting`: stable sorting of each 256-key tile by four
  individual bits. Each Hillis–Steele scan uses 17 barriers plus a barrier after
  the partition. Including initialization and histogram extraction: 76 barriers,
  5248 bytes of local memory. Ranks are written back in original input order.
  Padding with `UINT_MAX` handles the final partial tile.
- The histogram is digit-major, `counts[digit * groups + group]`. Scanning this
  flattened array produces both bucket bases and prefixes within each bucket.
- `radix_sort_02_global_prefixes_scan_sum_reduction`: pairwise sums; 23 launches
  for 6,250,000 counts. Its intermediate levels occupy about 25 MB.
- `fill_buffer_with_zeros` and
  `radix_sort_03_global_prefixes_scan_accumulation`: initialize the result and add
  the contribution of each of 24 tree levels over the entire histogram.
- `radix_sort_04_scatter`: rank plus histogram prefix gives the stable global
  destination. It reads from immutable `input_gpu` on the first pass; subsequent
  passes alternate two separate work/output buffers. Eight passes end in output.
- Total: 50 launches/pass, 400/sort. Every baseline `exec()` waits on its event.
  The CUDA and Vulkan radix kernels remain assignment stubs; this experiment
  uses OpenCL only. A+B and the other APIs are not part of the sorting path.

## Experiment 1: hierarchical global scan

`RadixHistogramScan` scans 1024 counts per workgroup: 256 threads each load a
`uint4`, scan their four-value totals using a padded Blelloch tree, then write
four exclusive prefixes. It uses 18 barriers and 1056 bytes of local memory.
Tail elements are masked; all threads still reach every barrier.

For 100,000,000 keys the hierarchy is 6,250,000 -> 6104 -> 6 counts. The upper
levels scan their sum buffers in place; two downward kernels add block offsets.
Only two scratch buffers (24,440 bytes total) are needed. The top total is not
stored. The main histogram is read/written for the local scan and updated once
for global offsets, with contiguous vector accesses. No prefix initialization.

Five scan launches/pass, 56 total launches/sort including unchanged counting and
scatter. Scan kernels are enqueued asynchronously on the existing in-order queue;
the synchronous scatter waits for them, so timings include their execution.
`TASK05_SCAN=baseline` preserves the original global scan for A/B comparison.

## Local validation, 2026-10-27

Windows, Release/MSVC 19.50, OpenCL Intel(R) Graphics, Core Ultra 7 255U laptop.
These are laptop measurements, **not Tesla V100 results**. The V100 target of
less than 28 ms is unverified. No teacher-server submission has been performed.

All six required sizes passed comparison with `std::sort` and verification that
`input_gpu` was unchanged. Each case performs ten complete sorts of the same
immutable input. Default input generation is kept, with guaranteed duplicates.

| n | Baseline median, ms | Hierarchical scan median, ms |
|---:|---:|---:|
| 10 | 0.8703 | 0.2708 |
| 256 | 1.0123 | 0.2664 |
| 257 | 1.0267 | 0.2679 |
| 600 | 1.1849 | 0.3265 |
| 65537 | 3.3582 | 3.0052 |
| 100000000 | 2323.61 | 2089.81 |

An additional 100,000,000-element test with full-range uint32 values, explicit
0, 0x80000000, UINT_MAX and duplicates passed both paths: baseline 2292.50 ms,
hierarchical 2415.03 ms. This run regressed by 5.3%; reducing launches alone does
not establish a full-sort performance improvement for every workload/device.
Do not extrapolate the laptop timings to V100.

Both scan modes also passed `full-u32`, `duplicates` (16 repeated uint32 values),
`equal` (all UINT_MAX), and `descending` at 10, 256, 257, 600 and 65537 elements.
Direct scan tests passed 14 lengths: 1, 10, 16, 255, 256, 257, 1023, 1024, 1025,
6104, 1048575, 1048576, 1048577, 6250000. Each checks every prefix against a CPU
exclusive sum, unchanged input, and buffer reuse for three histogram patterns.
Clang 21 accepted the scan kernels as OpenCL C 1.2.

## Reproduce

Build using the project's regular CMake configuration. Device selection remains
the original numeric CLI argument. Environment options:

- `TASK05_SIZE`: 1..100000000, default 100000000.
- `TASK05_PATTERN`: random (default), full-u32, duplicates, equal, descending.
- `TASK05_SCAN`: hierarchical (default), baseline.
- `TASK05_SCAN_TEST=1`: run direct histogram scan tests instead of sorting.

PowerShell (set dependency DLL search paths as needed):

```powershell
./scripts/windows/test_task05.ps1 -Device 0
./scripts/windows/test_task05.ps1 -Sizes 10,256,257,600,65537 `
    -Patterns full-u32,duplicates,equal,descending
./scripts/windows/test_task05.ps1 -Sizes 100000000 -Patterns full-u32
$env:TASK05_SCAN_TEST = '1'
./build/Release/main_radix_sort.exe 0
Remove-Item Env:TASK05_SCAN_TEST
```

The runner saves individual logs and `results.csv` under `build/task05-results`.
Use distinct `-OutputDirectory` paths to retain multiple runs. Default benchmark
behavior still sorts all 100 million values and retains ten measurements.
