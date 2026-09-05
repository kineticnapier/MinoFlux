# Placement Teacher v2 performance

The reproducible harness is `tools/benchmark_placement_teacher.py`. It runs the
real dataset generator, including candidate encoding and JSONL writing; no
mock scorer or reduced root search is used. Dataset SHA-256 and optional exact
byte comparison include all candidate scores, labels, record ordering and seeds.

## Measurement environment and reference

- Reference teacher: `46316f0` on `self-improve`.
- Linux, CPython 3.12.13, AMD EPYC 9V74 host, container CPU quota: 8 cores.
- Teacher depth 2, beam 24, max candidates 24, exact SRS, Hold enabled.
- Seed base 8000001, step 31; all other settings are the production defaults.
- Each timing starts with an empty reachability cache and includes worker startup.
- Throughput runs have no profiler enabled and run sequentially without another
  benchmark. These are individual measurements, not statistical confidence bounds.
- Reference mode loads the original teacher module from Git but uses current
  shared engine/reachability dependencies. This change adds feature helpers without
  changing the existing helpers used by the reference teacher.

The user's i7-12700F measurements (A approximately 27 s, B approximately 37 s)
come from another machine and must not be directly divided by these timings.

## Search invariants

Every legal root action is still enumerated with exact SRS and scored. Root
actions retain their original paths and complete transition breakdowns. The
depth-2 leaf returns the maximum of every legal immediate value, so its beam
width has no effect. Depth-3 intermediate nodes retain the original full sort,
beam slicing, and path tie-break; only their final leaves use the fast evaluator.

The leaf evaluator uses the engine's shared spin and B2B rules, reproduces combo,
attack (including released Surge), perfect clear and every teacher penalty, and
checks both hidden-row topout and the next piece's spawn collision. Its float
conversion and summation order matches the reference. A short queue or custom
Game subclass uses an actual cloned engine transition instead.

Hold reachability uses the existing read-only search view. An empty queue falls
back to real engine Hold so bag refill is preserved. Public entry points still
normalize configuration; internal loops reuse that normalized object. Parent row
masks and before-features are shared by all leaf candidates. Profiling lives only
in the standalone harness, adding no timers to normal dataset generation.

## Reproduce

```powershell
uv run python tools/benchmark_placement_teacher.py --reference-commit 46316f0 --games 12 --pieces 50 --workers 12 --output before-A.jsonl --report before-A.json
uv run python tools/benchmark_placement_teacher.py --games 12 --pieces 50 --workers 12 --output after-A.jsonl --compare before-A.jsonl --report after-A.json
```

For B use `--games 3 --pieces 100 --workers 3`; for the single-worker run use
`--games 1 --pieces 50 --workers 1`. `--compare` fails if even one output byte
differs. Use `--spawn` to exercise Windows-style process startup on Linux.

Profiling must be a separate single-worker run:

```powershell
uv run python tools/benchmark_placement_teacher.py --games 1 --pieces 5 --workers 1 --profile --output profile.jsonl --report profile.json
```

The report contains every cProfile function's call count, exclusive and inclusive
milliseconds, mean inclusive/exclusive microseconds per call, and percent of cProfile's
total time. It also contains the existing detailed ReachabilityProfile counters.
Inclusive percentages overlap (for example, legal action generation contains
reachability) and must not be added. Instrumentation changes timing proportions;
use the uninstrumented runs for before/after throughput.

## Baseline profile

One game, five pieces, 6.484 s instrumented wall time:

| Function | Inclusive share | Calls |
| --- | ---: | ---: |
| `reachable_placements` | 40.75% | 470 |
| `extract_board_features` | 34.57% | 10315 |
| `apply_search_action` | 14.72% | 10205 |
| `clone_game` | 6.98% | 10435 |
| Teacher config normalization | 0.99% | 10329 |
| `hold` | 0.41% | 4904 |
| List sorting | 0.20% | 697 |

`_legal_actions` includes 41.42%; its overhead outside reachability is small.
Full feature extraction spends 22.23% of total time in T-spin slot counting,
which the teacher objective does not use. Reachability makes 267860 BFS visits;
only 8/470 calls hit its existing cache (1.70%). Path reconstruction takes
22.3 ms for 9901 calls. Therefore full feature extraction, exact reachability,
and engine transitions dominate; sorting alone cannot deliver a large gain.

## Before / after datasets

| Case | Games × pieces | Workers | Before seconds | After seconds | Speedup | Time reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 12 × 50 | 12 | 40.714 | 17.396 | 2.340× | 57.27% |
| B | 3 × 100 | 3 | 43.963 | 19.051 | 2.308× | 56.67% |
| Single | 1 × 50 | 1 | 18.140 | 8.272 | 2.193× | 54.40% |

All three output files match the reference **byte for byte**. A has 600 records /
14382 candidates / 16089633 bytes; B has 300 / 7182 / 8029798; Single has
50 / 1200 / 1341812. The single-worker improvement shows the gain is from cheaper
search rather than increased multiprocessing. Different game lengths and seed sets
mean this table is not a same-workload worker-scaling curve.

Matching before/after SHA-256:

- A: `34f4b029c0a0e6ceadeb455f518e88a0e2363391e95db66936147cfa8980231d`
- B: `d080ec808cb4ad7aa2da76a00f10a461a215afc4841cb1563f389c30b29c815c`
- Single: `ef7235b39788b38d25c77e65b91a38faec6b75307dd738f43a6926a783a0e7c3`

An additional two-game × two-piece, two-worker `--spawn` run verifies that both
the original module loader and optimized dataset generator work with Windows-style
process startup. These outputs also match byte for byte. This checks spawn behavior
on Linux; it does not replace a native Windows performance measurement.

## After profile and remaining work

The same one-game, five-piece instrumented run takes 3.247 s and produces exactly
the same JSONL bytes:

| Function | Inclusive share | Calls |
| --- | ---: | ---: |
| `reachable_placements` | 83.11% | 470 |
| `extract_teacher_board_features_from_masks` | 9.25% | 10315 |
| `apply_search_action` | 1.19% | 355 |
| `clone_game` | 0.74% | 350 |
| `_held_search_game` | 0.18% | 235 |
| List sorting | 0.18% | 475 |

The leaf bitboard evaluator eliminates 9850 engine transitions and 10085 Game
clones in this sample. Exact teacher-only features replace full features, including
the unused T-spin-slot scan. Future nodes use no paths; path reconstruction falls
from 9901 calls / 22.3 ms to 230 calls / 0.53 ms. Leaf best-value selection no longer
retains child Games or sorts candidates. Depth-3 non-leaf beam ordering still uses
the original ranking and full paths for tie-breaking.

Exact SRS now dominates. Its 470 queries hit the cache zero times in this small
sample, versus eight previously: root queries still need paths while leaves do not,
and `include_paths` is part of the correct cache key. BFS visits increase slightly
(267860 to 272875) because those eight cross-mode cache hits disappear. This small
cost is included in the measured net speedup; no approximate placement generation
or heuristic filtering was introduced.

Further safe candidates require another measured, independently verified change:
moving the existing exact SRS traversal to compiled code; reducing allocation in
representative-state handling while preserving all rotation/kick metadata; or sharing
path-free placement data across cache modes with exact path reconstruction retained
for roots. Increasing the current cache alone is unlikely to help this workload's
low hit rate. Board-feature calculation is the next largest measured area.

The full 50-game × 300-piece production workload was not timed here; extrapolating
from shorter games is unreliable because later boards change reachability cost.
