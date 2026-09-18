# Fusion Oracle Teacher Design

## Goal

Add an optional cross-engine teacher path that labels MinoFlux states with MochBot/fusion's offline search oracle, while keeping MinoFlux's existing neural ranking dataset format unchanged.

The first deliverable is an adapter boundary and dataset generator, not a runtime playing dependency. MinoFlux must continue to run normally when fusion is absent.

## Scope

This feature spans two boundaries:

1. **fusion oracle output contract** — expose enough information about the selected root action to identify the exact Hold/direct placement.
2. **MinoFlux adapter and dataset generation** — serialize MinoFlux states into fusion requests, batch the oracle process, map returned actions back to exact MinoFlux `SearchAction`s, and write ordinary `minoflux_neural_ranking_dataset_v1` samples.

Out of scope for this iteration:

- embedding fusion as a Python extension or FFI library;
- changing MinoFlux model architecture;
- replacing existing heuristic, TETR.IO-capture, or human-review teachers;
- approximating unmatched oracle moves;
- using a fusion time budget during offline labeling;
- making fusion a required project dependency.

## Existing Contracts

MinoFlux neural ranking samples already encode the teacher choice using `expertIndex` / `expertIndices`, and each candidate move has the tuple `(use_hold, piece, x, y, rotation)`. The new teacher therefore does not need a new dataset schema.

MinoFlux candidate generation already exposes exact `SearchAction` objects containing `use_hold` and a reachable `Placement`.

fusion's `generate_policy_value_labels` input already accepts board rows, current piece, Hold, queue, combo, B2B, pending garbage, lines, and bag metadata. Its current output exposes `best_move_raw`, `best_value`, `position_complexity`, root scores, and policy probabilities. The current `best_move_raw` alone is not sufficient to distinguish every Hold/direct action, so the oracle output needs an additive action-identity field.

## Architecture

### 1. Fusion oracle output extension

Extend the label JSON with two additive fields:

```json
{
  "bestHoldUsed": true,
  "bestCells": [[3, 0], [4, 0], [4, 1], [5, 1]]
}
```

`bestHoldUsed` is copied from fusion's root search result. `bestCells` is the four occupied board cells represented in fusion's native bottom-up coordinate system after decoding the selected move.

Existing fields remain unchanged. Existing consumers that ignore unknown JSON fields remain compatible.

The adapter must reject output that lacks these fields unless it can prove a unique exact action match without them. No heuristic inference or nearest-action matching is allowed.

### 2. MinoFlux state conversion

Create `src/minoflux_ai/fusion_oracle.py` as the only module that knows the external wire format.

For each MinoFlux `Game` state, produce a fusion oracle request:

- `player_board_rows`: convert MinoFlux's 24 top-down rows to bottom-up 10-bit masks; only occupied/not-occupied matters.
- `opponent_board_rows`: empty for this first teacher integration.
- `current_piece`: lowercase current tetromino.
- `hold_piece`: lowercase Hold tetromino or null.
- `queue`: lowercase visible queue. The caller must supply enough queue for the requested fusion search depth; if insufficient, the sample is skipped rather than silently fabricated.
- `combo`: fusion stores the signed S2/TETR.IO combo value plus one, so map MinoFlux `combo == -1` to `0`, otherwise `game.combo + 1`.
- `b2b`: fusion stores the signed displayed B2B value plus one. Map inactive B2B to `0`; active MinoFlux B2B maps to `game.b2b_chain + 1`.
- `lines`: `game.lines`.
- `pending_garbage`: `0` in the first dataset generator because a normal solo MinoFlux `Game` does not retain an inbound versus queue at this boundary.
- `bag_number`: `0` initially; it is metadata for the current oracle binary and must not be guessed from an unrelated RNG representation.

### 3. Action identity conversion

Do not map pivot coordinates directly between engines.

Instead, map the selected fusion move to a normalized occupied-cell identity:

```text
(use_hold, piece, frozenset((x, y_bottom_up) for four occupied cells))
```

For each MinoFlux legal `SearchAction`, compute its four occupied cells using `placement.cells`, convert MinoFlux top-down `y` to bottom-up with:

```text
y_bottom_up = (game.height - 1) - y_top_down
```

and compare the full identity including `use_hold` and `piece`.

This avoids relying on engine-specific pivot definitions, O/I-piece origin offsets, rotation-number conventions, or T-spin sentinels.

Matching rules:

- exactly one matching MinoFlux action -> accepted teacher action;
- zero matches -> skip as `oracle-action-unmatched`;
- more than one match -> skip as `oracle-action-ambiguous`.

There is never a nearest placement fallback.

### 4. Candidate generation and dataset records

The generator enumerates MinoFlux's normal exact reachable direct + Hold actions using the same SRS/SRS+ path used by existing neural ranking data.

After the fusion teacher action is matched, candidate states are encoded with the existing neural encoder and written as ordinary `NeuralRankingSample`s. The teacher candidate must always be retained when candidate sampling/capping is applied.

The record uses the existing fields:

```json
{
  "format": "minoflux_neural_ranking_dataset_v1",
  "seed": 123,
  "pieceIndex": 42,
  "expertIndex": 3,
  "expertIndices": [3],
  "candidates": []
}
```

Additional provenance fields may be appended at record level:

```json
{
  "teacher": "fusion-offline-oracle",
  "fusionOracle": {
    "beamWidth": 2000,
    "depth": 18,
    "timeBudgetMs": null
  }
}
```

The training loader must not require these provenance fields.

### 5. Batch execution

The adapter runs fusion as an external process, but not once per sample.

Generation flow:

1. Build all valid oracle requests for a shard.
2. Write one request JSONL per worker shard.
3. Launch up to `workers` oracle processes, one process per shard.
4. Force `RAYON_NUM_THREADS=1` for each child process so outer process parallelism controls concurrency.
5. Do not pass `--time-budget-ms`; the fusion implementation switches to iterative widening when a budget is enabled, making offline full-width labeling significantly slower.
6. Parse outputs in the same line order and preserve each source `(seed, piece_index)` identity.
7. Merge labeled samples deterministically in source order.

Default worker count should be configurable rather than hard-coded. The Ryzen 5 5600G benchmark showed throughput still increasing through 12 outer workers, but machine-specific defaults do not belong in the library API.

### 6. Error handling

Every rejected state is counted by a stable reason string. At minimum:

- `insufficient-queue`
- `invalid-state`
- `oracle-process-failed`
- `oracle-output-missing`
- `oracle-output-invalid`
- `oracle-action-unmatched`
- `oracle-action-ambiguous`
- `teacher-not-encodable`

A failed shard must not silently produce a partially trusted dataset. The generator reports the failure and either aborts or, under an explicit skip-failures option, records skipped counts. Default behavior is fail-fast for process/contract failures and skip-only for state/action mismatch.

### 7. CLI surface

Expose the feature under the existing neural CLI rather than adding another top-level executable. The intended command shape is:

```text
uv run minoflux-neural fusion-dataset \
  --oracle-bin /path/to/generate_policy_value_labels \
  --output data/neural/fusion-oracle.jsonl \
  --games 200 \
  --max-pieces 500 \
  --workers 12 \
  --allow-180
```

Generation follows learner or teacher trajectories only when explicitly selected. The first implementation should use normal seeded MinoFlux game trajectories and make the trajectory policy explicit in metadata.

### 8. Testing

Tests must cover the boundary rather than trusting either engine implicitly.

#### Unit tests

- board row conversion on empty, single-cell, and mixed 24-row boards;
- all seven tetrominoes and rotations for occupied-cell normalization;
- Hold false / occupied Hold / empty Hold action identity;
- combo `-1`, `0`, `1` conversion;
- inactive B2B, active x0, x1, and x4 conversion;
- exact unique match;
- deliberately unmatched move;
- deliberately ambiguous action identity;
- output parsing with required action identity fields;
- deterministic shard merge order.

#### Integration test

Use a tiny fake oracle executable in tests. It reads request JSONL and emits deterministic labels, allowing CI to exercise batching and process failure handling without requiring Rust or the external fusion repository.

A local/manual integration check may additionally point at a real release build of `generate_policy_value_labels`.

## Compatibility and rollout

The adapter is opt-in and imports no fusion package. Existing commands, datasets, training, and benchmarks remain unchanged.

The fusion output change is additive. If upstream fusion cannot be modified or the user is running an older binary, the adapter reports a clear contract mismatch instead of guessing Hold usage.

The feature should land independently of the open TETR.IO Hold-distillation work. Both ultimately produce the same ranking dataset format but have different source-state and teacher boundaries.

## Success criteria

The feature is ready when:

1. A MinoFlux state can be serialized to fusion, labeled in a batch, and mapped back to exactly one MinoFlux `SearchAction`.
2. Hold and direct choices remain distinguishable end-to-end.
3. Unmatched cross-engine moves are skipped rather than approximated.
4. Produced samples load through the existing neural ranking training path without schema changes.
5. Batch generation supports configurable multi-process workers and runs with no fusion internal time budget by default.
6. CI covers the adapter with a fake oracle and does not require the fusion repository or Rust toolchain.
