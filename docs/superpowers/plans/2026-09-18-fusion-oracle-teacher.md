# Fusion Oracle Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in MinoFlux dataset generator that labels seeded MinoFlux states with the MochBot/fusion offline oracle and writes ordinary neural ranking samples without changing the training schema.

**Architecture:** `src/minoflux_ai/fusion_oracle.py` owns the cross-engine wire contract, legacy `best_move_raw` decoding, exact occupied-cell action matching, batched shard execution, and dataset generation. `src/minoflux/neural_cli.py` exposes a `fusion-dataset` command. Tests use a fake executable so CI never requires Rust or a fusion checkout; a real fusion binary is only used for a manual smoke test.

**Tech Stack:** Python 3.11+, stdlib `subprocess` / `concurrent.futures`, MinoFlux `Game`, `SearchAction`, `reachable_placements`, existing neural ranking dataset schema; optional external MochBot/fusion release binary.

**Spec:** `docs/superpowers/specs/2026-09-18-fusion-oracle-teacher-design.md`

## Global Constraints

- Do not change `minoflux_neural_ranking_dataset_v1`.
- fusion remains optional and is never imported as a Python package.
- Do not pass `--time-budget-ms` for offline full-width labeling.
- Set `RAYON_NUM_THREADS=1` for each oracle child process.
- Never approximate an unmatched or ambiguous cross-engine move.
- Existing commands and training paths must remain unchanged.
- Legacy fusion output containing only `best_move_raw` may be accepted only when it maps to exactly one MinoFlux root action.
- Extended fusion output with `bestHoldUsed` and `bestCells` takes precedence when present.

---

### Task 1: Cross-engine state and action adapter

**Files:**
- Create: `src/minoflux_ai/fusion_oracle.py`
- Create: `tests/test_fusion_oracle.py`

**Interfaces:**
- Produces: `FusionOracleConfig`, `FusionOracleLabel`, `game_to_fusion_request(game, *, request_id, queue_length)`, `parse_fusion_label(value)`, `match_fusion_action(game, actions, label)`.
- Consumes: `minoflux_engine.Game`, `minoflux_ai.search.SearchAction`, `minoflux_engine.state.Placement`.

- [ ] **Step 1: Write failing board/chain conversion tests**

Add tests that construct a `Game`, replace its board with known occupied cells, set combo/B2B fields, and assert the exact request object:

```python
def test_game_to_fusion_request_converts_board_and_chain_state():
    game = Game(123)
    game.board = [[None] * 10 for _ in range(game.height)]
    game.board[23][0] = "G"
    game.board[22][9] = "T"
    game.current = "T"
    game.hold_piece = "I"
    game.queue = deque(["O", "S", "Z", "J", "L", "T", "I"] * 4)
    game.combo = 1
    game.back_to_back = True
    game.b2b_chain = 4
    game.lines = 12

    request = game_to_fusion_request(game, request_id="seed123:0", queue_length=18)

    assert request["player_board_rows"][:2] == [1, 1 << 9]
    assert request["current_piece"] == "t"
    assert request["hold_piece"] == "i"
    assert request["combo"] == 2
    assert request["b2b"] == 5
    assert request["lines"] == 12
    assert request["pending_garbage"] == 0
```

Also cover combo `-1/0/1`, inactive B2B, active x0/x1/x4, and insufficient queue.

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: collection/import failure because `minoflux_ai.fusion_oracle` does not exist.

- [ ] **Step 3: Implement request conversion minimally**

Create `fusion_oracle.py` with:

```python
@dataclass(frozen=True, slots=True)
class FusionOracleConfig:
    queue_length: int = 18
    max_candidates: int = 24
    workers: int = 1
    allow_180: bool = True
    reachability_node_limit: int = 8_000


def game_to_fusion_request(game: Game, *, request_id: str, queue_length: int) -> dict[str, object]:
    if len(game.queue) < queue_length:
        raise ValueError("insufficient-queue")
    rows = []
    for row in reversed(game.board):
        mask = 0
        for x, cell in enumerate(row):
            if cell is not None:
                mask |= 1 << x
        rows.append(mask)
    while rows and rows[-1] == 0:
        rows.pop()
    return {
        "schema_version": "phase1-v1",
        "replay_id": request_id,
        "round_id": 0,
        "player_id": 0,
        "frame_id": int(game.pieces_placed),
        "group_id": request_id,
        "player_board_rows": rows,
        "opponent_board_rows": [],
        "current_piece": game.current.lower(),
        "hold_piece": None if game.hold_piece is None else game.hold_piece.lower(),
        "queue": [piece.lower() for piece in list(game.queue)[:queue_length]],
        "combo": max(0, int(game.combo) + 1),
        "b2b": 0 if not game.back_to_back else int(game.b2b_chain) + 1,
        "lines": int(game.lines),
        "pending_garbage": 0,
        "bag_number": 0,
    }
```

- [ ] **Step 4: Run conversion tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: conversion tests pass; later action tests are not present yet.

- [ ] **Step 5: Write failing action decode/match tests**

Cover extended and legacy output. Legacy decoding follows fusion's `Move` bit layout:

```python
def _legacy_move(raw: int) -> tuple[int, int, int, int]:
    y = raw & 0x3F
    x = (raw >> 6) & 0xF
    piece_raw = (raw >> 10) & 0x7
    rotation = (raw >> 13) & 0x3
    return piece_raw, rotation, x, y
```

Tests must assert:

- extended `bestHoldUsed/bestCells` selects the exact Hold candidate;
- legacy `best_move_raw` selects a unique direct action when only one candidate has the decoded piece/cells;
- legacy output returns `oracle-action-ambiguous` when direct and Hold produce the same action identity;
- no candidate returns `oracle-action-unmatched`.

- [ ] **Step 6: Run action tests and verify RED**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: failures because label parsing/action matching functions are missing.

- [ ] **Step 7: Implement label parsing and exact action matching**

Add:

```python
@dataclass(frozen=True, slots=True)
class FusionOracleLabel:
    best_move_raw: int
    best_value: float
    best_hold_used: bool | None
    best_cells: frozenset[tuple[int, int]] | None
```

Normalize MinoFlux placement cells with:

```python
def _action_cells_bottom_up(game: Game, action: SearchAction) -> frozenset[tuple[int, int]]:
    return frozenset((x, (game.height - 1) - y) for x, y in action.placement.cells)
```

For legacy output, decode piece/rotation/x/y from `best_move_raw`, reconstruct the four fusion cells using the fusion piece geometry copied as a small immutable lookup table, and compare piece + occupied cells while leaving Hold unspecified. Accept only one matching action.

For extended output, compare `(use_hold, piece, cells)` exactly. Return a small result object or raise a stable `FusionOracleMatchError(reason)` with only `oracle-action-unmatched` or `oracle-action-ambiguous`.

- [ ] **Step 8: Run adapter tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 9: Commit Task 1**

```bash
git add src/minoflux_ai/fusion_oracle.py tests/test_fusion_oracle.py
git commit -m "feat: add fusion oracle action adapter"
```

---

### Task 2: Batch oracle execution and deterministic dataset generation

**Files:**
- Modify: `src/minoflux_ai/fusion_oracle.py`
- Modify: `tests/test_fusion_oracle.py`

**Interfaces:**
- Produces: `write_fusion_oracle_dataset(output_path, oracle_bin, config, *, games, max_pieces, seed_base, seed_step, trajectory)`.
- Uses Task 1 request and action matching functions.

- [ ] **Step 1: Write a fake oracle executable fixture and failing batch test**

Inside `tests/test_fusion_oracle.py`, create a temporary Python executable/script that reads every JSONL request and emits one label per input in the same order. Have it choose a deterministic legal identity supplied by the test via environment/file fixture. Assert the runner:

- batches multiple requests into one child invocation per shard;
- sets `RAYON_NUM_THREADS=1`;
- preserves source order after merging shards;
- does not pass `--time-budget-ms`;
- raises on child non-zero exit by default.

- [ ] **Step 2: Run batch tests and verify RED**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: failure because the batch runner does not exist.

- [ ] **Step 3: Implement shard execution**

Use `ProcessPoolExecutor` or `ThreadPoolExecutor` only to launch independent subprocesses. Each shard writes one input JSONL and receives one output JSONL. Child command shape:

```python
cmd = [str(oracle_bin), "--skip-failures", str(request_path), str(output_path)]
env = os.environ.copy()
env["RAYON_NUM_THREADS"] = "1"
```

Validate exact input/output line correspondence; a missing line is `oracle-output-missing`, malformed JSON is `oracle-output-invalid`, and non-zero exit is `oracle-process-failed`.

- [ ] **Step 4: Run batch tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: batch tests pass.

- [ ] **Step 5: Write failing dataset-generation test**

Generate a short seeded MinoFlux trajectory, enumerate legal direct + Hold actions with existing search/reachability logic, have the fake oracle select an exact candidate, and assert the output line:

```python
assert record["format"] == NEURAL_DATASET_FORMAT
assert record["teacher"] == "fusion-offline-oracle"
assert record["expertIndices"] == [record["expertIndex"]]
assert record["candidates"][record["expertIndex"]]["move"] == expected_move
```

Also assert skip counters for `insufficient-queue`, `oracle-action-unmatched`, and `teacher-not-encodable`.

- [ ] **Step 6: Run dataset test and verify RED**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: failure because `write_fusion_oracle_dataset` is missing.

- [ ] **Step 7: Implement dataset generation**

For each seeded game state:

1. clone/store the state before applying the trajectory action;
2. enumerate root actions with existing `rank_search_actions(..., limit=None)` so direct/Hold roots use the same exact reachability path as current neural data;
3. build a fusion request;
4. batch-label valid requests;
5. match the teacher action exactly;
6. encode candidate child states with the existing neural encoder;
7. keep the teacher candidate under any candidate cap;
8. write `NeuralRankingSample.to_dict()` plus:

```python
record["teacher"] = "fusion-offline-oracle"
record["fusionOracle"] = {
    "beamWidth": 2000,
    "depth": 18,
    "timeBudgetMs": None,
}
```

For the first implementation, support `trajectory="heuristic"` only and record it in metadata. Use the existing heuristic top-ranked root to advance each game; do not introduce a second learner dependency in this task.

- [ ] **Step 8: Run focused and neighboring tests**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py tests/test_neural_dataset.py tests/test_search.py -q
```

If either neighboring file name does not exist, use the repository's actual neural/search test modules discovered from `tests/` and keep the same coverage intent.

Expected: all selected tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add src/minoflux_ai/fusion_oracle.py tests/test_fusion_oracle.py
git commit -m "feat: generate fusion oracle ranking datasets"
```

---

### Task 3: CLI integration

**Files:**
- Modify: `src/minoflux/neural_cli.py`
- Modify: `tests/test_fusion_oracle.py` or the repository's neural CLI test module if one exists

**Interfaces:**
- Produces command: `uv run minoflux-neural fusion-dataset ...`.
- Consumes: `write_fusion_oracle_dataset` from Task 2.

- [ ] **Step 1: Write failing parser test**

Assert the parser accepts:

```text
fusion-dataset
--oracle-bin <path>
--output <path>
--games 2
--max-pieces 50
--seed-base 6000001
--seed-step 97
--workers 4
--max-candidates 24
--allow-180
```

and defaults `workers=1`, `trajectory="heuristic"`, and no time-budget option.

- [ ] **Step 2: Run parser test and verify RED**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: parser rejects unknown command `fusion-dataset`.

- [ ] **Step 3: Add CLI handler and parser**

Import:

```python
from minoflux_ai.fusion_oracle import FusionOracleConfig, write_fusion_oracle_dataset
```

Add `_fusion_dataset(args)` that calls the generator and prints JSON summary. Add parser arguments:

```python
fusion = subparsers.add_parser("fusion-dataset", help="Generate neural ranking data with the fusion offline oracle")
fusion.add_argument("--oracle-bin", required=True)
fusion.add_argument("--output", default="data/neural/fusion-oracle.jsonl")
fusion.add_argument("--games", type=int, default=40)
fusion.add_argument("--max-pieces", type=int, default=500)
fusion.add_argument("--seed-base", type=int, default=6_000_001)
fusion.add_argument("--seed-step", type=int, default=97)
fusion.add_argument("--workers", type=int, default=1)
fusion.add_argument("--max-candidates", type=int, default=24)
fusion.add_argument("--trajectory", choices=("heuristic",), default="heuristic")
fusion.add_argument("--allow-180", action="store_true")
fusion.add_argument("--reachability-nodes", type=int, default=8_000)
fusion.set_defaults(func=_fusion_dataset)
```

- [ ] **Step 4: Run CLI and adapter tests**

Run:

```bash
uv run pytest tests/test_fusion_oracle.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add src/minoflux/neural_cli.py tests/test_fusion_oracle.py
git commit -m "feat: expose fusion oracle dataset CLI"
```

---

### Task 4: Compatibility documentation and real-oracle smoke test

**Files:**
- Modify: `docs/neural-training.md`
- Modify: `src/minoflux_ai/fusion_oracle.py` only if the real binary reveals a contract issue
- Modify: `tests/test_fusion_oracle.py` first if a contract bug is found

**Interfaces:**
- Documents current legacy fallback and optional extended action fields.

- [ ] **Step 1: Document the command and current binary compatibility**

Add a section showing:

```bash
uv run minoflux-neural fusion-dataset \
  --oracle-bin /home/USER/dev/fusion/target/release/generate_policy_value_labels \
  --output data/neural/fusion-oracle.jsonl \
  --games 200 \
  --max-pieces 500 \
  --workers 12 \
  --allow-180
```

Document that current fusion binaries are accepted through unique `best_move_raw` matching, while extended `bestHoldUsed/bestCells` removes Hold ambiguity. State explicitly that ambiguous legacy actions are skipped.

- [ ] **Step 2: Run the full Python test suite**

Run:

```bash
uv run pytest -q
```

Expected: PASS.

- [ ] **Step 3: Run a real fusion smoke test locally**

From MinoFlux with the already-built fusion release binary:

```bash
uv run minoflux-neural fusion-dataset \
  --oracle-bin ~/dev/fusion/target/release/generate_policy_value_labels \
  --output data/neural/fusion-oracle-smoke.jsonl \
  --games 1 \
  --max-pieces 20 \
  --workers 4 \
  --allow-180
```

Verify summary reports non-zero samples, no process/contract failure, and any skips are explicit stable reasons.

- [ ] **Step 4: Validate output with the existing trainer loader**

Run one minimal training/load pass using the generated dataset, for example the repository's shortest supported training invocation with `--epochs 1`, and confirm no schema error occurs.

- [ ] **Step 5: Commit Task 4**

```bash
git add docs/neural-training.md
# Include code/tests only if the smoke test required a test-first compatibility fix.
git commit -m "docs: document fusion oracle teacher workflow"
```

- [ ] **Step 6: Final verification**

Run:

```bash
uv run pytest -q
uv run minoflux-neural fusion-dataset --help
```

Expected: test suite passes and help shows the new command without a time-budget argument.
