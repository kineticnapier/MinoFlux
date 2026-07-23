# MinoFlux

A pure-Python tetromino game and AI learning laboratory.

MinoFlux uses one shared rule engine: the Pygame client, headless simulations, search agents, replay viewer, and CEM trainer all call `minoflux_engine.Game` directly. The rules are Tetris-like rather than a strict Guideline clone.

## Included

- dependency-free 10×20 game engine with four hidden rows
- deterministic 7-bag, Hold, CW/CCW/180 rotation, lock delay, combo, B2B, attack, and approximate spins
- exact 90-degree SRS kick tables and project-specific 180-degree kicks
- configurable DAS, ARR, soft-drop speed, and key bindings
- board-only heuristic placement evaluation
- Hold-aware action generation
- configurable future-piece lookahead and beam search
- parallel fixed-seed benchmarks and deterministic JSON replays
- CEM weight training with worker processes and candidate screening
- Attack/Spin-oriented fitness and champion-versus-candidate model promotion
- Pygame game/replay clients and a Gradio experiment lab

## Windows

Install `uv`, clone the repository, then run:

```text
start-game.bat
start-lab.bat
```

The launchers use `uv sync` and `uv run`; they do not invoke `pip` directly.

Manual setup:

```powershell
uv sync --extra ui --extra dev
uv run --no-sync minoflux-lab
```

## Hold, lookahead, and beam search

`SearchConfig.lookahead_pieces` counts pieces after the current action:

```text
0 = current placement only
1 = current placement + next piece
2 = current placement + next two pieces
```

After each ply, only the strongest `beam_width` states remain. The search is deterministic for the same engine state, weights, and configuration.

```python
from minoflux_ai import SearchConfig, apply_search_action, choose_search_action
from minoflux_engine import Game

game = Game(seed=1234)
config = SearchConfig(
    allow_hold=True,
    lookahead_pieces=1,
    beam_width=4,
    discount=0.90,
)
choice = choose_search_action(game, config=config)
if choice is not None:
    apply_search_action(game, choice.action)
```

The benchmark defaults to Hold plus one future piece and beam width 4. Independent games run in separate worker processes.

```powershell
uv run minoflux benchmark `
  --games 20 `
  --max-pieces 1000 `
  --workers 0 `
  --fitness-profile attack_spin `
  --hold `
  --lookahead-pieces 1 `
  --beam-width 4 `
  --lookahead-discount 0.9 `
  --replay-out data/replays/best.json
```

The old direct-drop greedy behavior remains available:

```powershell
uv run minoflux benchmark --no-hold --lookahead-pieces 0 --beam-width 1
```

## Attack and Spin fitness

The default `attack_spin` profile directly rewards:

- Attack
- Spin count
- lines cleared by Spins
- perfect clears
- survival, with a smaller weight than the offensive terms

The older survival-oriented formula remains available as `balanced`.

Benchmark output now includes `spins`, `spinLines`, `perfectClears`, their per-game means, and the selected fitness score.

## Champion model protection

CEM no longer overwrites the main model unconditionally.

```text
training result
→ data/models/candidate-cem.json
→ promotion benchmark on separate unseen seeds
→ winner replaces data/models/champion-cem.json
→ rejected candidates remain in data/models/history/
```

The promotion check compares candidate and champion on identical seeds and rejects candidates that lose more completed games than the configured safety limit.

The stronger model recovered from the 2026-07-23 benchmark is stored at:

```text
presets/recovered-attack-20260723.json
```

On the first launch after upgrading, it bootstraps `champion-cem.json` when no champion exists. `data/models/latest-cem.json` is maintained as a compatibility alias.

## CEM training

Train a Hold-aware Attack/Spin candidate and challenge the champion:

```powershell
uv run minoflux train-cem `
  --generations 15 `
  --population 32 `
  --workers 0 `
  --screen-games 1 `
  --screen-max-pieces 60 `
  --screen-fraction 0.5 `
  --fitness-profile attack_spin `
  --hold `
  --lookahead-pieces 0 `
  --beam-width 4 `
  --promotion-games 10 `
  --promotion-max-pieces 1000
```

Training defaults to no future lookahead because evaluating every candidate with lookahead is expensive. A smaller follow-up run can refine a stable champion under lookahead:

```powershell
uv run minoflux train-cem `
  --generations 5 `
  --population 16 `
  --games 2 `
  --max-pieces 150 `
  --lookahead-pieces 1 `
  --beam-width 2
```

Use `--no-promote` for an experimental run that should only save a candidate.

## Replays

New replays use `minoflux_replay_v2` and record Hold, Spin, and perfect-clear metadata for every placement. The loader remains compatible with older v1 and v2 files.

```powershell
uv run minoflux replay data/replays/best.json
```

Controls:

```text
Space          pause / resume
Left / Right   previous / next placement
Up / Down      playback speed
Home / End     first / final state
Esc            quit
```

## Current search limits

Search candidates still come from `Game.legal_placements()`, so each selected piece is dropped vertically into its final position. The search does not yet enumerate placements that require a sequence of SRS moves or rotations to reach.

## Recommended next steps

1. Add SRS-reachable placement-path generation.
2. Add transposition caching for deeper beam search.
3. Train separate greedy, Hold, and lookahead champions.
4. Add garbage and versus-state evaluation.
5. Then add imitation learning or reinforcement learning.
