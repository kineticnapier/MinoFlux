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
- fixed-seed benchmarks and deterministic JSON replays
- CEM weight training with worker processes and candidate screening
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

The benchmark defaults to Hold plus one future piece and beam width 4:

```powershell
uv run minoflux benchmark `
  --games 20 `
  --max-pieces 1000 `
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

Lookahead is substantially more expensive than greedy evaluation. Increase `beam-width` only after measuring runtime on representative games.

## CEM training

CEM can train weights under the same Hold/search policy. Training defaults to Hold enabled but no future lookahead, because evaluating every candidate with lookahead is expensive.

```powershell
uv run minoflux train-cem `
  --initial-model data/models/latest-cem.json `
  --generations 20 `
  --population 64 `
  --workers 0 `
  --screen-games 1 `
  --screen-max-pieces 60 `
  --screen-fraction 0.5 `
  --hold `
  --lookahead-pieces 0 `
  --beam-width 4 `
  --model-out data/models/latest-cem.json `
  --replay-out data/replays/cem-best.json
```

After a stable Hold-aware model is trained, a smaller lookahead run can refine it:

```powershell
uv run minoflux train-cem `
  --initial-model data/models/latest-cem.json `
  --generations 5 `
  --population 16 `
  --games 2 `
  --max-pieces 150 `
  --lookahead-pieces 1 `
  --beam-width 2
```

## Replays

New replays use `minoflux_replay_v2` and record whether Hold was used before every placement. The loader remains compatible with `minoflux_replay_v1` files.

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
3. Train and compare separate greedy, Hold, and lookahead models.
4. Add garbage and versus-state evaluation.
5. Then add imitation learning or reinforcement learning.
