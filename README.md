# MinoFlux

A pure-Python tetromino game and AI learning laboratory.

MinoFlux uses one shared rule engine: the Pygame client, headless simulations, search agents, and future reinforcement-learning environments all call `minoflux_engine.Game` directly. The rules are Tetris-like rather than a strict Guideline clone.

## Included

- dependency-free game engine
- 10×20 visible board with four hidden rows
- deterministic 7-bag randomizer
- movement, soft/hard drop, hold, CW/CCW/180 rotation
- configurable DAS, ARR, and soft-drop speed
- frame-rate-independent held-key repeat and in-game key rebinding
- 500 ms lock delay with a 15-reset move/rotation limit
- JLSTZ and I-piece SRS kick tables
- line clears, combo, B2B, attack, and approximate spin detection
- direct legal-placement actions for search and reinforcement learning
- board feature extraction and a deterministic heuristic placement bot
- fixed-seed heuristic benchmark in CLI and Gradio
- CEM heuristic-weight training with separate validation seeds
- best-game replay recording and a Pygame replay viewer
- Pygame game client, Gradio experiment lab, and local JSON run folders

## Windows

Install `uv`, clone the repository, then run:

```text
start-game.bat
start-lab.bat
```

The launchers use `uv sync` and `uv run`; they do not invoke `pip` directly.

## Manual setup

```powershell
uv sync --extra game
uv run --no-sync minoflux-game
```

Lab:

```powershell
uv sync --extra ui
uv run --no-sync minoflux-lab
```

Headless commands:

```powershell
uv run minoflux info
uv run minoflux smoke --games 4 --max-pieces 200 --save
uv run minoflux benchmark --games 8 --max-pieces 500 --save
```

A saved benchmark contains its exact seeds, built-in weights, aggregate result, per-game result, and best replay under `data/runs/`.

## Input settings

Press `F1` while playing.

- Up / Down: select a setting
- Left / Right: adjust DAS, ARR, or SDS
- Enter: rebind the selected action, then press the new key
- Backspace: restore all defaults
- F1 / Esc: save and close

Handling values:

- **DAS**: delay before horizontal auto-repeat starts
- **ARR**: delay between repeated horizontal moves; `0` moves instantly to the wall after DAS
- **SDS**: delay per soft-drop cell; `0` moves instantly to the floor without locking

Settings are stored at `~/.minoflux/settings.json`. Set `MINOFLUX_SETTINGS` to use another path.

## Baseline AI

The first learning baseline considers the direct-drop placements returned by `Game.legal_placements()`. It does not use Hold or search SRS movement routes yet.

Each candidate is simulated on an independent copy of the game and scored from:

- aggregate and maximum stack height
- holes, hole depth, and newly created holes
- surface bumpiness and well depth
- cleared lines, attack, spin lines, and perfect clears
- a large game-over penalty

```python
from minoflux_ai import choose_placement
from minoflux_engine import Game

game = Game(seed=1234)
while not game.game_over:
    choice = choose_placement(game)
    if choice is None:
        break
    game.place(choice.placement)
```

Weights use the JSON format `minoflux_heuristic_v1`; `save_weights()` and `load_weights()` are used by the CEM trainer.

## Heuristic benchmark replay

Benchmarks can record the strongest game in `minoflux_replay_v1` format.

```powershell
uv run minoflux benchmark --games 8 --max-pieces 500 --replay-out data/replays/best.json
uv run minoflux replay data/replays/best.json
```

The Pygame replay viewer supports pause, single-step forward/backward, speed changes, and jump-to-start/end. The Gradio benchmark tab saves the best replay automatically and provides a **Replay best game** button.

## CEM weight training

The Cross-Entropy Method samples heuristic weight vectors, evaluates every candidate on the same fixed training seeds, keeps the elite fraction, and updates the sampling mean and standard deviation. A separate validation seed set is used after training.

```powershell
uv run minoflux train-cem --generations 8 --population 16 --games 3 --max-pieces 200 --model-out data/models/cem.json --replay-out data/replays/cem-best.json
```

Small settings are useful for checking the pipeline. Larger populations, more games per candidate, and longer piece limits produce steadier weights but take substantially longer. The Gradio **CEM weight training** tab saves a model and the best validation replay.

## Python API

```python
from minoflux_engine import Game

game = Game(seed=1234)
game.move_left()
game.rotate_cw()
result = game.hard_drop()

placements = game.legal_placements()
result = game.place(placements[0])
```

## Recommended next steps

1. Add Hold candidates.
2. Add SRS-reachable placement generation rather than direct drops only.
3. Parallelize CEM candidate evaluation and rotate training seed sets.
4. Add replay timelines and placement-score inspection.
5. Add imitation learning and reinforcement-learning environments.
