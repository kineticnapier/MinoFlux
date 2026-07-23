# MinoFlux

A pure-Python tetromino game and AI learning laboratory.

MinoFlux uses one shared rule engine: the Pygame client, headless simulations, search agents, and future reinforcement-learning environments all call `minoflux_engine.Game` directly. The rules are Tetris-like rather than a strict Guideline clone.

## Included

- dependency-free game engine
- 10×20 visible board with four hidden rows
- deterministic 7-bag randomizer
- movement, soft/hard drop, hold, CW/CCW/180 rotation
- configurable DAS, ARR, soft-drop speed, and in-game key rebinding
- 500 ms lock delay with movement/rotation reset and a 15-reset limit
- exact SRS 90-degree kick tables, separated for I and JLSTZ pieces
- project-specific 180-degree kicks because SRS does not define 180-degree rotation
- line clears, combo, B2B, attack, and approximate spin detection
- board feature extraction and a deterministic heuristic placement bot
- board-only candidate simulation without copying the full `Game`
- fixed-seed benchmarks and deterministic JSON replays
- CEM weight training with worker processes and optional candidate screening
- Pygame game/replay clients, Gradio experiment lab, and local JSON run folders

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
uv run minoflux train-cem --generations 8 --population 16 --save
```

## Input settings

Press `F1` while playing.

- Up / Down: select a setting
- Left / Right: adjust DAS, ARR, or SDS
- Enter: rebind the selected action, then press the new key
- Backspace: restore all defaults
- F1 / Esc: save and close

Settings are stored at `~/.minoflux/settings.json`. Set `MINOFLUX_SETTINGS` to use another path.

Default bindings:

```text
Left / Right     move
Down             soft drop
Z                rotate counter-clockwise
X                rotate clockwise
A                rotate 180 degrees
C                hold
Space            hard drop
P                pause
R                restart
F1               settings
Esc              quit
```

## Lock behavior

`Game` defaults to a 500 ms lock delay and permits 15 successful grounded movement/rotation resets per piece. Hard drop and direct placement actions still lock immediately.

The engine owns the timer. Real-time clients call `game.advance_time(delta_ms)` every frame, while deterministic tests or environments can advance it with controlled values.

## Baseline AI

The baseline considers the direct-drop placements returned by `Game.legal_placements()`. It does not use Hold or search SRS movement routes yet.

Each candidate is scored from aggregate/maximum height, holes, hole depth, new holes, bumpiness, wells, lines, attack, spin lines, perfect clears, and game over.

Candidate evaluation copies only the 24×10 board and reproduces the immediate line-clear, combo, B2B, attack, spin, perfect-clear, and top-out result. It no longer calls `deepcopy(Game)` for every possible placement.

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

## Accelerated CEM

CEM candidate evaluations are independent, so MinoFlux can distribute them across worker processes. `--workers 0` automatically uses the available logical CPUs minus one; use `--workers 1` for serial execution.

An optional screening round first evaluates every candidate on shorter games. Only the strongest fraction receives the full expensive evaluation.

```powershell
uv run minoflux train-cem `
  --generations 20 `
  --population 64 `
  --workers 0 `
  --screen-games 1 `
  --screen-max-pieces 60 `
  --screen-fraction 0.5 `
  --games 4 `
  --max-pieces 500 `
  --model-out data/models/cem.json `
  --replay-out data/replays/cem-best.json
```

Screening can miss a candidate that starts poorly but performs well over long games. Disable it for maximum evaluation accuracy:

```powershell
uv run minoflux train-cem --screen-games 0
```

Training results include the resolved worker count, total elapsed time, each generation's elapsed time, the number of fully evaluated candidates, and the number rejected by screening.

## Replays

A benchmark or CEM validation run can save its best game:

```powershell
uv run minoflux benchmark --replay-out data/replays/best.json
uv run minoflux replay data/replays/best.json
```

Replay controls:

```text
Space          pause / resume
Left / Right   previous / next placement
Up / Down      playback speed
Home / End     first / final state
Esc            quit
```

## Recommended next steps

1. Add Hold candidates to the heuristic search.
2. Add SRS-reachable placement generation rather than direct drops only.
3. Benchmark process count and screening settings on representative hardware.
4. Add lookahead and beam search.
5. Then add imitation learning or reinforcement learning.
