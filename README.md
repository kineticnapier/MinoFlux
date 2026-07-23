# MinoFlux

A pure-Python tetromino game and AI learning laboratory.

MinoFlux uses one shared rule engine: the Pygame client, headless simulations, search agents, and future reinforcement-learning environments all call `minoflux_engine.Game` directly. The rules are Tetris-like rather than a strict Guideline clone.

## Included

- dependency-free game engine
- 10×20 visible board with four hidden rows
- deterministic 7-bag randomizer
- movement, soft/hard drop, hold, CW/CCW/180 rotation
- configurable DAS, ARR, and soft-drop speed
- frame-rate-independent held-key repeat
- in-game key rebinding
- 500 ms lock delay with move/rotation reset and a 15-reset limit
- exact SRS 90-degree kick tables, separated for I and JLSTZ pieces
- project-specific 180-degree kicks because SRS does not define 180-degree rotation
- line clears, combo, B2B, attack, and approximate spin detection
- direct legal-placement actions for search and reinforcement learning
- Pygame game client
- Gradio experiment lab
- local JSON run folders

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
```

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

```python
from minoflux_engine import Game

game = Game(seed=1234, lock_delay_ms=500, lock_reset_limit=15)
game.gravity_step()
result = game.advance_time(16.67)
```

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

1. Add configurable gravity, 20G behavior, and level progression.
2. Add replay recording and deterministic playback.
3. Tighten T-Spin and All-Spin classification using kick metadata.
4. Add board features and a baseline placement heuristic.
5. Add Python CEM, then imitation learning and reinforcement learning.
