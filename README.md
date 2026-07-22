# MinoFlux

A pure-Python tetromino game and AI learning laboratory.

MinoFlux starts with one deliberately shared rule engine: the Pygame client, headless simulations, search agents, and future reinforcement-learning environments all use `minoflux_engine.Game` directly. The rules are Tetris-like rather than a strict Guideline clone.

## Included

- dependency-free game engine
- 10×20 visible board with four hidden rows
- deterministic 7-bag randomizer
- movement, soft/hard drop, hold, CW/CCW/180 rotation
- simple wall and floor kicks
- line clears, combo, B2B, attack, and approximate spin detection
- direct legal-placement actions for search and reinforcement learning
- Pygame game client
- Gradio experiment lab
- local JSON run folders

## Windows

```text
start-game.bat
start-lab.bat
```

## Manual setup

```powershell
py -3 -m venv .venv
.venv\Scripts\python -m pip install -e ".[ui]"
.venv\Scripts\python -m minoflux.game
```

Headless commands:

```powershell
.venv\Scripts\python -m minoflux info
.venv\Scripts\python -m minoflux smoke --games 4 --max-pieces 200 --save
```

## Controls

```text
Left / Right     move
Down             soft drop
Z                rotate counter-clockwise
X / Up           rotate clockwise
A                rotate 180 degrees
C / Shift        hold
Space            hard drop
P                pause
R                restart
Esc              quit
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

## Planned learning layers

1. Board features and a hand-written placement heuristic.
2. CEM implemented entirely in Python.
3. Vectorized placement environments and action masks.
4. Imitation datasets from heuristic or search agents.
5. Neural value models and reinforcement learning.
