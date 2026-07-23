# MinoFlux

A pure-Python tetromino game and AI learning laboratory.

MinoFlux uses one shared rule engine: the Pygame clients, headless simulations, search agents, replay viewer, versus mode, and CEM trainer all call `minoflux_engine.Game` directly. The rules are Tetris-like rather than a strict Guideline clone.

## Included

- dependency-free 10×20 game engine with four hidden rows
- deterministic 7-bag, Hold, CW/CCW/180 rotation, lock delay, combo, attack, and TETR.IO-style B2B Charging
- exact 90-degree SRS kick tables and project-specific 180-degree kicks
- configurable DAS, ARR, soft-drop speed, and key bindings
- SRS-reachable action generation with recorded movement and rotation paths
- exact T-spin three-corner, front-corner, Mini, and fifth-kick classification
- configurable future-piece lookahead and beam search
- parallel fixed-seed benchmarks and deterministic JSON replays
- CEM weight training with worker processes and candidate screening
- Attack/T-spin-oriented fitness and champion-versus-candidate model promotion
- local player-versus-AI mode with garbage cancellation and asynchronous AI search
- Pygame game/replay/versus clients and a Gradio experiment lab

## Windows

Install `uv`, clone the repository, then run:

```text
start-game.bat
start-versus.bat
start-lab.bat
```

The launchers use `uv sync` and `uv run`; they do not invoke `pip` directly.

Manual setup:

```powershell
uv sync --extra ui --extra dev
uv run --no-sync minoflux-lab
uv run --no-sync minoflux-versus --ai-pps 1.0
```

## B2B Charging

The multiplayer attack engine follows the standard TETR.IO B2B Charging shape:

- the first difficult clear activates B2B at displayed `x0`
- each later difficult clear gives the attack `+1` and increments the displayed chain
- at `B2B x4`, Surge starts at 4 lines
- each further B2B level increases Surge by one
- a no-clear placement preserves B2B and the stored Surge
- a non-difficult line clear breaks B2B and releases Surge
- Surge is split into three ordered packets; early packets receive any remainder
- an All Clear preserves/starts B2B and adds exactly two displayed levels

Examples:

```text
Surge 4 -> 2, 1, 1
Surge 5 -> 2, 2, 1
Surge 8 -> 3, 3, 2
```

`LockResult.attack_packets` keeps the normal attack and the three Surge segments separate. Versus cancellation and garbage-hole selection therefore happen per attack packet rather than after combining everything into one number.

## Versus AI

Launch with:

```text
start-versus.bat
```

or:

```powershell
uv run minoflux-versus `
  --ai-pps 1.0 `
  --ai-lookahead 0 `
  --ai-beam 2 `
  --garbage-cap 8
```

Both player panels use the same layout:

```text
[HOLD] [BOARD] [NEXT]
```

The AI uses the champion model when available, then the recovered attack model, then built-in weights. Its search runs on a worker thread so player input and rendering continue while SRS paths are evaluated.

Controls:

```text
Your configured game bindings   move / rotate / Hold / hard drop
[ and ]                         lower / raise AI PPS by 0.1
Pause binding                   pause
Restart binding                 new round
Esc                             quit
```

Garbage behavior:

- outgoing attack cancels the sender's queued garbage first
- uncanceled packet remnants are queued on the opponent
- separate packets use separate garbage holes
- a line clear blocks queued garbage for that placement
- a no-clear placement accepts up to the configured garbage cap

## SRS reachability

Search starts from the real spawn position and explores:

```text
left / right
soft drop
CW / CCW rotation with the exact SRS kick order
optional project-specific 180 rotation
hard drop to the final resting position
```

Every candidate stores its input path, whether the final manipulation was a rotation, the SRS kick index, and the rotation transition. Placements that cannot be reached through this state graph are not offered to the AI.

```python
from minoflux_ai import SearchConfig, apply_search_action, choose_search_action
from minoflux_engine import Game

game = Game(seed=1234)
config = SearchConfig(
    allow_hold=True,
    lookahead_pieces=1,
    beam_width=4,
    discount=0.90,
    srs_reachable=True,
    allow_180=False,
)
choice = choose_search_action(game, config=config)
if choice is not None:
    print(choice.action.placement.path)
    apply_search_action(game, choice.action)
```

The reachability search is geometrically exact for movement and kicks. It does not simulate elapsed lock-delay time or the 15-reset timing budget while finding a route.

Set `srs_reachable=False` to use the old vertical-drop candidate generator. The old direct greedy behavior is represented by `DIRECT_SEARCH_CONFIG`.

## Hold, lookahead, and beam search

`SearchConfig.lookahead_pieces` counts pieces after the current action:

```text
0 = current placement only
1 = current placement + next piece
2 = current placement + next two pieces
```

After each ply, only the strongest `beam_width` states remain. The search is deterministic for the same engine state, weights, and configuration.

The benchmark defaults to Hold, SRS reachability, one future piece, and beam width 4. Independent games run in separate worker processes.

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

SRS route enumeration is substantially more expensive than the old direct-drop generator. For quick experiments, use fewer games or no future lookahead.

## Exact T-spin classification

A T-spin requires:

- the final successful manipulation to be a rotation
- at least three occupied pivot corners
- front-corner classification for Full versus Mini
- fifth SRS kick upgrade from Mini to Full

Benchmark output separates:

```text
tSpinMinis
tSpinMiniSingles
tSpinSingles
tSpinDoubles
tSpinTriples
```

The previous broad All-spin approximation was removed from AI statistics. Results produced before version 0.8.0 are therefore not directly comparable with the new T-spin counts.

## Attack and T-spin fitness

The default `attack_spin` profile rewards:

- Attack, including released Surge
- exact T-spin Singles
- exact T-spin Doubles
- exact T-spin Triples
- Mini events at a much smaller value
- perfect clears
- survival, with a smaller weight than the offensive terms

T-spin Double and Triple receive larger explicit bonuses than raw spin count. The older survival-oriented formula remains available as `balanced`.

## Champion model protection

CEM does not overwrite the main model unconditionally.

```text
training result
-> data/models/candidate-cem.json
-> promotion benchmark on separate unseen seeds
-> winner replaces data/models/champion-cem.json
-> rejected candidates remain in data/models/history/
```

The promotion check compares candidate and champion on identical seeds and rejects candidates that lose more completed games than the configured safety limit.

The stronger model recovered from the 2026-07-23 benchmark is stored at:

```text
presets/recovered-attack-20260723.json
```

On the first launch after upgrading, it bootstraps `champion-cem.json` when no champion exists. `data/models/latest-cem.json` is maintained as a compatibility alias.

## CEM training

Train an SRS-aware Attack/T-spin candidate and challenge the champion:

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

Training defaults to no future lookahead because SRS enumeration is already more expensive than direct-drop evaluation. Models from before version 0.9.0 were trained without Surge valuation, so a new candidate should be trained before judging B2B release strategy.

Use `--no-promote` for an experimental run that should only save a candidate.

## Replays

New replays use `minoflux_replay_v3`. Each placement records:

- Hold use
- final position
- exact input path
- final-rotation flag
- SRS kick index and rotation transition
- exact T-spin event
- line clear, Attack, score, and perfect clear

The loader remains compatible with v1 and v2 files, but exact strict validation is used only for v3. Old replays may produce different Attack totals after B2B Charging was added.

```powershell
uv run minoflux replay data/replays/best.json
```

## Recommended next steps

1. Add versus-aware AI features for pending garbage, cancellation, and opponent pressure.
2. Add transposition caching for deeper SRS beam search.
3. Add lock-delay-aware route feasibility for timing-sensitive placements.
4. Train separate greedy, Hold, lookahead, and versus champions.
5. Then add imitation learning or reinforcement learning.
