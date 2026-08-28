from __future__ import annotations

from dataclasses import replace
import json
import os

import minoflux_ai.heuristic as heuristic
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import extract_board_features

MARKER = 8.000001
TEST_WEIGHTS = replace(DEFAULT_WEIGHTS, perfect_clear=MARKER)
ORIGINAL_CONTEXT = heuristic._context_score
ORIGINAL_FEATURES = heuristic._placement_features_fast
EXTRA: dict[int, dict[str, float]] = {}
CURRENT = "baseline"


def _board_after(game, placement):
    board = [row.copy() for row in game.board]
    for x, y in placement.cells:
        if y >= 0:
            board[y][x] = placement.piece
    full = {i for i, row in enumerate(board) if all(cell is not None for cell in row)}
    if full:
        board = [[None] * game.width for _ in full] + [row for i, row in enumerate(board) if i not in full]
    return board


def _slot_count(board) -> float:
    if not board:
        return 0.0
    h, w = len(board), len(board[0])
    def occ(x, y):
        return x < 0 or x >= w or y < 0 or y >= h or board[y][x] is not None
    def empty(x, y):
        return 0 <= x < w and 0 <= y < h and board[y][x] is None
    shapes = (
        ((0,-1),(-1,0),(0,0),(1,0)), ((0,-1),(0,0),(1,0),(0,1)),
        ((-1,0),(0,0),(1,0),(0,1)), ((0,-1),(-1,0),(0,0),(0,1)),
    )
    count = 0.0
    for y in range(h):
        for x in range(w):
            if board[y][x] is not None:
                continue
            corners = (occ(x-1,y-1), occ(x+1,y-1), occ(x-1,y+1), occ(x+1,y+1))
            if sum(corners) >= 3 and any(all(empty(x+dx,y+dy) for dx,dy in shape) for shape in shapes):
                count += 1.0
    return count


def patched_features(game, placement, before):
    result = ORIGINAL_FEATURES(game, placement, before)
    after = _board_after(game, placement)
    af = extract_board_features(after)
    bf = extract_board_features(game.board)
    EXTRA[id(result)] = {
        "before_holes": float(bf.holes), "after_holes": float(af.holes),
        "before_depth": float(bf.hole_depth), "after_depth": float(af.hole_depth),
        "before_height": float(bf.max_height), "after_height": float(af.max_height),
        "before_bump": float(bf.bumpiness), "after_bump": float(af.bumpiness),
        "before_slots": _slot_count(game.board), "after_slots": _slot_count(after),
    }
    return result


def _modifier(game, f):
    e = EXTRA.get(id(f), {})
    bh, ah = e.get("before_holes",0), e.get("after_holes",0)
    bd, ad = e.get("before_depth",0), e.get("after_depth",0)
    bm, am = e.get("before_height",0), e.get("after_height",0)
    bb, ab = e.get("before_bump",0), e.get("after_bump",0)
    bs, ass = e.get("before_slots",0), e.get("after_slots",0)
    hole_relief = max(0.0, bh-ah)
    depth_relief = max(0.0, bd-ad)
    height_relief = max(0.0, bm-am)
    bump_relief = max(0.0, bb-ab)
    slot_keep = min(bs, ass)
    slot_gain = max(0.0, ass-bs)
    damage = max(0.0, ah-bh) + 0.15*max(0.0, ad-bd) + 0.25*max(0.0, am-bm)
    attack = float(f.attack)
    difficult = float(f.spin_lines > 0 or f.lines == 4)
    danger = max(0.0, (bm-9.0)/7.0)
    clean = 1.0/(1.0 + ah + 0.12*ad + 0.08*ab)

    if CURRENT == "pressure_clean_exit": return 0.38*attack*clean
    if CURRENT == "pressure_hole_relief": return 0.18*attack*hole_relief
    if CURRENT == "pressure_depth_relief": return 0.025*attack*min(20.0, depth_relief)
    if CURRENT == "pressure_height_relief": return 0.16*attack*height_relief
    if CURRENT == "pressure_surface_relief": return 0.045*attack*min(12.0, bump_relief)
    if CURRENT == "pressure_damage_cost": return -0.16*attack*damage
    if CURRENT == "spin_clean_exit": return 0.28*f.spin_lines*clean + 0.08*attack*clean
    if CURRENT == "b2b_clean_pressure": return 0.24*difficult*attack*clean if game.back_to_back else 0.0
    if CURRENT == "danger_pressure_recovery": return danger*(0.16*attack*height_relief + 0.08*attack*hole_relief)
    if CURRENT == "pressure_slot_preserve": return 0.12*attack*slot_keep
    if CURRENT == "pressure_slot_create": return 0.10*attack*slot_gain/(1.0+ah)
    if CURRENT == "tspin_slot_exit": return (0.18*f.spin_lines*slot_keep + 0.06*attack*slot_keep) if f.spin_lines else 0.0
    if CURRENT == "downstack_pressure_exit": return 0.08*f.lines*(hole_relief + 0.06*depth_relief) + 0.05*attack*hole_relief
    if CURRENT == "pressure_consistency": return 0.12*attack*(hole_relief + 0.35*height_relief) - 0.10*attack*damage
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    if weights.perfect_clear > 8.0000005:
        return base + _modifier(game, features)
    return base


heuristic._placement_features_fast = patched_features
heuristic._context_score = patched_context
cfg = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.9)
stage = os.environ.get("STAGE", "short")
candidates = [x for x in os.environ.get("CANDIDATES", "baseline").split(",") if x]
if stage == "short": games, pieces, seed = 2, 120, 113_771
elif stage == "fresh": games, pieces, seed = 3, 280, 418_903
else: raise SystemExit(stage)

results = []
for candidate in candidates:
    CURRENT = candidate
    EXTRA.clear()
    weights = DEFAULT_WEIGHTS if candidate == "baseline" else TEST_WEIGHTS
    r = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed, seed_step=97, weights=weights, search_config=cfg, workers=1)
    results.append({
        "candidate": candidate, "attack": r.attack, "pieces": r.pieces,
        "attackPerPiece": r.attack/max(1,r.pieces), "topouts": r.topouts,
        "completed": r.completed, "spins": r.spins, "spinLines": r.spin_lines,
        "tSpinDoubles": r.t_spin_doubles, "tSpinTriples": r.t_spin_triples,
        "meanHoles": r.mean_holes, "meanHoleDepth": r.mean_hole_depth,
        "meanMaxHeight": r.mean_max_height, "highStackFraction": r.high_stack_fraction,
    })

with open("self-improve-results.json", "w", encoding="utf-8") as f:
    json.dump({"stage":stage,"results":results}, f, indent=2)
print(json.dumps(results, separators=(",",":")))
