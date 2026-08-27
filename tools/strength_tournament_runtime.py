from __future__ import annotations

import json
import os

from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE
import minoflux_ai.heuristic as heuristic
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action, choose_search_action

CANDIDATE = os.environ.get("CANDIDATE", "baseline")
PHASE = os.environ.get("PHASE", "short")


def extra_score(features):
    b = features.board
    positive_delta = max(0, features.t_spin_slot_delta)
    extras = {
        "baseline": 0.0,
        "tsd_ready": 0.50 if features.spin == T_SPIN_DOUBLE else 0.0,
        "tst_ready": 0.75 if features.spin == T_SPIN_TRIPLE else 0.0,
        "spin_efficiency": 0.55 * features.spin_lines / (1 + b.holes),
        "slot_height_quality": 0.70 * b.t_spin_slots / (1.0 + b.max_height / 6.0),
        "slot_depth_quality": 0.70 * b.t_spin_slots / (1.0 + b.hole_depth / 4.0),
        "slot_delta_clean": 0.60 * positive_delta / (1 + features.new_holes),
        "slot_delta_holefree": 0.50 * positive_delta if b.holes <= 1 else 0.0,
        "clean_attack": 0.22 * features.attack / (1 + features.new_holes),
        "danger_attack": 0.18 * features.attack * max(0, b.max_height - 10) / 10.0,
        "holes_height": -0.045 * b.holes * b.max_height,
        "depth_height": -0.012 * b.hole_depth * b.max_height,
        "spin_clean": 0.42 * features.spin_lines / (1 + features.new_holes),
        "efficient_clear": 0.24 * max(0, features.attack - features.lines),
        "slot_attack_combo": 0.11 * features.attack * b.t_spin_slots / (1 + b.holes),
        "safe_slot": 0.45 * b.t_spin_slots / (1 + max(0, b.max_height - 8)),
    }
    return extras[CANDIDATE]


_original_score = heuristic.score_features
heuristic.score_features = lambda features, weights=DEFAULT_WEIGHTS: _original_score(features, weights) + extra_score(features)


def run_game(seed: int, pieces: int):
    game = Game(seed)
    spins = tsd = tst = spin_lines = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < pieces:
        choice = choose_search_action(game, DEFAULT_WEIGHTS, DEFAULT_SEARCH_CONFIG)
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
        tsd += int(result.spin == T_SPIN_DOUBLE)
        tst += int(result.spin == T_SPIN_TRIPLE)
        max_b2b = max(max_b2b, game.b2b_chain)
    return {
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "topout": int(game.game_over),
        "completed": int(not game.game_over and game.pieces_placed >= pieces),
        "spins": spins,
        "spin_lines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "max_b2b": max_b2b,
    }


if PHASE == "short":
    seeds, pieces = (71003, 71034), 100
else:
    seeds, pieces = (91009, 91046, 91083), 150

rows = [run_game(seed, pieces) for seed in seeds]
total_pieces = sum(r["pieces"] for r in rows)
summary = {
    "candidate": CANDIDATE,
    "phase": PHASE,
    "games": len(rows),
    "pieces": total_pieces,
    "attack": sum(r["attack"] for r in rows),
    "attack_per_piece": (sum(r["attack"] for r in rows) / total_pieces) if total_pieces else 0.0,
    "topouts": sum(r["topout"] for r in rows),
    "completed": sum(r["completed"] for r in rows),
    "spins": sum(r["spins"] for r in rows),
    "spin_lines": sum(r["spin_lines"] for r in rows),
    "tsd": sum(r["tsd"] for r in rows),
    "tst": sum(r["tst"] for r in rows),
    "max_b2b": max(r["max_b2b"] for r in rows),
    "rows": rows,
}
print("TOURNAMENT_RESULT=" + json.dumps(summary, sort_keys=True))
