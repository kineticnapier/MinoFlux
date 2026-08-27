from __future__ import annotations

import argparse
import json
from dataclasses import replace

import minoflux_ai.heuristic as heuristic
from minoflux_ai.benchmark import run_heuristic_benchmark

BASE_SCORE = heuristic.score_features


def bonus(name, f):
    b = f.board
    if name == "baseline": return 0.0
    if name == "slot_low_clean": return 0.90 * b.t_spin_slots / (1.0 + b.holes + b.max_height / 6.0)
    if name == "slot_stability": return 0.65 * b.t_spin_slots / (1.0 + b.bumpiness / 8.0)
    if name == "slot_shallow_holes": return 0.55 * b.t_spin_slots / (1.0 + b.hole_depth / 4.0)
    if name == "slot_delta_quality": return 0.85 * max(0, f.t_spin_slot_delta) / (1.0 + b.holes + b.max_height / 8.0)
    if name == "preserve_unspent_slot": return -0.80 * max(0, -f.t_spin_slot_delta) * (1 if f.spin_lines == 0 else 0)
    if name == "cashout_slot": return 0.55 * f.spin_lines * max(0, -f.t_spin_slot_delta)
    if name == "attack_clean": return 0.45 * f.attack / (1.0 + f.new_holes)
    if name == "attack_low": return 0.40 * f.attack / (1.0 + b.max_height / 10.0)
    if name == "attack_slot": return 0.22 * f.attack * (1.0 + min(2, b.t_spin_slots))
    if name == "deep_hole_mean": return -0.38 * b.hole_depth / (1.0 + b.holes)
    if name == "danger_newhole": return -0.12 * f.new_holes * max(0, b.max_height - 8) ** 2
    if name == "rough_high": return -0.025 * b.bumpiness * max(0, b.max_height - 7)
    if name == "safe_clear": return 0.30 * f.lines / (1.0 + b.max_height / 8.0 + b.holes)
    if name == "spin_efficiency": return 0.38 * f.attack * f.spin_lines / (1.0 + f.lines)
    raise ValueError(name)


def patched_score(features, weights=heuristic.DEFAULT_WEIGHTS):
    return BASE_SCORE(features, weights) + bonus(ARGS.candidate, features)


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("--games", type=int, default=2)
    p.add_argument("--pieces", type=int, default=90)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--step", type=int, default=7919)
    ARGS = p.parse_args()
    heuristic.score_features = patched_score
    r = run_heuristic_benchmark(games=ARGS.games, max_pieces=ARGS.pieces, seed_base=ARGS.seed, seed_step=ARGS.step, workers=1)
    payload = {
        "candidate": ARGS.candidate,
        "games": r.games,
        "pieces": r.pieces,
        "attack": r.attack,
        "attack_per_piece": r.attack / max(1, r.pieces),
        "topouts": r.topouts,
        "completed": r.completed,
        "spins": r.spins,
        "spin_lines": r.spin_lines,
        "tsd": r.t_spin_doubles,
        "tst": r.t_spin_triples,
    }
    print("RESULT=" + json.dumps(payload, sort_keys=True))

if __name__ == "__main__":
    main()
