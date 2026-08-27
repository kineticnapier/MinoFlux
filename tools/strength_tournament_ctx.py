from __future__ import annotations

import argparse
import json
from dataclasses import replace

import minoflux_ai.heuristic as heuristic
import minoflux_ai.search as search
from minoflux_ai.benchmark import run_heuristic_benchmark


CANDIDATES = {
    "baseline": (0.0, "baseline"),
    "safe_attack": (0.55, "safe_attack"),
    "attack_no_holes": (0.65, "attack_no_holes"),
    "spin_safety": (0.75, "spin_safety"),
    "slot_bump_quality": (0.70, "slot_bump_quality"),
    "slot_depth_quality": (0.80, "slot_depth_quality"),
    "slot_supply_match": (0.55, "slot_supply_match"),
    "slot_timing": (0.75, "slot_timing"),
    "slot_single_focus": (0.55, "slot_single_focus"),
    "danger_attack_escape": (0.60, "danger_attack_escape"),
    "clean_spin_attack": (0.65, "clean_spin_attack"),
    "line_safety": (0.50, "line_safety"),
    "deep_hole_danger": (-0.050, "deep_hole_danger"),
    "nonspin_clear_tax": (-0.45, "nonspin_clear_tax"),
    "slot_clear_conflict": (-0.55, "slot_clear_conflict"),
}

_ORIGINAL_RANK = heuristic.rank_placements


def _next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def _bonus(name: str, weight: float, game, ev) -> float:
    f = ev.features
    b = f.board
    if name == "baseline":
        return 0.0
    if name == "safe_attack":
        return weight * f.attack / (1.0 + f.new_holes + b.max_height / 12.0)
    if name == "attack_no_holes":
        return weight * f.attack / (1.0 + 2.0 * f.new_holes)
    if name == "spin_safety":
        return weight * f.spin_lines / (1.0 + b.holes + b.max_height / 8.0)
    if name == "slot_bump_quality":
        return weight * b.t_spin_slots / (1.0 + b.bumpiness / 4.0 + b.max_height / 8.0)
    if name == "slot_depth_quality":
        return weight * b.t_spin_slots / (1.0 + b.holes + b.hole_depth / 6.0)
    if name == "slot_supply_match":
        d = _next_t_distance(game)
        availability = max(0.0, (6.0 - d) / 6.0)
        excess = max(0, b.t_spin_slots - 1)
        return weight * (availability * min(1, b.t_spin_slots) - 0.35 * excess * (1.0 - availability))
    if name == "slot_timing":
        d = _next_t_distance(game)
        timing = max(0.0, (5.0 - d) / 5.0)
        quality = b.t_spin_slots / (1.0 + b.holes + b.max_height / 6.0)
        return weight * timing * quality
    if name == "slot_single_focus":
        return weight * (1.0 if b.t_spin_slots == 1 else 0.0) / (1.0 + b.holes + b.max_height / 8.0)
    if name == "danger_attack_escape":
        danger = max(0.0, (b.max_height - 9.0) / 8.0)
        return weight * danger * f.attack
    if name == "clean_spin_attack":
        return weight * f.attack * (1.0 if f.spin_lines else 0.0) / (1.0 + b.holes)
    if name == "line_safety":
        return weight * f.lines / (1.0 + b.holes + b.max_height / 10.0)
    if name == "deep_hole_danger":
        danger = 1.0 + max(0.0, (b.max_height - 10.0) / 6.0)
        return weight * (b.hole_depth ** 2) * danger
    if name == "nonspin_clear_tax":
        return weight * f.lines if f.lines and not f.spin_lines and b.t_spin_slots > 0 else 0.0
    if name == "slot_clear_conflict":
        d = _next_t_distance(game)
        return weight * f.lines * b.t_spin_slots if f.lines and d <= 3 and not f.spin_lines else 0.0
    raise KeyError(name)


def install_candidate(candidate: str) -> None:
    weight, name = CANDIDATES[candidate]
    if candidate == "baseline":
        return

    def rank(game, weights=heuristic.DEFAULT_WEIGHTS, *, placements=None, limit=None):
        ranked = list(_ORIGINAL_RANK(game, weights, placements=placements, limit=None))
        adjusted = [replace(ev, score=ev.score + _bonus(name, weight, game, ev)) for ev in ranked]
        adjusted.sort(key=heuristic._placement_key, reverse=True)
        if limit is not None:
            adjusted = adjusted[: max(0, int(limit))]
        return tuple(adjusted)

    heuristic.rank_placements = rank
    search.rank_placements = rank


def summarize(result) -> dict[str, object]:
    pieces = result.pieces
    return {
        "attack": result.attack,
        "pieces": pieces,
        "attack_per_piece": result.attack / pieces if pieces else 0.0,
        "topouts": result.topouts,
        "completed": result.completed,
        "spins": result.spins,
        "spin_lines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    p.add_argument("--phase", choices=("screen", "fresh"), default="screen")
    args = p.parse_args()
    install_candidate(args.candidate)
    if args.phase == "screen":
        result = run_heuristic_benchmark(games=2, max_pieces=180, seed_base=41003, seed_step=97, workers=1)
    else:
        result = run_heuristic_benchmark(games=3, max_pieces=450, seed_base=880301, seed_step=131, workers=1)
    print("TOURNAMENT_RESULT=" + json.dumps({"candidate": args.candidate, "phase": args.phase, **summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
