from __future__ import annotations

from dataclasses import replace
import json

import minoflux_ai.heuristic as h
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.versus_benchmark import run_versus_benchmark

BASE = h.DEFAULT_WEIGHTS
ORIGINAL_CONTEXT = h._context_score
STEP = 1e-6
CANDIDATES = (
    "i_arrival_tetris_conversion",
    "i_supply_well_match",
    "hold_i_well_balance",
    "tetris_ready_clear_restraint",
    "safe_zero_attack_clear_restraint",
    "attack_per_clear_efficiency",
    "danger_attack_priority",
    "danger_height_recovery_attack",
    "hole_recovery_attack",
    "tspin_tetris_choice_quality",
    "clean_difficult_clear_value",
    "t_supply_reserve_without_slot",
    "t_slot_surplus_cost",
    "attack_preserves_t_slots",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def piece_distances(game, piece: str):
    out = [0.0] if game.current == piece else []
    out.extend(i + 1.0 for i, p in enumerate(list(game.queue)[:6]) if p == piece)
    if game.hold_piece == piece:
        out.append(0.5)
    return out


def bonus(i, game, f):
    before = h.extract_board_features(game.board)
    after = f.board
    danger = max(0.0, (before.max_height - 9) / 6.0) + before.holes / 4.0
    safe = 1.0 / (1.0 + after.holes + after.max_height / 8.0)
    hole_drop = max(0, before.holes - after.holes)
    depth_drop = max(0, before.hole_depth - after.hole_depth)
    height_drop = max(0, before.max_height - after.max_height)
    i_dists = piece_distances(game, "I")
    t_dists = piece_distances(game, "T")
    nearest_i = min(i_dists) if i_dists else 8.0
    nearest_t = min(t_dists) if t_dists else 8.0
    well_ready = min(2.0, before.wells / 4.0)
    slots = after.t_spin_slots
    destroyed = max(0, -f.t_spin_slot_delta)

    if i == 0:
        if game.current != "I":
            return 0.0
        if f.lines == 4:
            return 0.75 + 0.12 * f.attack
        return -0.18 * well_ready if before.wells >= 4 and f.lines < 4 else 0.0
    if i == 1:
        supply = sum(max(0.0, (7.0 - d) / 7.0) for d in i_dists)
        return 0.24 * min(well_ready, supply) - 0.16 * max(0.0, well_ready - supply)
    if i == 2:
        if not game.hold_used:
            return 0.0
        held_i = 1.0 if game.hold_piece == "I" else 0.0
        return 0.20 * held_i * well_ready - 0.10 * (1.0 - held_i) * max(0.0, well_ready - 0.75)
    if i == 3:
        if before.holes or before.max_height > 11 or before.wells < 4 or nearest_i > 3:
            return 0.0
        return -0.22 * f.lines if 0 < f.lines < 4 and f.attack <= 1 else 0.0
    if i == 4:
        if before.holes or before.max_height > 9:
            return 0.0
        return -0.20 * f.lines if f.lines and f.attack == 0 else 0.0
    if i == 5:
        if not f.lines:
            return 0.0
        return 0.10 * f.attack / f.lines - 0.04 * max(0, f.lines - f.attack)
    if i == 6:
        return 0.16 * f.attack * min(2.0, danger)
    if i == 7:
        return 0.12 * f.attack * min(2.0, danger) + 0.08 * height_drop * min(2.0, danger)
    if i == 8:
        return 0.13 * f.attack * min(2, hole_drop) + 0.025 * depth_drop
    if i == 9:
        difficult = bool(f.spin_lines or f.lines == 4)
        return (0.16 * f.attack + 0.15 * f.spin_lines) * safe if difficult else 0.0
    if i == 10:
        difficult = bool(f.spin_lines or f.lines == 4)
        return 0.20 * f.attack * safe if difficult and after.holes == 0 else 0.0
    if i == 11:
        if game.current != "T" or slots > 0:
            return 0.0
        t_soon = nearest_t <= 3
        return -0.16 if not f.spin_lines and not t_soon and after.max_height < 12 else 0.0
    if i == 12:
        t_supply = len(t_dists)
        return -0.18 * max(0, slots - t_supply - 1) * (1.0 + after.holes / 4.0)
    if i == 13:
        return 0.13 * f.attack * (1.0 + min(2, slots)) - 0.20 * destroyed if f.attack else 0.0
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    i = candidate_index(weights)
    return base if i is None else base + bonus(i, game, features)


h._context_score = patched_context


def summary(r):
    return {"pieces": r.pieces, "attack": r.attack, "app": r.attack / max(1, r.pieces),
            "topouts": r.topouts, "completed": r.completed, "spins": r.spins,
            "spin_lines": r.spin_lines, "tsd": r.t_spin_doubles, "tst": r.t_spin_triples}


def key(row):
    s = row["summary"]
    return s["app"] - 0.05 * s["topouts"] + 0.002 * s["completed"] + 0.0015 * s["tsd"] + 0.0005 * s["spin_lines"]


def stage(games, pieces, seed_base, seed_step, indices):
    rows = [{"name": "baseline", "index": None, "summary": summary(run_heuristic_benchmark(
        games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step, weights=BASE, workers=1))}]
    for i in indices:
        r = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step,
                                    weights=candidate_weights(i), workers=1)
        rows.append({"name": CANDIDATES[i], "index": i, "summary": summary(r)})
    return rows


def versus_summary(r):
    rows = r.per_game
    n = len(rows)
    return {"candidate_wins": r.player_wins, "baseline_wins": r.ai_wins, "draws": r.draws,
            "candidate_attack": r.player_mean_attack, "baseline_attack": r.ai_mean_attack,
            "candidate_sent": r.player_mean_sent, "baseline_sent": r.ai_mean_sent,
            "candidate_cancel": sum(x.player_canceled for x in rows) / n,
            "baseline_cancel": sum(x.ai_canceled for x in rows) / n,
            "candidate_received": sum(x.player_received for x in rows) / n,
            "baseline_received": sum(x.ai_received for x in rows) / n,
            "candidate_b2b": sum(x.player_max_b2b for x in rows) / n,
            "baseline_b2b": sum(x.ai_max_b2b for x in rows) / n,
            "candidate_topouts": sum(x.winner == "ai" for x in rows),
            "baseline_topouts": sum(x.winner == "player" for x in rows)}


def main():
    short = stage(2, 100, 1401127, 157, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 200, 1513901, 211, survivors)
    base = fresh[0]["summary"]
    viable = [x for x in fresh[1:] if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(games=8, max_turns=90, seed_base=151807, seed_step=227,
                                 player_weights=candidate_weights(x["index"]), ai_weights=BASE)
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print("TOURNAMENT_RESULT=" + json.dumps({"candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [x["name"] for x in finalists], "versus": versus}, sort_keys=True))


if __name__ == "__main__":
    main()
