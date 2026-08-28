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
    "t_timing_weighted_supply", "slot_timing_capacity", "current_t_clean_conversion",
    "tspin_rebuild_after_clear", "near_t_preserve_low_clean", "tspin_attack_rebuild",
    "high_stack_tspin_escape", "safe_attack_efficiency", "hole_depth_recovery",
    "danger_hole_depth_recovery", "downstack_attack_synergy", "height_drop_under_danger",
    "clean_line_escape", "slot_supply_hole_recovery",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def t_distances(game):
    out = [0.0] if game.current == "T" else []
    out.extend(i + 1.0 for i, p in enumerate(list(game.queue)[:6]) if p == "T")
    if game.hold_piece == "T":
        out.append(0.5)
    return out


def bonus(i, game, f):
    a = f.board
    slots = a.t_spin_slots
    dists = t_distances(game)
    supply = sum(max(0.0, (7.0 - d) / 7.0) for d in dists)
    safe = 1.0 / (1.0 + a.holes + a.max_height / 8.0)
    low_clean = slots / (1.0 + a.holes + a.max_height / 6.0)
    destroyed = max(0, -f.t_spin_slot_delta)
    created = max(0, f.t_spin_slot_delta)
    nearest = min(dists) if dists else 8.0

    if i == 0:
        return 0.34 * min(slots, supply) - 0.22 * max(0.0, slots - supply)
    if i == 1:
        capacity = sum(1.0 if d <= 3 else 0.45 for d in dists)
        return 0.28 * min(slots, capacity) - 0.30 * max(0.0, slots - capacity)
    if i == 2:
        return (0.65 * f.spin_lines + 0.18 * f.attack) * safe if game.current == "T" and f.spin_lines else 0.0
    if i == 3:
        return 0.42 * min(2, slots) * safe if game.current == "T" and f.spin_lines else 0.0
    if i == 4:
        return 0.0 if nearest > 2 or game.current == "T" else 0.38 * low_clean - 0.48 * destroyed / (1.0 + a.holes)
    if i == 5:
        return 0.20 * f.attack + 0.32 * min(1, slots) * safe if game.current == "T" and f.spin_lines else 0.0
    if i == 7:
        return 0.24 * f.attack * safe

    # Delta/recovery candidates need the pre-placement board; avoid recomputing it for every other candidate.
    b = h.extract_board_features(game.board)
    depth_drop = max(0, b.hole_depth - a.hole_depth)
    hole_drop = max(0, b.holes - a.holes)
    height_drop = max(0, b.max_height - a.max_height)
    danger = max(0.0, (b.max_height - 9) / 7.0) + b.holes / 5.0
    if i == 6:
        return 0.30 * f.spin_lines * max(0.0, (b.max_height - 10) / 6.0) if f.spin_lines else 0.0
    if i == 8:
        return 0.045 * depth_drop
    if i == 9:
        return 0.055 * depth_drop * (1.0 + danger)
    if i == 10:
        return 0.16 * f.attack * min(2.0, depth_drop / 3.0 + hole_drop)
    if i == 11:
        return 0.18 * height_drop * danger
    if i == 12:
        return 0.0 if not f.lines or f.new_holes else 0.16 * f.lines * (1.0 + danger) * (1.0 + hole_drop)
    if i == 13:
        return 0.26 * created * min(1.0, supply) + 0.035 * depth_drop - 0.16 * created * a.holes
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
    return s["app"] - 0.04 * s["topouts"] + 0.002 * s["completed"] + 0.001 * s["tsd"]


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
    short = stage(2, 80, 1181003, 149, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 160, 1299709, 197, survivors)
    base = fresh[0]["summary"]
    viable = [x for x in fresh[1:] if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(games=6, max_turns=70, seed_base=130363, seed_step=223,
                                 player_weights=candidate_weights(x["index"]), ai_weights=BASE)
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print("TOURNAMENT_RESULT=" + json.dumps({"candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [x["name"] for x in finalists], "versus": versus}, sort_keys=True))


if __name__ == "__main__":
    main()
