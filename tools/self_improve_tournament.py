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
    "difficult_clear_slot_preservation",
    "attack_slot_creation_synergy",
    "clean_low_slot_creation",
    "danger_depth_recovery_attack",
    "danger_height_escape_without_attack",
    "aggregate_height_clear_efficiency",
    "hole_clear_efficiency",
    "t_arrival_postspin_cleanliness",
    "t_arrival_multi_slot_conversion",
    "b2b_chain_attack_quality",
    "b2b_chain_slot_preservation",
    "combo_difficult_clear_quality",
    "safe_attack_density",
    "spin_followup_slot_quality",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def bonus(i, game, f):
    before = h.extract_board_features(game.board)
    after = f.board
    slots_before = before.t_spin_slots
    slots_after = after.t_spin_slots
    slot_gain = max(0, f.t_spin_slot_delta)
    slot_loss = max(0, -f.t_spin_slot_delta)
    hole_drop = max(0, before.holes - after.holes)
    depth_drop = max(0, before.hole_depth - after.hole_depth)
    height_drop = max(0, before.max_height - after.max_height)
    agg_drop = max(0, before.aggregate_height - after.aggregate_height)
    danger = min(2.5, max(0.0, (before.max_height - 8) / 5.0) + before.holes / 3.0)
    clean = 1.0 / (1.0 + after.holes + after.max_height / 7.0)
    difficult = bool(f.spin_lines or f.lines == 4)

    if i == 0:
        return (0.18 * f.attack + 0.16 * min(2, slots_after) - 0.28 * slot_loss) if difficult else 0.0
    if i == 1:
        return 0.18 * f.attack * slot_gain - 0.16 * f.attack * slot_loss
    if i == 2:
        return 0.30 * slot_gain * clean if after.holes == 0 and after.max_height <= 9 else 0.0
    if i == 3:
        return danger * (0.07 * depth_drop + 0.13 * hole_drop + 0.07 * f.attack)
    if i == 4:
        if danger < 0.8 or f.attack:
            return 0.0
        return 0.10 * height_drop + 0.025 * agg_drop - 0.10 * f.new_holes
    if i == 5:
        if not f.lines:
            return 0.0
        return 0.020 * agg_drop / f.lines - 0.05 * f.new_holes
    if i == 6:
        if not f.lines:
            return 0.0
        return 0.20 * hole_drop / f.lines + 0.018 * depth_drop / f.lines
    if i == 7:
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.32 * clean + 0.08 * hole_drop + 0.04 * height_drop
    if i == 8:
        if game.current != "T":
            return 0.0
        return 0.22 * f.spin_lines * min(2, slots_before) - 0.22 * slot_loss if slots_before >= 2 else 0.0
    if i == 9:
        if not game.back_to_back or not difficult:
            return 0.0
        return 0.08 * f.attack * (1.0 + min(4, game.b2b_chain) / 4.0)
    if i == 10:
        if not game.back_to_back:
            return 0.0
        return 0.15 * min(2, slots_after) - 0.24 * slot_loss
    if i == 11:
        if game.combo < 1 or not difficult:
            return 0.0
        return 0.06 * f.attack * min(4, game.combo + 1) + 0.05 * f.spin_lines
    if i == 12:
        return 0.15 * f.attack * clean - 0.08 * f.new_holes if f.attack else 0.0
    if i == 13:
        if not f.spin_lines:
            return 0.0
        return 0.18 * min(2, slots_after) * clean + 0.10 * slot_gain
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
    return s["app"] - 0.06 * s["topouts"] + 0.002 * s["completed"] + 0.0015 * s["tsd"] + 0.0005 * s["spin_lines"]


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
    short = stage(2, 100, 1638119, 173, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 220, 1745203, 223, survivors)
    base = fresh[0]["summary"]
    viable = [x for x in fresh[1:] if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(games=8, max_turns=100, seed_base=175319, seed_step=229,
                                 player_weights=candidate_weights(x["index"]), ai_weights=BASE)
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print("TOURNAMENT_RESULT=" + json.dumps({"candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [x["name"] for x in finalists], "versus": versus}, sort_keys=True))

if __name__ == "__main__":
    main()
