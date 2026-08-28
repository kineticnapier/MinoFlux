from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing as mp

import minoflux_ai.heuristic as h
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.versus_benchmark import run_versus_benchmark

BASE = h.DEFAULT_WEIGHTS
ORIGINAL_CONTEXT = h._context_score
STEP = 1e-6
CANDIDATES = (
    "t_arrival_chain_setup",
    "t_arrival_clean_exit",
    "t_arrival_low_exit",
    "t_arrival_b2b_conversion",
    "spin_attack_risk_adjusted",
    "difficult_clear_clean_exit",
    "difficult_clear_low_exit",
    "b2b_clean_difficult_exit",
    "t_supply_distance_weighted_match",
    "t_supply_danger_deficit",
    "hold_t_clean_ready_balance",
    "slot_attack_efficiency",
    "slot_conversion_efficiency",
    "danger_attack_clean_exit",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def next_t_distance(game):
    if game.current == "T":
        return 0
    if game.hold_piece == "T" and not game.hold_used:
        return 1
    for j, p in enumerate(game.queue[:6]):
        if p == "T":
            return j + 1
    return 8


def weighted_t_supply(game):
    score = 1.0 if game.current == "T" else 0.0
    if game.hold_piece == "T" and not game.hold_used:
        score += 0.85
    for j, p in enumerate(game.queue[:6]):
        if p == "T":
            score += max(0.15, 0.75 - 0.10 * j)
    return score


def bonus(i, game, f):
    before = h.extract_board_features(game.board)
    after = f.board
    hole_drop = max(0, before.holes - after.holes)
    height_drop = max(0, before.max_height - after.max_height)
    slot_gain = max(0, f.t_spin_slot_delta)
    slot_loss = max(0, -f.t_spin_slot_delta)
    clean = 1.0 / (1.0 + after.holes + after.max_height / 6.0)
    low = 1.0 / (1.0 + after.max_height / 6.0)
    danger = max(0.0, (before.max_height - 10) / 4.0) + before.holes / 4.0
    difficult = bool(f.spin_lines or f.lines == 4)
    td = next_t_distance(game)
    urgency = max(0.0, (7.0 - td) / 7.0)
    supply = weighted_t_supply(game)

    if i == 0:
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.24 * min(2, after.t_spin_slots) + 0.08 * slot_gain
    if i == 1:
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.40 * clean + 0.08 * hole_drop
    if i == 2:
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.32 * low + 0.05 * height_drop
    if i == 3:
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return (0.16 + 0.035 * min(8, game.b2b_chain)) * f.attack
    if i == 4:
        if not f.spin_lines:
            return 0.0
        return 0.18 * f.attack * clean + 0.06 * f.spin_lines * low
    if i == 5:
        if not difficult:
            return 0.0
        return 0.22 * clean + 0.05 * hole_drop
    if i == 6:
        if not difficult:
            return 0.0
        return 0.20 * low + 0.05 * height_drop
    if i == 7:
        if not game.back_to_back or not difficult:
            return 0.0
        return (0.12 + 0.02 * min(8, game.b2b_chain)) * clean * (1 + 0.20 * f.attack)
    if i == 8:
        matched = min(after.t_spin_slots, supply)
        deficit = max(0.0, after.t_spin_slots - supply)
        return 0.20 * matched * urgency - 0.16 * deficit * (1.0 - 0.4 * urgency)
    if i == 9:
        deficit = max(0.0, after.t_spin_slots - supply)
        return -0.20 * deficit * min(2.0, danger) + 0.05 * min(after.t_spin_slots, int(supply)) * clean
    if i == 10:
        if game.hold_piece != "T" or game.hold_used:
            return 0.0
        return 0.14 * min(2, after.t_spin_slots) * clean - 0.08 * slot_loss
    if i == 11:
        return 0.07 * f.attack * (1.0 + min(2, after.t_spin_slots) * clean)
    if i == 12:
        if game.current != "T":
            return 0.0
        if f.spin_lines:
            return 0.10 * f.attack + 0.08 * after.t_spin_slots + 0.06 * clean
        return -0.10 * slot_loss * urgency
    if i == 13:
        if danger < 0.5:
            return 0.0
        return min(2.0, danger) * (0.06 * f.attack + 0.08 * hole_drop + 0.06 * height_drop + 0.10 * clean)
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
    return s["app"] - 0.08 * s["topouts"] + 0.002 * s["completed"] + 0.0015 * s["tsd"] + 0.0005 * s["spin_lines"]


def _bench_task(task):
    games, pieces, seed_base, seed_step, i = task
    weights = BASE if i is None else candidate_weights(i)
    r = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step,
                                weights=weights, workers=1)
    return {"name": "baseline" if i is None else CANDIDATES[i], "index": i, "summary": summary(r)}


def stage(games, pieces, seed_base, seed_step, indices):
    tasks = [(games, pieces, seed_base, seed_step, None)] + [(games, pieces, seed_base, seed_step, i) for i in indices]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(8, len(tasks)), mp_context=ctx) as ex:
        return list(ex.map(_bench_task, tasks))


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
    short = stage(2, 120, 2297111, 193, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 260, 2419003, 251, survivors)
    base = fresh[0]["summary"]
    viable = [x for x in fresh[1:] if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(games=8, max_turns=120, seed_base=247013, seed_step=257,
                                 player_weights=candidate_weights(x["index"]), ai_weights=BASE)
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print("TOURNAMENT_RESULT=" + json.dumps({"candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [x["name"] for x in finalists], "versus": versus}, sort_keys=True))

if __name__ == "__main__":
    main()
