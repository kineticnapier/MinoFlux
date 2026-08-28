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
    "t_slot_supply_ratio",
    "t_slot_overbooking_danger",
    "t_arrival_slot_preserve_after_clear",
    "t_arrival_attack_efficiency",
    "queue_t_urgency_low_stack",
    "queue_t_urgency_clean_stack",
    "danger_attack_escape",
    "danger_spin_escape_quality",
    "danger_hole_recovery_efficiency",
    "b2b_difficult_clear_survival",
    "b2b_slot_ready_bonus",
    "combo_attack_cleanliness",
    "downstack_attack_hole_reduction",
    "slot_creation_supply_gated",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def t_supply(game):
    count = (1 if game.current == "T" else 0) + (1 if game.hold_piece == "T" else 0)
    for j, p in enumerate(game.queue):
        if j >= 6:
            break
        count += p == "T"
    return int(count)


def next_t_distance(game):
    if game.current == "T":
        return 0
    if game.hold_piece == "T" and not game.hold_used:
        return 1
    for j, p in enumerate(game.queue):
        if j >= 6:
            break
        if p == "T":
            return j + 1
    return 8


def bonus(i, game, f):
    before = h.extract_board_features(game.board)
    after = f.board
    slots0, slots1 = before.t_spin_slots, after.t_spin_slots
    gain = max(0, f.t_spin_slot_delta)
    loss = max(0, -f.t_spin_slot_delta)
    hole_drop = max(0, before.holes - after.holes)
    depth_drop = max(0, before.hole_depth - after.hole_depth)
    height_drop = max(0, before.max_height - after.max_height)
    supply = t_supply(game)
    dist = next_t_distance(game)
    danger = max(0.0, (before.max_height - 9) / 4.0) + before.holes / 4.0
    clean = 1.0 / (1.0 + after.holes + after.max_height / 6.0)
    difficult = bool(f.spin_lines or f.lines == 4)
    if i == 0:
        # Prefer slot counts that are proportionate to nearby T supply.
        if slots1 == 0 and supply == 0:
            return 0.0
        return 0.28 * min(slots1, supply) - 0.24 * max(0, slots1 - supply)
    if i == 1:
        # Extra prepared slots are especially expensive while the board is dangerous.
        return -0.18 * danger * max(0, slots1 - supply)
    if i == 2:
        # When T actually clears, preserve a usable follow-up slot if possible.
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.24 * min(1, slots1) * clean - 0.18 * loss
    if i == 3:
        # Reward efficient conversion of a T arrival into attack without dirtying the board.
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.14 * f.attack + 0.18 * clean - 0.10 * f.new_holes
    if i == 4:
        # If T is imminent, prefer keeping the stack low enough to actually use the slot.
        urgency = max(0.0, (7.0 - dist) / 7.0)
        return urgency * min(2, slots1) * max(0.0, (11.0 - after.max_height) / 11.0) * 0.30
    if i == 5:
        urgency = max(0.0, (7.0 - dist) / 7.0)
        return 0.34 * urgency * min(2, slots1) / (1.0 + after.holes)
    if i == 6:
        if danger < 0.8:
            return 0.0
        return danger * (0.10 * f.attack + 0.07 * height_drop + 0.09 * hole_drop - 0.08 * f.new_holes)
    if i == 7:
        if danger < 0.7 or not f.spin_lines:
            return 0.0
        return danger * (0.13 * f.spin_lines + 0.08 * f.attack + 0.10 * clean)
    if i == 8:
        if danger < 0.6:
            return 0.0
        return danger * (0.16 * hole_drop + 0.025 * depth_drop + 0.04 * height_drop)
    if i == 9:
        if not game.back_to_back or not difficult:
            return 0.0
        return 0.10 * f.attack + 0.08 * height_drop + 0.10 * hole_drop - 0.08 * f.new_holes
    if i == 10:
        if not game.back_to_back:
            return 0.0
        return 0.18 * min(1, slots1) * clean + (0.05 * f.attack if difficult else 0.0) - 0.16 * loss
    if i == 11:
        if game.combo < 1 or not f.attack:
            return 0.0
        return 0.08 * f.attack * min(4, game.combo + 1) * clean - 0.06 * f.new_holes
    if i == 12:
        if not f.lines:
            return 0.0
        return 0.12 * f.attack * hole_drop + 0.015 * depth_drop - 0.05 * f.new_holes
    if i == 13:
        if gain <= 0:
            return 0.0
        availability = min(1.0, supply / max(1, slots1))
        return 0.32 * gain * availability * clean
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
    short = stage(2, 110, 1859107, 179, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 240, 1964209, 227, survivors)
    base = fresh[0]["summary"]
    viable = [x for x in fresh[1:] if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(games=8, max_turns=110, seed_base=197311, seed_step=233,
                                 player_weights=candidate_weights(x["index"]), ai_weights=BASE)
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print("TOURNAMENT_RESULT=" + json.dumps({"candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [x["name"] for x in finalists], "versus": versus}, sort_keys=True))

if __name__ == "__main__":
    main()
