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
    "garbage_clear_value",
    "garbage_attack_cancel_proxy",
    "garbage_hole_recovery",
    "garbage_depth_recovery",
    "surge_release_value",
    "surge_safe_charge",
    "b2b_chain_length_value",
    "b2b_clean_difficult_value",
    "combo_danger_cashout",
    "attack_hole_depth_efficiency",
    "attack_height_recovery",
    "spin_attack_clean_exchange",
    "t_arrival_garbage_conversion",
    "hold_t_danger_release",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def board_stats(game):
    garbage_rows = 0
    garbage_cells = 0
    top_g = game.height
    for y, row in enumerate(game.board):
        n = sum(cell == "G" for cell in row)
        if n:
            garbage_rows += 1
            garbage_cells += n
            top_g = min(top_g, y)
    garbage_pressure = min(1.5, garbage_rows / 5.0 + garbage_cells / 40.0)
    garbage_depth = 0.0 if top_g == game.height else (game.height - top_g) / game.height
    return garbage_rows, garbage_cells, garbage_pressure, garbage_depth


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
    hole_drop = max(0, before.holes - after.holes)
    depth_drop = max(0, before.hole_depth - after.hole_depth)
    height_drop = max(0, before.max_height - after.max_height)
    danger = max(0.0, (before.max_height - 9) / 4.0) + before.holes / 4.0
    clean = 1.0 / (1.0 + after.holes + after.max_height / 6.0)
    difficult = bool(f.spin_lines or f.lines == 4)
    garbage_rows, garbage_cells, gp, gd = board_stats(game)
    if i == 0:
        return gp * (0.16 * f.lines + 0.08 * f.attack)
    if i == 1:
        return gp * (0.18 * f.attack + 0.06 * f.spin_lines) - 0.05 * gp * f.new_holes
    if i == 2:
        return gp * (0.22 * hole_drop + 0.020 * depth_drop + 0.05 * f.lines)
    if i == 3:
        return gd * (0.16 * height_drop + 0.018 * depth_drop + 0.07 * f.lines)
    if i == 4:
        if game.surge_charge <= 0 or not difficult:
            return 0.0
        return min(8, game.surge_charge) * (0.035 * f.attack + 0.025 * f.spin_lines)
    if i == 5:
        if game.surge_charge <= 0:
            return 0.0
        return min(8, game.surge_charge) * (0.025 * clean + 0.018 * height_drop - 0.022 * f.new_holes)
    if i == 6:
        if not difficult:
            return 0.0
        return 0.045 * min(8, game.b2b_chain + 1) * (1.0 + 0.25 * f.attack)
    if i == 7:
        if not game.back_to_back or not difficult:
            return 0.0
        return min(6, game.b2b_chain + 1) * (0.035 * f.attack + 0.060 * clean)
    if i == 8:
        if game.combo < 1 or danger < 0.5:
            return 0.0
        return min(5, game.combo + 1) * danger * (0.040 * f.attack + 0.025 * height_drop + 0.030 * hole_drop)
    if i == 9:
        return 0.08 * f.attack * (1.0 + 0.25 * depth_drop) - 0.020 * max(0, after.hole_depth - before.hole_depth)
    if i == 10:
        return 0.055 * f.attack * (1.0 + height_drop) + 0.055 * height_drop - 0.035 * f.new_holes
    if i == 11:
        if not f.spin_lines:
            return 0.0
        return 0.10 * f.attack * clean + 0.08 * hole_drop + 0.04 * height_drop
    if i == 12:
        if game.current != "T" or not f.spin_lines or gp <= 0:
            return 0.0
        return gp * (0.12 * f.attack + 0.10 * f.spin_lines + 0.06 * hole_drop)
    if i == 13:
        if game.hold_piece != "T" or game.current == "T" or danger < 0.6:
            return 0.0
        urgency = max(0.0, (7.0 - next_t_distance(game)) / 7.0)
        return -0.10 * danger * urgency * min(2, after.t_spin_slots)
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
    short = stage(2, 110, 2061107, 181, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 240, 2174209, 229, survivors)
    base = fresh[0]["summary"]
    viable = [x for x in fresh[1:] if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(games=8, max_turns=110, seed_base=218311, seed_step=239,
                                 player_weights=candidate_weights(x["index"]), ai_weights=BASE)
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print("TOURNAMENT_RESULT=" + json.dumps({"candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [x["name"] for x in finalists], "versus": versus}, sort_keys=True))

if __name__ == "__main__":
    main()
