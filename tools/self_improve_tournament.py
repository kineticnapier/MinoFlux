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
    "post_tspin_ready_clean_followup",
    "post_tspin_single_slot_followup",
    "post_tspin_b2b_followup",
    "b2b_break_ready_slot_cost",
    "b2b_break_clean_stack_cost",
    "b2b_start_t_supply",
    "tsd_over_tss_value",
    "tst_safe_value",
    "spin_slot_rebuild",
    "attack_slot_preserve",
    "line_clear_slot_damage_cost",
    "danger_abandon_excess_slots",
    "clean_downstack_preserve_slot",
    "spin_clean_low_efficiency",
)


def candidate_weights(i: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (i + 1) * STEP)


def candidate_index(weights):
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    i = round(delta / STEP) - 1
    return i if 0 <= i < len(CANDIDATES) and abs(delta - (i + 1) * STEP) < 1e-9 else None


def future_t_distance(game):
    # Distance to the next T after the current piece. Hold counts as immediately available.
    if game.hold_piece == "T" and not game.hold_used:
        return 1
    for j, p in enumerate(game.queue):
        if j >= 6:
            break
        if p == "T":
            return j + 1
    return 8


def t_supply(game):
    count = 0
    if game.hold_piece == "T" and not game.hold_used:
        count += 1
    for j, p in enumerate(game.queue):
        if j >= 6:
            break
        if p == "T":
            count += 1
    return count


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
    fd = future_t_distance(game)
    urgency = max(0.0, (7.0 - fd) / 7.0)
    supply = t_supply(game)
    matched = min(after.t_spin_slots, supply)
    excess = max(0, after.t_spin_slots - supply)

    if i == 0:  # Convert one T-spin while leaving a clean follow-up setup for the next T.
        if game.current != "T" or not f.spin_lines:
            return 0.0
        return 0.28 * urgency * min(1, after.t_spin_slots) * clean
    if i == 1:  # Prefer exactly one follow-up slot when one near-future T can consume it.
        if game.current != "T" or not f.spin_lines or supply <= 0:
            return 0.0
        return 0.22 * urgency * min(1, after.t_spin_slots) - 0.12 * max(0, after.t_spin_slots - 1)
    if i == 2:  # Reward a T-spin that keeps B2B and leaves a supplied follow-up slot.
        if game.current != "T" or not f.spin_lines or not game.back_to_back:
            return 0.0
        return 0.12 * f.attack + 0.20 * matched * urgency * clean
    if i == 3:  # Breaking B2B is more expensive when a ready T-spin can soon continue it.
        if not game.back_to_back or f.lines <= 0 or difficult:
            return 0.0
        return -0.30 * urgency * min(2, before.t_spin_slots) * (1.0 + 0.15 * min(6, game.b2b_chain))
    if i == 4:  # Avoid gratuitous B2B breaks on already safe/clean boards.
        if not game.back_to_back or f.lines <= 0 or difficult:
            return 0.0
        safe_before = 1.0 / (1.0 + before.holes + before.max_height / 6.0)
        return -0.24 * safe_before * (1.0 + 0.10 * min(6, game.b2b_chain))
    if i == 5:  # Starting B2B is more valuable if a future T can use a remaining slot.
        if game.back_to_back or not difficult:
            return 0.0
        return 0.20 * matched * urgency * clean + 0.04 * f.attack
    if i == 6:  # Bias spin conversions toward TSD rather than TSS when otherwise comparable.
        if game.current != "T" or not f.spin_lines:
            return 0.0
        if f.spin_lines == 2:
            return 0.34 + 0.05 * f.attack
        if f.spin_lines == 1:
            return -0.08 * min(1, after.t_spin_slots)
        return 0.0
    if i == 7:  # TST is valuable only when the exit remains reasonably safe.
        if game.current != "T" or f.spin_lines != 3:
            return 0.0
        safety = 1.0 / (1.0 + after.holes + max(0, after.max_height - 8) / 4.0)
        return 0.48 * safety + 0.05 * f.attack
    if i == 8:  # After cashing a spin, reward rebuilding the next clean slot immediately.
        if not f.spin_lines:
            return 0.0
        return 0.22 * slot_gain * clean + 0.10 * min(1, after.t_spin_slots) * urgency * clean
    if i == 9:  # Attack that preserves an existing setup is worth more than attack that destroys it.
        if f.attack <= 0 or before.t_spin_slots <= 0:
            return 0.0
        preserved = max(0, min(before.t_spin_slots, after.t_spin_slots))
        return 0.07 * f.attack * preserved * clean - 0.10 * slot_loss
    if i == 10:  # Delay ordinary line clears that destroy a ready slot when T is close.
        if f.lines <= 0 or f.spin_lines or slot_loss <= 0:
            return 0.0
        return -0.26 * slot_loss * urgency
    if i == 11:  # In danger, stop overbooking more T-spin slots than the visible T supply can service.
        if danger < 0.5:
            return 0.0
        return -0.18 * min(2.0, danger) * excess + 0.06 * matched * clean
    if i == 12:  # Downstacking is best when it reaches holes without destroying prepared T-spin value.
        if f.lines <= 0:
            return 0.0
        return 0.12 * hole_drop + 0.05 * height_drop - 0.14 * slot_loss * urgency
    if i == 13:  # Spin attack efficiency, gated by a clean and low exit instead of raw attack alone.
        if not f.spin_lines:
            return 0.0
        return 0.11 * f.attack * clean + 0.08 * f.spin_lines * low
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    i = candidate_index(weights)
    return base if i is None else base + bonus(i, game, features)


h._context_score = patched_context


def summary(r):
    return {
        "pieces": r.pieces,
        "attack": r.attack,
        "app": r.attack / max(1, r.pieces),
        "topouts": r.topouts,
        "completed": r.completed,
        "spins": r.spins,
        "spin_lines": r.spin_lines,
        "tsd": r.t_spin_doubles,
        "tst": r.t_spin_triples,
    }


def key(row):
    s = row["summary"]
    return s["app"] - 0.08 * s["topouts"] + 0.002 * s["completed"] + 0.0015 * s["tsd"] + 0.0005 * s["spin_lines"]


def _bench_task(task):
    games, pieces, seed_base, seed_step, i = task
    weights = BASE if i is None else candidate_weights(i)
    r = run_heuristic_benchmark(
        games=games,
        max_pieces=pieces,
        seed_base=seed_base,
        seed_step=seed_step,
        weights=weights,
        workers=1,
    )
    return {"name": "baseline" if i is None else CANDIDATES[i], "index": i, "summary": summary(r)}


def stage(games, pieces, seed_base, seed_step, indices):
    tasks = [(games, pieces, seed_base, seed_step, None)] + [
        (games, pieces, seed_base, seed_step, i) for i in indices
    ]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(8, len(tasks)), mp_context=ctx) as ex:
        return list(ex.map(_bench_task, tasks))


def versus_summary(r):
    rows = r.per_game
    n = len(rows)
    return {
        "candidate_wins": r.player_wins,
        "baseline_wins": r.ai_wins,
        "draws": r.draws,
        "candidate_attack": r.player_mean_attack,
        "baseline_attack": r.ai_mean_attack,
        "candidate_sent": r.player_mean_sent,
        "baseline_sent": r.ai_mean_sent,
        "candidate_cancel": sum(x.player_canceled for x in rows) / n,
        "baseline_cancel": sum(x.ai_canceled for x in rows) / n,
        "candidate_received": sum(x.player_received for x in rows) / n,
        "baseline_received": sum(x.ai_received for x in rows) / n,
        "candidate_b2b": sum(x.player_max_b2b for x in rows) / n,
        "baseline_b2b": sum(x.ai_max_b2b for x in rows) / n,
        "candidate_topouts": sum(x.winner == "ai" for x in rows),
        "baseline_topouts": sum(x.winner == "player" for x in rows),
    }


def main():
    short = stage(2, 120, 2573119, 263, range(len(CANDIDATES)))
    survivors = [x["index"] for x in sorted(short[1:], key=key, reverse=True)[:3]]
    fresh = stage(3, 260, 2690141, 271, survivors)
    base = fresh[0]["summary"]
    viable = [
        x for x in fresh[1:]
        if x["summary"]["app"] >= base["app"] and x["summary"]["topouts"] <= base["topouts"]
    ]
    finalists = sorted(viable, key=key, reverse=True)[:2]
    versus = []
    for x in finalists:
        r = run_versus_benchmark(
            games=8,
            max_turns=120,
            seed_base=2718281,
            seed_step=277,
            player_weights=candidate_weights(x["index"]),
            ai_weights=BASE,
        )
        versus.append({"name": x["name"], "index": x["index"], "summary": versus_summary(r)})
    print(
        "TOURNAMENT_RESULT="
        + json.dumps(
            {
                "candidates": list(CANDIDATES),
                "short": short,
                "survivors": [CANDIDATES[i] for i in survivors],
                "fresh": fresh,
                "finalists": [x["name"] for x in finalists],
                "versus": versus,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
