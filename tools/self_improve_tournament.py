from __future__ import annotations

from dataclasses import replace
import json

import minoflux_ai.heuristic as h
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.versus_benchmark import run_versus_benchmark

BASE = h.DEFAULT_WEIGHTS
ORIGINAL_CONTEXT = h._context_score
SENTINEL_STEP = 1e-6

CANDIDATES = (
    "queue_quality_match",
    "near_t_low_clean",
    "arrival_slot_preservation",
    "arrival_safe_conversion",
    "hold_supply_clean_balance",
    "supply_overcommit_squared",
    "idle_t_demand",
    "two_t_two_slot_alignment",
    "queue_t_deadline",
    "slot_delta_supply_quality",
    "slot_destroy_supply",
    "slot_density_queue",
    "arrival_attack_efficiency",
    "queue_mismatch_danger",
)


def candidate_weights(index: int):
    # Tiny marker so patched context can distinguish candidate from baseline in mirrored versus.
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (index + 1) * SENTINEL_STEP)


def candidate_index(weights) -> int | None:
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    index = round(delta / SENTINEL_STEP) - 1
    if 0 <= index < len(CANDIDATES) and abs(delta - (index + 1) * SENTINEL_STEP) < 1e-9:
        return index
    return None


def t_distance(game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if i >= 6:
            break
        if piece == "T":
            return i + 1
    return 8


def t_supply(game) -> int:
    count = int(game.current == "T") + int(game.hold_piece == "T")
    count += sum(piece == "T" for piece in list(game.queue)[:6])
    return count


def bonus(index: int, game, f) -> float:
    b = f.board
    slots = b.t_spin_slots
    supply = t_supply(game)
    dist = t_distance(game)
    low_clean = slots / (1.0 + b.holes + b.max_height / 6.0)
    density = b.t_spin_slot_density
    destroyed = max(0, -f.t_spin_slot_delta)
    created = max(0, f.t_spin_slot_delta)
    safe = 1.0 / (1.0 + b.max_height / 8.0)

    if index == 0:  # exact nearby T supply should matter more when the prepared slots are actually clean/safe
        matched = min(slots, supply)
        mismatch = abs(slots - supply)
        return 0.36 * matched * safe / (1 + b.holes) - 0.24 * mismatch * (1.0 + b.holes / 4.0)
    if index == 1:  # deadline pressure: preserve quality when T is imminent
        urgency = max(0.0, (3.0 - dist) / 3.0)
        return 0.72 * urgency * low_clean
    if index == 2:  # current T should not casually destroy a ready clean slot
        if game.current != "T" or f.spin_lines:
            return 0.0
        return -0.75 * destroyed * (1.0 + safe)
    if index == 3:  # reward conversions that leave a survivable board
        if game.current == "T" and f.spin_lines:
            return 0.32 * f.attack * safe + 0.42 * f.spin_lines * safe
        return 0.0
    if index == 4:  # after using Hold, unmet clean slots are especially bad in danger
        if not game.hold_used:
            return 0.0
        deficit = max(0, slots - supply)
        held = int(game.hold_piece == "T")
        danger = 1.0 + max(0, b.max_height - 10) / 6.0 + b.holes / 5.0
        return 0.18 * held * min(slots, 1) - 0.24 * deficit * danger
    if index == 5:  # nonlinear overcommit: many slots with too little T supply are disproportionately wasteful
        deficit = max(0, slots - supply)
        return -0.17 * deficit * deficit
    if index == 6:  # nearby T with no prepared slot is idle attacking resource
        if slots:
            return 0.0
        urgency = max(0.0, (5.0 - dist) / 5.0)
        return -0.32 * urgency
    if index == 7:  # explicitly recognize multiple-slot/multiple-T alignment
        pairs = min(slots, supply)
        return 0.22 * max(0, pairs - 1) * safe
    if index == 8:  # hard deadline quality, only current/next T
        urgency = 1.0 if dist == 0 else 0.65 if dist == 1 else 0.0
        return 0.55 * urgency * low_clean
    if index == 9:  # create slots only when nearby T supply can realistically consume them
        available = min(1.0, supply / max(1, slots))
        return 0.42 * created * available * safe / (1.0 + b.holes / 3.0)
    if index == 10:  # destroying a slot is worse if a T is already on the way/held
        available = 1.0 if (game.hold_piece == "T" or dist <= 3) else 0.0
        return -0.52 * destroyed * available
    if index == 11:  # density is useful only to the extent the queue can service it
        return 0.30 * density * min(slots, supply)
    if index == 12:  # prefer efficient current-T conversions, moderated by danger
        if game.current == "T" and f.spin_lines:
            efficiency = f.attack / max(1, f.spin_lines)
            return 0.22 * efficiency * safe
        return 0.0
    if index == 13:  # supply mismatch becomes more costly as topout danger rises
        mismatch = abs(slots - supply)
        danger = max(0.0, (b.max_height - 8) / 8.0) + b.holes / 8.0
        return -0.20 * mismatch * danger
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    index = candidate_index(weights)
    return base if index is None else base + bonus(index, game, features)


h._context_score = patched_context


def summarize(result):
    tsd = result.t_spin_doubles
    tst = result.t_spin_triples
    return {
        "pieces": result.pieces,
        "attack": result.attack,
        "app": result.attack / max(1, result.pieces),
        "topouts": result.topouts,
        "completed": result.completed,
        "spins": result.spins,
        "spin_lines": result.spin_lines,
        "tsd": tsd,
        "tst": tst,
    }


def strength_key(item):
    s = item["summary"]
    # Firepower first, but a topout is expensive enough that fragile short-seed spikes do not dominate.
    return s["app"] - 0.035 * s["topouts"] + 0.002 * s["completed"] + 0.001 * s["tsd"]


def run_stage(games, pieces, seed_base, seed_step, indices):
    rows = []
    baseline = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step, weights=BASE, workers=1)
    rows.append({"name": "baseline", "index": None, "summary": summarize(baseline)})
    for index in indices:
        result = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step, weights=candidate_weights(index), workers=1)
        rows.append({"name": CANDIDATES[index], "index": index, "summary": summarize(result)})
    return rows


def versus_summary(result):
    rows = result.per_game
    return {
        "candidate_wins": result.player_wins,
        "baseline_wins": result.ai_wins,
        "draws": result.draws,
        "candidate_attack": result.player_mean_attack,
        "baseline_attack": result.ai_mean_attack,
        "candidate_sent": result.player_mean_sent,
        "baseline_sent": result.ai_mean_sent,
        "candidate_cancel": sum(r.player_canceled for r in rows) / len(rows),
        "baseline_cancel": sum(r.ai_canceled for r in rows) / len(rows),
        "candidate_received": sum(r.player_received for r in rows) / len(rows),
        "baseline_received": sum(r.ai_received for r in rows) / len(rows),
        "candidate_b2b": sum(r.player_max_b2b for r in rows) / len(rows),
        "baseline_b2b": sum(r.ai_max_b2b for r in rows) / len(rows),
        "candidate_topouts": sum(r.winner == "ai" for r in rows),
        "baseline_topouts": sum(r.winner == "player" for r in rows),
    }


def main():
    short = run_stage(2, 150, 814001, 137, range(len(CANDIDATES)))
    ranked = sorted(short[1:], key=strength_key, reverse=True)
    survivors = [row["index"] for row in ranked[:3]]

    fresh = run_stage(3, 260, 927503, 193, survivors)
    base_fresh = fresh[0]["summary"]
    viable = []
    for row in fresh[1:]:
        s = row["summary"]
        # Require fresh firepower to match/beat baseline and do not accept a clear survival regression.
        if s["app"] >= base_fresh["app"] and s["topouts"] <= base_fresh["topouts"] + 0:
            viable.append(row)
    viable.sort(key=strength_key, reverse=True)
    finalists = viable[:2]

    versus = []
    for row in finalists:
        index = row["index"]
        result = run_versus_benchmark(
            games=8,
            max_turns=110,
            seed_base=104729,
            seed_step=211,
            player_weights=candidate_weights(index),
            ai_weights=BASE,
        )
        versus.append({"name": row["name"], "index": index, "summary": versus_summary(result)})

    payload = {
        "candidates": list(CANDIDATES),
        "short": short,
        "survivors": [CANDIDATES[i] for i in survivors],
        "fresh": fresh,
        "finalists": [row["name"] for row in finalists],
        "versus": versus,
    }
    print("TOURNAMENT_RESULT=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
