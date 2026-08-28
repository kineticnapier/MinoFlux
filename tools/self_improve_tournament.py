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
    "t_timing_weighted_supply",
    "slot_timing_capacity",
    "current_t_clean_conversion",
    "tspin_rebuild_after_clear",
    "near_t_preserve_low_clean",
    "tspin_attack_rebuild",
    "high_stack_tspin_escape",
    "safe_attack_efficiency",
    "hole_depth_recovery",
    "danger_hole_depth_recovery",
    "downstack_attack_synergy",
    "height_drop_under_danger",
    "clean_line_escape",
    "slot_supply_hole_recovery",
)


def candidate_weights(index: int):
    return replace(BASE, t_spin_slot_queue_match=BASE.t_spin_slot_queue_match + (index + 1) * SENTINEL_STEP)


def candidate_index(weights) -> int | None:
    delta = weights.t_spin_slot_queue_match - BASE.t_spin_slot_queue_match
    index = round(delta / SENTINEL_STEP) - 1
    if 0 <= index < len(CANDIDATES) and abs(delta - (index + 1) * SENTINEL_STEP) < 1e-9:
        return index
    return None


def t_distances(game):
    out = []
    if game.current == "T":
        out.append(0)
    for i, piece in enumerate(list(game.queue)[:6]):
        if piece == "T":
            out.append(i + 1)
    if game.hold_piece == "T":
        out.append(0.5)
    return out


def bonus(index: int, game, f) -> float:
    after = f.board
    before = h.extract_board_features(game.board)
    slots = after.t_spin_slots
    dists = t_distances(game)
    weighted_supply = sum(max(0.0, (7.0 - d) / 7.0) for d in dists)
    low_clean = slots / (1.0 + after.holes + after.max_height / 6.0)
    destroyed = max(0, -f.t_spin_slot_delta)
    created = max(0, f.t_spin_slot_delta)
    hole_depth_drop = max(0, before.hole_depth - after.hole_depth)
    hole_drop = max(0, before.holes - after.holes)
    height_drop = max(0, before.max_height - after.max_height)
    danger = max(0.0, (before.max_height - 9) / 7.0) + before.holes / 5.0
    safe = 1.0 / (1.0 + after.holes + after.max_height / 8.0)
    nearest_t = min(dists) if dists else 8.0

    if index == 0:  # distinguish one close T from several distant Ts
        return 0.34 * min(slots, weighted_supply) - 0.22 * max(0.0, slots - weighted_supply)
    if index == 1:  # prepared slots should not exceed T supply that can arrive soon enough
        capacity = sum(1.0 if d <= 3 else 0.45 for d in dists)
        return 0.28 * min(slots, capacity) - 0.30 * max(0.0, slots - capacity)
    if index == 2:  # when T arrives, prefer conversions that finish on a low/clean board
        if game.current == "T" and f.spin_lines:
            return 0.65 * f.spin_lines * safe + 0.18 * f.attack * safe
        return 0.0
    if index == 3:  # reward a T-spin clear that leaves another usable setup behind
        if game.current == "T" and f.spin_lines:
            return 0.42 * min(2, slots) * safe
        return 0.0
    if index == 4:  # preserve only genuinely good setups when T is very near
        if nearest_t > 2 or game.current == "T":
            return 0.0
        return 0.38 * low_clean - 0.48 * destroyed / (1.0 + after.holes)
    if index == 5:  # conversion quality = immediate firepower plus next setup readiness
        if game.current == "T" and f.spin_lines:
            return 0.20 * f.attack + 0.32 * min(1, slots) * safe
        return 0.0
    if index == 6:  # T-spins are especially valuable as an escape while already high
        if f.spin_lines:
            return 0.30 * f.spin_lines * max(0.0, (before.max_height - 10) / 6.0)
        return 0.0
    if index == 7:  # reward attack only when the resulting board remains survivable
        return 0.24 * f.attack * safe
    if index == 8:  # prioritize exposing buried cells, not merely counting holes
        return 0.045 * hole_depth_drop
    if index == 9:  # hole-depth recovery matters more under genuine danger
        return 0.055 * hole_depth_drop * (1.0 + danger)
    if index == 10:  # line attack that simultaneously reaches garbage/deep holes
        return 0.16 * f.attack * min(2.0, hole_depth_drop / 3.0 + hole_drop)
    if index == 11:  # nonlinear emergency preference for dropping the ceiling
        return 0.18 * height_drop * danger
    if index == 12:  # under danger, clean line clears with no new holes are valuable
        if not f.lines or f.new_holes:
            return 0.0
        return 0.16 * f.lines * (1.0 + danger) * (1.0 + hole_drop)
    if index == 13:  # setup creation is best when it also improves buried-hole access
        supply = min(1.0, weighted_supply)
        return 0.26 * created * supply + 0.035 * hole_depth_drop - 0.16 * created * after.holes
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    index = candidate_index(weights)
    return base if index is None else base + bonus(index, game, features)


h._context_score = patched_context


def summarize(result):
    return {
        "pieces": result.pieces,
        "attack": result.attack,
        "app": result.attack / max(1, result.pieces),
        "topouts": result.topouts,
        "completed": result.completed,
        "spins": result.spins,
        "spin_lines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def strength_key(item):
    s = item["summary"]
    return s["app"] - 0.04 * s["topouts"] + 0.002 * s["completed"] + 0.001 * s["tsd"]


def run_stage(games, pieces, seed_base, seed_step, indices):
    rows = []
    base = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step, weights=BASE, workers=1)
    rows.append({"name": "baseline", "index": None, "summary": summarize(base)})
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
    short = run_stage(2, 150, 1181003, 149, range(len(CANDIDATES)))
    ranked = sorted(short[1:], key=strength_key, reverse=True)
    survivors = [row["index"] for row in ranked[:3]]
    fresh = run_stage(3, 270, 1299709, 197, survivors)
    base_fresh = fresh[0]["summary"]
    viable = [row for row in fresh[1:] if row["summary"]["app"] >= base_fresh["app"] and row["summary"]["topouts"] <= base_fresh["topouts"]]
    viable.sort(key=strength_key, reverse=True)
    finalists = viable[:2]
    versus = []
    for row in finalists:
        result = run_versus_benchmark(games=8, max_turns=115, seed_base=130363, seed_step=223, player_weights=candidate_weights(row["index"]), ai_weights=BASE)
        versus.append({"name": row["name"], "index": row["index"], "summary": versus_summary(result)})
    print("TOURNAMENT_RESULT=" + json.dumps({
        "candidates": list(CANDIDATES), "short": short,
        "survivors": [CANDIDATES[i] for i in survivors], "fresh": fresh,
        "finalists": [row["name"] for row in finalists], "versus": versus,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
