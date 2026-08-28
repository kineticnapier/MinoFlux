from __future__ import annotations

from dataclasses import replace
import json

import minoflux_ai.heuristic as heuristic
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_search import VersusSearchConfig

BASE_GAME_OVER = DEFAULT_WEIGHTS.game_over
ORIGINAL_CONTEXT = heuristic._context_score

CANDIDATES = (
    "slot_preserve_until_t",
    "b2b_break_cost_dynamic",
    "b2b_attack_conversion",
    "danger_hole_depth_product",
    "danger_new_hole_product",
    "downstack_depth_recovery",
    "downstack_hole_recovery",
    "t_ready_low_dirty_cost",
    "hold_t_release_pressure",
    "hold_t_reserve_when_slots",
    "slot_supply_near_t_cap",
    "spin_conversion_efficiency",
    "clean_attack_followthrough",
    "safe_difficult_clear_chain",
)


def candidate_weights(index: int):
    # Tiny sentinel difference only; it is far below any meaningful terminal-score scale.
    return replace(DEFAULT_WEIGHTS, game_over=BASE_GAME_OVER - (index + 1) * 1e-3)


def candidate_index(weights) -> int | None:
    delta = BASE_GAME_OVER - weights.game_over
    if delta < 5e-4:
        return None
    index = round(delta / 1e-3) - 1
    return index if 0 <= index < len(CANDIDATES) else None


def next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def extra_score(name: str, game, f) -> float:
    before = extract_board_features(game.board)
    after = f.board
    slot_loss = max(0, before.t_spin_slots - after.t_spin_slots)
    slot_gain = max(0, after.t_spin_slots - before.t_spin_slots)
    holes_recovered = max(0, before.holes - after.holes)
    depth_recovered = max(0.0, before.hole_depth - after.hole_depth)
    height = after.max_height
    danger = max(0.0, (height - 8.0) / 8.0)
    t_dist = next_t_distance(game)
    t_urgency = max(0.0, (7.0 - t_dist) / 7.0)
    difficult = f.spin_lines > 0 or f.lines == 4
    held_t = game.hold_piece == "T"

    if name == "slot_preserve_until_t":
        return -0.70 * t_urgency * slot_loss + 0.10 * t_urgency * slot_gain
    if name == "b2b_break_cost_dynamic":
        if game.back_to_back and f.lines > 0 and not difficult:
            return -0.45 * (1.0 + min(4, game.b2b_chain) * 0.20)
        return 0.0
    if name == "b2b_attack_conversion":
        return (0.22 * f.attack + 0.30 * f.spin_lines) if game.back_to_back and difficult else 0.0
    if name == "danger_hole_depth_product":
        return -0.025 * danger * after.holes * max(1.0, after.hole_depth)
    if name == "danger_new_hole_product":
        return -0.90 * danger * f.new_holes
    if name == "downstack_depth_recovery":
        return 0.055 * depth_recovered * (1.0 + 0.5 * danger)
    if name == "downstack_hole_recovery":
        return 0.60 * holes_recovered * (1.0 + 0.4 * danger)
    if name == "t_ready_low_dirty_cost":
        return 0.28 * t_urgency * after.t_spin_slots / (1.0 + after.holes + max(0, height - 6) * 0.25)
    if name == "hold_t_release_pressure":
        if held_t and t_urgency > 0 and after.t_spin_slots > 0 and game.current != "T":
            return -0.14 * min(2, after.t_spin_slots)
        return 0.0
    if name == "hold_t_reserve_when_slots":
        if game.hold_used and held_t and after.t_spin_slots > 0:
            return 0.18 * min(2, after.t_spin_slots) / (1.0 + after.holes)
        return 0.0
    if name == "slot_supply_near_t_cap":
        supply = (1 if game.current == "T" else 0) + sum(1 for p in list(game.queue)[:6] if p == "T") + (1 if held_t else 0)
        return 0.16 * min(after.t_spin_slots, supply) - 0.20 * max(0, after.t_spin_slots - supply)
    if name == "spin_conversion_efficiency":
        if game.current == "T" and f.spin_lines:
            return 0.24 * f.attack + 0.38 * f.spin_lines - 0.08 * after.holes
        return 0.0
    if name == "clean_attack_followthrough":
        if f.attack <= 0:
            return 0.0
        return 0.08 * f.attack * (1.0 + holes_recovered) - 0.06 * f.attack * danger
    if name == "safe_difficult_clear_chain":
        if difficult:
            return 0.18 * (1.0 + min(4, game.b2b_chain) * 0.15) / (1.0 + danger + after.holes * 0.15)
        return 0.0
    raise AssertionError(name)


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    index = candidate_index(weights)
    if index is None:
        return base
    return base + extra_score(CANDIDATES[index], game, features)


heuristic._context_score = patched_context

SOLO = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4, discount=0.9, srs_reachable=True)
VERSUS_CFG = VersusSearchConfig(placement_search=SOLO, candidate_width=6, opponent_reply_width=1)


def summary(result):
    return {
        "pieces": result.pieces,
        "attack": result.attack,
        "app": result.attack / max(1, result.pieces),
        "topouts": result.topouts,
        "completed": result.completed,
        "spins": result.spins,
        "spinLines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
        "meanHoles": result.mean_holes,
        "meanHoleDepth": result.mean_hole_depth,
        "meanMaxHeight": result.mean_max_height,
    }


def stage_key(item):
    s = item[1]
    return (s["completed"], -s["topouts"], s["app"], s["tsd"], s["spinLines"])


def main():
    baseline_short = run_heuristic_benchmark(games=2, max_pieces=120, seed_base=31001, seed_step=31, weights=DEFAULT_WEIGHTS, search_config=SOLO, workers=1)
    short = []
    for i, name in enumerate(CANDIDATES):
        r = run_heuristic_benchmark(games=2, max_pieces=120, seed_base=31001, seed_step=31, weights=candidate_weights(i), search_config=SOLO, workers=1)
        short.append((name, summary(r), i))
    survivors = sorted(short, key=stage_key, reverse=True)[:3]

    baseline_fresh = run_heuristic_benchmark(games=3, max_pieces=260, seed_base=731003, seed_step=97, weights=DEFAULT_WEIGHTS, search_config=SOLO, workers=1)
    fresh = []
    for name, _, i in survivors:
        r = run_heuristic_benchmark(games=3, max_pieces=260, seed_base=731003, seed_step=97, weights=candidate_weights(i), search_config=SOLO, workers=1)
        fresh.append((name, summary(r), i))

    base_f = summary(baseline_fresh)
    eligible = [x for x in fresh if x[1]["completed"] >= base_f["completed"] and x[1]["topouts"] <= base_f["topouts"] and x[1]["app"] >= base_f["app"] * 0.99]
    finalists = sorted(eligible, key=stage_key, reverse=True)[:2]

    versus = []
    for name, _, i in finalists:
        v = run_versus_benchmark(games=8, max_turns=120, seed_base=12000052, seed_step=193, player_weights=candidate_weights(i), ai_weights=DEFAULT_WEIGHTS, player_config=VERSUS_CFG, ai_config=VERSUS_CFG, garbage_cap=8)
        versus.append({
            "name": name,
            "wins": v.player_wins,
            "losses": v.ai_wins,
            "draws": v.draws,
            "attack": [v.player_mean_attack, v.ai_mean_attack],
            "sent": [v.player_mean_sent, v.ai_mean_sent],
            "canceled": [sum(g.player_canceled for g in v.per_game)/v.games, sum(g.ai_canceled for g in v.per_game)/v.games],
            "received": [sum(g.player_received for g in v.per_game)/v.games, sum(g.ai_received for g in v.per_game)/v.games],
            "maxB2B": [sum(g.player_max_b2b for g in v.per_game)/v.games, sum(g.ai_max_b2b for g in v.per_game)/v.games],
            "topouts": [sum(g.winner == "ai" for g in v.per_game), sum(g.winner == "player" for g in v.per_game)],
        })

    payload = {
        "candidateCount": len(CANDIDATES),
        "baselineShort": summary(baseline_short),
        "short": [{"name": n, **s} for n, s, _ in short],
        "survivors": [n for n, _, _ in survivors],
        "baselineFresh": base_f,
        "fresh": [{"name": n, **s} for n, s, _ in fresh],
        "finalists": [n for n, _, _ in finalists],
        "versus": versus,
    }
    print("TOURNAMENT_RESULT=" + json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
