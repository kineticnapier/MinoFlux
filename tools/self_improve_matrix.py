from __future__ import annotations

from dataclasses import replace
import json
import sys

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


def candidate_weights(name: str):
    if name == "baseline":
        return DEFAULT_WEIGHTS
    index = CANDIDATES.index(name)
    return replace(DEFAULT_WEIGHTS, game_over=BASE_GAME_OVER - (index + 1) * 1e-3)


def candidate_name(weights) -> str | None:
    delta = BASE_GAME_OVER - weights.game_over
    if delta < 5e-4:
        return None
    index = round(delta / 1e-3) - 1
    return CANDIDATES[index] if 0 <= index < len(CANDIDATES) else None


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
    danger = max(0.0, (after.max_height - 8.0) / 8.0)
    t_urgency = max(0.0, (7.0 - next_t_distance(game)) / 7.0)
    difficult = f.spin_lines > 0 or f.lines == 4
    held_t = game.hold_piece == "T"

    if name == "slot_preserve_until_t":
        return -0.70 * t_urgency * slot_loss + 0.10 * t_urgency * slot_gain
    if name == "b2b_break_cost_dynamic":
        return -0.45 * (1.0 + min(4, game.b2b_chain) * 0.20) if game.back_to_back and f.lines > 0 and not difficult else 0.0
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
        return 0.28 * t_urgency * after.t_spin_slots / (1.0 + after.holes + max(0, after.max_height - 6) * 0.25)
    if name == "hold_t_release_pressure":
        return -0.14 * min(2, after.t_spin_slots) if held_t and t_urgency > 0 and after.t_spin_slots > 0 and game.current != "T" else 0.0
    if name == "hold_t_reserve_when_slots":
        return 0.18 * min(2, after.t_spin_slots) / (1.0 + after.holes) if game.hold_used and held_t and after.t_spin_slots > 0 else 0.0
    if name == "slot_supply_near_t_cap":
        supply = (1 if game.current == "T" else 0) + sum(1 for p in list(game.queue)[:6] if p == "T") + (1 if held_t else 0)
        return 0.16 * min(after.t_spin_slots, supply) - 0.20 * max(0, after.t_spin_slots - supply)
    if name == "spin_conversion_efficiency":
        return 0.24 * f.attack + 0.38 * f.spin_lines - 0.08 * after.holes if game.current == "T" and f.spin_lines else 0.0
    if name == "clean_attack_followthrough":
        return 0.08 * f.attack * (1.0 + holes_recovered) - 0.06 * f.attack * danger if f.attack > 0 else 0.0
    if name == "safe_difficult_clear_chain":
        return 0.18 * (1.0 + min(4, game.b2b_chain) * 0.15) / (1.0 + danger + after.holes * 0.15) if difficult else 0.0
    raise AssertionError(name)


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    name = candidate_name(weights)
    return base if name is None else base + extra_score(name, game, features)


heuristic._context_score = patched_context
SOLO = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4, discount=0.9, srs_reachable=True)
VERSUS_CFG = VersusSearchConfig(placement_search=SOLO, candidate_width=6, opponent_reply_width=1)


def solo_summary(name: str, stage: str):
    if stage == "short":
        games, pieces, seed, step = 2, 80, 31001, 31
    elif stage == "fresh":
        games, pieces, seed, step = 3, 160, 731003, 97
    else:
        raise ValueError(stage)
    result = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed, seed_step=step, weights=candidate_weights(name), search_config=SOLO, workers=1)
    return {
        "stage": stage, "name": name, "pieces": result.pieces, "attack": result.attack,
        "app": result.attack / max(1, result.pieces), "topouts": result.topouts,
        "completed": result.completed, "spins": result.spins, "spinLines": result.spin_lines,
        "tsd": result.t_spin_doubles, "tst": result.t_spin_triples,
        "meanHoles": result.mean_holes, "meanHoleDepth": result.mean_hole_depth,
        "meanMaxHeight": result.mean_max_height,
    }


def versus_summary(name: str):
    result = run_versus_benchmark(games=6, max_turns=90, seed_base=12000052, seed_step=193, player_weights=candidate_weights(name), ai_weights=DEFAULT_WEIGHTS, player_config=VERSUS_CFG, ai_config=VERSUS_CFG, garbage_cap=8)
    return {
        "stage": "versus", "name": name, "wins": result.player_wins, "losses": result.ai_wins,
        "draws": result.draws, "attack": [result.player_mean_attack, result.ai_mean_attack],
        "sent": [result.player_mean_sent, result.ai_mean_sent],
        "canceled": [sum(g.player_canceled for g in result.per_game)/result.games, sum(g.ai_canceled for g in result.per_game)/result.games],
        "received": [sum(g.player_received for g in result.per_game)/result.games, sum(g.ai_received for g in result.per_game)/result.games],
        "maxB2B": [sum(g.player_max_b2b for g in result.per_game)/result.games, sum(g.ai_max_b2b for g in result.per_game)/result.games],
        "topouts": [sum(g.winner == "ai" for g in result.per_game), sum(g.winner == "player" for g in result.per_game)],
    }


def main():
    stage, name = sys.argv[1], sys.argv[2]
    if name != "baseline" and name not in CANDIDATES:
        raise ValueError(name)
    payload = versus_summary(name) if stage == "versus" else solo_summary(name, stage)
    print("EVAL_RESULT=" + json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
