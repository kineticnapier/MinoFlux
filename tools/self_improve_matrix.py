from __future__ import annotations

from dataclasses import replace
import json
import sys

import minoflux_ai.heuristic as heuristic
import minoflux_ai.versus_benchmark as versus_benchmark
import minoflux_ai.versus_search as versus_search
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_search import VersusSearchConfig

ORIGINAL_CONTEXT = heuristic._context_score
ORIGINAL_VERSUS_SCORE = versus_search.score_versus_state
ORIGINAL_CHOOSE_VERSUS = versus_search.choose_versus_action
BASE_GAME_OVER = DEFAULT_WEIGHTS.game_over
ACTIVE_NAME = "baseline"
APPLY_VERSUS_EXTRA = False

CANDIDATES = (
    "t_window_reserve",
    "slot_survival_margin",
    "attack_reserve_margin",
    "b2b_clean_continuation",
    "downstack_access_gain",
    "deep_hole_emergency",
    "clean_slot_after_downstack",
    "slot_attack_balance",
    "t_supply_safety_margin",
    "spin_exit_reserve",
    "height_recovery_with_slot",
    "hole_depth_recovery_attack",
    "net_pending_advantage",
    "attack_race_margin",
)
VERSUS_ONLY = {"net_pending_advantage", "attack_race_margin"}


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


def t_supply(game) -> int:
    count = int(game.current == "T") + int(game.hold_piece == "T")
    count += sum(piece == "T" for i, piece in enumerate(game.queue) if i < 6)
    return count


def extra_context_score(name: str, game, f) -> float:
    if name in VERSUS_ONLY:
        return 0.0
    before = extract_board_features(game.board)
    after = f.board
    slots = after.t_spin_slots
    holes_recovered = max(0.0, before.holes - after.holes)
    depth_recovered = max(0.0, before.hole_depth - after.hole_depth)
    height_recovered = max(0.0, before.max_height - after.max_height)
    tdist = next_t_distance(game)
    urgency = max(0.0, (4.0 - tdist) / 4.0)
    reserve = max(0.0, 12.0 - after.max_height) / 12.0
    difficult = f.spin_lines > 0 or f.lines == 4

    if name == "t_window_reserve":
        return 0.52 * urgency * min(2, slots) * reserve / (1.0 + 0.3 * after.holes + 0.05 * after.hole_depth)
    if name == "slot_survival_margin":
        return 0.42 * min(2, slots) * reserve / (1.0 + after.hole_depth / 10.0)
    if name == "attack_reserve_margin":
        return 0.10 * f.attack * reserve / (1.0 + 0.25 * after.holes)
    if name == "b2b_clean_continuation":
        if not difficult or not game.back_to_back:
            return 0.0
        return 0.34 * (1.0 + min(2, slots)) / (1.0 + after.holes + after.max_height / 10.0)
    if name == "downstack_access_gain":
        return 0.11 * f.lines * depth_recovered / (1.0 + 0.5 * f.new_holes + after.max_height / 12.0)
    if name == "deep_hole_emergency":
        danger = max(0.0, (after.max_height - 9.0) / 7.0)
        return -0.0045 * danger * after.hole_depth * after.hole_depth
    if name == "clean_slot_after_downstack":
        return 0.18 * min(2, slots) * (holes_recovered + 0.12 * depth_recovered) / (1.0 + after.max_height / 10.0)
    if name == "slot_attack_balance":
        return 0.09 * f.attack * min(2, slots) / (1.0 + after.holes + after.max_height / 8.0)
    if name == "t_supply_safety_margin":
        supply_gap = abs(slots - t_supply(game))
        return 0.26 * reserve * min(slots, t_supply(game)) - 0.10 * supply_gap * (1.0 - reserve)
    if name == "spin_exit_reserve":
        if f.spin_lines <= 0:
            return 0.0
        return 0.24 * f.spin_lines * reserve / (1.0 + 0.3 * after.holes + after.hole_depth / 18.0)
    if name == "height_recovery_with_slot":
        return 0.16 * height_recovered * (1.0 + 0.5 * min(2, slots))
    if name == "hole_depth_recovery_attack":
        return 0.018 * depth_recovered * (1.0 + 0.35 * f.attack)
    raise AssertionError(name)


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    name = candidate_name(weights)
    return base if name is None else base + extra_context_score(name, game, features)


def _versus_sides(match, root_side):
    return (match.player, match.ai) if root_side == "player" else (match.ai, match.player)


def patched_versus_score(match, root_side, **kwargs):
    base = ORIGINAL_VERSUS_SCORE(match, root_side, **kwargs)
    if not APPLY_VERSUS_EXTRA or ACTIVE_NAME not in VERSUS_ONLY or abs(base) >= 900_000:
        return base
    own, opponent = _versus_sides(match, root_side)
    own_board = extract_board_features(own.game.board)
    opp_board = extract_board_features(opponent.game.board)
    own_pending = own.pending.pending_lines
    opp_pending = opponent.pending.pending_lines
    if ACTIVE_NAME == "net_pending_advantage":
        safety = max(0.0, 12.0 - own_board.max_height) / 12.0
        return base + 0.22 * (opp_pending - own_pending) * (0.5 + safety)
    if ACTIVE_NAME == "attack_race_margin":
        pressure = opp_pending * max(0.0, opp_board.max_height - 6.0)
        danger = own_pending * max(0.0, own_board.max_height - 6.0)
        return base + 0.16 * (pressure - danger)
    return base


def patched_choose_versus(match, side_name, heuristic_weights=DEFAULT_WEIGHTS, config=versus_search.DEFAULT_VERSUS_SEARCH_CONFIG, versus_weights=versus_search.DEFAULT_VERSUS_WEIGHTS):
    global APPLY_VERSUS_EXTRA
    previous = APPLY_VERSUS_EXTRA
    APPLY_VERSUS_EXTRA = candidate_name(heuristic_weights) == ACTIVE_NAME and ACTIVE_NAME in VERSUS_ONLY
    try:
        return ORIGINAL_CHOOSE_VERSUS(match, side_name, heuristic_weights, config, versus_weights)
    finally:
        APPLY_VERSUS_EXTRA = previous


heuristic._context_score = patched_context
versus_search.score_versus_state = patched_versus_score
versus_search.choose_versus_action = patched_choose_versus
versus_benchmark.choose_versus_action = patched_choose_versus

SOLO = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4, discount=0.9, srs_reachable=True)
VERSUS_CFG = VersusSearchConfig(placement_search=SOLO, candidate_width=6, opponent_reply_width=1)


def solo_summary(name: str, stage: str):
    global ACTIVE_NAME
    ACTIVE_NAME = name
    if stage == "short":
        games, pieces, seed, step = 2, 100, 51017, 31
    elif stage == "fresh":
        games, pieces, seed, step = 3, 260, 1141019, 97
    else:
        raise ValueError(stage)
    result = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed, seed_step=step, weights=candidate_weights(name), search_config=SOLO, workers=1)
    return {
        "stage": stage, "name": name, "pieces": result.pieces, "attack": result.attack,
        "app": result.attack / max(1, result.pieces), "topouts": result.topouts,
        "completed": result.completed, "spins": result.spins, "spinLines": result.spin_lines,
        "tsd": result.t_spin_doubles, "tst": result.t_spin_triples,
        "meanHoles": result.mean_holes, "meanHoleDepth": result.mean_hole_depth,
        "meanBumpiness": result.mean_bumpiness, "meanMaxHeight": result.mean_max_height,
    }


def versus_summary(name: str):
    global ACTIVE_NAME
    ACTIVE_NAME = name
    result = run_versus_benchmark(games=8, max_turns=120, seed_base=16000063, seed_step=193, player_weights=candidate_weights(name), ai_weights=DEFAULT_WEIGHTS, player_config=VERSUS_CFG, ai_config=VERSUS_CFG, garbage_cap=8)
    return {
        "stage": "versus", "name": name, "wins": result.player_wins, "losses": result.ai_wins, "draws": result.draws,
        "attack": [result.player_mean_attack, result.ai_mean_attack], "sent": [result.player_mean_sent, result.ai_mean_sent],
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
