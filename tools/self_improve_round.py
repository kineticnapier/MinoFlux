from __future__ import annotations

from dataclasses import replace
import json
import os

import minoflux_ai.heuristic as heuristic
import minoflux_ai.versus_benchmark as versus_benchmark
import minoflux_ai.versus_search as versus_search
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.versus_search import DEFAULT_VERSUS_WEIGHTS, VersusSearchConfig

CANDIDATE = os.environ.get("CANDIDATE", "baseline")
STAGE = os.environ.get("STAGE", "short")
ORIGINAL_CONTEXT = heuristic._context_score
ORIGINAL_VERSUS_SCORE = versus_search.score_versus_state
ORIGINAL_CHOOSE_VERSUS = versus_search.choose_versus_action
MARKER = 8.000001
CANDIDATE_WEIGHTS = replace(DEFAULT_WEIGHTS, perfect_clear=MARKER)
ACTIVE_VERSUS_MODIFIER = False

VERSUS_ONLY = {
    "pending_height_coupling",
    "pending_hole_coupling",
    "ko_sent_pressure",
    "cancel_survival_margin",
}


def _next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 5:
            break
    return 7


def _t_supply(game) -> int:
    count = int(game.current == "T") + int(game.hold_piece == "T")
    count += sum(1 for piece in list(game.queue)[:6] if piece == "T")
    return count


def _modifier(game, features) -> float:
    if CANDIDATE in VERSUS_ONLY:
        return 0.0
    before = extract_board_features(game.board)
    after = features.board
    holes_added = max(0, after.holes - before.holes)
    hole_relief = max(0, before.holes - after.holes)
    depth_relief = max(0.0, before.hole_depth - after.hole_depth)
    height_relief = max(0, before.max_height - after.max_height)
    slot_kept = min(before.t_spin_slots, after.t_spin_slots)
    slot_loss = max(0, before.t_spin_slots - after.t_spin_slots)
    tdist = _next_t_distance(game)
    urgency = max(0.0, (7.0 - tdist) / 7.0)
    far_t = min(1.0, max(0, tdist - 2) / 4.0)
    supply = _t_supply(game)
    clean = 1.0 / (1.0 + after.holes + after.hole_depth / 8.0)
    low = 1.0 / (1.0 + after.max_height / 8.0)
    difficult = features.spin_lines > 0 or features.lines == 4

    if CANDIDATE == "emergency_slot_liquidation":
        danger = max(0.0, (before.max_height - 11) / 5.0)
        return 0.42 * danger * far_t * slot_loss * (height_relief + min(2.0, depth_relief / 8.0)) - 0.28 * danger * far_t * slot_kept
    if CANDIDATE == "setup_relief_synergy":
        relief = min(2.0, depth_relief / 8.0) + min(2, height_relief) + hole_relief
        return 0.18 * slot_kept * relief - 0.22 * slot_loss * max(0, before.max_height - 9)
    if CANDIDATE == "imminent_t_depth_guard":
        depth_per_hole = after.hole_depth / max(1, after.holes)
        return 0.52 * urgency * min(2, after.t_spin_slots) / (1.0 + depth_per_hole / 5.0) - 0.18 * urgency * holes_added
    if CANDIDATE == "attack_slot_option":
        option = min(after.t_spin_slots, max(1, supply))
        return 0.14 * min(6, features.attack) * option * clean - 0.18 * features.attack * slot_loss
    if CANDIDATE == "downstack_combo_relief":
        if game.combo < 0 or features.lines == 0:
            return 0.0
        relief = hole_relief + min(2.0, depth_relief / 8.0) + 0.5 * min(2, height_relief)
        return 0.22 * (1 + min(3, game.combo + 1)) * relief
    if CANDIDATE == "difficult_depth_exit":
        if not difficult:
            return 0.0
        return 0.22 * min(3.0, depth_relief / 6.0) + 0.18 * hole_relief + 0.12 * height_relief - 0.20 * holes_added
    if CANDIDATE == "t_arrival_stack_reset":
        if game.current != "T":
            return 0.0
        if features.spin_lines > 0:
            return 0.24 * features.attack + 0.18 * height_relief + 0.16 * hole_relief + 0.08 * min(3.0, depth_relief / 6.0)
        return 0.16 * height_relief + 0.12 * hole_relief - 0.34 * slot_loss
    if CANDIDATE == "slot_single_supply_low":
        target = min(1, supply)
        exact = 1.0 if after.t_spin_slots == target else -0.22 * abs(after.t_spin_slots - target)
        return 0.30 * urgency * exact * low * (1.0 + clean)
    if CANDIDATE == "setup_b2b_reentry_clean":
        if game.back_to_back or not difficult:
            return 0.0
        return 0.34 * (1 + slot_kept) * clean * low - 0.24 * holes_added
    if CANDIDATE == "hold_escape_clean":
        if not game.hold_used:
            return 0.0
        relief = hole_relief + min(2, height_relief) + min(2.0, depth_relief / 8.0)
        held_t_factor = 0.75 if game.hold_piece == "T" and after.t_spin_slots > 0 else 1.0
        return 0.18 * relief * held_t_factor - 0.14 * holes_added
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    if weights.perfect_clear > 8.0000005:
        return base + _modifier(game, features)
    return base


def patched_versus_score(match, root_side, *, weights=DEFAULT_VERSUS_WEIGHTS, resolution=None, solo_score=0.0, path_length=0, action_side=None):
    base = ORIGINAL_VERSUS_SCORE(
        match,
        root_side,
        weights=weights,
        resolution=resolution,
        solo_score=solo_score,
        path_length=path_length,
        action_side=action_side,
    )
    if not ACTIVE_VERSUS_MODIFIER:
        return base
    own = match.player if root_side == "player" else match.ai
    opp = match.ai if root_side == "player" else match.player
    own_board = extract_board_features(own.game.board)
    opp_board = extract_board_features(opp.game.board)
    if CANDIDATE == "pending_height_coupling":
        return base - 0.55 * own.pending.pending_lines * max(0, own_board.max_height - 8)
    if CANDIDATE == "pending_hole_coupling":
        return base - 0.70 * own.pending.pending_lines * own_board.holes
    if CANDIDATE == "ko_sent_pressure" and resolution is not None:
        direction = 1.0 if resolution.side == root_side else -1.0
        return base + direction * 0.55 * resolution.sent_lines * max(0, opp_board.max_height - 9)
    if CANDIDATE == "cancel_survival_margin" and resolution is not None:
        direction = 1.0 if resolution.side == root_side else -1.0
        danger = max(0, own_board.max_height - 9) + 0.5 * own_board.holes
        return base + direction * 0.65 * resolution.canceled_lines * danger
    return base


def patched_choose_versus(match, side, weights, config):
    global ACTIVE_VERSUS_MODIFIER
    previous = ACTIVE_VERSUS_MODIFIER
    ACTIVE_VERSUS_MODIFIER = CANDIDATE in VERSUS_ONLY and weights.perfect_clear > 8.0000005
    try:
        return ORIGINAL_CHOOSE_VERSUS(match, side, weights, config)
    finally:
        ACTIVE_VERSUS_MODIFIER = previous


heuristic._context_score = patched_context
versus_search.score_versus_state = patched_versus_score
versus_benchmark.choose_versus_action = patched_choose_versus
cfg = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.9)
weights = DEFAULT_WEIGHTS if CANDIDATE == "baseline" else CANDIDATE_WEIGHTS

if STAGE == "versus":
    vcfg = VersusSearchConfig(placement_search=cfg, candidate_width=6, opponent_reply_width=1)
    result = versus_benchmark.run_versus_benchmark(
        games=8,
        max_turns=120,
        seed_base=113_971,
        seed_step=193,
        player_weights=weights,
        ai_weights=DEFAULT_WEIGHTS,
        player_config=vcfg,
        ai_config=vcfg,
        garbage_cap=8,
    )
    match_games = result.per_game
    payload = result.to_dict()
    payload.update({
        "candidate": CANDIDATE,
        "stage": STAGE,
        "playerMeanCanceled": sum(g.player_canceled for g in match_games) / len(match_games),
        "aiMeanCanceled": sum(g.ai_canceled for g in match_games) / len(match_games),
        "playerMeanReceived": sum(g.player_received for g in match_games) / len(match_games),
        "aiMeanReceived": sum(g.ai_received for g in match_games) / len(match_games),
        "playerMeanMaxB2B": sum(g.player_max_b2b for g in match_games) / len(match_games),
        "aiMeanMaxB2B": sum(g.ai_max_b2b for g in match_games) / len(match_games),
        "playerTopouts": result.ai_wins,
        "aiTopouts": result.player_wins,
    })
    print("RESULT=" + json.dumps(payload, separators=(",", ":")))
    raise SystemExit

if STAGE == "short":
    games, pieces, seed = 2, 120, 53_113
elif STAGE == "fresh":
    games, pieces, seed = 3, 280, 88_031
else:
    raise SystemExit(f"unknown STAGE={STAGE}")
result = run_heuristic_benchmark(
    games=games,
    max_pieces=pieces,
    seed_base=seed,
    seed_step=97,
    weights=weights,
    search_config=cfg,
    workers=1,
)
payload = result.to_dict()
payload.update({"candidate": CANDIDATE, "stage": STAGE, "attackPerPiece": result.attack / max(1, result.pieces)})
print("RESULT=" + json.dumps(payload, separators=(",", ":")))
