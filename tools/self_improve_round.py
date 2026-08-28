from __future__ import annotations

from dataclasses import replace
import json
import os

import minoflux_ai.heuristic as heuristic
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import extract_board_features

CANDIDATE = os.environ.get("CANDIDATE", "baseline")
STAGE = os.environ.get("STAGE", "short")

ORIGINAL_CONTEXT = heuristic._context_score
MARKER = 8.000001
CANDIDATE_WEIGHTS = replace(DEFAULT_WEIGHTS, perfect_clear=MARKER)


def _next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def _modifier(game, features) -> float:
    before = extract_board_features(game.board)
    after = features.board
    hole_relief = max(0, before.holes - after.holes)
    depth_relief = max(0, before.hole_depth - after.hole_depth)
    height_relief = max(0, before.max_height - after.max_height)
    aggregate_relief = max(0, before.aggregate_height - after.aggregate_height)
    slot_kept = min(before.t_spin_slots, after.t_spin_slots)
    slot_created = max(0, after.t_spin_slots - before.t_spin_slots)
    t_distance = _next_t_distance(game)
    t_urgency = max(0.0, (7.0 - t_distance) / 7.0)
    clean = 1.0 / (1.0 + after.holes + after.hole_depth / 8.0)
    danger = max(0.0, after.max_height - 10.0) / 10.0
    difficult = features.spin_lines > 0 or features.lines == 4

    if CANDIDATE == "clear_depth_relief":
        return 0.18 * features.lines * min(8.0, depth_relief)
    if CANDIDATE == "clear_hole_access":
        return 0.50 * features.lines * hole_relief + 0.08 * features.lines * min(6.0, depth_relief)
    if CANDIDATE == "clear_compaction":
        return 0.035 * features.lines * min(20.0, aggregate_relief)
    if CANDIDATE == "attack_clean_exit":
        return 0.22 * features.attack * clean
    if CANDIDATE == "attack_height_recovery":
        return 0.16 * min(4, features.attack) * min(4, height_relief)
    if CANDIDATE == "attack_slot_exit":
        return 0.18 * min(4, features.attack) * slot_kept * clean
    if CANDIDATE == "t_near_depth_guard":
        return 0.45 * t_urgency * min(2, after.t_spin_slots) / (1.0 + after.hole_depth / 6.0)
    if CANDIDATE == "t_near_clear_path":
        return t_urgency * (0.35 * hole_relief + 0.06 * min(8.0, depth_relief)) * max(1, after.t_spin_slots)
    if CANDIDATE == "slot_creation_escape":
        return 0.38 * slot_created / (1.0 + after.holes + danger * 3.0)
    if CANDIDATE == "slot_retention_downstack":
        return 0.30 * slot_kept * (hole_relief + min(1.5, depth_relief / 8.0))
    if CANDIDATE == "b2b_clean_difficult":
        return (0.42 if game.back_to_back and difficult else 0.0) * (1.0 + clean)
    if CANDIDATE == "b2b_break_damage":
        if game.back_to_back and features.lines > 0 and not difficult:
            return -0.35 * (1.0 + after.holes / 3.0 + danger)
        return 0.0
    if CANDIDATE == "t_arrival_downstack":
        if game.current == "T" and features.spin_lines > 0:
            return 0.16 * min(8.0, depth_relief) + 0.30 * hole_relief
        return 0.0
    if CANDIDATE == "hold_t_stack_buffer":
        if game.hold_piece == "T" and game.current != "T":
            return -0.14 * max(0, after.max_height - 11) - 0.10 * max(0, after.holes - before.holes)
        return 0.0
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    if weights.perfect_clear > 8.0000005:
        return base + _modifier(game, features)
    return base


heuristic._context_score = patched_context

if STAGE == "short":
    games, pieces, seed = 2, 120, 41001
elif STAGE == "fresh":
    games, pieces, seed = 3, 260, 73117
else:
    raise SystemExit(f"unknown STAGE={STAGE}")

cfg = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.9)
weights = DEFAULT_WEIGHTS if CANDIDATE == "baseline" else CANDIDATE_WEIGHTS
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
payload["candidate"] = CANDIDATE
payload["stage"] = STAGE
payload["attackPerPiece"] = result.attack / max(1, result.pieces)
print("RESULT=" + json.dumps(payload, separators=(",", ":")))
