from __future__ import annotations

from dataclasses import replace
import json
import os

import minoflux_ai.heuristic as heuristic
import minoflux_ai.versus_benchmark as versus_benchmark
import minoflux_ai.versus_search as versus_search
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import column_heights, extract_board_features
from minoflux_ai.versus_search import DEFAULT_VERSUS_WEIGHTS, VersusSearchConfig

CANDIDATE = os.environ.get("CANDIDATE", "baseline")
STAGE = os.environ.get("STAGE", "short")
MARKER = 8.000001
CANDIDATE_WEIGHTS = replace(DEFAULT_WEIGHTS, perfect_clear=MARKER)
ORIGINAL_CONTEXT = heuristic._context_score
ORIGINAL_FEATURES = heuristic._placement_features_fast
ORIGINAL_VERSUS_SCORE = versus_search.score_versus_state
ORIGINAL_CHOOSE_VERSUS = versus_search.choose_versus_action
ACTIVE_VERSUS_MODIFIER = False
EXTRA: dict[int, dict[str, float]] = {}

VERSUS_ONLY = {"pending_slot_escape"}


def _occupied_or_wall(board, x: int, y: int) -> bool:
    h = len(board)
    w = len(board[0]) if board else 0
    return x < 0 or x >= w or y < 0 or y >= h or board[y][x] is not None


def _empty(board, x: int, y: int) -> bool:
    h = len(board)
    w = len(board[0]) if board else 0
    return 0 <= x < w and 0 <= y < h and board[y][x] is None


def _slot_metrics(board) -> dict[str, float]:
    if not board:
        return {"slots": 0.0, "open": 0.0, "shallow": 0.0, "kick": 0.0, "accessible": 0.0}
    h = len(board)
    w = len(board[0])
    heights = column_heights(board)
    orientations = (
        ((0, -1), (-1, 0), (0, 0), (1, 0)),
        ((0, -1), (0, 0), (1, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((0, -1), (-1, 0), (0, 0), (0, 1)),
    )
    slots = open_top = shallow = kick = accessible = 0.0
    for py in range(h):
        for px in range(w):
            if board[py][px] is not None:
                continue
            corners = (
                _occupied_or_wall(board, px - 1, py - 1),
                _occupied_or_wall(board, px + 1, py - 1),
                _occupied_or_wall(board, px - 1, py + 1),
                _occupied_or_wall(board, px + 1, py + 1),
            )
            if sum(corners) < 3:
                continue
            if not any(all(_empty(board, px + dx, py + dy) for dx, dy in cells) for cells in orientations):
                continue
            slots += 1.0
            column_top = h - heights[px]
            depth = max(0, py - column_top)
            is_open = all(board[y][px] is None for y in range(0, py + 1))
            side_space = sum(_empty(board, px + dx, py + dy) for dx, dy in ((-1, 0), (1, 0), (-1, -1), (1, -1)))
            open_top += float(is_open)
            shallow_score = 1.0 / (1.0 + depth)
            shallow += shallow_score
            kick_score = min(1.0, side_space / 2.0)
            kick += kick_score
            accessible += float(is_open) * (0.65 + 0.35 * kick_score) * shallow_score
    return {"slots": slots, "open": open_top, "shallow": shallow, "kick": kick, "accessible": accessible}


def _hole_metrics(board) -> dict[str, float]:
    if not board:
        return {"cover": 0.0, "bottom_cover": 0.0, "isolated_peaks": 0.0, "roofed_valleys": 0.0}
    h = len(board)
    w = len(board[0])
    cover = 0.0
    bottom_cover = 0.0
    for x in range(w):
        blocks = 0
        for y in range(h):
            if board[y][x] is not None:
                blocks += 1
            elif blocks:
                cover += blocks
                bottom_weight = 1.0 + y / max(1, h - 1)
                bottom_cover += blocks * bottom_weight
    heights = column_heights(board)
    isolated = 0.0
    valleys = 0.0
    for x, height in enumerate(heights):
        left = heights[x - 1] if x > 0 else height
        right = heights[x + 1] if x + 1 < w else height
        shoulder = min(left, right)
        if 0 < x < w - 1 and height >= max(left, right) + 3:
            isolated += height - max(left, right) - 2
        if 0 < x < w - 1 and shoulder >= height + 4:
            valleys += shoulder - height - 3
    return {"cover": cover, "bottom_cover": bottom_cover, "isolated_peaks": isolated, "roofed_valleys": valleys}


def _board_after(game, placement):
    board = [row.copy() for row in game.board]
    for x, y in placement.cells:
        if y >= 0:
            board[y][x] = placement.piece
    full = {i for i, row in enumerate(board) if all(cell is not None for cell in row)}
    if full:
        board = [[None] * game.width for _ in full] + [row for i, row in enumerate(board) if i not in full]
    return board


def patched_features(game, placement, before):
    result = ORIGINAL_FEATURES(game, placement, before)
    after_board = _board_after(game, placement)
    slots = _slot_metrics(after_board)
    holes = _hole_metrics(after_board)
    EXTRA[id(result)] = {**slots, **holes}
    return result


def _next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def _t_supply(game) -> int:
    return int(game.current == "T") + int(game.hold_piece == "T") + sum(1 for p in list(game.queue)[:6] if p == "T")


def _modifier(game, features) -> float:
    if CANDIDATE in VERSUS_ONLY:
        return 0.0
    extra = EXTRA.get(id(features), {})
    before_slot = _slot_metrics(game.board)
    before_holes = _hole_metrics(game.board)
    accessible = extra.get("accessible", 0.0)
    open_top = extra.get("open", 0.0)
    shallow = extra.get("shallow", 0.0)
    kick = extra.get("kick", 0.0)
    danger = max(0.0, (features.board.max_height - 9) / 7.0)
    urgency = max(0.0, (7.0 - _next_t_distance(game)) / 7.0)
    access_delta = accessible - before_slot["accessible"]
    cover_relief = max(0.0, before_holes["cover"] - extra.get("cover", 0.0))
    bottom_relief = max(0.0, before_holes["bottom_cover"] - extra.get("bottom_cover", 0.0))

    if CANDIDATE == "slot_open_top_access":
        return 0.42 * open_top - 0.18 * max(0.0, extra.get("slots", 0.0) - open_top)
    if CANDIDATE == "slot_shallow_access":
        return 0.46 * shallow
    if CANDIDATE == "slot_kick_space":
        return 0.34 * kick
    if CANDIDATE == "slot_access_combo":
        return 0.62 * accessible
    if CANDIDATE == "slot_entrance_preserve":
        return 0.58 * urgency * access_delta - 0.34 * urgency * max(0.0, -access_delta)
    if CANDIDATE == "hole_cover_relief":
        return 0.10 * min(12.0, cover_relief) - 0.04 * max(0.0, extra.get("cover", 0.0) - before_holes["cover"])
    if CANDIDATE == "bottom_hole_cover_relief":
        return 0.065 * min(18.0, bottom_relief)
    if CANDIDATE == "isolated_peak_risk":
        return -0.38 * extra.get("isolated_peaks", 0.0) * (1.0 + 0.6 * danger)
    if CANDIDATE == "roofed_valley_risk":
        return -0.28 * extra.get("roofed_valleys", 0.0) * (1.0 + 0.5 * danger)
    if CANDIDATE == "danger_accessible_slot":
        return 0.52 * danger * accessible - 0.22 * danger * features.board.holes
    if CANDIDATE == "t_arrival_accessible_conversion":
        if game.current != "T":
            return 0.0
        return 0.36 * features.spin_lines * (1.0 + accessible) + 0.18 * features.attack * accessible - 0.42 * max(0.0, -access_delta)
    if CANDIDATE == "b2b_accessible_reserve":
        if not game.back_to_back:
            return 0.0
        difficult = features.spin_lines > 0 or features.lines == 4
        return 0.38 * accessible + (0.22 if difficult else -0.18) * max(0.0, before_slot["accessible"] - accessible)
    if CANDIDATE == "hold_accessible_supply":
        if not game.hold_used:
            return 0.0
        supply = _t_supply(game)
        matched = min(float(supply), accessible)
        shortage = max(0.0, accessible - supply)
        return 0.30 * matched - 0.36 * shortage
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    if weights.perfect_clear > 8.0000005:
        return base + _modifier(game, features)
    return base


def patched_versus_score(match, root_side, *, weights=DEFAULT_VERSUS_WEIGHTS, resolution=None, solo_score=0.0, path_length=0, action_side=None):
    base = ORIGINAL_VERSUS_SCORE(match, root_side, weights=weights, resolution=resolution, solo_score=solo_score, path_length=path_length, action_side=action_side)
    if not ACTIVE_VERSUS_MODIFIER:
        return base
    own = match.player if root_side == "player" else match.ai
    if CANDIDATE == "pending_slot_escape":
        pending = own.pending.pending_lines
        slots = _slot_metrics(own.game.board)["accessible"]
        board = extract_board_features(own.game.board)
        danger = max(0.0, (board.max_height - 8) / 6.0)
        return base + 0.55 * pending * danger * slots - 0.30 * pending * danger * board.holes
    return base


def patched_choose_versus(match, side, weights, config):
    global ACTIVE_VERSUS_MODIFIER
    previous = ACTIVE_VERSUS_MODIFIER
    ACTIVE_VERSUS_MODIFIER = CANDIDATE in VERSUS_ONLY and weights.perfect_clear > 8.0000005
    try:
        return ORIGINAL_CHOOSE_VERSUS(match, side, weights, config)
    finally:
        ACTIVE_VERSUS_MODIFIER = previous


heuristic._placement_features_fast = patched_features
heuristic._context_score = patched_context
versus_search.score_versus_state = patched_versus_score
versus_benchmark.choose_versus_action = patched_choose_versus
cfg = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.9)
weights = DEFAULT_WEIGHTS if CANDIDATE == "baseline" else CANDIDATE_WEIGHTS

if STAGE == "versus":
    vcfg = VersusSearchConfig(placement_search=cfg, candidate_width=6, opponent_reply_width=1)
    result = versus_benchmark.run_versus_benchmark(games=8, max_turns=120, seed_base=271_337, seed_step=193, player_weights=weights, ai_weights=DEFAULT_WEIGHTS, player_config=vcfg, ai_config=vcfg, garbage_cap=8)
    games = result.per_game
    payload = result.to_dict()
    payload.update({"candidate": CANDIDATE, "stage": STAGE, "playerMeanCanceled": sum(g.player_canceled for g in games) / len(games), "aiMeanCanceled": sum(g.ai_canceled for g in games) / len(games), "playerMeanReceived": sum(g.player_received for g in games) / len(games), "aiMeanReceived": sum(g.ai_received for g in games) / len(games), "playerMeanMaxB2B": sum(g.player_max_b2b for g in games) / len(games), "aiMeanMaxB2B": sum(g.ai_max_b2b for g in games) / len(games), "playerTopouts": result.ai_wins, "aiTopouts": result.player_wins})
else:
    if STAGE == "short":
        games, pieces, seed = 2, 120, 61_303
    elif STAGE == "fresh":
        games, pieces, seed = 3, 280, 94_109
    else:
        raise SystemExit(f"unknown STAGE={STAGE}")
    result = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed, seed_step=97, weights=weights, search_config=cfg, workers=1)
    payload = result.to_dict()
    payload.update({"candidate": CANDIDATE, "stage": STAGE, "attackPerPiece": result.attack / max(1, result.pieces)})

path = os.environ.get("RESULT_PATH", "result.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print("RESULT=" + json.dumps(payload, separators=(",", ":")))
