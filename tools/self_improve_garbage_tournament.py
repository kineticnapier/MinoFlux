from __future__ import annotations

from dataclasses import replace
from heapq import nlargest
import json
import math
from pathlib import Path
import sys

import minoflux_ai.heuristic as heuristic
import minoflux_ai.search as search
from minoflux_ai import run_heuristic_benchmark, run_versus_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_search import VersusSearchConfig

CANDIDATES = (
    "garbage_profile_min",
    "garbage_profile_mean",
    "garbage_profile_variance",
    "garbage_tspin_recovery",
    "garbage_hole_access",
    "garbage_b2b_recovery",
    "t_near_garbage_ready",
    "attack_resilient_exit",
    "spin_resilient_exit",
    "danger_attack_cancel_proxy",
    "adaptive_spike_2_4_6",
    "worst_spike_2_4_6",
    "hold_garbage_flex",
    "garbage_clean_setup_balance",
)

ACTIVE = "baseline"
CANDIDATE_WEIGHTS = replace(heuristic.DEFAULT_WEIGHTS)
ORIGINAL_RANK = search.rank_placements


def _post_board(game, placement):
    board = [row.copy() for row in game.board]
    for x, y in placement.cells:
        if y >= 0:
            board[y][x] = placement.piece
    full = {i for i, row in enumerate(board) if all(cell is not None for cell in row)}
    if full:
        board = [[None] * game.width for _ in full] + [row for i, row in enumerate(board) if i not in full]
    return board


def _stressed(board, width, hole, lines):
    value = [row.copy() for row in board]
    hole = min(width - 1, max(0, int(hole)))
    for _ in range(lines):
        value.pop(0)
        row = ["G"] * width
        row[hole] = None
        value.append(row)
    return extract_board_features(value)


def _quality(f):
    return -(0.080 * f.holes + 0.020 * f.hole_depth + 0.030 * f.max_height)


def _next_t_distance(game):
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def _extra(name, game, evaluation):
    f = evaluation.features
    board = _post_board(game, evaluation.placement)
    holes = (0, 3, 6, 9)
    profiles = [_stressed(board, game.width, hole, 4) for hole in holes]
    qualities = [_quality(item) for item in profiles]
    mean_q = sum(qualities) / len(qualities)
    worst_q = min(qualities)
    variance = sum((q - mean_q) ** 2 for q in qualities) / len(qualities)
    stressed_slots = sum(item.t_spin_slots / (1.0 + item.holes + item.max_height / 6.0) for item in profiles) / len(profiles)
    mean_holes = sum(item.holes for item in profiles) / len(profiles)
    mean_depth = sum(item.hole_depth for item in profiles) / len(profiles)
    mean_height = sum(item.max_height for item in profiles) / len(profiles)
    safety = 1.0 / (1.0 + mean_holes / 4.0 + mean_depth / 24.0 + mean_height / 12.0)
    urgency = max(0.0, (7.0 - _next_t_distance(game)) / 7.0)

    if name == "garbage_profile_min":
        return 0.55 * worst_q
    if name == "garbage_profile_mean":
        return 0.55 * mean_q
    if name == "garbage_profile_variance":
        return 0.35 * mean_q - 0.80 * variance
    if name == "garbage_tspin_recovery":
        return 1.10 * stressed_slots + 0.20 * mean_q
    if name == "garbage_hole_access":
        return -0.030 * mean_depth - 0.060 * mean_holes
    if name == "garbage_b2b_recovery":
        difficult = bool(f.spin is not None or f.lines == 4)
        return (0.45 * mean_q + 0.35 * int(difficult) * safety) if game.back_to_back else 0.15 * mean_q
    if name == "t_near_garbage_ready":
        return 1.25 * urgency * stressed_slots + 0.15 * mean_q
    if name == "attack_resilient_exit":
        return 0.75 * f.attack * safety + 0.12 * mean_q
    if name == "spin_resilient_exit":
        return 1.10 * f.spin_lines * safety + 0.10 * mean_q
    if name == "danger_attack_cancel_proxy":
        danger = max(0.0, (f.board.max_height - 9.0) / 7.0)
        return 0.40 * f.attack * danger - 0.18 * f.new_holes * (1.0 + danger)
    if name == "adaptive_spike_2_4_6":
        center = max(0, game.width // 2 - 1)
        qs = [_quality(_stressed(board, game.width, center, n)) for n in (2, 4, 6)]
        return 0.50 * sum(qs) / 3.0
    if name == "worst_spike_2_4_6":
        center = max(0, game.width // 2 - 1)
        qs = [_quality(_stressed(board, game.width, center, n)) for n in (2, 4, 6)]
        return 0.60 * min(qs)
    if name == "hold_garbage_flex":
        if not game.hold_used:
            return 0.0
        held_t = 1.0 if game.hold_piece == "T" else 0.0
        return 0.75 * stressed_slots + 0.25 * held_t * safety + 0.10 * f.attack * safety
    if name == "garbage_clean_setup_balance":
        return 1.00 * stressed_slots - 0.035 * mean_depth - 0.040 * mean_holes
    return 0.0


def _key(e):
    return (
        e.score,
        e.features.attack,
        e.features.spin_lines,
        e.features.lines,
        -e.features.board.holes,
        -e.features.board.max_height,
        -e.placement.rotation,
        -e.placement.x,
    )


def _rank(game, weights=heuristic.DEFAULT_WEIGHTS, *, placements=None, limit=None):
    values = ORIGINAL_RANK(game, weights, placements=placements, limit=None)
    if weights is CANDIDATE_WEIGHTS and ACTIVE != "baseline":
        values = tuple(replace(e, score=e.score + _extra(ACTIVE, game, e)) for e in values)
    if limit is not None:
        count = max(0, int(limit))
        if count == 0:
            return ()
        if count < len(values):
            return tuple(nlargest(count, values, key=_key))
    return tuple(sorted(values, key=_key, reverse=True))


search.rank_placements = _rank

SOLO_CFG = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.90)
VERSUS_CFG = VersusSearchConfig(
    placement_search=SOLO_CFG,
    candidate_width=6,
    opponent_reply_width=1,
)


def _solo(name, *, games, pieces, seed):
    global ACTIVE
    ACTIVE = name
    weights = heuristic.DEFAULT_WEIGHTS if name == "baseline" else CANDIDATE_WEIGHTS
    r = run_heuristic_benchmark(
        games=games,
        max_pieces=pieces,
        seed_base=seed,
        seed_step=97,
        weights=weights,
        search_config=SOLO_CFG,
        workers=1,
    )
    return {
        "name": name,
        "pieces": r.pieces,
        "attack": r.attack,
        "app": r.attack / max(1, r.pieces),
        "topouts": r.topouts,
        "completed": r.completed,
        "spin_lines": r.spin_lines,
        "t_spin_minis": r.t_spin_minis,
        "t_spin_mini_singles": r.t_spin_mini_singles,
        "t_spin_singles": r.t_spin_singles,
        "t_spin_doubles": r.t_spin_doubles,
        "t_spin_triples": r.t_spin_triples,
        "mean_holes": r.mean_holes,
        "mean_hole_depth": r.mean_hole_depth,
        "mean_max_height": r.mean_max_height,
        "high_stack_fraction": r.high_stack_fraction,
    }


def _solo_key(item):
    return (
        item["completed"],
        -item["topouts"],
        item["app"],
        item["spin_lines"],
        item["t_spin_doubles"],
        item["pieces"],
    )


def _versus(name):
    global ACTIVE
    ACTIVE = name
    r = run_versus_benchmark(
        games=6,
        max_turns=120,
        seed_base=880301,
        seed_step=131,
        player_weights=CANDIDATE_WEIGHTS,
        ai_weights=heuristic.DEFAULT_WEIGHTS,
        player_config=VERSUS_CFG,
        ai_config=VERSUS_CFG,
        garbage_cap=8,
    )
    games = r.per_game
    return {
        "name": name,
        "wins": r.player_wins,
        "losses": r.ai_wins,
        "draws": r.draws,
        "attack": r.player_mean_attack,
        "baseline_attack": r.ai_mean_attack,
        "sent": r.player_mean_sent,
        "baseline_sent": r.ai_mean_sent,
        "canceled": sum(g.player_canceled for g in games) / len(games),
        "baseline_canceled": sum(g.ai_canceled for g in games) / len(games),
        "received": sum(g.player_received for g in games) / len(games),
        "baseline_received": sum(g.ai_received for g in games) / len(games),
        "max_b2b": sum(g.player_max_b2b for g in games) / len(games),
        "baseline_max_b2b": sum(g.ai_max_b2b for g in games) / len(games),
        "topouts": r.ai_wins,
        "baseline_topouts": r.player_wins,
    }


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "tournament-result.json")
    short = [_solo("baseline", games=2, pieces=100, seed=771001)]
    for name in CANDIDATES:
        short.append(_solo(name, games=2, pieces=100, seed=771001))
    baseline_short = short[0]
    eligible = [
        item for item in short[1:]
        if item["topouts"] <= baseline_short["topouts"]
        and item["completed"] >= baseline_short["completed"]
    ]
    top3 = sorted(eligible, key=_solo_key, reverse=True)[:3]

    fresh = [_solo("baseline", games=3, pieces=220, seed=991003)]
    for item in top3:
        fresh.append(_solo(item["name"], games=3, pieces=220, seed=991003))
    baseline_fresh = fresh[0]
    finalists = [
        item for item in fresh[1:]
        if item["topouts"] <= baseline_fresh["topouts"]
        and item["completed"] >= baseline_fresh["completed"]
        and item["app"] >= baseline_fresh["app"]
    ]
    finalists = sorted(finalists, key=_solo_key, reverse=True)[:2]
    versus = [_versus(item["name"]) for item in finalists]
    payload = {"candidateCount": len(CANDIDATES), "short": short, "top3": [x["name"] for x in top3], "fresh": fresh, "finalists": [x["name"] for x in finalists], "versus": versus}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("RESULT_JSON=" + json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
