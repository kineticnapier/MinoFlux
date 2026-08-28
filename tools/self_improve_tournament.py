from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

from minoflux_ai import HeuristicWeights, SearchConfig, run_heuristic_game
from minoflux_ai.features import extract_board_features
from minoflux_ai import heuristic as heuristic_mod
from minoflux_ai import search as search_mod
from minoflux_ai import versus_search as versus_search_mod
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_search import VersusSearchConfig

BASE = HeuristicWeights()
MARKER = 1e-6
CANDIDATE_WEIGHTS = replace(BASE, perfect_clear=BASE.perfect_clear + MARKER)
ORIGINAL_RANK = heuristic_mod.rank_placements


def next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for i, p in enumerate(game.queue):
        if p == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def candidate_bonus(name, game, ev, before) -> float:
    f = ev.features
    b = f.board
    slot_loss = max(0, -f.t_spin_slot_delta)
    slot_gain = max(0, f.t_spin_slot_delta)
    hole_drop = max(0, before.holes - b.holes)
    depth_drop = max(0, before.hole_depth - b.hole_depth)
    height_drop = max(0, before.max_height - b.max_height)
    tdist = next_t_distance(game)
    tnear = max(0.0, (6.0 - tdist) / 6.0)
    spin_clear = f.spin_lines > 0
    difficult_context = game.back_to_back or game.b2b_chain > 0

    if name == "baseline":
        return 0.0
    if name == "slot_clean_exit":
        return 0.55 * slot_gain / (1 + b.holes) + 0.18 * hole_drop
    if name == "slot_rebuild_after_spin":
        return (0.48 * b.t_spin_slots / (1 + b.holes)) if spin_clear else 0.0
    if name == "t_arrival_clean_conversion":
        if game.current != "T" or not spin_clear:
            return 0.0
        return 0.22 * depth_drop + 0.20 * height_drop + 0.30 * b.t_spin_slots
    if name == "t_near_preserve_exit":
        return -0.72 * tnear * slot_loss + 0.18 * tnear * hole_drop
    if name == "non_t_setup_preservation":
        return -0.52 * tnear * slot_loss if game.current != "T" else 0.0
    if name == "b2b_tspin_conversion":
        return 0.52 * f.spin_lines + 0.10 * f.attack if spin_clear and difficult_context else 0.0
    if name == "b2b_break_safety":
        if not difficult_context or f.lines == 0:
            return 0.0
        return 0.28 * f.spin_lines + 0.14 * hole_drop - 0.18 * max(0, b.holes - before.holes)
    if name == "downstack_hole_recovery":
        return (0.24 * hole_drop + 0.035 * depth_drop) * (1.0 + min(1.0, before.max_height / 12.0))
    if name == "downstack_attack_recovery":
        return 0.10 * f.attack * hole_drop + 0.025 * depth_drop
    if name == "danger_clean_clear":
        danger = max(0.0, (before.max_height - 9) / 8.0)
        return danger * (0.20 * f.lines + 0.22 * hole_drop + 0.025 * depth_drop)
    if name == "danger_height_escape":
        danger = max(0.0, (before.max_height - 10) / 7.0)
        return danger * (0.16 * height_drop + 0.12 * hole_drop)
    if name == "hold_t_clean_reserve":
        held_t = game.hold_piece == "T"
        return (0.22 * min(2, b.t_spin_slots) / (1 + b.holes)) if held_t else 0.0
    if name == "t_window_clean_capacity":
        capacity = min(2, b.t_spin_slots)
        return 0.30 * tnear * capacity / (1 + b.holes) - 0.20 * tnear * max(0, b.t_spin_slots - 2)
    if name == "spin_downstack_followthrough":
        return (0.12 * depth_drop + 0.20 * hole_drop + 0.12 * height_drop) if spin_clear else 0.0
    raise ValueError(name)


def custom_rank(game, weights=BASE, *, placements=None, limit=None):
    ranked = ORIGINAL_RANK(game, weights, placements=placements, limit=None)
    if abs(weights.perfect_clear - (BASE.perfect_clear + MARKER)) > 1e-9:
        if limit is None:
            return ranked
        return ranked[: max(0, int(limit))]
    name = os.environ["CANDIDATE"]
    before = extract_board_features(game.board)
    adjusted = [replace(ev, score=ev.score + candidate_bonus(name, game, ev, before)) for ev in ranked]
    adjusted.sort(key=lambda ev: (
        ev.score, ev.features.attack, ev.features.spin_lines, ev.features.lines,
        -ev.features.board.holes, -ev.features.board.max_height,
        -ev.placement.rotation, -ev.placement.x,
    ), reverse=True)
    if limit is not None:
        adjusted = adjusted[: max(0, int(limit))]
    return tuple(adjusted)


heuristic_mod.rank_placements = custom_rank
search_mod.rank_placements = custom_rank
versus_search_mod.rank_search_actions = search_mod.rank_search_actions


def summarize(games):
    pieces = sum(g.pieces for g in games)
    return {
        "pieces": pieces,
        "attack": sum(g.attack for g in games),
        "attackPerPiece": sum(g.attack for g in games) / max(1, pieces),
        "topouts": sum(g.topout for g in games),
        "completed": sum(g.completed for g in games),
        "spins": sum(g.spins for g in games),
        "spinLines": sum(g.spin_lines for g in games),
        "tSpinDoubles": sum(g.t_spin_doubles for g in games),
        "tSpinTriples": sum(g.t_spin_triples for g in games),
        "meanHoles": sum(g.mean_holes for g in games) / len(games),
        "meanHoleDepth": sum(g.mean_hole_depth for g in games) / len(games),
        "meanMaxHeight": sum(g.mean_max_height for g in games) / len(games),
    }


def solo(stage):
    if stage == "short":
        seeds, max_pieces = [43100, 43197], 100
    else:
        seeds, max_pieces = [93211, 93308, 93405], 240
    cfg = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4)
    games = [run_heuristic_game(s, max_pieces, CANDIDATE_WEIGHTS, cfg) for s in seeds]
    return {"stage": stage, "candidate": os.environ["CANDIDATE"], "solo": summarize(games)}


def versus():
    placement = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4)
    cfg = VersusSearchConfig(placement_search=placement, candidate_width=6, opponent_reply_width=1)
    r = run_versus_benchmark(
        games=8, max_turns=120, seed_base=143211, seed_step=193,
        player_weights=CANDIDATE_WEIGHTS, ai_weights=BASE,
        player_config=cfg, ai_config=cfg, garbage_cap=8,
    )
    per = r.per_game
    return {
        "stage": "versus", "candidate": os.environ["CANDIDATE"],
        "versus": {
            "wins": r.player_wins, "losses": r.ai_wins, "draws": r.draws,
            "attack": r.player_mean_attack, "baselineAttack": r.ai_mean_attack,
            "sent": r.player_mean_sent, "baselineSent": r.ai_mean_sent,
            "canceled": sum(x.player_canceled for x in per)/len(per),
            "baselineCanceled": sum(x.ai_canceled for x in per)/len(per),
            "received": sum(x.player_received for x in per)/len(per),
            "baselineReceived": sum(x.ai_received for x in per)/len(per),
            "maxB2B": sum(x.player_max_b2b for x in per)/len(per),
            "baselineMaxB2B": sum(x.ai_max_b2b for x in per)/len(per),
            "topouts": sum(x.winner == "ai" for x in per),
            "baselineTopouts": sum(x.winner == "player" for x in per),
        }
    }


stage = os.environ.get("STAGE", "short")
result = versus() if stage == "versus" else solo(stage)
Path("result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
