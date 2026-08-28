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
    holes_added = max(0, b.holes - before.holes)
    holes_removed = max(0, before.holes - b.holes)
    depth_added = max(0, b.hole_depth - before.hole_depth)
    depth_removed = max(0, before.hole_depth - b.hole_depth)
    height_added = max(0, b.max_height - before.max_height)
    height_removed = max(0, before.max_height - b.max_height)
    bump_removed = max(0, before.bumpiness - b.bumpiness)
    tdist = next_t_distance(game)
    tnear = max(0.0, (6.0 - tdist) / 6.0)
    spin_clear = f.spin_lines > 0
    line_clear = f.lines > 0
    danger = max(0.0, (before.max_height - 8.0) / 10.0)
    deep_tail_before = max(0.0, before.hole_depth - 4.0 * before.holes)
    deep_tail_after = max(0.0, b.hole_depth - 4.0 * b.holes)
    deep_tail_removed = max(0.0, deep_tail_before - deep_tail_after)

    if name == "baseline":
        return 0.0
    # Long-horizon cleanliness: distinguish a few shallow holes from pathological buried holes.
    if name == "deep_hole_tail_recovery":
        return 0.045 * deep_tail_removed - 0.030 * max(0.0, deep_tail_after - deep_tail_before)
    if name == "hole_depth_per_hole":
        before_ratio = before.hole_depth / max(1, before.holes)
        after_ratio = b.hole_depth / max(1, b.holes)
        return 0.055 * max(0.0, before_ratio - after_ratio) - 0.035 * max(0.0, after_ratio - before_ratio)
    if name == "clean_clear_compaction":
        return (0.16 * height_removed + 0.12 * holes_removed + 0.020 * depth_removed) if line_clear and holes_added == 0 else -0.16 * holes_added
    if name == "low_damage_attack":
        damage = 0.35 * holes_added + 0.025 * depth_added + 0.055 * height_added
        return 0.10 * f.attack - damage if f.attack > 0 else -0.35 * damage
    if name == "danger_deep_hole_escape":
        return danger * (0.060 * deep_tail_removed + 0.14 * height_removed + 0.10 * holes_removed)
    if name == "danger_smooth_escape":
        return danger * (0.13 * height_removed + 0.025 * bump_removed + 0.11 * holes_removed)
    if name == "t_near_damage_avoidance":
        return -tnear * (0.32 * holes_added + 0.030 * depth_added + 0.08 * height_added)
    if name == "t_near_clean_conversion_margin":
        if game.current != "T":
            return 0.0
        return (0.22 * f.spin_lines + 0.08 * f.attack + 0.10 * holes_removed + 0.015 * depth_removed) if spin_clear else -0.16 * max(0, -f.t_spin_slot_delta)
    if name == "slot_vs_deep_hole_tradeoff":
        return 0.13 * max(0, f.t_spin_slot_delta) - 0.035 * max(0.0, deep_tail_after - deep_tail_before) - 0.13 * holes_added
    if name == "safe_slot_reserve":
        safe = 1.0 / (1.0 + 0.18 * b.max_height + 0.10 * b.holes + 0.010 * b.hole_depth)
        return 0.32 * min(2, b.t_spin_slots) * safe
    if name == "clean_surface_recovery":
        return 0.10 * holes_removed + 0.018 * depth_removed + 0.020 * bump_removed + 0.08 * height_removed
    if name == "nonlinear_damage_guard":
        return -0.018 * (b.holes * b.holes - before.holes * before.holes) - 0.0015 * (b.hole_depth - before.hole_depth) * max(1, b.holes)
    if name == "high_stack_clean_attack":
        return 0.08 * f.attack + danger * (0.13 * height_removed + 0.12 * holes_removed + 0.020 * depth_removed) - danger * 0.20 * holes_added
    if name == "spin_exit_quality":
        if not spin_clear:
            return 0.0
        return 0.08 * f.attack + 0.12 * height_removed + 0.11 * holes_removed + 0.020 * depth_removed + 0.10 * b.t_spin_slots
    raise ValueError(name)


def custom_rank(game, weights=BASE, *, placements=None, limit=None):
    ranked = ORIGINAL_RANK(game, weights, placements=placements, limit=None)
    if abs(weights.perfect_clear - (BASE.perfect_clear + MARKER)) > 1e-9:
        return ranked if limit is None else ranked[: max(0, int(limit))]
    name = os.environ["CANDIDATE"]
    before = extract_board_features(game.board)
    adjusted = [replace(ev, score=ev.score + candidate_bonus(name, game, ev, before)) for ev in ranked]
    adjusted.sort(key=lambda ev: (
        ev.score, ev.features.attack, ev.features.spin_lines, ev.features.lines,
        -ev.features.board.holes, -ev.features.board.max_height,
        -ev.placement.rotation, -ev.placement.x,
    ), reverse=True)
    return tuple(adjusted if limit is None else adjusted[: max(0, int(limit))])


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
        "meanBumpiness": sum(g.mean_bumpiness for g in games) / len(games),
        "meanMaxHeight": sum(g.mean_max_height for g in games) / len(games),
    }


def solo(stage):
    if stage == "short":
        seeds, max_pieces = [58103, 58200], 110
    else:
        seeds, max_pieces = [118211, 118308, 118405], 300
    cfg = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4)
    games = [run_heuristic_game(s, max_pieces, CANDIDATE_WEIGHTS, cfg) for s in seeds]
    return {"stage": stage, "candidate": os.environ["CANDIDATE"], "solo": summarize(games)}


def versus():
    placement = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4)
    cfg = VersusSearchConfig(placement_search=placement, candidate_width=6, opponent_reply_width=1)
    r = run_versus_benchmark(
        games=8, max_turns=120, seed_base=218211, seed_step=193,
        player_weights=CANDIDATE_WEIGHTS, ai_weights=BASE,
        player_config=cfg, ai_config=cfg, garbage_cap=8,
    )
    per = r.per_game
    return {"stage":"versus","candidate":os.environ["CANDIDATE"],"versus":{
        "wins":r.player_wins,"losses":r.ai_wins,"draws":r.draws,
        "attack":r.player_mean_attack,"baselineAttack":r.ai_mean_attack,
        "sent":r.player_mean_sent,"baselineSent":r.ai_mean_sent,
        "canceled":sum(x.player_canceled for x in per)/len(per),
        "baselineCanceled":sum(x.ai_canceled for x in per)/len(per),
        "received":sum(x.player_received for x in per)/len(per),
        "baselineReceived":sum(x.ai_received for x in per)/len(per),
        "maxB2B":sum(x.player_max_b2b for x in per)/len(per),
        "baselineMaxB2B":sum(x.ai_max_b2b for x in per)/len(per),
        "topouts":sum(x.winner=="ai" for x in per),"baselineTopouts":sum(x.winner=="player" for x in per)}}

stage = os.environ.get("STAGE", "short")
result = versus() if stage == "versus" else solo(stage)
Path("result.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
print(json.dumps(result, sort_keys=True))
