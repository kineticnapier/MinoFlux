from __future__ import annotations

from dataclasses import replace
import json
import os

import minoflux_ai.heuristic as heuristic
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_search import VersusSearchConfig

CANDIDATE = os.environ.get("CANDIDATE", "baseline")
STAGE = os.environ.get("STAGE", "short")
ORIGINAL_CONTEXT = heuristic._context_score
MARKER = 8.000001
CANDIDATE_WEIGHTS = replace(DEFAULT_WEIGHTS, perfect_clear=MARKER)


def _next_t_distance(game) -> int:
    if game.current == "T": return 0
    for i, piece in enumerate(game.queue):
        if piece == "T": return i + 1
        if i >= 5: break
    return 7


def _t_supply(game) -> int:
    count = int(game.current == "T") + int(game.hold_piece == "T")
    count += sum(1 for p in list(game.queue)[:6] if p == "T")
    return count


def _modifier(game, features) -> float:
    before = extract_board_features(game.board)
    after = features.board
    holes_added = max(0, after.holes - before.holes)
    hole_relief = max(0, before.holes - after.holes)
    depth_relief = max(0, before.hole_depth - after.hole_depth)
    height_relief = max(0, before.max_height - after.max_height)
    slot_kept = min(before.t_spin_slots, after.t_spin_slots)
    slot_loss = max(0, before.t_spin_slots - after.t_spin_slots)
    slot_created = max(0, after.t_spin_slots - before.t_spin_slots)
    tdist = _next_t_distance(game)
    urgency = max(0.0, (7.0 - tdist) / 7.0)
    clean = 1.0 / (1.0 + after.holes + after.hole_depth / 8.0)
    smooth = 1.0 / (1.0 + after.bumpiness / 8.0)
    low = 1.0 / (1.0 + after.max_height / 8.0)
    difficult = features.spin_lines > 0 or features.lines == 4
    supply = _t_supply(game)

    if CANDIDATE == "t_ready_smooth":
        return 0.70 * urgency * min(2, after.t_spin_slots) * smooth
    if CANDIDATE == "t_ready_no_new_holes":
        return 0.55 * urgency * min(2, after.t_spin_slots) - 0.75 * urgency * holes_added
    if CANDIDATE == "t_ready_low_well":
        return 0.50 * urgency * min(2, after.t_spin_slots) / (1.0 + after.wells / 4.0)
    if CANDIDATE == "b2b_slot_supply_clean":
        return (0.50 if game.back_to_back and difficult else 0.0) * min(after.t_spin_slots, supply) * clean
    if CANDIDATE == "b2b_safe_preserve":
        return (0.40 if game.back_to_back and difficult and holes_added == 0 else 0.0) * (1.0 + slot_kept)
    if CANDIDATE == "hold_t_overreserve":
        excess = max(0, after.t_spin_slots - supply)
        return (-0.34 * excess if game.hold_piece == "T" else 0.0) + 0.14 * slot_kept * int(game.hold_piece == "T")
    if CANDIDATE == "hold_t_release_pressure":
        if game.hold_piece == "T" and game.current != "T" and urgency > 0.55:
            return -0.22 * max(0, after.t_spin_slots - 1) - 0.12 * max(0, after.max_height - 10)
        return 0.0
    if CANDIDATE == "safe_attack_slot_preserve":
        return 0.20 * min(5, features.attack) * slot_kept * clean - 0.30 * features.attack * slot_loss
    if CANDIDATE == "attack_without_new_holes":
        return 0.24 * min(5, features.attack) * (1.0 if holes_added == 0 else -0.6 * holes_added)
    if CANDIDATE == "difficult_clean_low":
        return (0.48 if difficult else 0.0) * clean * low
    if CANDIDATE == "difficult_slot_preserve":
        return (0.36 if difficult else 0.0) * slot_kept - 0.30 * int(difficult) * slot_loss
    if CANDIDATE == "recovery_keep_setup":
        return 0.16 * slot_kept * (hole_relief + min(2.0, depth_relief / 8.0) + min(2, height_relief))
    if CANDIDATE == "dirty_slot_growth_cap":
        dirty = after.holes + after.hole_depth / 8.0
        return 0.30 * slot_created / (1.0 + dirty) - 0.16 * slot_created * max(0.0, dirty - 3.0)
    if CANDIDATE == "queue_gap_slot_cost":
        gap = max(0, tdist - 2)
        return -0.12 * gap * max(0, after.t_spin_slots - supply) + 0.10 * urgency * min(after.t_spin_slots, supply)
    return 0.0


def patched_context(game, features, weights):
    base = ORIGINAL_CONTEXT(game, features, weights)
    return base + (_modifier(game, features) if weights.perfect_clear > 8.0000005 else 0.0)

heuristic._context_score = patched_context
cfg = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.9)
weights = DEFAULT_WEIGHTS if CANDIDATE == "baseline" else CANDIDATE_WEIGHTS

if STAGE == "versus":
    vcfg = VersusSearchConfig(placement_search=cfg, candidate_width=6, opponent_reply_width=1)
    result = run_versus_benchmark(games=8,max_turns=120,seed_base=92831,seed_step=193,player_weights=weights,ai_weights=DEFAULT_WEIGHTS,player_config=vcfg,ai_config=vcfg,garbage_cap=8)
    games = result.per_game
    p = result.to_dict(); p.update({"candidate":CANDIDATE,"stage":STAGE,"playerMeanCanceled":sum(g.player_canceled for g in games)/len(games),"aiMeanCanceled":sum(g.ai_canceled for g in games)/len(games),"playerMeanReceived":sum(g.player_received for g in games)/len(games),"aiMeanReceived":sum(g.ai_received for g in games)/len(games),"playerMeanMaxB2B":sum(g.player_max_b2b for g in games)/len(games),"aiMeanMaxB2B":sum(g.ai_max_b2b for g in games)/len(games),"playerTopouts":result.ai_wins,"aiTopouts":result.player_wins})
    print("RESULT="+json.dumps(p,separators=(",",":"))); raise SystemExit

if STAGE == "short": games,pieces,seed=2,120,42113
elif STAGE == "fresh": games,pieces,seed=3,280,75431
else: raise SystemExit(f"unknown STAGE={STAGE}")
result=run_heuristic_benchmark(games=games,max_pieces=pieces,seed_base=seed,seed_step=97,weights=weights,search_config=cfg,workers=1)
p=result.to_dict(); p.update({"candidate":CANDIDATE,"stage":STAGE,"attackPerPiece":result.attack/max(1,result.pieces)})
print("RESULT="+json.dumps(p,separators=(",",":")))
