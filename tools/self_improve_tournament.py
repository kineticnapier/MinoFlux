from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json

import minoflux_ai.heuristic as heuristic
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.versus_benchmark import run_versus_benchmark

CANDIDATES = (
    "t_supply_exact_match", "t_supply_shortage", "t_supply_surplus",
    "next_t_urgency", "post_spin_next_t_chain", "spin_conversion_efficiency",
    "safe_attack_efficiency", "safe_spike", "clean_height_recovery",
    "deep_hole_repair", "danger_attack_tradeoff", "danger_spin_escape",
    "hold_t_supply_balance", "hold_non_t_rescue",
)

def _next_t_distance(game):
    if game.current == "T": return 0
    for i, p in enumerate(game.queue[:7]):
        if p == "T": return i + 1
    return 8

def _t_count(game):
    return (1 if game.current == "T" else 0) + sum(1 for p in game.queue[:6] if p == "T")

def _extra_score(name, game, features):
    before = extract_board_features(game.board)
    after = features.board
    slots = after.t_spin_slots
    slot_delta = features.t_spin_slot_delta
    tdist = _next_t_distance(game)
    tcount = _t_count(game)
    hdrop = max(0, before.max_height - after.max_height)
    holes_removed = max(0, before.holes - after.holes)
    depth_removed = max(0, before.hole_depth - after.hole_depth)
    danger = max(0.0, (before.max_height - 10.0) / 6.0)
    spin = game.current == "T" and features.spin_lines > 0
    clean = 1.0 / (1.0 + after.holes + after.max_height / 7.0)

    if name == "t_supply_exact_match":
        return 0.40 * min(slots, tcount) - 0.35 * abs(slots - tcount)
    if name == "t_supply_shortage":
        return 0.55 * max(0, min(tcount, 2) - slots) * max(0, slot_delta)
    if name == "t_supply_surplus":
        return -0.35 * max(0, slots - tcount) + 0.20 * min(slots, tcount)
    if name == "next_t_urgency":
        urgency = max(0.0, (6.0 - tdist) / 6.0)
        return urgency * (0.42 * max(0, slot_delta) + 0.16 * slots) - (1-urgency) * 0.20 * max(0, slots-1)
    if name == "post_spin_next_t_chain":
        if not spin: return 0.0
        next_supply = sum(1 for p in game.queue[:6] if p == "T")
        return 0.45 * features.spin_lines + 0.30 * min(slots, next_supply) * clean
    if name == "spin_conversion_efficiency":
        if not spin: return 0.0
        return (0.62 * features.attack + 0.38 * features.spin_lines) * clean
    if name == "safe_attack_efficiency":
        return 0.20 * features.attack / (1.0 + features.new_holes + after.max_height / 8.0)
    if name == "safe_spike":
        return 0.22 * max(0, features.attack - 2) * clean
    if name == "clean_height_recovery":
        if before.max_height < 11: return 0.0
        return 0.20 * hdrop + 0.18 * holes_removed - 0.16 * features.new_holes
    if name == "deep_hole_repair":
        return 0.10 * depth_removed + 0.20 * holes_removed - 0.12 * features.new_holes
    if name == "danger_attack_tradeoff":
        return danger * (0.18 * features.attack + 0.18 * hdrop + 0.15 * holes_removed - 0.22 * features.new_holes)
    if name == "danger_spin_escape":
        if before.max_height < 11: return 0.0
        return 0.28 * features.spin_lines + 0.12 * features.attack + 0.16 * hdrop - 0.18 * features.new_holes
    if name == "hold_t_supply_balance":
        if not game.hold_used: return 0.0
        held_t = 1 if game.hold_piece == "T" else 0
        need = max(0, slots - (tcount + held_t))
        return -0.28 * need + 0.12 * held_t * min(1, slots)
    if name == "hold_non_t_rescue":
        if not game.hold_used or game.hold_piece == "T" or before.max_height < 11: return 0.0
        return 0.16 * hdrop + 0.12 * holes_removed + 0.08 * features.attack - 0.15 * features.new_holes
    raise ValueError(name)

def _install(name):
    marker = replace(DEFAULT_WEIGHTS)
    mid = id(marker)
    original = heuristic._context_score
    def patched(game, features, weights):
        base = original(game, features, weights)
        return base + (_extra_score(name, game, features) if id(weights) == mid else 0.0)
    heuristic._context_score = patched
    return marker

def _solo(name, games, pieces, seed_base, seed_step):
    weights = DEFAULT_WEIGHTS if name == "baseline" else _install(name)
    r = run_heuristic_benchmark(games=games, max_pieces=pieces, seed_base=seed_base, seed_step=seed_step, weights=weights, workers=1)
    return {"name":name,"pieces":r.pieces,"attack":r.attack,"attackPerPiece":r.attack/r.pieces if r.pieces else 0.0,"topouts":r.topouts,"completed":r.completed,"tSpins":r.spins,"spinLines":r.spin_lines,"tsd":r.t_spin_doubles,"tst":r.t_spin_triples}

def _key(row, games):
    return (-row["topouts"], row["completed"]/games, row["attackPerPiece"], row["tsd"]+1.5*row["tst"], row["spinLines"])

def _parallel(names, games, pieces, seed_base, seed_step):
    rows=[]
    with ProcessPoolExecutor(max_workers=min(8,len(names))) as pool:
        fs=[pool.submit(_solo,n,games,pieces,seed_base,seed_step) for n in names]
        for f in as_completed(fs): rows.append(f.result())
    return sorted(rows,key=lambda r:r["name"])

def _versus(name):
    w=_install(name)
    r=run_versus_benchmark(games=6,max_turns=110,seed_base=1_780_021,seed_step=113,player_weights=w,ai_weights=DEFAULT_WEIGHTS)
    g=r.per_game
    return {"name":name,"wins":r.player_wins,"losses":r.ai_wins,"draws":r.draws,"attack":r.player_mean_attack,"baselineAttack":r.ai_mean_attack,"sent":r.player_mean_sent,"baselineSent":r.ai_mean_sent,"cancel":sum(x.player_canceled for x in g)/len(g),"baselineCancel":sum(x.ai_canceled for x in g)/len(g),"received":sum(x.player_received for x in g)/len(g),"baselineReceived":sum(x.ai_received for x in g)/len(g),"maxB2B":sum(x.player_max_b2b for x in g)/len(g),"baselineMaxB2B":sum(x.ai_max_b2b for x in g)/len(g),"topouts":r.ai_wins,"baselineTopouts":r.player_wins}

def main():
    names=("baseline",*CANDIDATES)
    short=_parallel(names,2,170,620_011,89)
    survivors=[r["name"] for r in sorted((x for x in short if x["name"]!="baseline"),key=lambda x:_key(x,2),reverse=True)[:3]]
    fresh=_parallel(("baseline",*survivors),3,280,1_020_033,109)
    base=next(x for x in fresh if x["name"]=="baseline")
    candidates=sorted((x for x in fresh if x["name"]!="baseline"),key=lambda x:_key(x,3),reverse=True)
    finalists=[]
    for x in candidates:
        if x["topouts"]<=base["topouts"] and x["completed"]>=base["completed"] and x["attackPerPiece"]>=base["attackPerPiece"]:
            finalists.append(x["name"])
            if len(finalists)==2: break
    payload={"candidateCount":len(CANDIDATES),"short":short,"survivors":survivors,"fresh":fresh,"finalists":finalists,"versus":[_versus(n) for n in finalists]}
    print("RESULT_JSON="+json.dumps(payload,sort_keys=True))
if __name__=="__main__": main()
