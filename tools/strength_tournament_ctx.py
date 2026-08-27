from __future__ import annotations

import argparse
import json
from dataclasses import replace

from minoflux_engine import VersusMatch
import minoflux_ai.heuristic as heuristic
import minoflux_ai.search as search
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.search import apply_search_action
from minoflux_ai.versus_search import choose_versus_action

CANDIDATES = {
    "baseline": (0.0, "baseline"),
    "safe_attack": (0.55, "safe_attack"),
    "attack_no_holes": (0.65, "attack_no_holes"),
    "spin_safety": (0.75, "spin_safety"),
    "slot_bump_quality": (0.70, "slot_bump_quality"),
    "slot_depth_quality": (0.80, "slot_depth_quality"),
    "slot_supply_match": (0.55, "slot_supply_match"),
    "slot_timing": (0.75, "slot_timing"),
    "slot_single_focus": (0.55, "slot_single_focus"),
    "danger_attack_escape": (0.60, "danger_attack_escape"),
    "clean_spin_attack": (0.65, "clean_spin_attack"),
    "line_safety": (0.50, "line_safety"),
    "deep_hole_danger": (-0.050, "deep_hole_danger"),
    "nonspin_clear_tax": (-0.45, "nonspin_clear_tax"),
    "slot_clear_conflict": (-0.55, "slot_clear_conflict"),
}
_ORIGINAL_RANK = heuristic.rank_placements

def _next_t_distance(game) -> int:
    if game.current == "T": return 0
    for i, piece in enumerate(game.queue):
        if piece == "T": return i + 1
        if i >= 5: break
    return 7

def _bonus(name: str, weight: float, game, ev) -> float:
    f, b = ev.features, ev.features.board
    if name == "baseline": return 0.0
    if name == "safe_attack": return weight * f.attack / (1.0 + f.new_holes + b.max_height / 12.0)
    if name == "attack_no_holes": return weight * f.attack / (1.0 + 2.0 * f.new_holes)
    if name == "spin_safety": return weight * f.spin_lines / (1.0 + b.holes + b.max_height / 8.0)
    if name == "slot_bump_quality": return weight * b.t_spin_slots / (1.0 + b.bumpiness / 4.0 + b.max_height / 8.0)
    if name == "slot_depth_quality": return weight * b.t_spin_slots / (1.0 + b.holes + b.hole_depth / 6.0)
    if name == "slot_supply_match":
        d = _next_t_distance(game); a = max(0.0, (6.0 - d) / 6.0); e = max(0, b.t_spin_slots - 1)
        return weight * (a * min(1, b.t_spin_slots) - 0.35 * e * (1.0 - a))
    if name == "slot_timing":
        d = _next_t_distance(game); t = max(0.0, (5.0 - d) / 5.0)
        return weight * t * b.t_spin_slots / (1.0 + b.holes + b.max_height / 6.0)
    if name == "slot_single_focus": return weight * (1.0 if b.t_spin_slots == 1 else 0.0) / (1.0 + b.holes + b.max_height / 8.0)
    if name == "danger_attack_escape": return weight * max(0.0, (b.max_height - 9.0) / 8.0) * f.attack
    if name == "clean_spin_attack": return weight * f.attack * (1.0 if f.spin_lines else 0.0) / (1.0 + b.holes)
    if name == "line_safety": return weight * f.lines / (1.0 + b.holes + b.max_height / 10.0)
    if name == "deep_hole_danger": return weight * (b.hole_depth ** 2) * (1.0 + max(0.0, (b.max_height - 10.0) / 6.0))
    if name == "nonspin_clear_tax": return weight * f.lines if f.lines and not f.spin_lines and b.t_spin_slots > 0 else 0.0
    if name == "slot_clear_conflict": return weight * f.lines * b.t_spin_slots if f.lines and _next_t_distance(game) <= 3 and not f.spin_lines else 0.0
    raise KeyError(name)

def install_candidate(candidate: str) -> None:
    heuristic.rank_placements = _ORIGINAL_RANK; search.rank_placements = _ORIGINAL_RANK
    weight, name = CANDIDATES[candidate]
    if candidate == "baseline": return
    def rank(game, weights=heuristic.DEFAULT_WEIGHTS, *, placements=None, limit=None):
        ranked = list(_ORIGINAL_RANK(game, weights, placements=placements, limit=None))
        adjusted = [replace(ev, score=ev.score + _bonus(name, weight, game, ev)) for ev in ranked]
        adjusted.sort(key=heuristic._placement_key, reverse=True)
        return tuple(adjusted if limit is None else adjusted[:max(0, int(limit))])
    heuristic.rank_placements = rank; search.rank_placements = rank

def summarize(result):
    return {"attack": result.attack, "pieces": result.pieces, "attack_per_piece": result.attack/result.pieces if result.pieces else 0.0,
            "topouts": result.topouts, "completed": result.completed, "spins": result.spins, "spin_lines": result.spin_lines,
            "tsd": result.t_spin_doubles, "tst": result.t_spin_triples}

def run_mirrored(candidate: str):
    rows=[]
    for leg in range(6):
        swapped = leg % 2 == 1; seed = 990071 + (leg//2)*173
        match=VersusMatch(seed, garbage_cap=8); turn="player"; max_b2b={"player":0,"ai":0}; turns=0
        candidate_side = "ai" if swapped else "player"
        while match.winner is None and turns < 160:
            install_candidate(candidate if turn == candidate_side else "baseline")
            choice=choose_versus_action(match, turn)
            if choice is None:
                match.side(turn).game.game_over=True; match._update_winner(); break
            result=apply_search_action(match.side(turn).game, choice.action); match.resolve_lock(turn, result); turns += 1
            max_b2b["player"]=max(max_b2b["player"],match.player.game.b2b_chain); max_b2b["ai"]=max(max_b2b["ai"],match.ai.game.b2b_chain)
            turn="ai" if turn=="player" else "player"
        c=match.side(candidate_side); b=match.side("ai" if candidate_side=="player" else "player")
        winner="candidate" if match.winner==candidate_side else ("baseline" if match.winner in ("player","ai") else "draw")
        rows.append({"winner":winner,"candidate_attack":c.game.attack,"baseline_attack":b.game.attack,"candidate_sent":c.sent,"baseline_sent":b.sent,
                     "candidate_canceled":c.canceled,"baseline_canceled":b.canceled,"candidate_received":c.received,"baseline_received":b.received,
                     "candidate_b2b":max_b2b[candidate_side],"baseline_b2b":max_b2b["ai" if candidate_side=="player" else "player"],
                     "candidate_topout":int(c.game.game_over),"baseline_topout":int(b.game.game_over)})
    n=len(rows)
    keys=("candidate_attack","baseline_attack","candidate_sent","baseline_sent","candidate_canceled","baseline_canceled","candidate_received","baseline_received","candidate_b2b","baseline_b2b")
    out={"candidate":candidate,"phase":"versus","wins":sum(r["winner"]=="candidate" for r in rows),"losses":sum(r["winner"]=="baseline" for r in rows),"draws":sum(r["winner"]=="draw" for r in rows),
         "candidate_topouts":sum(r["candidate_topout"] for r in rows),"baseline_topouts":sum(r["baseline_topout"] for r in rows)}
    out.update({k:sum(r[k] for r in rows)/n for k in keys}); return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--candidate", choices=sorted(CANDIDATES), required=True); p.add_argument("--phase", choices=("screen","fresh","versus"), default="screen"); a=p.parse_args()
    if a.phase=="versus": print("TOURNAMENT_RESULT="+json.dumps(run_mirrored(a.candidate),sort_keys=True)); return
    install_candidate(a.candidate)
    r=run_heuristic_benchmark(games=2,max_pieces=180,seed_base=41003,seed_step=97,workers=1) if a.phase=="screen" else run_heuristic_benchmark(games=3,max_pieces=250,seed_base=880301,seed_step=131,workers=1)
    print("TOURNAMENT_RESULT="+json.dumps({"candidate":a.candidate,"phase":a.phase,**summarize(r)},sort_keys=True))
if __name__=="__main__": main()
