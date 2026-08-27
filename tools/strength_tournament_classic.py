from __future__ import annotations

import argparse, json, math
from dataclasses import replace

from minoflux_engine import VersusMatch
import minoflux_ai.heuristic as heuristic
import minoflux_ai.search as search
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.search import apply_search_action
from minoflux_ai.versus_search import choose_versus_action

CANDIDATES = {
    "baseline": (0.0, "baseline"),
    "row_transitions": (-0.18, "row_transitions"),
    "column_transitions": (-0.16, "column_transitions"),
    "hole_rows": (-0.42, "hole_rows"),
    "covered_holes": (-0.20, "covered_holes"),
    "deepest_hole": (-0.16, "deepest_hole"),
    "landing_height": (-0.10, "landing_height"),
    "surface_variance": (-0.07, "surface_variance"),
    "high_columns": (-0.22, "high_columns"),
    "low_t_slot_transition": (-0.11, "low_t_slot_transition"),
    "eroded_cells": (0.28, "eroded_cells"),
    "clean_lines": (0.22, "clean_lines"),
    "attack_safety": (0.20, "attack_safety"),
    "shallow_hole_ratio": (0.45, "shallow_hole_ratio"),
    "classic_combo": (1.0, "classic_combo"),
}

_ORIGINAL_RANK = heuristic.rank_placements


def _after_board(game, placement):
    board = [row.copy() for row in game.board]
    for x, y in placement.cells:
        if y >= 0:
            board[y][x] = placement.piece
    full = [i for i, row in enumerate(board) if all(c is not None for c in row)]
    if full:
        s = set(full)
        board = [[None] * game.width for _ in full] + [row for i, row in enumerate(board) if i not in s]
    return board, len(full)


def _heights(board):
    h, w = len(board), len(board[0])
    out=[]
    for x in range(w):
        top=h
        for y in range(h):
            if board[y][x] is not None:
                top=y; break
        out.append(h-top)
    return out


def _classic(board, placement, lines):
    h, w = len(board), len(board[0])
    heights = _heights(board)
    row_trans = 0
    for y in range(h):
        prev = True
        for x in range(w):
            cur = board[y][x] is not None
            row_trans += cur != prev
            prev = cur
        row_trans += prev != True
    col_trans = 0
    for x in range(w):
        prev = True
        for y in range(h):
            cur = board[y][x] is not None
            col_trans += cur != prev
            prev = cur
        col_trans += prev != True
    hole_rows=set(); covered=0; deepest=0; shallow=0; holes=0
    for x in range(w):
        seen=0
        for y in range(h):
            if board[y][x] is not None:
                seen += 1
            elif seen:
                holes += 1; hole_rows.add(y); covered += seen; deepest=max(deepest, seen)
                if seen <= 2: shallow += 1
    mean=sum(heights)/w
    variance=sum((v-mean)**2 for v in heights)/w
    high_cols=sum(max(0, v-10) for v in heights)
    landing=max((y for _, y in placement.cells), default=0)
    eroded=lines * sum(1 for _, y in placement.cells if y >= 0 and y in range(h) and all(c is not None for c in game.board[y]) if False)
    # exact eroded cells need pre-clear full rows; approximate with piece cells in cleared rows reconstructed from placement.
    pre=[row.copy() for row in game.board]
    for x,y in placement.cells:
        if y>=0: pre[y][x]=placement.piece
    full={i for i,row in enumerate(pre) if all(c is not None for c in row)}
    eroded=lines*sum(1 for _,y in placement.cells if y in full)
    return {
        "row_transitions": row_trans,
        "column_transitions": col_trans,
        "hole_rows": len(hole_rows),
        "covered_holes": covered,
        "deepest_hole": deepest,
        "landing_height": landing,
        "surface_variance": variance,
        "high_columns": high_cols,
        "eroded_cells": eroded,
        "shallow_hole_ratio": (shallow / holes) if holes else 1.0,
    }


def _bonus(name, weight, game, ev):
    if name == "baseline": return 0.0
    board, lines = _after_board(game, ev.placement)
    m = _classic(board, ev.placement, lines)
    b = ev.features.board
    if name in m: return weight * m[name]
    if name == "low_t_slot_transition": return weight * m["row_transitions"] * b.t_spin_slots / (1+b.max_height/6)
    if name == "clean_lines": return weight * ev.features.lines / (1+b.holes+b.max_height/10)
    if name == "attack_safety": return weight * ev.features.attack / (1+b.holes+b.max_height/8)
    if name == "classic_combo":
        return (-0.10*m["row_transitions"] -0.08*m["column_transitions"] -0.28*m["hole_rows"] +0.20*m["eroded_cells"])
    raise KeyError(name)


def install(candidate):
    heuristic.rank_placements = _ORIGINAL_RANK
    search.rank_placements = _ORIGINAL_RANK
    weight,name=CANDIDATES[candidate]
    if candidate=="baseline": return
    def rank(game, weights=heuristic.DEFAULT_WEIGHTS, *, placements=None, limit=None):
        ranked=list(_ORIGINAL_RANK(game,weights,placements=placements,limit=None))
        adjusted=[replace(ev,score=ev.score+_bonus(name,weight,game,ev)) for ev in ranked]
        adjusted.sort(key=heuristic._placement_key,reverse=True)
        return tuple(adjusted if limit is None else adjusted[:max(0,int(limit))])
    heuristic.rank_placements=rank; search.rank_placements=rank


def summarize(r):
    return {"attack":r.attack,"pieces":r.pieces,"attack_per_piece":r.attack/r.pieces if r.pieces else 0.0,"topouts":r.topouts,"completed":r.completed,"spins":r.spins,"spin_lines":r.spin_lines,"tsd":r.t_spin_doubles,"tst":r.t_spin_triples}


def versus(candidate):
    rows=[]
    for leg in range(6):
        swapped=leg%2==1; seed=181003+(leg//2)*211; match=VersusMatch(seed,garbage_cap=8); turn="player"; turns=0
        cand="ai" if swapped else "player"; max_b2b={"player":0,"ai":0}
        while match.winner is None and turns<160:
            install(candidate if turn==cand else "baseline")
            choice=choose_versus_action(match,turn)
            if choice is None:
                match.side(turn).game.game_over=True; match._update_winner(); break
            result=apply_search_action(match.side(turn).game,choice.action); match.resolve_lock(turn,result); turns+=1
            max_b2b["player"]=max(max_b2b["player"],match.player.game.b2b_chain); max_b2b["ai"]=max(max_b2b["ai"],match.ai.game.b2b_chain)
            turn="ai" if turn=="player" else "player"
        other="ai" if cand=="player" else "player"; c=match.side(cand); b=match.side(other)
        winner="candidate" if match.winner==cand else ("baseline" if match.winner in ("player","ai") else "draw")
        rows.append((winner,c,b,max_b2b[cand],max_b2b[other]))
    n=len(rows)
    return {"candidate":candidate,"phase":"versus","wins":sum(r[0]=="candidate" for r in rows),"losses":sum(r[0]=="baseline" for r in rows),"draws":sum(r[0]=="draw" for r in rows),"candidate_attack":sum(r[1].game.attack for r in rows)/n,"baseline_attack":sum(r[2].game.attack for r in rows)/n,"candidate_sent":sum(r[1].sent for r in rows)/n,"baseline_sent":sum(r[2].sent for r in rows)/n,"candidate_canceled":sum(r[1].canceled for r in rows)/n,"baseline_canceled":sum(r[2].canceled for r in rows)/n,"candidate_received":sum(r[1].received for r in rows)/n,"baseline_received":sum(r[2].received for r in rows)/n,"candidate_b2b":sum(r[3] for r in rows)/n,"baseline_b2b":sum(r[4] for r in rows)/n,"candidate_topouts":sum(int(r[1].game.game_over) for r in rows),"baseline_topouts":sum(int(r[2].game.game_over) for r in rows)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--candidate",choices=sorted(CANDIDATES),required=True); p.add_argument("--phase",choices=("screen","fresh","versus"),default="screen"); a=p.parse_args()
    if a.phase=="versus": print("TOURNAMENT_RESULT="+json.dumps(versus(a.candidate),sort_keys=True)); return
    install(a.candidate)
    r=run_heuristic_benchmark(games=2,max_pieces=140,seed_base=61003,seed_step=101,workers=1) if a.phase=="screen" else run_heuristic_benchmark(games=3,max_pieces=240,seed_base=1200301,seed_step=137,workers=1)
    print("TOURNAMENT_RESULT="+json.dumps({"candidate":a.candidate,"phase":a.phase,**summarize(r)},sort_keys=True))
if __name__=="__main__": main()
