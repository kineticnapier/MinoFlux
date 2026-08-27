from __future__ import annotations
import json, os
from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action, choose_search_action
from minoflux_ai.versus_benchmark import run_versus_benchmark

CANDIDATE=os.environ.get('CANDIDATE','baseline'); PHASE=os.environ.get('PHASE','short')
os.environ['MINOFLUX_TOURNAMENT_CANDIDATE']=CANDIDATE

def run_game(seed,pieces):
    g=Game(seed); spins=tsd=tst=spin_lines=0; max_b2b=0
    while not g.game_over and g.pieces_placed<pieces:
        c=choose_search_action(g,DEFAULT_WEIGHTS,DEFAULT_SEARCH_CONFIG)
        if c is None: break
        r=apply_search_action(g,c.action)
        if r.spin is not None: spins+=1; spin_lines+=r.lines
        tsd+=int(r.spin==T_SPIN_DOUBLE); tst+=int(r.spin==T_SPIN_TRIPLE); max_b2b=max(max_b2b,g.b2b_chain)
    return {'pieces':g.pieces_placed,'attack':g.attack,'topout':int(g.game_over),'completed':int(not g.game_over and g.pieces_placed>=pieces),'spins':spins,'spin_lines':spin_lines,'tsd':tsd,'tst':tst,'max_b2b':max_b2b}

def solo():
    seeds,pieces=((73001,73031),90) if PHASE=='short' else ((93011,93047,93083),150)
    rows=[run_game(s,pieces) for s in seeds]; tp=sum(r['pieces'] for r in rows); ta=sum(r['attack'] for r in rows)
    return {'candidate':CANDIDATE,'phase':PHASE,'games':len(rows),'pieces':tp,'attack':ta,'attack_per_piece':ta/tp if tp else 0,'topouts':sum(r['topout'] for r in rows),'completed':sum(r['completed'] for r in rows),'spins':sum(r['spins'] for r in rows),'spin_lines':sum(r['spin_lines'] for r in rows),'tsd':sum(r['tsd'] for r in rows),'tst':sum(r['tst'] for r in rows),'max_b2b':max(r['max_b2b'] for r in rows),'rows':rows}

def versus():
    res=run_versus_benchmark(games=6,max_turns=90,seed_base=130013,seed_step=47,player_weights=DEFAULT_WEIGHTS,ai_weights=DEFAULT_WEIGHTS)
    rows=res.per_game
    return {'candidate':CANDIDATE,'phase':'versus','games':res.games,'candidate_wins':res.player_wins,'baseline_wins':res.ai_wins,'draws':res.draws,'candidate_attack':res.player_mean_attack,'baseline_attack':res.ai_mean_attack,'candidate_sent':res.player_mean_sent,'baseline_sent':res.ai_mean_sent,'candidate_canceled':sum(r.player_canceled for r in rows)/len(rows),'baseline_canceled':sum(r.ai_canceled for r in rows)/len(rows),'candidate_received':sum(r.player_received for r in rows)/len(rows),'baseline_received':sum(r.ai_received for r in rows)/len(rows),'candidate_max_b2b':sum(r.player_max_b2b for r in rows)/len(rows),'baseline_max_b2b':sum(r.ai_max_b2b for r in rows)/len(rows),'candidate_topouts':sum(r.winner=='ai' for r in rows),'baseline_topouts':sum(r.winner=='player' for r in rows)}
print('TOURNAMENT_RESULT='+json.dumps(versus() if PHASE=='versus' else solo(),sort_keys=True))
