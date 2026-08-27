from __future__ import annotations
import json, os
from minoflux_engine import Game, VersusMatch, T_SPIN_DOUBLE, T_SPIN_TRIPLE
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action, choose_search_action
from minoflux_ai.versus_search import DEFAULT_VERSUS_SEARCH_CONFIG, choose_versus_action

CANDIDATE=os.environ.get('CANDIDATE','baseline'); PHASE=os.environ.get('PHASE','short')

def set_candidate(enabled: bool):
    os.environ['MINOFLUX_TOURNAMENT_CANDIDATE']=CANDIDATE if enabled else 'baseline'

def run_game(seed,pieces):
    set_candidate(CANDIDATE!='baseline')
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

def versus_game(seed: int, swapped: bool):
    match=VersusMatch(seed,garbage_cap=8); turn='player'; turns=0; max_b2b={'player':0,'ai':0}
    candidate_side='ai' if swapped else 'player'
    while match.winner is None and turns<90:
        set_candidate(turn==candidate_side)
        choice=choose_versus_action(match,turn,DEFAULT_WEIGHTS,DEFAULT_VERSUS_SEARCH_CONFIG)
        if choice is None:
            match.side(turn).game.game_over=True; match._update_winner(); break
        side=match.side(turn); result=apply_search_action(side.game,choice.action); match.resolve_lock(turn,result); turns+=1
        max_b2b['player']=max(max_b2b['player'],match.player.game.b2b_chain); max_b2b['ai']=max(max_b2b['ai'],match.ai.game.b2b_chain)
        turn='ai' if turn=='player' else 'player'
    c=match.side(candidate_side); b=match.side('player' if candidate_side=='ai' else 'ai')
    winner=match.winner or 'draw'; cw=int(winner==candidate_side); bw=int(winner not in ('draw',candidate_side))
    return {'candidate_win':cw,'baseline_win':bw,'draw':int(winner=='draw'),'candidate_attack':c.game.attack,'baseline_attack':b.game.attack,'candidate_sent':c.sent,'baseline_sent':b.sent,'candidate_canceled':c.canceled,'baseline_canceled':b.canceled,'candidate_received':c.received,'baseline_received':b.received,'candidate_max_b2b':max_b2b[candidate_side],'baseline_max_b2b':max_b2b['player' if candidate_side=='ai' else 'ai'],'candidate_topout':int(c.game.game_over),'baseline_topout':int(b.game.game_over)}

def versus():
    rows=[versus_game(130013+(i//2)*47,bool(i%2)) for i in range(6)]; n=len(rows)
    avg=lambda k: sum(r[k] for r in rows)/n
    return {'candidate':CANDIDATE,'phase':'versus','games':n,'candidate_wins':sum(r['candidate_win'] for r in rows),'baseline_wins':sum(r['baseline_win'] for r in rows),'draws':sum(r['draw'] for r in rows),'candidate_attack':avg('candidate_attack'),'baseline_attack':avg('baseline_attack'),'candidate_sent':avg('candidate_sent'),'baseline_sent':avg('baseline_sent'),'candidate_canceled':avg('candidate_canceled'),'baseline_canceled':avg('baseline_canceled'),'candidate_received':avg('candidate_received'),'baseline_received':avg('baseline_received'),'candidate_max_b2b':avg('candidate_max_b2b'),'baseline_max_b2b':avg('baseline_max_b2b'),'candidate_topouts':sum(r['candidate_topout'] for r in rows),'baseline_topouts':sum(r['baseline_topout'] for r in rows),'rows':rows}
print('TOURNAMENT_RESULT='+json.dumps(versus() if PHASE=='versus' else solo(),sort_keys=True))
