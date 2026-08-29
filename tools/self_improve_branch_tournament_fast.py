from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig
from minoflux_ai.search import SearchAction, apply_search_action, clone_game, rank_search_actions
from minoflux_ai.versus_search import VersusSearchConfig, clone_versus_match, score_versus_state
from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE, VersusMatch

SEARCH = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4, discount=0.9, srs_reachable=True, allow_180=False, reachability_node_limit=8000)
VERSUS = VersusSearchConfig(placement_search=SEARCH, candidate_width=6, opponent_reply_width=1)
ROOT_WIDTH = 5
NEXT_WIDTH = 4
CANDIDATES = (
    "future_survival", "future_clean", "future_recovery", "future_attack_peak",
    "future_attack_floor", "future_score_floor", "future_slot_reserve",
    "future_t_conversion", "future_b2b_options", "future_clear_options",
    "future_low_stack", "future_hold_flex", "future_balanced", "future_t_window",
)


def next_t_distance(game: Game) -> int:
    if game.current == "T": return 0
    for i, piece in enumerate(game.queue):
        if piece == "T": return i + 1
        if i >= 3: break
    return 5


def future_stats(game: Game, action: SearchAction) -> dict[str, float]:
    child = clone_game(game)
    apply_search_action(child, action)
    if child.game_over:
        return {"survival":0,"clean":0,"recovery":0,"attack_peak":0,"attack_floor":0,"score_floor":-40,"slots":0,"t_convert":0,"b2b":0,"clear":0,"low":0,"hold_flex":0,"t_dist":float(next_t_distance(child))}
    ranked = rank_search_actions(child, DEFAULT_WEIGHTS, SEARCH, limit=NEXT_WIDTH)
    if not ranked:
        return {"survival":0,"clean":0,"recovery":0,"attack_peak":0,"attack_floor":0,"score_floor":-40,"slots":0,"t_convert":0,"b2b":0,"clear":0,"low":0,"hold_flex":0,"t_dist":float(next_t_distance(child))}
    actions = [a for a, _ in ranked]
    evals = [e for _, e in ranked]
    n = len(evals)
    attacks = sorted((e.features.attack for e in evals), reverse=True)
    scores = sorted((e.score for e in evals), reverse=True)
    best_holes = min(e.features.board.holes for e in evals)
    best_depth = min(e.features.board.hole_depth for e in evals)
    return {
        "survival": sum(not e.features.game_over for e in evals) / n,
        "clean": sum(e.features.new_holes == 0 for e in evals) / n,
        "recovery": max(0, 3-best_holes) + max(0, 8-best_depth)/4,
        "attack_peak": max(attacks, default=0),
        "attack_floor": min(attacks[:3], default=0),
        "score_floor": max(-40, min(40, min(scores[:3], default=-40))),
        "slots": mean(e.features.board.t_spin_slots for e in evals),
        "t_convert": sum(e.features.spin_lines > 0 for e in evals) if child.current == "T" else 0,
        "b2b": sum(e.features.spin_lines > 0 or e.features.lines == 4 for e in evals),
        "clear": sum(e.features.lines > 0 for e in evals),
        "low": sum(e.features.board.max_height <= 12 for e in evals) / n,
        "hold_flex": float(any(a.use_hold for a in actions) and any(not a.use_hold for a in actions)),
        "t_dist": float(next_t_distance(child)),
    }


def bonus(mode: str, s: dict[str, float], b2b_active: bool) -> float:
    return {
        "future_survival": 2.5*s["survival"],
        "future_clean": 2.0*s["clean"],
        "future_recovery": 0.4*s["recovery"],
        "future_attack_peak": 0.4*s["attack_peak"],
        "future_attack_floor": 0.7*s["attack_floor"],
        "future_score_floor": 0.07*s["score_floor"],
        "future_slot_reserve": 0.6*s["slots"],
        "future_t_conversion": 0.9*s["t_convert"],
        "future_b2b_options": (0.65 if b2b_active else 0.25)*s["b2b"],
        "future_clear_options": 0.35*s["clear"],
        "future_low_stack": 1.8*s["low"],
        "future_hold_flex": 1.0*s["hold_flex"],
        "future_balanced": 1.0*s["survival"] + 0.8*s["clean"] + 0.3*s["attack_floor"] + 0.15*s["recovery"],
        "future_t_window": max(0,(3-s["t_dist"])/3)*(0.6*s["slots"]+0.8*s["clean"]+0.4*s["t_convert"]),
    }[mode]


def choose(game: Game, mode: str | None) -> SearchAction | None:
    ranked = rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=1 if mode is None else ROOT_WIDTH)
    if not ranked: return None
    if mode is None: return ranked[0][0]
    best = None
    best_key = None
    for action, ev in ranked:
        s = future_stats(game, action)
        key = (ev.score + bonus(mode,s,game.back_to_back), ev.features.attack, ev.features.spin_lines, ev.features.lines, -ev.features.board.holes, -ev.features.board.max_height, -int(action.use_hold))
        if best_key is None or key > best_key:
            best_key, best = key, action
    return best


def solo_game(seed: int, pieces: int, mode: str | None) -> dict[str, object]:
    game = Game(seed)
    spins=spin_lines=tsd=tst=0; max_b2b=0
    while not game.game_over and game.pieces_placed < pieces:
        action=choose(game,mode)
        if action is None: break
        result=apply_search_action(game,action)
        if result.spin is not None: spins+=1; spin_lines+=result.lines
        tsd += int(result.spin == T_SPIN_DOUBLE); tst += int(result.spin == T_SPIN_TRIPLE)
        max_b2b=max(max_b2b,game.b2b_chain)
    return {"pieces":game.pieces_placed,"attack":game.attack,"topout":game.game_over,"completed":not game.game_over and game.pieces_placed>=pieces,"spins":spins,"spin_lines":spin_lines,"tsd":tsd,"tst":tst,"max_b2b":max_b2b}


def solo(mode: str | None, seeds: tuple[int,...], pieces: int) -> dict[str, object]:
    gs=[solo_game(s,pieces,mode) for s in seeds]; p=sum(g["pieces"] for g in gs); a=sum(g["attack"] for g in gs)
    return {"mode":mode or "baseline","pieces":p,"attack":a,"app":a/max(1,p),"topouts":sum(g["topout"] for g in gs),"completed":sum(g["completed"] for g in gs),"spins":sum(g["spins"] for g in gs),"spin_lines":sum(g["spin_lines"] for g in gs),"tsd":sum(g["tsd"] for g in gs),"tst":sum(g["tst"] for g in gs),"max_b2b":max(g["max_b2b"] for g in gs)}


def solo_task(args): return solo(*args)

def qkey(x): return (-x["topouts"],x["completed"],x["app"],x["attack"],x["tsd"]+1.5*x["tst"],x["spin_lines"],x["max_b2b"])


def other(side: str) -> str: return "ai" if side=="player" else "player"

def baseline_versus_action(match, side):
    from minoflux_ai.versus_search import choose_versus_action
    c=choose_versus_action(match,side,DEFAULT_WEIGHTS,VERSUS); return c.action if c else None


def candidate_versus_action(match, side, mode):
    own=match.side(side); opp=other(side); ranked=rank_search_actions(own.game,DEFAULT_WEIGHTS,SEARCH,limit=VERSUS.candidate_width)
    best=None; best_key=None
    for action,ev in ranked:
        s=future_stats(own.game,action); after=clone_versus_match(match); r=apply_search_action(after.side(side).game,action); res=after.resolve_lock(side,r)
        score=score_versus_state(after,side,resolution=res,solo_score=ev.score+bonus(mode,s,own.game.back_to_back),path_length=len(action.placement.path),action_side=side)
        if after.winner is None:
            replies=rank_search_actions(after.side(opp).game,DEFAULT_WEIGHTS,SEARCH,limit=1)
            for ra,re in replies:
                replied=clone_versus_match(after); rr=apply_search_action(replied.side(opp).game,ra); rres=replied.resolve_lock(opp,rr)
                score=score_versus_state(replied,side,resolution=rres,solo_score=-re.score,path_length=len(ra.placement.path),action_side=opp)
        key=(score,res.sent_lines,res.canceled_lines,-len(action.placement.path),-int(action.use_hold))
        if best_key is None or key>best_key: best_key,best=key,action
    return best


def versus_game(seed,mode,swap,max_turns=90):
    m=VersusMatch(seed,garbage_cap=8); cand="ai" if swap else "player"; turn="player"; turns=0; mb={"player":0,"ai":0}
    while m.winner is None and turns<max_turns:
        a=candidate_versus_action(m,turn,mode) if turn==cand else baseline_versus_action(m,turn)
        if a is None: m.side(turn).game.game_over=True; m._update_winner(); break
        r=apply_search_action(m.side(turn).game,a); m.resolve_lock(turn,r); mb[turn]=max(mb[turn],m.side(turn).game.b2b_chain); turns+=1; turn=other(turn)
    base=other(cand); w=m.winner or "draw"; w="candidate" if w==cand else "baseline" if w==base else "draw"
    cs=m.side(cand); bs=m.side(base)
    return {"winner":w,"ca":cs.game.attack,"ba":bs.game.attack,"cs":cs.sent,"bs":bs.sent,"cc":cs.canceled,"bc":bs.canceled,"cr":cs.received,"br":bs.received,"cb":mb[cand],"bb":mb[base],"ct":cs.game.game_over,"bt":bs.game.game_over}


def versus(mode):
    gs=[versus_game(810001+(i//2)*193,mode,bool(i%2)) for i in range(6)]
    av=lambda k:sum(g[k] for g in gs)/len(gs)
    return {"mode":mode,"candidate_wins":sum(g["winner"]=="candidate" for g in gs),"baseline_wins":sum(g["winner"]=="baseline" for g in gs),"draws":sum(g["winner"]=="draw" for g in gs),"candidate_attack":av("ca"),"baseline_attack":av("ba"),"candidate_sent":av("cs"),"baseline_sent":av("bs"),"candidate_canceled":av("cc"),"baseline_canceled":av("bc"),"candidate_received":av("cr"),"baseline_received":av("br"),"candidate_b2b":av("cb"),"baseline_b2b":av("bb"),"candidate_topouts":sum(g["ct"] for g in gs),"baseline_topouts":sum(g["bt"] for g in gs)}


def main():
    short_seeds=(41001,41098); fresh_seeds=(141001,141098,141195)
    baseline_short=solo(None,short_seeds,60)
    with ProcessPoolExecutor(max_workers=4) as ex:
        short=list(ex.map(solo_task,[(m,short_seeds,60) for m in CANDIDATES]))
    eligible=[x for x in short if x["topouts"]<=baseline_short["topouts"] and x["completed"]>=baseline_short["completed"]]
    eligible.sort(key=qkey,reverse=True); finalists=[x["mode"] for x in eligible[:3]]
    baseline_fresh=solo(None,fresh_seeds,130)
    with ProcessPoolExecutor(max_workers=3) as ex:
        fresh=list(ex.map(solo_task,[(m,fresh_seeds,130) for m in finalists]))
    viable=[x for x in fresh if x["topouts"]<=baseline_fresh["topouts"] and x["completed"]>=baseline_fresh["completed"] and x["app"]>=baseline_fresh["app"]*1.01]
    viable.sort(key=qkey,reverse=True); vm=[x["mode"] for x in viable[:2]]
    vs=[versus(m) for m in vm]
    out={"candidate_count":len(CANDIDATES),"candidates":list(CANDIDATES),"short_baseline":baseline_short,"short":short,"short_finalists":finalists,"fresh_baseline":baseline_fresh,"fresh":fresh,"versus_modes":vm,"versus":vs}
    Path("tournament-results.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__": main()
