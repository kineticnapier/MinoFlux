from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, apply_search_action, clone_game, extract_board_features, rank_search_actions
from minoflux_ai.versus_search import _simulate_action, score_versus_state
from minoflux_engine import Game, VersusMatch, T_SPIN_DOUBLE, T_SPIN_TRIPLE

SEARCH = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4, discount=0.9, srs_reachable=True, allow_180=False, reachability_node_limit=8000)
HOLES = (0, 2, 4, 6, 9)
CENTER_HOLES = (3, 4, 5, 6)
EDGE_HOLES = (0, 1, 8, 9)

CANDIDATES = (
    "worst4_quality", "worst4_holes", "worst4_depth", "worst4_height",
    "worst4_tslot", "range4_stability", "tail2_6", "avg_worst_mix",
    "edge_worst", "center_worst", "tspin_worst_clean", "recovery_margin",
    "survival_floor", "attack_resilience",
)

@dataclass
class Stress:
    holes: list[int]
    depth: list[int]
    height: list[int]
    bump: list[int]
    slots: list[int]


def stress(board, lines: int, holes=HOLES) -> Stress:
    width = len(board[0])
    out = Stress([], [], [], [], [])
    for hole in holes:
        b = [row.copy() for row in board]
        for _ in range(lines):
            b.pop(0)
            row = ["G"] * width
            row[min(width - 1, hole)] = None
            b.append(row)
        f = extract_board_features(b)
        out.holes.append(f.holes)
        out.depth.append(f.hole_depth)
        out.height.append(f.max_height)
        out.bump.append(f.bumpiness)
        out.slots.append(f.t_spin_slots)
    return out


def danger(s: Stress) -> list[float]:
    return [2.2*h + 0.26*d + 0.72*mh + 0.08*b for h,d,mh,b in zip(s.holes,s.depth,s.height,s.bump)]


def bonus(name: str, child: Game, immediate) -> float:
    if name == "baseline":
        return 0.0
    s4 = stress(child.board, 4)
    d4 = danger(s4)
    if name == "worst4_quality": return -0.18 * max(d4)
    if name == "worst4_holes": return -0.72 * max(s4.holes)
    if name == "worst4_depth": return -0.075 * max(s4.depth)
    if name == "worst4_height": return -0.34 * max(s4.height)
    if name == "worst4_tslot": return 0.90 * min(s4.slots) - 0.055 * max(s4.depth)
    if name == "range4_stability": return -0.30 * (max(d4) - min(d4))
    if name == "tail2_6":
        return -0.075 * (max(danger(stress(child.board, 2))) + max(danger(stress(child.board, 6))))
    if name == "avg_worst_mix": return -0.070 * mean(d4) - 0.105 * max(d4)
    if name == "edge_worst": return -0.15 * max(danger(stress(child.board, 4, EDGE_HOLES)))
    if name == "center_worst": return -0.15 * max(danger(stress(child.board, 4, CENTER_HOLES)))
    if name == "tspin_worst_clean":
        q = [slot / (1.0 + h + mh / 6.0) for slot,h,mh in zip(s4.slots,s4.holes,s4.height)]
        return 2.4 * min(q)
    if name == "recovery_margin":
        now = extract_board_features(child.board)
        now_d = 2.2*now.holes + 0.26*now.hole_depth + 0.72*now.max_height + 0.08*now.bumpiness
        return 0.13 * (now_d - max(d4))
    if name == "survival_floor":
        return -0.26 * max(0, max(s4.height) - 16) ** 2 - 0.22 * max(s4.holes)
    if name == "attack_resilience":
        return 0.55 * immediate.features.attack - 0.105 * max(d4)
    raise KeyError(name)


def choose(game: Game, name: str):
    ranked = rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=8)
    best = None
    best_key = None
    for action, ev in ranked:
        child = clone_game(game)
        apply_search_action(child, action)
        score = ev.score + bonus(name, child, ev)
        key = (score, ev.features.attack, ev.features.spin_lines, -len(action.placement.path), -int(action.use_hold))
        if best is None or key > best_key:
            best = (action, ev)
            best_key = key
    return best


def solo(name: str, games: int, max_pieces: int, seed_base: int, seed_step: int):
    attack = pieces = spins = spin_lines = tsd = tst = topouts = completed = max_b2b = 0
    for i in range(games):
        g = Game(seed_base + i * seed_step)
        local_b2b = 0
        while not g.game_over and g.pieces_placed < max_pieces:
            picked = choose(g, name)
            if picked is None: break
            result = apply_search_action(g, picked[0])
            if result.spin is not None:
                spins += 1
                spin_lines += result.lines
            tsd += int(result.spin == T_SPIN_DOUBLE)
            tst += int(result.spin == T_SPIN_TRIPLE)
            local_b2b = max(local_b2b, g.b2b_chain)
        attack += g.attack
        pieces += g.pieces_placed
        topouts += int(g.game_over)
        completed += int((not g.game_over) and g.pieces_placed >= max_pieces)
        max_b2b += local_b2b
    return {"name":name,"games":games,"pieces":pieces,"attack":attack,"app":attack/max(1,pieces),"topouts":topouts,"completed":completed,"spins":spins,"spinLines":spin_lines,"tsd":tsd,"tst":tst,"meanMaxB2B":max_b2b/games}


def choose_vs(match: VersusMatch, side: str, name: str):
    own = match.side(side)
    ranked = rank_search_actions(own.game, DEFAULT_WEIGHTS, SEARCH, limit=6)
    best = None
    best_key = None
    for action, ev in ranked:
        after, resolution = _simulate_action(match, side, action)
        child = after.side(side).game
        score = score_versus_state(after, side, resolution=resolution, solo_score=ev.score, path_length=len(action.placement.path), action_side=side)
        score += bonus(name, child, ev)
        key = (score, resolution.sent_lines, resolution.canceled_lines, -len(action.placement.path), -int(action.use_hold))
        if best is None or key > best_key:
            best = action
            best_key = key
    return best


def versus(candidate: str, games=8, max_turns=120, seed_base=880001, seed_step=97):
    wins=losses=draws=0
    ca=ba=cs=bs=cc=bc=cr=br=ct=bt=cb2=bb2=0
    for i in range(games):
        swapped = i % 2 == 1
        seed = seed_base + (i//2)*seed_step
        match = VersusMatch(seed, garbage_cap=8)
        turn = "player"
        turns = 0
        c_side = "ai" if swapped else "player"
        b_side = "player" if swapped else "ai"
        cmax=bmax=0
        while match.winner is None and turns < max_turns:
            name = candidate if turn == c_side else "baseline"
            action = choose_vs(match, turn, name)
            if action is None:
                match.side(turn).game.game_over = True
                match._update_winner(); break
            result = apply_search_action(match.side(turn).game, action)
            match.resolve_lock(turn, result)
            cmax=max(cmax, match.side(c_side).game.b2b_chain)
            bmax=max(bmax, match.side(b_side).game.b2b_chain)
            turns += 1
            turn = "ai" if turn == "player" else "player"
        w=match.winner or "draw"
        if w == c_side: wins += 1
        elif w == b_side: losses += 1
        else: draws += 1
        c=match.side(c_side); b=match.side(b_side)
        ca+=c.game.attack; ba+=b.game.attack; cs+=c.sent; bs+=b.sent; cc+=c.canceled; bc+=b.canceled; cr+=c.received; br+=b.received
        ct+=int(c.game.game_over); bt+=int(b.game.game_over); cb2+=cmax; bb2+=bmax
    n=float(games)
    return {"name":candidate,"games":games,"wins":wins,"losses":losses,"draws":draws,"attack":[ca/n,ba/n],"sent":[cs/n,bs/n],"canceled":[cc/n,bc/n],"received":[cr/n,br/n],"topouts":[ct,bt],"maxB2B":[cb2/n,bb2/n]}


def rank_key(r):
    survival = r["completed"] - 1.5*r["topouts"]
    return (survival, r["app"], r["spinLines"], r["tsd"], r["meanMaxB2B"])


def main():
    names=("baseline",)+CANDIDATES
    short=[solo(n,2,110,610001,37) for n in names]
    baseline=short[0]
    viable=[r for r in short[1:] if r["topouts"] <= baseline["topouts"] and r["completed"] >= baseline["completed"]]
    top=sorted(viable,key=rank_key,reverse=True)[:3]
    fresh_names=["baseline"]+[r["name"] for r in top]
    fresh=[solo(n,3,260,710003,53) for n in fresh_names]
    fb=fresh[0]
    finals=[r for r in fresh[1:] if r["topouts"] <= fb["topouts"] and r["completed"] >= fb["completed"] and r["app"] >= fb["app"]*0.995]
    finals=sorted(finals,key=rank_key,reverse=True)[:2]
    vs=[versus(r["name"]) for r in finals]
    result={"short":short,"top3":[r["name"] for r in top],"fresh":fresh,"finalists":[r["name"] for r in finals],"versus":vs}
    print(json.dumps(result,indent=2))

if __name__ == "__main__": main()
