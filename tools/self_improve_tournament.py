from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from typing import Sequence

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, apply_search_action, clone_game, rank_search_actions
from minoflux_ai.features import extract_board_features
from minoflux_ai.search import PlacementEvaluation, SearchAction, SearchScorer, choose_search_action
from minoflux_ai.versus_search import DEFAULT_VERSUS_SEARCH_CONFIG, DEFAULT_VERSUS_WEIGHTS, clone_versus_match, score_versus_state
from minoflux_engine import Game, VersusMatch

CANDIDATES = (
    "baseline",
    "garbage_slot_floor",
    "garbage_slot_supply_floor",
    "garbage_danger_variance",
    "edge_garbage_robustness",
    "t_near_worst_recovery",
    "hold_t_emergency_flex",
    "b2b_clean_conversion",
    "t_conversion_survival",
    "slot_preserve_safe_height",
    "downstack_slot_bridge",
    "garbage_b2b_floor",
    "garbage_hole_spread",
    "t_supply_stress_gap",
    "danger_attack_balance",
)
SHORT_SEEDS = (810001, 810019)
FRESH_SEEDS = (930001, 930041, 930083)
SEARCH = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4, discount=0.9, srs_reachable=True, allow_180=False, reachability_node_limit=8000)


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue[:6]):
        if piece == "T":
            return i + 1
    return 7


def _t_supply(game: Game) -> int:
    return int(game.current == "T") + int(game.hold_piece == "T") + sum(piece == "T" for piece in game.queue[:6])


def _stress_board(board: Sequence[Sequence[str | None]], width: int, hole: int, lines: int = 4):
    stressed = [list(row) for row in board]
    for _ in range(lines):
        stressed.pop(0)
        row: list[str | None] = ["G"] * width
        row[hole] = None
        stressed.append(row)
    return stressed


def _danger(features) -> float:
    return 2.1 * features.holes + 0.25 * features.hole_depth + 0.70 * features.max_height + 0.08 * features.bumpiness


def _slot_quality(features) -> float:
    return features.t_spin_slots / (1.0 + features.holes + features.max_height / 6.0)


def _candidate_child(game: Game, evaluation: PlacementEvaluation):
    child = clone_game(game)
    child.place(evaluation.placement)
    return child


class CandidateScorer(SearchScorer):
    def __init__(self, name: str):
        self.name = name

    def score_many(self, game: Game, evaluations: Sequence[PlacementEvaluation]) -> Sequence[float]:
        if self.name == "baseline":
            return [ev.score for ev in evaluations]
        result: list[float] = []
        before = extract_board_features(game.board)
        next_t = _next_t_distance(game)
        supply = _t_supply(game)
        stress_names = {"garbage_slot_floor", "garbage_slot_supply_floor", "garbage_danger_variance", "edge_garbage_robustness", "t_near_worst_recovery", "garbage_b2b_floor", "garbage_hole_spread", "t_supply_stress_gap"}
        for ev in evaluations:
            f = ev.features
            b = f.board
            bonus = 0.0
            if self.name in stress_names:
                child = _candidate_child(game, ev)
                holes = (3, 4, 5, 6) if child.width >= 7 else tuple(range(child.width))
                if self.name == "edge_garbage_robustness":
                    holes = (0, 1, child.width - 2, child.width - 1)
                stressed = [extract_board_features(_stress_board(child.board, child.width, hole)) for hole in holes]
                dangers = [_danger(x) for x in stressed]
                slot_floor = min((x.t_spin_slots for x in stressed), default=0)
                quality_floor = min((_slot_quality(x) for x in stressed), default=0.0)
                if self.name == "garbage_slot_floor":
                    bonus = 0.95 * quality_floor
                elif self.name == "garbage_slot_supply_floor":
                    bonus = 0.30 * min(slot_floor, supply) - 0.22 * max(0, slot_floor - supply)
                elif self.name == "garbage_danger_variance":
                    mean = sum(dangers) / max(1, len(dangers))
                    variance = sum((x - mean) ** 2 for x in dangers) / max(1, len(dangers))
                    bonus = -0.010 * math.sqrt(variance)
                elif self.name == "edge_garbage_robustness":
                    bonus = -0.026 * max(dangers, default=0.0)
                elif self.name == "t_near_worst_recovery":
                    urgency = max(0.0, (3.0 - next_t) / 3.0)
                    bonus = urgency * (0.70 * quality_floor - 0.010 * max(dangers, default=0.0))
                elif self.name == "garbage_b2b_floor":
                    bonus = (0.18 * quality_floor - 0.010 * max(dangers, default=0.0)) if game.back_to_back else 0.0
                elif self.name == "garbage_hole_spread":
                    bonus = -0.010 * (max(dangers, default=0.0) - min(dangers, default=0.0))
                else:
                    bonus = 0.26 * min(slot_floor, supply) - 0.30 * max(0, slot_floor - supply) - 0.012 * max(dangers, default=0.0)
            elif self.name == "hold_t_emergency_flex":
                high = max(0.0, (before.max_height - 12) / 6.0)
                held_t = int(game.hold_piece == "T")
                safety_gain = (before.max_height - b.max_height) + 0.6 * max(0, before.holes - b.holes)
                bonus = high * held_t * (0.18 * safety_gain + 0.08 * min(1, b.t_spin_slots)) - high * held_t * 0.14 * max(0, f.new_holes)
            elif self.name == "b2b_clean_conversion":
                difficult = f.spin_lines > 0 or f.lines == 4
                if game.back_to_back:
                    bonus = (0.30 if difficult else (-0.28 if f.lines > 0 else 0.0)) + 0.07 * f.attack - 0.035 * b.holes
            elif self.name == "t_conversion_survival":
                if game.current == "T":
                    clean = 1.0 / (1.0 + b.holes + b.max_height / 8.0)
                    bonus = clean * (0.38 * f.spin_lines + 0.09 * f.attack) - 0.09 * max(0, -f.t_spin_slot_delta)
            elif self.name == "slot_preserve_safe_height":
                safe = max(0.0, (17.0 - b.max_height) / 17.0)
                bonus = 0.22 * safe * max(0, f.t_spin_slot_delta) - 0.16 * max(0, -f.t_spin_slot_delta) * (1.0 - safe)
            elif self.name == "downstack_slot_bridge":
                hole_relief = max(0, before.hole_depth - b.hole_depth)
                bonus = 0.032 * hole_relief * (1.0 + 0.30 * min(2, b.t_spin_slots)) if f.lines > 0 else 0.0
            elif self.name == "danger_attack_balance":
                danger = max(0.0, (b.max_height - 12) / 6.0) + 0.15 * b.holes
                bonus = 0.09 * f.attack / (1.0 + danger) + 0.05 * f.spin_lines - 0.06 * max(0, f.new_holes) * danger
            result.append(ev.score + bonus)
        return result


def _solo_one(args):
    candidate, seed, max_pieces = args
    game = Game(seed)
    scorer = CandidateScorer(candidate)
    spins = tsd = tst = spin_lines = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < max_pieces:
        choice = choose_search_action(game, DEFAULT_WEIGHTS, SEARCH, scorer=scorer)
        if choice is None:
            break
        lock = apply_search_action(game, choice.action)
        if lock.spin is not None:
            spins += 1
            spin_lines += lock.lines
            tsd += int(lock.lines == 2)
            tst += int(lock.lines == 3)
        max_b2b = max(max_b2b, game.b2b_chain)
    features = extract_board_features(game.board)
    return {"seed": seed, "pieces": game.pieces_placed, "attack": game.attack, "topout": int(game.game_over), "completed": int(not game.game_over and game.pieces_placed >= max_pieces), "tSpins": spins, "spinLines": spin_lines, "tsd": tsd, "tst": tst, "maxB2B": max_b2b, "holes": features.holes, "holeDepth": features.hole_depth, "maxHeight": features.max_height}


def solo(candidate: str, seeds: Sequence[int], max_pieces: int):
    rows = [_solo_one((candidate, seed, max_pieces)) for seed in seeds]
    pieces = sum(x["pieces"] for x in rows)
    attack = sum(x["attack"] for x in rows)
    n = len(rows)
    return {"candidate": candidate, "games": n, "pieces": pieces, "attack": attack, "attackPerPiece": attack / max(1, pieces), "topouts": sum(x["topout"] for x in rows), "completed": sum(x["completed"] for x in rows), "tSpins": sum(x["tSpins"] for x in rows), "spinLines": sum(x["spinLines"] for x in rows), "tsd": sum(x["tsd"] for x in rows), "tst": sum(x["tst"] for x in rows), "maxB2B": max((x["maxB2B"] for x in rows), default=0), "meanHoles": sum(x["holes"] for x in rows) / n, "meanHoleDepth": sum(x["holeDepth"] for x in rows) / n, "meanMaxHeight": sum(x["maxHeight"] for x in rows) / n, "perGame": rows}


def _solo_candidate_job(args):
    return solo(*args)


def _solo_parallel(candidates: Sequence[str], seeds: Sequence[int], max_pieces: int):
    with ProcessPoolExecutor(max_workers=min(len(candidates), 12)) as pool:
        results = list(pool.map(_solo_candidate_job, [(c, tuple(seeds), max_pieces) for c in candidates]))
    return {x["candidate"]: x for x in results}


def _solo_rank(x):
    return (x["completed"] - x["topouts"], x["attackPerPiece"], x["spinLines"], x["tsd"] + x["tst"], x["maxB2B"])


def _simulate_versus_action(match, side_name: str, action: SearchAction):
    simulated = clone_versus_match(match)
    side = simulated.player if side_name == "player" else simulated.ai
    lock = apply_search_action(side.game, action)
    resolution = simulated.resolve_lock(side_name, lock)
    return simulated, resolution


def _choose_versus(match, side_name: str, candidate: str):
    cfg = DEFAULT_VERSUS_SEARCH_CONFIG.normalized()
    own = match.player if side_name == "player" else match.ai
    opponent_name = "ai" if side_name == "player" else "player"
    ranked = rank_search_actions(own.game, DEFAULT_WEIGHTS, cfg.placement_search, limit=cfg.candidate_width, scorer=CandidateScorer(candidate))
    best = None
    for action, evaluation in ranked:
        after, resolution = _simulate_versus_action(match, side_name, action)
        score = score_versus_state(after, side_name, weights=DEFAULT_VERSUS_WEIGHTS, resolution=resolution, solo_score=evaluation.score, path_length=len(action.placement.path), action_side=side_name)
        if cfg.opponent_reply_width > 0 and after.winner is None:
            opp = after.player if opponent_name == "player" else after.ai
            replies = rank_search_actions(opp.game, DEFAULT_WEIGHTS, cfg.placement_search, limit=cfg.opponent_reply_width)
            worst = None
            for reply, reply_eval in replies:
                replied, reply_resolution = _simulate_versus_action(after, opponent_name, reply)
                reply_score = score_versus_state(replied, side_name, weights=DEFAULT_VERSUS_WEIGHTS, resolution=reply_resolution, solo_score=-reply_eval.score, path_length=len(reply.placement.path), action_side=opponent_name)
                worst = reply_score if worst is None else min(worst, reply_score)
            if worst is not None:
                score = worst
        key = (score, resolution.sent_lines, resolution.canceled_lines, -len(action.placement.path), -int(action.use_hold))
        if best is None or key > best[0]:
            best = (key, action)
    return None if best is None else best[1]


def _versus_leg(seed: int, candidate: str, swapped: bool, max_turns: int = 100):
    match = VersusMatch(seed, garbage_cap=8)
    turn = "player"
    turns = 0
    pb = ab = 0
    while match.winner is None and turns < max_turns:
        candidate_side = "ai" if swapped else "player"
        action = _choose_versus(match, turn, candidate if turn == candidate_side else "baseline")
        if action is None:
            side = match.player if turn == "player" else match.ai
            side.game.game_over = True
            match._update_winner()
            break
        side = match.player if turn == "player" else match.ai
        lock = apply_search_action(side.game, action)
        match.resolve_lock(turn, lock)
        pb = max(pb, match.player.game.b2b_chain)
        ab = max(ab, match.ai.game.b2b_chain)
        turns += 1
        turn = "ai" if turn == "player" else "player"
    if not swapped:
        winner = match.winner or "draw"; c, b = match.player, match.ai; cb, bb = pb, ab
    else:
        winner = {"player": "ai", "ai": "player", None: "draw", "draw": "draw"}[match.winner]; c, b = match.ai, match.player; cb, bb = ab, pb
    return {"winner": winner, "candidateAttack": c.game.attack, "baselineAttack": b.game.attack, "candidateSent": c.sent, "baselineSent": b.sent, "candidateCanceled": c.canceled, "baselineCanceled": b.canceled, "candidateReceived": c.received, "baselineReceived": b.received, "candidateTopout": int(c.game.game_over), "baselineTopout": int(b.game.game_over), "candidateMaxB2B": cb, "baselineMaxB2B": bb}


def versus(candidate: str, games: int = 6):
    rows = [_versus_leg(970001 + (i // 2) * 47, candidate, bool(i % 2)) for i in range(games)]
    n = len(rows)
    mean = lambda key: sum(x[key] for x in rows) / n
    return {"candidate": candidate, "games": n, "wins": sum(x["winner"] == "player" for x in rows), "losses": sum(x["winner"] == "ai" for x in rows), "draws": sum(x["winner"] == "draw" for x in rows), "candidateAttack": mean("candidateAttack"), "baselineAttack": mean("baselineAttack"), "candidateSent": mean("candidateSent"), "baselineSent": mean("baselineSent"), "candidateCanceled": mean("candidateCanceled"), "baselineCanceled": mean("baselineCanceled"), "candidateReceived": mean("candidateReceived"), "baselineReceived": mean("baselineReceived"), "candidateTopouts": sum(x["candidateTopout"] for x in rows), "baselineTopouts": sum(x["baselineTopout"] for x in rows), "candidateMaxB2B": max(x["candidateMaxB2B"] for x in rows), "baselineMaxB2B": max(x["baselineMaxB2B"] for x in rows), "perGame": rows}


def main():
    short = _solo_parallel(CANDIDATES, SHORT_SEEDS, 80)
    base_short = short["baseline"]
    eligible = [short[n] for n in CANDIDATES[1:] if short[n]["topouts"] <= base_short["topouts"] and short[n]["completed"] >= base_short["completed"]]
    eligible.sort(key=_solo_rank, reverse=True)
    finalists = [x["candidate"] for x in eligible[:3]]
    fresh = _solo_parallel(["baseline", *finalists], FRESH_SEEDS, 180)
    base_fresh = fresh["baseline"]
    versus_names = []
    for name in finalists:
        x = fresh[name]
        safer = x["topouts"] < base_fresh["topouts"] or x["completed"] > base_fresh["completed"]
        firepower = x["attackPerPiece"] >= base_fresh["attackPerPiece"] * 1.01
        neutral = x["topouts"] <= base_fresh["topouts"] and x["completed"] >= base_fresh["completed"]
        if neutral and (safer or firepower):
            versus_names.append(name)
    versus_names.sort(key=lambda n: _solo_rank(fresh[n]), reverse=True)
    versus_names = versus_names[:2]
    versus_results = {name: versus(name) for name in versus_names}
    accepted = None
    for name in versus_names:
        v, f = versus_results[name], fresh[name]
        fresh_ok = f["attackPerPiece"] >= base_fresh["attackPerPiece"] and f["topouts"] <= base_fresh["topouts"] and f["completed"] >= base_fresh["completed"]
        pressure_ok = v["candidateSent"] >= v["baselineSent"] and v["candidateReceived"] <= v["baselineReceived"]
        safety_ok = v["candidateTopouts"] <= v["baselineTopouts"]
        if fresh_ok and safety_ok and (v["wins"] > v["losses"] or pressure_ok):
            accepted = name
            break
    output = {"candidates": list(CANDIDATES), "short": short, "shortFinalists": finalists, "fresh": fresh, "versusFinalists": versus_names, "versus": versus_results, "accepted": accepted}
    path = Path("data/self-improve/latest-tournament.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
