from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from typing import Sequence

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, apply_search_action, clone_game, rank_search_actions
from minoflux_ai.features import extract_board_features
from minoflux_ai.search import PlacementEvaluation, SearchAction, SearchScorer, choose_search_action
from minoflux_ai.versus_search import (
    DEFAULT_VERSUS_SEARCH_CONFIG,
    DEFAULT_VERSUS_WEIGHTS,
    clone_versus_match,
    score_versus_state,
)
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
    "next_clean_frontier2",
    "next_attack_frontier2",
    "next_tspin_frontier2",
    "garbage_clean_frontier",
)

SHORT_SEEDS = (810001, 810019)
FRESH_SEEDS = (930001, 930041, 930083)
SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
)


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def _t_supply(game: Game) -> int:
    total = int(game.current == "T") + int(game.hold_piece == "T")
    for i, piece in enumerate(game.queue):
        if i >= 6:
            break
        total += int(piece == "T")
    return total


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


def _next_frontier(game: Game, mode: str) -> float:
    ranked = rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=6)
    if not ranked:
        return -4.0
    if mode == "clean":
        clean = sum(ev.features.new_holes == 0 and not ev.features.game_over for _, ev in ranked)
        safe = sum(ev.features.board.max_height <= 15 and not ev.features.game_over for _, ev in ranked)
        return 0.16 * clean + 0.06 * safe
    if mode == "attack":
        attacks = sorted((ev.features.attack for _, ev in ranked), reverse=True)
        return 0.12 * sum(attacks[:3]) + 0.10 * sum(ev.features.spin_lines for _, ev in ranked)
    spins = sum(ev.features.spin_lines > 0 for _, ev in ranked)
    slots = max((ev.features.board.t_spin_slots for _, ev in ranked), default=0)
    return 0.24 * spins + 0.08 * slots


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
        for ev in evaluations:
            f = ev.features
            b = f.board
            bonus = 0.0
            child = None
            if self.name in {
                "garbage_slot_floor", "garbage_slot_supply_floor", "garbage_danger_variance",
                "edge_garbage_robustness", "t_near_worst_recovery", "garbage_clean_frontier",
            }:
                child = _candidate_child(game, ev)
                stressed = []
                holes = (3, 4, 5, 6) if game.width >= 7 else tuple(range(game.width))
                if self.name == "edge_garbage_robustness":
                    holes = (0, 1, game.width - 2, game.width - 1)
                for hole in holes:
                    stressed.append(extract_board_features(_stress_board(child.board, child.width, hole)))
                if self.name == "garbage_slot_floor":
                    bonus = 1.10 * min((_slot_quality(x) for x in stressed), default=0.0)
                elif self.name == "garbage_slot_supply_floor":
                    floor = min((x.t_spin_slots for x in stressed), default=0)
                    bonus = 0.34 * min(floor, supply) - 0.24 * max(0, floor - supply)
                elif self.name == "garbage_danger_variance":
                    vals = [_danger(x) for x in stressed]
                    mean = sum(vals) / max(1, len(vals))
                    variance = sum((x - mean) ** 2 for x in vals) / max(1, len(vals))
                    bonus = -0.012 * math.sqrt(variance)
                elif self.name == "edge_garbage_robustness":
                    bonus = -0.030 * max((_danger(x) for x in stressed), default=0.0)
                elif self.name == "t_near_worst_recovery":
                    urgency = max(0.0, (3.0 - next_t) / 3.0)
                    bonus = urgency * (0.85 * min((_slot_quality(x) for x in stressed), default=0.0) - 0.012 * max((_danger(x) for x in stressed), default=0.0))
                else:
                    # Count clean immediate replies after a representative 4-line center spike.
                    stressed_game = clone_game(child)
                    stressed_game.board = _stress_board(child.board, child.width, child.width // 2 - 1)
                    bonus = _next_frontier(stressed_game, "clean")
            elif self.name == "hold_t_emergency_flex":
                high = max(0.0, (before.max_height - 12) / 6.0)
                held_t = int(game.hold_piece == "T")
                safety_gain = (before.max_height - b.max_height) + 0.6 * max(0, before.holes - b.holes)
                slot_term = min(1, b.t_spin_slots)
                bonus = high * held_t * (0.20 * safety_gain + 0.10 * slot_term) - high * held_t * 0.15 * max(0, f.new_holes)
            elif self.name == "b2b_clean_conversion":
                difficult = f.spin_lines > 0 or f.lines == 4
                if game.back_to_back:
                    bonus = (0.34 if difficult else (-0.30 if f.lines > 0 else 0.0)) + 0.08 * f.attack - 0.04 * b.holes
            elif self.name == "t_conversion_survival":
                if game.current == "T":
                    clean = 1.0 / (1.0 + b.holes + b.max_height / 8.0)
                    bonus = clean * (0.42 * f.spin_lines + 0.10 * f.attack) - 0.10 * max(0, -f.t_spin_slot_delta)
            elif self.name == "slot_preserve_safe_height":
                safe = max(0.0, (17.0 - b.max_height) / 17.0)
                bonus = 0.24 * safe * max(0, f.t_spin_slot_delta) - 0.18 * max(0, -f.t_spin_slot_delta) * (1.0 - safe)
            elif self.name == "downstack_slot_bridge":
                hole_relief = max(0, before.hole_depth - b.hole_depth)
                bonus = 0.035 * hole_relief * (1.0 + 0.35 * min(2, b.t_spin_slots)) if f.lines > 0 else 0.0
            elif self.name.startswith("next_"):
                child = _candidate_child(game, ev)
                mode = "clean" if self.name == "next_clean_frontier2" else "attack" if self.name == "next_attack_frontier2" else "tspin"
                bonus = _next_frontier(child, mode)
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
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "topout": int(game.game_over),
        "completed": int(not game.game_over and game.pieces_placed >= max_pieces),
        "tSpins": spins,
        "spinLines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "maxB2B": max_b2b,
        "holes": features.holes,
        "holeDepth": features.hole_depth,
        "maxHeight": features.max_height,
    }


def solo(candidate: str, seeds: Sequence[int], max_pieces: int):
    rows = [_solo_one((candidate, seed, max_pieces)) for seed in seeds]
    pieces = sum(x["pieces"] for x in rows)
    attack = sum(x["attack"] for x in rows)
    n = len(rows)
    return {
        "candidate": candidate,
        "games": n,
        "pieces": pieces,
        "attack": attack,
        "attackPerPiece": attack / max(1, pieces),
        "topouts": sum(x["topout"] for x in rows),
        "completed": sum(x["completed"] for x in rows),
        "tSpins": sum(x["tSpins"] for x in rows),
        "spinLines": sum(x["spinLines"] for x in rows),
        "tsd": sum(x["tsd"] for x in rows),
        "tst": sum(x["tst"] for x in rows),
        "maxB2B": max((x["maxB2B"] for x in rows), default=0),
        "meanHoles": sum(x["holes"] for x in rows) / n,
        "meanHoleDepth": sum(x["holeDepth"] for x in rows) / n,
        "meanMaxHeight": sum(x["maxHeight"] for x in rows) / n,
        "perGame": rows,
    }


def _solo_parallel(candidates: Sequence[str], seeds: Sequence[int], max_pieces: int):
    workers = min(len(candidates), 12)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_solo_candidate_job, [(c, tuple(seeds), max_pieces) for c in candidates]))
    return {x["candidate"]: x for x in results}


def _solo_candidate_job(args):
    return solo(*args)


def _solo_rank(item):
    # Safety first, then firepower, then useful spin/B2B output.
    return (
        item["completed"] - item["topouts"],
        item["attackPerPiece"],
        item["spinLines"],
        item["tsd"] + item["tst"],
        item["maxB2B"],
    )


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
    scorer = CandidateScorer(candidate)
    ranked = rank_search_actions(own.game, DEFAULT_WEIGHTS, cfg.placement_search, limit=cfg.candidate_width, scorer=scorer)
    if not ranked:
        return None
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
                if worst is None or reply_score < worst:
                    worst = reply_score
            if worst is not None:
                score = worst
        key = (score, resolution.sent_lines, resolution.canceled_lines, -len(action.placement.path), -int(action.use_hold))
        if best is None or key > best[0]:
            best = (key, action)
    return None if best is None else best[1]


def _versus_leg(seed: int, candidate: str, swapped: bool, max_turns: int = 120):
    match = VersusMatch(seed, garbage_cap=8)
    turn = "player"
    turns = 0
    player_max_b2b = ai_max_b2b = 0
    while match.winner is None and turns < max_turns:
        logical_candidate_side = "ai" if swapped else "player"
        model = candidate if turn == logical_candidate_side else "baseline"
        action = _choose_versus(match, turn, model)
        if action is None:
            side = match.player if turn == "player" else match.ai
            side.game.game_over = True
            match._update_winner()
            break
        side = match.player if turn == "player" else match.ai
        lock = apply_search_action(side.game, action)
        match.resolve_lock(turn, lock)
        player_max_b2b = max(player_max_b2b, match.player.game.b2b_chain)
        ai_max_b2b = max(ai_max_b2b, match.ai.game.b2b_chain)
        turns += 1
        turn = "ai" if turn == "player" else "player"
    # Remap physical sides to candidate=player, baseline=ai.
    if not swapped:
        winner = match.winner or "draw"
        c, b = match.player, match.ai
        cb2b, bb2b = player_max_b2b, ai_max_b2b
    else:
        winner = {"player": "ai", "ai": "player", None: "draw", "draw": "draw"}[match.winner]
        c, b = match.ai, match.player
        cb2b, bb2b = ai_max_b2b, player_max_b2b
    return {
        "winner": winner,
        "candidateAttack": c.game.attack,
        "baselineAttack": b.game.attack,
        "candidateSent": c.sent,
        "baselineSent": b.sent,
        "candidateCanceled": c.canceled,
        "baselineCanceled": b.canceled,
        "candidateReceived": c.received,
        "baselineReceived": b.received,
        "candidateTopout": int(c.game.game_over),
        "baselineTopout": int(b.game.game_over),
        "candidateMaxB2B": cb2b,
        "baselineMaxB2B": bb2b,
    }


def versus(candidate: str, games: int = 8):
    rows = []
    for i in range(games):
        rows.append(_versus_leg(970001 + (i // 2) * 47, candidate, bool(i % 2)))
    n = len(rows)
    mean = lambda key: sum(x[key] for x in rows) / n
    return {
        "candidate": candidate,
        "games": n,
        "wins": sum(x["winner"] == "player" for x in rows),
        "losses": sum(x["winner"] == "ai" for x in rows),
        "draws": sum(x["winner"] == "draw" for x in rows),
        "candidateAttack": mean("candidateAttack"),
        "baselineAttack": mean("baselineAttack"),
        "candidateSent": mean("candidateSent"),
        "baselineSent": mean("baselineSent"),
        "candidateCanceled": mean("candidateCanceled"),
        "baselineCanceled": mean("baselineCanceled"),
        "candidateReceived": mean("candidateReceived"),
        "baselineReceived": mean("baselineReceived"),
        "candidateTopouts": sum(x["candidateTopout"] for x in rows),
        "baselineTopouts": sum(x["baselineTopout"] for x in rows),
        "candidateMaxB2B": max(x["candidateMaxB2B"] for x in rows),
        "baselineMaxB2B": max(x["baselineMaxB2B"] for x in rows),
        "perGame": rows,
    }


def main():
    short = _solo_parallel(CANDIDATES, SHORT_SEEDS, 110)
    baseline_short = short["baseline"]
    nonbase = [short[name] for name in CANDIDATES if name != "baseline"]
    # Require no obvious safety regression in the cheap stage, then take best three.
    eligible = [x for x in nonbase if x["topouts"] <= baseline_short["topouts"] and x["completed"] >= baseline_short["completed"]]
    eligible.sort(key=_solo_rank, reverse=True)
    finalists = [x["candidate"] for x in eligible[:3]]

    fresh_names = ["baseline", *finalists]
    fresh = _solo_parallel(fresh_names, FRESH_SEEDS, 240)
    baseline_fresh = fresh["baseline"]
    versus_names = []
    for name in finalists:
        x = fresh[name]
        safer = x["topouts"] < baseline_fresh["topouts"] or x["completed"] > baseline_fresh["completed"]
        firepower = x["attackPerPiece"] >= baseline_fresh["attackPerPiece"] * 1.01
        neutral_safe = x["topouts"] <= baseline_fresh["topouts"] and x["completed"] >= baseline_fresh["completed"]
        if neutral_safe and (safer or firepower):
            versus_names.append(name)
    versus_names.sort(key=lambda n: _solo_rank(fresh[n]), reverse=True)
    versus_names = versus_names[:2]
    versus_results = {name: versus(name, 8) for name in versus_names}

    accepted = None
    for name in versus_names:
        v = versus_results[name]
        f = fresh[name]
        win_ok = v["wins"] > v["losses"]
        pressure_ok = v["candidateSent"] >= v["baselineSent"] and v["candidateReceived"] <= v["baselineReceived"]
        safety_ok = v["candidateTopouts"] <= v["baselineTopouts"]
        fresh_ok = f["attackPerPiece"] >= baseline_fresh["attackPerPiece"] and f["topouts"] <= baseline_fresh["topouts"] and f["completed"] >= baseline_fresh["completed"]
        if fresh_ok and safety_ok and (win_ok or pressure_ok):
            accepted = name
            break

    output = {
        "candidates": list(CANDIDATES),
        "short": short,
        "shortFinalists": finalists,
        "fresh": fresh,
        "versusFinalists": versus_names,
        "versus": versus_results,
        "accepted": accepted,
    }
    path = Path("data/self-improve/latest-tournament.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
