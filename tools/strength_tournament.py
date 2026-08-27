from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from minoflux_engine import VersusMatch
from minoflux_ai import heuristic, search
from minoflux_ai.benchmark import run_heuristic_game
from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action
from minoflux_ai.versus_search import DEFAULT_VERSUS_SEARCH_CONFIG, choose_versus_action

ORIGINAL_RANK = heuristic.rank_placements
ACTIVE = "baseline"


def _distance_to_t(game) -> int:
    pieces = [game.current, *list(game.queue)]
    try:
        return pieces.index("T")
    except ValueError:
        return 8


def _bonus(game, ev: PlacementEvaluation, name: str) -> float:
    f = ev.features
    b = f.board
    before = extract_board_features(game.board)
    holes_removed = max(0, before.holes - b.holes)
    depth_removed = max(0, before.hole_depth - b.hole_depth)
    difficult = (f.spin is not None and f.lines > 0) or f.lines == 4
    tdist = _distance_to_t(game)
    active_b2b = bool(getattr(game, "back_to_back", False))

    if name == "baseline": return 0.0
    if name == "t_proximity_slots": return 0.65 * b.t_spin_slots / (1.0 + min(tdist, 6))
    if name == "t_proximity_preserve": return 0.55 * max(0, -f.t_spin_slot_delta) * (-1 if tdist <= 3 else 0)
    if name == "t_ready_b2b": return 0.45 * b.t_spin_slots * (1.5 if active_b2b else 1.0) / (1.0 + min(tdist, 5))
    if name == "b2b_break_guard": return -0.65 if active_b2b and f.lines > 0 and not difficult else 0.0
    if name == "b2b_extend": return 0.50 if active_b2b and difficult else 0.0
    if name == "b2b_start": return 0.35 if (not active_b2b) and difficult else 0.0
    if name == "downstack_holes": return 0.55 * holes_removed
    if name == "downstack_depth": return 0.10 * depth_removed
    if name == "downstack_combo": return 0.40 * holes_removed + 0.12 * depth_removed + (0.18 if f.lines and holes_removed else 0.0)
    if name == "danger_softcap": return -0.16 * max(0, b.max_height - 12) ** 2
    if name == "danger_holes": return -0.08 * max(0, b.max_height - 10) * b.holes
    if name == "clean_stack_flat": return -0.05 * b.bumpiness if b.holes == 0 and b.max_height <= 10 else 0.0
    if name == "clean_t_well": return 0.22 * b.t_spin_slots - 0.03 * b.wells if b.holes == 0 else 0.0
    if name == "slot_downstack": return 0.28 * max(0, f.t_spin_slot_delta) + 0.38 * holes_removed + 0.08 * depth_removed
    raise ValueError(name)


def experimental_rank(game, weights=DEFAULT_WEIGHTS, *, placements=None, limit=None):
    ranked = ORIGINAL_RANK(game, weights, placements=placements, limit=None)
    adjusted = [replace(ev, score=ev.score + _bonus(game, ev, ACTIVE)) for ev in ranked]
    adjusted.sort(key=heuristic._placement_key, reverse=True)
    if limit is None:
        return tuple(adjusted)
    return tuple(adjusted[: max(0, int(limit))])


def run(candidate: str, seeds: list[int], pieces: int) -> dict[str, object]:
    global ACTIVE
    ACTIVE = candidate
    search.rank_placements = experimental_rank
    games = [run_heuristic_game(seed, pieces, DEFAULT_WEIGHTS, DEFAULT_SEARCH_CONFIG) for seed in seeds]
    total_pieces = sum(g.pieces for g in games)
    attack = sum(g.attack for g in games)
    return {
        "candidate": candidate,
        "games": len(games),
        "pieces": total_pieces,
        "attack": attack,
        "attackPerPiece": attack / total_pieces if total_pieces else 0.0,
        "topouts": sum(g.topout for g in games),
        "completed": sum(g.completed for g in games),
        "spins": sum(g.spins for g in games),
        "spinLines": sum(g.spin_lines for g in games),
        "tsd": sum(g.t_spin_doubles for g in games),
        "tst": sum(g.t_spin_triples for g in games),
        "perGame": [g.__dict__ if hasattr(g, "__dict__") else {k: getattr(g, k) for k in g.__dataclass_fields__} for g in games],
    }


def _play_versus_leg(candidate: str, seed: int, max_turns: int, *, candidate_player: bool) -> dict[str, object]:
    global ACTIVE
    search.rank_placements = experimental_rank
    match = VersusMatch(seed, garbage_cap=8)
    turn_side = "player"
    turns = 0
    max_b2b = {"player": 0, "ai": 0}
    while match.winner is None and turns < max_turns:
        candidate_turn = (turn_side == "player") == candidate_player
        ACTIVE = candidate if candidate_turn else "baseline"
        choice = choose_versus_action(match, turn_side, DEFAULT_WEIGHTS, DEFAULT_VERSUS_SEARCH_CONFIG)
        if choice is None:
            match.side(turn_side).game.game_over = True
            match._update_winner()
            break
        side = match.side(turn_side)
        result = apply_search_action(side.game, choice.action)
        match.resolve_lock(turn_side, result)
        turns += 1
        max_b2b["player"] = max(max_b2b["player"], match.player.game.b2b_chain)
        max_b2b["ai"] = max(max_b2b["ai"], match.ai.game.b2b_chain)
        turn_side = "ai" if turn_side == "player" else "player"

    cand_side = match.player if candidate_player else match.ai
    base_side = match.ai if candidate_player else match.player
    physical_winner = match.winner or "draw"
    if physical_winner == "draw":
        winner = "draw"
    elif (physical_winner == "player") == candidate_player:
        winner = "candidate"
    else:
        winner = "baseline"
    cand_b2b = max_b2b["player" if candidate_player else "ai"]
    base_b2b = max_b2b["ai" if candidate_player else "player"]
    return {
        "seed": seed,
        "candidatePlayer": candidate_player,
        "winner": winner,
        "turns": turns,
        "candidateAttack": cand_side.game.attack,
        "baselineAttack": base_side.game.attack,
        "candidateSent": cand_side.sent,
        "baselineSent": base_side.sent,
        "candidateCanceled": cand_side.canceled,
        "baselineCanceled": base_side.canceled,
        "candidateReceived": cand_side.received,
        "baselineReceived": base_side.received,
        "candidateMaxB2B": cand_b2b,
        "baselineMaxB2B": base_b2b,
        "candidateTopout": cand_side.game.game_over,
        "baselineTopout": base_side.game.game_over,
    }


def run_versus(candidate: str, seeds: list[int], max_turns: int) -> dict[str, object]:
    legs = []
    for seed in seeds:
        legs.append(_play_versus_leg(candidate, seed, max_turns, candidate_player=True))
        legs.append(_play_versus_leg(candidate, seed, max_turns, candidate_player=False))
    count = len(legs)
    def mean(key: str) -> float:
        return sum(float(item[key]) for item in legs) / count
    return {
        "candidate": candidate,
        "games": count,
        "maxTurns": max_turns,
        "candidateWins": sum(item["winner"] == "candidate" for item in legs),
        "baselineWins": sum(item["winner"] == "baseline" for item in legs),
        "draws": sum(item["winner"] == "draw" for item in legs),
        "candidateMeanAttack": mean("candidateAttack"),
        "baselineMeanAttack": mean("baselineAttack"),
        "candidateMeanSent": mean("candidateSent"),
        "baselineMeanSent": mean("baselineSent"),
        "candidateMeanCanceled": mean("candidateCanceled"),
        "baselineMeanCanceled": mean("baselineCanceled"),
        "candidateMeanReceived": mean("candidateReceived"),
        "baselineMeanReceived": mean("baselineReceived"),
        "candidateMeanMaxB2B": mean("candidateMaxB2B"),
        "baselineMeanMaxB2B": mean("baselineMaxB2B"),
        "candidateTopouts": sum(bool(item["candidateTopout"]) for item in legs),
        "baselineTopouts": sum(bool(item["baselineTopout"]) for item in legs),
        "perGame": legs,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--pieces", type=int)
    p.add_argument("--versus-turns", type=int)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if args.versus_turns is not None:
        result = run_versus(args.candidate, seeds, args.versus_turns)
    else:
        if args.pieces is None:
            p.error("--pieces is required outside versus mode")
        result = run(args.candidate, seeds, args.pieces)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
