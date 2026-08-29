from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean
from typing import Callable

from minoflux_ai import DEFAULT_WEIGHTS, HeuristicWeights, SearchConfig
from minoflux_ai.search import SearchAction, apply_search_action, clone_game, rank_search_actions
from minoflux_ai.versus_search import (
    VersusSearchConfig,
    clone_versus_match,
    score_versus_state,
)
from minoflux_engine import T_SPIN_DOUBLE, T_SPIN_TRIPLE, VersusMatch

SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
)
VERSUS = VersusSearchConfig(placement_search=SEARCH, candidate_width=6, opponent_reply_width=1)
ROOT_WIDTH = 8
NEXT_WIDTH = 6

CANDIDATES = (
    "next_survival_width",
    "next_clean_width",
    "next_recovery_width",
    "next_attack_ceiling",
    "next_attack_floor",
    "next_score_floor",
    "next_slot_survival",
    "next_t_conversion_width",
    "next_b2b_width",
    "next_clear_width",
    "next_low_stack_width",
    "hold_branch_flexibility",
    "balanced_future_options",
    "t_window_branch_reserve",
)


def _next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 4:
            break
    return 6


def _future_stats(game, action: SearchAction, weights: HeuristicWeights) -> dict[str, float]:
    child = clone_game(game)
    apply_search_action(child, action)
    if child.game_over:
        return {
            "survival": 0.0,
            "clean": 0.0,
            "recovery": -10.0,
            "attack_ceiling": 0.0,
            "attack_floor": 0.0,
            "score_floor": -100.0,
            "slot_survival": 0.0,
            "t_conversion": 0.0,
            "b2b_options": 0.0,
            "clear_options": 0.0,
            "low_stack": 0.0,
            "hold_flex": 0.0,
            "t_distance": float(_next_t_distance(child)),
        }
    next_ranked = rank_search_actions(child, weights, SEARCH, limit=NEXT_WIDTH)
    if not next_ranked:
        return {
            "survival": 0.0,
            "clean": 0.0,
            "recovery": -10.0,
            "attack_ceiling": 0.0,
            "attack_floor": 0.0,
            "score_floor": -100.0,
            "slot_survival": 0.0,
            "t_conversion": 0.0,
            "b2b_options": 0.0,
            "clear_options": 0.0,
            "low_stack": 0.0,
            "hold_flex": 0.0,
            "t_distance": float(_next_t_distance(child)),
        }
    evals = [evaluation for _, evaluation in next_ranked]
    actions = [next_action for next_action, _ in next_ranked]
    n = float(len(evals))
    current_holes = min(item.features.board.holes for item in evals)
    current_depth = min(item.features.board.hole_depth for item in evals)
    attacks = sorted((item.features.attack for item in evals), reverse=True)
    scores = sorted((item.score for item in evals), reverse=True)
    slot_counts = [item.features.board.t_spin_slots for item in evals]
    return {
        "survival": sum(not item.features.game_over for item in evals) / n,
        "clean": sum(item.features.new_holes == 0 for item in evals) / n,
        "recovery": max(0.0, 3.0 - current_holes) + max(0.0, 8.0 - current_depth) / 4.0,
        "attack_ceiling": float(max(attacks, default=0)),
        "attack_floor": float(min(attacks[:3], default=0)),
        "score_floor": float(max(-40.0, min(40.0, min(scores[:3], default=-40.0)))),
        "slot_survival": float(mean(slot_counts)) if slot_counts else 0.0,
        "t_conversion": float(sum(item.features.spin_lines > 0 for item in evals)) if child.current == "T" else 0.0,
        "b2b_options": float(sum(item.features.spin_lines > 0 or item.features.lines == 4 for item in evals)),
        "clear_options": float(sum(item.features.lines > 0 for item in evals)),
        "low_stack": sum(item.features.board.max_height <= 12 for item in evals) / n,
        "hold_flex": float(any(a.use_hold for a in actions) and any(not a.use_hold for a in actions)),
        "t_distance": float(_next_t_distance(child)),
    }


def _bonus(mode: str, stats: dict[str, float], *, back_to_back: bool) -> float:
    if mode == "next_survival_width":
        return 3.0 * stats["survival"]
    if mode == "next_clean_width":
        return 2.5 * stats["clean"]
    if mode == "next_recovery_width":
        return 0.45 * stats["recovery"]
    if mode == "next_attack_ceiling":
        return 0.45 * stats["attack_ceiling"]
    if mode == "next_attack_floor":
        return 0.75 * stats["attack_floor"]
    if mode == "next_score_floor":
        return 0.08 * stats["score_floor"]
    if mode == "next_slot_survival":
        return 0.65 * stats["slot_survival"]
    if mode == "next_t_conversion_width":
        return 0.90 * stats["t_conversion"]
    if mode == "next_b2b_width":
        return (0.70 if back_to_back else 0.30) * stats["b2b_options"]
    if mode == "next_clear_width":
        return 0.40 * stats["clear_options"]
    if mode == "next_low_stack_width":
        return 2.0 * stats["low_stack"]
    if mode == "hold_branch_flexibility":
        return 1.25 * stats["hold_flex"]
    if mode == "balanced_future_options":
        return 1.3 * stats["survival"] + 1.0 * stats["clean"] + 0.35 * stats["attack_floor"] + 0.20 * stats["recovery"]
    if mode == "t_window_branch_reserve":
        urgency = max(0.0, (3.0 - stats["t_distance"]) / 3.0)
        return urgency * (0.70 * stats["slot_survival"] + 1.10 * stats["clean"] + 0.45 * stats["t_conversion"])
    raise ValueError(mode)


def choose_action(game, mode: str | None, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> SearchAction | None:
    if mode is None:
        ranked = rank_search_actions(game, weights, SEARCH, limit=1)
        return ranked[0][0] if ranked else None
    ranked = rank_search_actions(game, weights, SEARCH, limit=ROOT_WIDTH)
    if not ranked:
        return None
    best = None
    best_key = None
    for action, evaluation in ranked:
        stats = _future_stats(game, action, weights)
        score = evaluation.score + _bonus(mode, stats, back_to_back=game.back_to_back)
        key = (
            score,
            evaluation.features.attack,
            evaluation.features.spin_lines,
            evaluation.features.lines,
            -evaluation.features.board.holes,
            -evaluation.features.board.max_height,
            -int(action.use_hold),
            -action.placement.rotation * 100 - action.placement.x,
        )
        if best_key is None or key > best_key:
            best_key = key
            best = action
    return best


def play_solo(seed: int, pieces: int, mode: str | None) -> dict[str, object]:
    from minoflux_engine import Game

    game = Game(seed)
    spins = tsd = tst = spin_lines = perfect_clears = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < pieces:
        action = choose_action(game, mode)
        if action is None:
            break
        result = apply_search_action(game, action)
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
        tsd += int(result.spin == T_SPIN_DOUBLE)
        tst += int(result.spin == T_SPIN_TRIPLE)
        perfect_clears += int(result.perfect_clear)
        max_b2b = max(max_b2b, game.b2b_chain)
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "app": game.attack / max(1, game.pieces_placed),
        "topout": game.game_over,
        "completed": (not game.game_over and game.pieces_placed >= pieces),
        "spins": spins,
        "spin_lines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "perfect_clears": perfect_clears,
        "max_b2b": max_b2b,
    }


def aggregate_solo(mode: str | None, seeds: list[int], pieces: int) -> dict[str, object]:
    games = [play_solo(seed, pieces, mode) for seed in seeds]
    total_pieces = sum(int(item["pieces"]) for item in games)
    total_attack = sum(int(item["attack"]) for item in games)
    return {
        "mode": mode or "baseline",
        "games": len(games),
        "pieces": total_pieces,
        "attack": total_attack,
        "app": total_attack / max(1, total_pieces),
        "topouts": sum(bool(item["topout"]) for item in games),
        "completed": sum(bool(item["completed"]) for item in games),
        "spins": sum(int(item["spins"]) for item in games),
        "spin_lines": sum(int(item["spin_lines"]) for item in games),
        "tsd": sum(int(item["tsd"]) for item in games),
        "tst": sum(int(item["tst"]) for item in games),
        "perfect_clears": sum(int(item["perfect_clears"]) for item in games),
        "max_b2b": max((int(item["max_b2b"]) for item in games), default=0),
        "per_game": games,
    }


def _side_name_other(name: str) -> str:
    return "ai" if name == "player" else "player"


def choose_candidate_versus(match: VersusMatch, side_name: str, mode: str) -> SearchAction | None:
    own = match.side(side_name)
    ranked = rank_search_actions(own.game, DEFAULT_WEIGHTS, SEARCH, limit=VERSUS.candidate_width)
    if not ranked:
        return None
    opponent_name = _side_name_other(side_name)
    best_action = None
    best_key = None
    for action, evaluation in ranked:
        stats = _future_stats(own.game, action, DEFAULT_WEIGHTS)
        after = clone_versus_match(match)
        result = apply_search_action(after.side(side_name).game, action)
        resolution = after.resolve_lock(side_name, result)
        score = score_versus_state(
            after,
            side_name,
            resolution=resolution,
            solo_score=evaluation.score + _bonus(mode, stats, back_to_back=own.game.back_to_back),
            path_length=len(action.placement.path),
            action_side=side_name,
        )
        if VERSUS.opponent_reply_width > 0 and after.winner is None:
            opponent = after.side(opponent_name)
            replies = rank_search_actions(opponent.game, DEFAULT_WEIGHTS, SEARCH, limit=VERSUS.opponent_reply_width)
            worst = None
            for reply, reply_eval in replies:
                replied = clone_versus_match(after)
                reply_result = apply_search_action(replied.side(opponent_name).game, reply)
                reply_resolution = replied.resolve_lock(opponent_name, reply_result)
                reply_score = score_versus_state(
                    replied,
                    side_name,
                    resolution=reply_resolution,
                    solo_score=-reply_eval.score,
                    path_length=len(reply.placement.path),
                    action_side=opponent_name,
                )
                worst = reply_score if worst is None else min(worst, reply_score)
            if worst is not None:
                score = worst
        key = (
            score,
            resolution.sent_lines,
            resolution.canceled_lines,
            -len(action.placement.path),
            -int(action.use_hold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_action = action
    return best_action


def choose_baseline_versus(match: VersusMatch, side_name: str) -> SearchAction | None:
    from minoflux_ai.versus_search import choose_versus_action
    choice = choose_versus_action(match, side_name, DEFAULT_WEIGHTS, VERSUS)
    return None if choice is None else choice.action


def play_versus(seed: int, mode: str, swapped: bool, max_turns: int = 100) -> dict[str, object]:
    match = VersusMatch(seed, garbage_cap=8)
    candidate_side = "ai" if swapped else "player"
    turn_side = "player"
    turns = 0
    max_b2b = {"player": 0, "ai": 0}
    while match.winner is None and turns < max_turns:
        if turn_side == candidate_side:
            action = choose_candidate_versus(match, turn_side, mode)
        else:
            action = choose_baseline_versus(match, turn_side)
        if action is None:
            match.side(turn_side).game.game_over = True
            match._update_winner()
            break
        result = apply_search_action(match.side(turn_side).game, action)
        match.resolve_lock(turn_side, result)
        max_b2b[turn_side] = max(max_b2b[turn_side], match.side(turn_side).game.b2b_chain)
        turns += 1
        turn_side = _side_name_other(turn_side)
    winner = match.winner or "draw"
    if candidate_side == "ai":
        winner = {"player": "baseline", "ai": "candidate", "draw": "draw"}[winner]
        candidate = match.ai
        baseline = match.player
        candidate_b2b = max_b2b["ai"]
        baseline_b2b = max_b2b["player"]
    else:
        winner = {"player": "candidate", "ai": "baseline", "draw": "draw"}[winner]
        candidate = match.player
        baseline = match.ai
        candidate_b2b = max_b2b["player"]
        baseline_b2b = max_b2b["ai"]
    return {
        "seed": seed,
        "swapped": swapped,
        "winner": winner,
        "turns": turns,
        "candidate_attack": candidate.game.attack,
        "baseline_attack": baseline.game.attack,
        "candidate_sent": candidate.sent,
        "baseline_sent": baseline.sent,
        "candidate_canceled": candidate.canceled,
        "baseline_canceled": baseline.canceled,
        "candidate_received": candidate.received,
        "baseline_received": baseline.received,
        "candidate_max_b2b": candidate_b2b,
        "baseline_max_b2b": baseline_b2b,
        "candidate_topout": candidate.game.game_over,
        "baseline_topout": baseline.game.game_over,
    }


def aggregate_versus(mode: str, games: int = 6, seed_base: int = 700001, seed_step: int = 193) -> dict[str, object]:
    results = []
    for index in range(games):
        results.append(play_versus(seed_base + (index // 2) * seed_step, mode, bool(index % 2)))
    def avg(key: str) -> float:
        return sum(float(item[key]) for item in results) / len(results)
    return {
        "mode": mode,
        "games": games,
        "candidate_wins": sum(item["winner"] == "candidate" for item in results),
        "baseline_wins": sum(item["winner"] == "baseline" for item in results),
        "draws": sum(item["winner"] == "draw" for item in results),
        "candidate_attack": avg("candidate_attack"),
        "baseline_attack": avg("baseline_attack"),
        "candidate_sent": avg("candidate_sent"),
        "baseline_sent": avg("baseline_sent"),
        "candidate_canceled": avg("candidate_canceled"),
        "baseline_canceled": avg("baseline_canceled"),
        "candidate_received": avg("candidate_received"),
        "baseline_received": avg("baseline_received"),
        "candidate_max_b2b": avg("candidate_max_b2b"),
        "baseline_max_b2b": avg("baseline_max_b2b"),
        "candidate_topouts": sum(bool(item["candidate_topout"]) for item in results),
        "baseline_topouts": sum(bool(item["baseline_topout"]) for item in results),
        "per_game": results,
    }


def quality_key(item: dict[str, object]) -> tuple[float, ...]:
    return (
        -float(item["topouts"]),
        float(item["completed"]),
        float(item["app"]),
        float(item["attack"]),
        float(item["tsd"]) + 1.5 * float(item["tst"]),
        float(item["spin_lines"]),
        float(item["max_b2b"]),
    )


def main() -> None:
    short_seeds = [31001, 31098]
    fresh_seeds = [91001, 91098, 91195]
    short = [aggregate_solo(None, short_seeds, 90)]
    short.extend(aggregate_solo(mode, short_seeds, 90) for mode in CANDIDATES)
    baseline_short = short[0]
    eligible = [
        item for item in short[1:]
        if int(item["topouts"]) <= int(baseline_short["topouts"])
        and int(item["completed"]) >= int(baseline_short["completed"])
    ]
    eligible.sort(key=quality_key, reverse=True)
    short_finalists = [str(item["mode"]) for item in eligible[:3]]

    fresh = [aggregate_solo(None, fresh_seeds, 180)]
    fresh.extend(aggregate_solo(mode, fresh_seeds, 180) for mode in short_finalists)
    baseline_fresh = fresh[0]
    viable = [
        item for item in fresh[1:]
        if int(item["topouts"]) <= int(baseline_fresh["topouts"])
        and int(item["completed"]) >= int(baseline_fresh["completed"])
        and float(item["app"]) >= float(baseline_fresh["app"]) * 1.01
    ]
    viable.sort(key=quality_key, reverse=True)
    versus_modes = [str(item["mode"]) for item in viable[:2]]
    versus = [aggregate_versus(mode) for mode in versus_modes]

    payload = {
        "candidate_count": len(CANDIDATES),
        "candidates": list(CANDIDATES),
        "short": short,
        "short_finalists": short_finalists,
        "fresh": fresh,
        "versus_modes": versus_modes,
        "versus": versus,
    }
    target = Path("tournament-results.json")
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
