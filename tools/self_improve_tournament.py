from __future__ import annotations

from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path

import minoflux_ai.heuristic as heuristic
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig
from minoflux_ai.search import apply_search_action, choose_search_action
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_search import VersusSearchConfig
from minoflux_engine import Game

CANDIDATES = (
    "baseline",
    "t_near_garbage_recovery",
    "t_near_garbage_resilience",
    "attack_garbage_reserve",
    "b2b_garbage_recovery",
    "b2b_break_resilience",
    "t_arrival_recovery",
    "slot_loss_garbage_risk",
    "slot_gain_garbage_quality",
    "garbage_safe_attack",
    "danger_garbage_resilience",
    "danger_tspin_recovery",
    "hold_t_garbage_buffer",
    "queue_supply_recovery",
    "clean_pressure_release",
)


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 5:
            break
    return 7


def _t_supply(game: Game) -> int:
    count = int(game.current == "T") + int(game.hold_piece == "T")
    for index, piece in enumerate(game.queue):
        if index >= 5:
            break
        count += int(piece == "T")
    return count


def experimental_score(name: str, game: Game, f: heuristic.PlacementFeatures) -> float:
    if name == "baseline":
        return 0.0
    board = f.board
    urgency = max(0.0, (7.0 - _next_t_distance(game)) / 7.0)
    danger = max(0.0, (board.max_height - 8.0) / 8.0)
    recovery = f.garbage_tspin_recovery
    resilience = f.center_garbage_resilience
    clean = 1.0 / (1.0 + board.holes + board.max_height / 8.0)
    slot_gain = max(0, f.t_spin_slot_delta)
    slot_loss = max(0, -f.t_spin_slot_delta)
    difficult = f.spin_lines > 0 or f.lines == 4

    if name == "t_near_garbage_recovery":
        return 0.70 * urgency * recovery
    if name == "t_near_garbage_resilience":
        return 0.55 * urgency * resilience
    if name == "attack_garbage_reserve":
        return 0.16 * f.attack * (0.65 * recovery + 0.35 * resilience)
    if name == "b2b_garbage_recovery":
        return 0.65 * int(game.back_to_back) * recovery
    if name == "b2b_break_resilience":
        return -0.55 * int(game.back_to_back and f.lines > 0 and not difficult) * (1.0 + danger)
    if name == "t_arrival_recovery":
        return 0.45 * int(game.current == "T") * (f.spin_lines * clean + 0.35 * recovery)
    if name == "slot_loss_garbage_risk":
        return -0.45 * slot_loss * (1.0 + danger + max(0.0, -resilience))
    if name == "slot_gain_garbage_quality":
        return 0.38 * slot_gain * clean * (1.0 + urgency)
    if name == "garbage_safe_attack":
        return 0.30 * f.attack * clean
    if name == "danger_garbage_resilience":
        return 0.90 * danger * resilience
    if name == "danger_tspin_recovery":
        return 0.75 * danger * recovery
    if name == "hold_t_garbage_buffer":
        return 0.32 * int(game.hold_piece == "T") * min(2, board.t_spin_slots) * clean * (1.0 + danger)
    if name == "queue_supply_recovery":
        matched = min(board.t_spin_slots, _t_supply(game))
        return 0.24 * matched * recovery - 0.12 * abs(board.t_spin_slots - _t_supply(game))
    if name == "clean_pressure_release":
        return 0.22 * (f.attack + f.lines) * clean * (1.0 + danger)
    raise ValueError(name)


def install_candidate(name: str):
    candidate_weights = replace(DEFAULT_WEIGHTS)
    original = heuristic._context_score

    def patched(game: Game, features: heuristic.PlacementFeatures, weights):
        score = original(game, features, weights)
        if weights is candidate_weights:
            score += experimental_score(name, game, features)
        return score

    heuristic._context_score = patched
    return candidate_weights, original


def solo(candidate: str, games: int, max_pieces: int, seed_base: int, seed_step: int) -> dict[str, object]:
    candidate_weights, original = install_candidate(candidate)
    config = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.90)
    totals = {
        "pieces": 0, "attack": 0, "lines": 0, "spins": 0, "spin_lines": 0,
        "t_spin_doubles": 0, "t_spin_triples": 0, "topouts": 0, "completed": 0,
        "max_b2b_sum": 0,
    }
    per_game: list[dict[str, object]] = []
    try:
        for index in range(games):
            seed = seed_base + index * seed_step
            game = Game(seed)
            spins = spin_lines = tsd = tst = max_b2b = 0
            while not game.game_over and game.pieces_placed < max_pieces:
                choice = choose_search_action(game, candidate_weights, config)
                if choice is None:
                    break
                result = apply_search_action(game, choice.action)
                if result.spin is not None:
                    spins += 1
                    spin_lines += result.lines
                    tsd += int(result.lines == 2)
                    tst += int(result.lines == 3)
                max_b2b = max(max_b2b, game.b2b_chain)
            completed = not game.game_over and game.pieces_placed >= max_pieces
            item = {
                "seed": seed, "pieces": game.pieces_placed, "attack": game.attack,
                "lines": game.lines, "spins": spins, "spin_lines": spin_lines,
                "t_spin_doubles": tsd, "t_spin_triples": tst,
                "topout": game.game_over, "completed": completed, "max_b2b": max_b2b,
            }
            per_game.append(item)
            for key in ("pieces", "attack", "lines", "spins", "spin_lines", "t_spin_doubles", "t_spin_triples"):
                totals[key] += int(item[key])
            totals["topouts"] += int(game.game_over)
            totals["completed"] += int(completed)
            totals["max_b2b_sum"] += max_b2b
    finally:
        heuristic._context_score = original
    pieces = int(totals["pieces"])
    return {
        "candidate": candidate, "stage": "solo", "games": games, "max_pieces": max_pieces,
        **totals,
        "attack_per_piece": totals["attack"] / pieces if pieces else 0.0,
        "mean_max_b2b": totals["max_b2b_sum"] / games,
        "per_game": per_game,
    }


def versus(candidate: str, games: int, max_turns: int, seed_base: int, seed_step: int) -> dict[str, object]:
    candidate_weights, original = install_candidate(candidate)
    cfg = VersusSearchConfig(
        placement_search=SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4),
        candidate_width=6,
        opponent_reply_width=1,
    )
    try:
        result = run_versus_benchmark(
            games=games, max_turns=max_turns, seed_base=seed_base, seed_step=seed_step,
            player_weights=candidate_weights, ai_weights=DEFAULT_WEIGHTS,
            player_config=cfg, ai_config=cfg, garbage_cap=8,
        )
    finally:
        heuristic._context_score = original
    per_game = [asdict(item) for item in result.per_game]
    return {
        "candidate": candidate, "stage": "versus", **result.to_dict(),
        "playerMeanCanceled": sum(item.player_canceled for item in result.per_game) / games,
        "aiMeanCanceled": sum(item.ai_canceled for item in result.per_game) / games,
        "playerMeanReceived": sum(item.player_received for item in result.per_game) / games,
        "aiMeanReceived": sum(item.ai_received for item in result.per_game) / games,
        "playerMeanMaxB2B": sum(item.player_max_b2b for item in result.per_game) / games,
        "aiMeanMaxB2B": sum(item.ai_max_b2b for item in result.per_game) / games,
        "playerTopouts": sum(item.winner == "ai" for item in result.per_game),
        "aiTopouts": sum(item.winner == "player" for item in result.per_game),
        "perGame": per_game,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--stage", choices=("screen", "fresh", "versus"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.stage == "screen":
        payload = solo(args.candidate, games=2, max_pieces=110, seed_base=41001, seed_step=97)
    elif args.stage == "fresh":
        payload = solo(args.candidate, games=3, max_pieces=260, seed_base=73019, seed_step=131)
    else:
        payload = versus(args.candidate, games=8, max_turns=120, seed_base=91009, seed_step=173)
    payload["requested_stage"] = args.stage
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
