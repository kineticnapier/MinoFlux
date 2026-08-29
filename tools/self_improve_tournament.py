from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import json
import math
import os

from minoflux_engine import Game
from minoflux_ai import heuristic
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementFeatures
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action, choose_search_action
from minoflux_ai.versus_benchmark import run_versus_benchmark

BASE_GAME_OVER = DEFAULT_WEIGHTS.game_over
MARKER_SCALE = 1e-4

CANDIDATES = {
    "baseline": 0,
    "t_near_garbage_option": 1,
    "fragile_attack_cost": 2,
    "fragile_new_hole_cost": 3,
    "b2b_garbage_reserve": 4,
    "hold_t_garbage_reserve": 5,
    "slot_growth_after_garbage": 6,
    "garbage_survival_floor": 7,
    "garbage_downstack_relief": 8,
    "t_arrival_garbage_conversion": 9,
    "high_stack_spike_cost": 10,
    "clean_pressure_after_spike": 11,
    "t_supply_spike_balance": 12,
    "garbage_recovery_consistency": 13,
    "attack_recovery_balance": 14,
}


def weights_for(name: str):
    marker = CANDIDATES[name]
    if marker == 0:
        return DEFAULT_WEIGHTS
    return replace(DEFAULT_WEIGHTS, game_over=BASE_GAME_OVER - marker * MARKER_SCALE)


def marker_from(weights) -> int:
    delta = BASE_GAME_OVER - weights.game_over
    if delta <= MARKER_SCALE / 2:
        return 0
    return int(round(delta / MARKER_SCALE))


def next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 5:
            break
    return 7


def t_supply(game: Game) -> int:
    supply = int(game.current == "T") + int(game.hold_piece == "T")
    for index, piece in enumerate(game.queue):
        if index >= 6:
            break
        supply += int(piece == "T")
    return supply


def experiment_score(marker: int, game: Game, f: PlacementFeatures) -> float:
    if marker == 0:
        return 0.0
    b = f.board
    # Accepted resilience features are negative when a board is fragile. Convert
    # them to a positive risk magnitude for interaction terms.
    center_risk = max(0.0, -f.center_garbage_resilience)
    recovery = f.garbage_tspin_recovery
    recovery_risk = max(0.0, -recovery)
    urgency = max(0.0, (7.0 - next_t_distance(game)) / 7.0)
    supply = t_supply(game)
    surplus_slots = max(0, b.t_spin_slots - supply)
    low_clean = 1.0 / (1.0 + b.holes + b.max_height / 6.0)
    safe_attack = f.attack * low_clean

    if marker == 1:  # preserve usable T-spin options when T is close and garbage lands
        return 0.55 * urgency * recovery
    if marker == 2:  # avoid taking nominal attack that leaves a fragile spike response
        return -0.22 * f.attack * center_risk
    if marker == 3:  # new holes are especially bad if the board also fails a spike test
        return -0.70 * f.new_holes * center_risk
    if marker == 4:  # protect the B2B state only when the post-garbage board remains usable
        return 0.42 * int(game.back_to_back) * recovery
    if marker == 5:  # a held T is more valuable when recovery slots survive garbage
        return 0.40 * int(game.hold_piece == "T") * urgency * recovery
    if marker == 6:  # creating a slot is useful only if it survives representative garbage
        return 0.48 * max(0, f.t_spin_slot_delta) * recovery
    if marker == 7:  # nonlinear floor: sharply reject boards beyond ordinary spike fragility
        excess = max(0.0, center_risk - 0.85)
        return -0.80 * excess * excess
    if marker == 8:  # when fragile, value line clears as immediate downstack access
        return 0.30 * f.lines * center_risk
    if marker == 9:  # T arrival should cash in only if the resulting board survives garbage
        if game.current != "T":
            return 0.0
        return 0.36 * f.spin_lines * recovery - 0.25 * max(0, -f.t_spin_slot_delta) * center_risk
    if marker == 10:  # nonlinear high-stack x spike fragility penalty
        return -0.045 * max(0, b.max_height - 10) * center_risk
    if marker == 11:  # pressure is preferred when it also leaves a clean spike response
        return 0.16 * safe_attack * (1.0 - min(1.0, center_risk))
    if marker == 12:  # avoid over-preparing slots when T supply is poor and spike risk is high
        return -0.34 * surplus_slots * (0.5 + center_risk)
    if marker == 13:  # require both center-hole and varied-hole garbage tests to agree
        disagreement = abs(center_risk - recovery_risk)
        return -0.35 * disagreement + 0.18 * recovery
    if marker == 14:  # balance immediate attack against multi-hole recovery quality
        return 0.12 * f.attack + 0.32 * recovery - 0.10 * f.attack * recovery_risk
    raise ValueError(marker)


_ORIGINAL_CONTEXT_SCORE = heuristic._context_score


def patched_context_score(game, features, weights):
    return _ORIGINAL_CONTEXT_SCORE(game, features, weights) + experiment_score(
        marker_from(weights), game, features
    )


heuristic._context_score = patched_context_score


def run_solo(name: str, games: int, max_pieces: int, seed_base: int, seed_step: int) -> dict:
    weights = weights_for(name)
    summaries = []
    for index in range(games):
        game = Game(seed_base + index * seed_step)
        spins = spin_lines = tsd = tst = tss = 0
        max_b2b = 0
        while not game.game_over and game.pieces_placed < max_pieces:
            choice = choose_search_action(game, weights, DEFAULT_SEARCH_CONFIG)
            if choice is None:
                break
            result = apply_search_action(game, choice.action)
            max_b2b = max(max_b2b, game.b2b_chain)
            if result.spin is not None:
                spins += 1
                spin_lines += result.lines
                if result.lines == 1:
                    tss += 1
                elif result.lines == 2:
                    tsd += 1
                elif result.lines == 3:
                    tst += 1
        summaries.append({
            "seed": game.seed,
            "pieces": game.pieces_placed,
            "attack": game.attack,
            "topout": game.game_over,
            "completed": (not game.game_over and game.pieces_placed >= max_pieces),
            "spins": spins,
            "spin_lines": spin_lines,
            "tss": tss,
            "tsd": tsd,
            "tst": tst,
            "max_b2b": max_b2b,
        })
    pieces = sum(x["pieces"] for x in summaries)
    attack = sum(x["attack"] for x in summaries)
    return {
        "name": name,
        "games": games,
        "max_pieces": max_pieces,
        "pieces": pieces,
        "attack": attack,
        "attack_per_piece": attack / pieces if pieces else 0.0,
        "topouts": sum(x["topout"] for x in summaries),
        "completed": sum(x["completed"] for x in summaries),
        "spins": sum(x["spins"] for x in summaries),
        "spin_lines": sum(x["spin_lines"] for x in summaries),
        "tss": sum(x["tss"] for x in summaries),
        "tsd": sum(x["tsd"] for x in summaries),
        "tst": sum(x["tst"] for x in summaries),
        "mean_max_b2b": sum(x["max_b2b"] for x in summaries) / games,
        "per_game": summaries,
    }


def solo_worker(args):
    return run_solo(*args)


def solo_stage(names, games, max_pieces, seed_base, seed_step):
    results = []
    workers = min(len(names), max(1, int(os.getenv("TOURNAMENT_WORKERS", "8"))))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(solo_worker, (name, games, max_pieces, seed_base, seed_step)): name
            for name in names
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda x: x["name"])


def rank_solo(results):
    # Safety first, then completion, then APP, then useful spin conversion.
    return sorted(
        results,
        key=lambda x: (
            -x["topouts"],
            x["completed"],
            x["attack_per_piece"],
            x["tsd"] + 1.5 * x["tst"],
            x["mean_max_b2b"],
        ),
        reverse=True,
    )


def versus(name: str) -> dict:
    candidate = weights_for(name)
    result = run_versus_benchmark(
        games=8,
        max_turns=120,
        seed_base=910003,
        seed_step=97,
        player_weights=candidate,
        ai_weights=DEFAULT_WEIGHTS,
        garbage_cap=8,
    )
    per = result.per_game
    return {
        "name": name,
        "wins": result.player_wins,
        "losses": result.ai_wins,
        "draws": result.draws,
        "mean_attack": result.player_mean_attack,
        "baseline_mean_attack": result.ai_mean_attack,
        "mean_sent": result.player_mean_sent,
        "baseline_mean_sent": result.ai_mean_sent,
        "mean_canceled": sum(x.player_canceled for x in per) / len(per),
        "baseline_mean_canceled": sum(x.ai_canceled for x in per) / len(per),
        "mean_received": sum(x.player_received for x in per) / len(per),
        "baseline_mean_received": sum(x.ai_received for x in per) / len(per),
        "mean_max_b2b": sum(x.player_max_b2b for x in per) / len(per),
        "baseline_mean_max_b2b": sum(x.ai_max_b2b for x in per) / len(per),
        "topouts": sum(x.winner == "ai" for x in per),
        "baseline_topouts": sum(x.winner == "player" for x in per),
    }


def main():
    names = list(CANDIDATES)
    short = solo_stage(names, games=2, max_pieces=120, seed_base=810001, seed_step=83)
    baseline_short = next(x for x in short if x["name"] == "baseline")
    contenders = [x for x in rank_solo(short) if x["name"] != "baseline"][:3]
    top_names = [x["name"] for x in contenders]

    fresh = solo_stage(["baseline", *top_names], games=3, max_pieces=280, seed_base=880003, seed_step=101)
    baseline_fresh = next(x for x in fresh if x["name"] == "baseline")
    # Only candidates that do not worsen topouts and show some meaningful solo signal advance.
    fresh_advancers = []
    for x in rank_solo(fresh):
        if x["name"] == "baseline":
            continue
        safe = x["topouts"] <= baseline_fresh["topouts"]
        completion_ok = x["completed"] >= baseline_fresh["completed"]
        firepower = x["attack_per_piece"] >= baseline_fresh["attack_per_piece"] * 1.01
        spin_or_b2b = (x["tsd"] + x["tst"] > baseline_fresh["tsd"] + baseline_fresh["tst"] or x["mean_max_b2b"] > baseline_fresh["mean_max_b2b"])
        if safe and completion_ok and (firepower or spin_or_b2b):
            fresh_advancers.append(x["name"])
        if len(fresh_advancers) >= 2:
            break

    versus_results = [versus(name) for name in fresh_advancers]
    report = {
        "format": "minoflux_self_improve_tournament_v1",
        "candidate_count": len(CANDIDATES) - 1,
        "short": short,
        "short_top3": top_names,
        "fresh": fresh,
        "versus": versus_results,
    }
    print("TOURNAMENT_RESULT=" + json.dumps(report, separators=(",", ":"), sort_keys=True))
    with open("tournament-result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
