from __future__ import annotations

from dataclasses import replace
import json
import os

from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action, choose_search_action
from minoflux_ai.versus_benchmark import run_versus_benchmark

CANDIDATE = os.environ.get("CANDIDATE", "baseline")
PHASE = os.environ.get("PHASE", "short")


def weights_for(candidate: str):
    if candidate == "slot_height_quality":
        return replace(DEFAULT_WEIGHTS, t_spin_slot_height_quality=0.70)
    return DEFAULT_WEIGHTS


def run_game(seed: int, pieces: int, candidate: str):
    game = Game(seed)
    weights = weights_for(candidate)
    spins = tsd = tst = spin_lines = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < pieces:
        choice = choose_search_action(game, weights, DEFAULT_SEARCH_CONFIG)
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
        tsd += int(result.spin == T_SPIN_DOUBLE)
        tst += int(result.spin == T_SPIN_TRIPLE)
        max_b2b = max(max_b2b, game.b2b_chain)
    return {
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "topout": int(game.game_over),
        "completed": int(not game.game_over and game.pieces_placed >= pieces),
        "spins": spins,
        "spin_lines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "max_b2b": max_b2b,
    }


def run_solo():
    if PHASE == "short":
        seeds, pieces = (71003, 71034), 100
    else:
        seeds, pieces = (91009, 91046, 91083), 150
    rows = [run_game(seed, pieces, CANDIDATE) for seed in seeds]
    total_pieces = sum(r["pieces"] for r in rows)
    return {
        "candidate": CANDIDATE,
        "phase": PHASE,
        "games": len(rows),
        "pieces": total_pieces,
        "attack": sum(r["attack"] for r in rows),
        "attack_per_piece": (sum(r["attack"] for r in rows) / total_pieces) if total_pieces else 0.0,
        "topouts": sum(r["topout"] for r in rows),
        "completed": sum(r["completed"] for r in rows),
        "spins": sum(r["spins"] for r in rows),
        "spin_lines": sum(r["spin_lines"] for r in rows),
        "tsd": sum(r["tsd"] for r in rows),
        "tst": sum(r["tst"] for r in rows),
        "max_b2b": max(r["max_b2b"] for r in rows),
        "rows": rows,
    }


def run_versus():
    result = run_versus_benchmark(
        games=6,
        max_turns=90,
        seed_base=120011,
        seed_step=43,
        player_weights=weights_for(CANDIDATE),
        ai_weights=DEFAULT_WEIGHTS,
    )
    rows = result.per_game
    return {
        "candidate": CANDIDATE,
        "phase": "versus",
        "games": result.games,
        "candidate_wins": result.player_wins,
        "baseline_wins": result.ai_wins,
        "draws": result.draws,
        "candidate_attack": result.player_mean_attack,
        "baseline_attack": result.ai_mean_attack,
        "candidate_sent": result.player_mean_sent,
        "baseline_sent": result.ai_mean_sent,
        "candidate_canceled": sum(r.player_canceled for r in rows) / len(rows),
        "baseline_canceled": sum(r.ai_canceled for r in rows) / len(rows),
        "candidate_received": sum(r.player_received for r in rows) / len(rows),
        "baseline_received": sum(r.ai_received for r in rows) / len(rows),
        "candidate_max_b2b": sum(r.player_max_b2b for r in rows) / len(rows),
        "baseline_max_b2b": sum(r.ai_max_b2b for r in rows) / len(rows),
        "candidate_topouts": sum(r.winner == "ai" for r in rows),
        "baseline_topouts": sum(r.winner == "player" for r in rows),
        "rows": [r.__dict__ if hasattr(r, "__dict__") else {
            "seed": r.seed,
            "winner": r.winner,
            "turns": r.turns,
            "player_attack": r.player_attack,
            "ai_attack": r.ai_attack,
            "player_sent": r.player_sent,
            "ai_sent": r.ai_sent,
            "player_canceled": r.player_canceled,
            "ai_canceled": r.ai_canceled,
            "player_received": r.player_received,
            "ai_received": r.ai_received,
            "player_max_b2b": r.player_max_b2b,
            "ai_max_b2b": r.ai_max_b2b,
            "models_swapped": r.models_swapped,
        } for r in rows],
    }


summary = run_versus() if PHASE == "versus" else run_solo()
print("TOURNAMENT_RESULT=" + json.dumps(summary, sort_keys=True))
