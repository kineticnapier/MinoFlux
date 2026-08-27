from __future__ import annotations

from dataclasses import replace
import json

from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.versus_benchmark import run_versus_benchmark


CANDIDATES = {
    "baseline": {},
    "chain_light": {"b2b_chain": 0.08},
    "chain_medium": {"b2b_chain": 0.16},
    "break_guard": {"b2b_break": -0.45},
    "break_guard_strong": {"b2b_break": -0.90},
    "difficult_light": {"b2b_difficult": 0.20},
    "difficult_medium": {"b2b_difficult": 0.40},
    "surge_light": {"surge_charge": 0.08},
    "surge_medium": {"surge_charge": 0.16},
    "chain_break": {"b2b_chain": 0.10, "b2b_break": -0.45},
    "chain_difficult": {"b2b_chain": 0.10, "b2b_difficult": 0.25},
    "surge_break": {"surge_charge": 0.10, "b2b_break": -0.45},
    "balanced_b2b": {"b2b_chain": 0.08, "b2b_break": -0.35, "b2b_difficult": 0.18},
    "surge_builder": {"b2b_chain": 0.06, "surge_charge": 0.12, "b2b_break": -0.35},
    "difficult_preserve": {"b2b_difficult": 0.28, "b2b_break": -0.55},
}


def weights_for(name: str):
    return replace(DEFAULT_WEIGHTS, **CANDIDATES[name])


def summarize(result):
    pieces = max(1, result.pieces)
    return {
        "attack": result.attack,
        "pieces": result.pieces,
        "attack_per_piece": result.attack / pieces,
        "topouts": result.topouts,
        "completed": result.completed,
        "spins": result.spins,
        "spin_lines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def solo_rank(item):
    name, data = item
    return (
        -data["topouts"],
        data["completed"],
        data["attack_per_piece"],
        data["tsd"],
        data["spin_lines"],
        name,
    )


def versus_summary(result):
    per = result.per_game
    n = max(1, len(per))
    return {
        "wins": result.player_wins,
        "losses": result.ai_wins,
        "draws": result.draws,
        "attack": result.player_mean_attack,
        "baseline_attack": result.ai_mean_attack,
        "sent": result.player_mean_sent,
        "baseline_sent": result.ai_mean_sent,
        "cancel": sum(x.player_canceled for x in per) / n,
        "baseline_cancel": sum(x.ai_canceled for x in per) / n,
        "received": sum(x.player_received for x in per) / n,
        "baseline_received": sum(x.ai_received for x in per) / n,
        "max_b2b": sum(x.player_max_b2b for x in per) / n,
        "baseline_max_b2b": sum(x.ai_max_b2b for x in per) / n,
        "topouts": sum(x.winner == "ai" for x in per),
        "baseline_topouts": sum(x.winner == "player" for x in per),
    }


def main():
    short = {}
    for name in CANDIDATES:
        result = run_heuristic_benchmark(
            games=2,
            max_pieces=140,
            seed_base=271828,
            seed_step=1009,
            weights=weights_for(name),
            workers=2,
        )
        short[name] = summarize(result)
    survivors = [name for name, _ in sorted(short.items(), key=solo_rank, reverse=True) if name != "baseline"][:3]

    fresh = {"baseline": summarize(run_heuristic_benchmark(
        games=3,
        max_pieces=190,
        seed_base=1618033,
        seed_step=1223,
        weights=weights_for("baseline"),
        workers=2,
    ))}
    for name in survivors:
        fresh[name] = summarize(run_heuristic_benchmark(
            games=3,
            max_pieces=190,
            seed_base=1618033,
            seed_step=1223,
            weights=weights_for(name),
            workers=2,
        ))

    finalists = [name for name, _ in sorted(
        ((name, data) for name, data in fresh.items() if name != "baseline"),
        key=solo_rank,
        reverse=True,
    )][:2]

    versus = {}
    for name in finalists:
        versus[name] = versus_summary(run_versus_benchmark(
            games=4,
            max_turns=80,
            seed_base=314159,
            seed_step=1301,
            player_weights=weights_for(name),
            ai_weights=weights_for("baseline"),
        ))

    print("TOURNAMENT_RESULT=" + json.dumps({
        "candidate_count": len(CANDIDATES) - 1,
        "short": short,
        "survivors": survivors,
        "fresh": fresh,
        "finalists": finalists,
        "versus": versus,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
