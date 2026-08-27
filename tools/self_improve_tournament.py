from __future__ import annotations

from dataclasses import replace
import json

from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.versus_benchmark import run_versus_benchmark

CANDIDATES = {
    "baseline": {},
    "tspin_clear_015": {"t_spin_clear_potential": 0.15},
    "tspin_clear_030": {"t_spin_clear_potential": 0.30},
    "tspin_clear_050": {"t_spin_clear_potential": 0.50},
    "deep_holes_020": {"deep_holes": -0.20},
    "deep_holes_040": {"deep_holes": -0.40},
    "deep_holes_060": {"deep_holes": -0.60},
    "danger_0003": {"danger_excess": -0.003},
    "danger_0006": {"danger_excess": -0.006},
    "danger_0010": {"danger_excess": -0.010},
    "b2b_continue_020": {"b2b_continue": 0.20},
    "b2b_continue_040": {"b2b_continue": 0.40},
    "b2b_continue_070": {"b2b_continue": 0.70},
    "tspin_danger_combo": {"t_spin_clear_potential": 0.30, "danger_excess": -0.006},
    "tspin_downstack_combo": {"t_spin_clear_potential": 0.30, "deep_holes": -0.40},
}


def weights(name: str):
    return replace(DEFAULT_WEIGHTS, **CANDIDATES[name])


def solo(name: str, *, games: int, pieces: int, seed_base: int, seed_step: int):
    r = run_heuristic_benchmark(
        games=games,
        max_pieces=pieces,
        seed_base=seed_base,
        seed_step=seed_step,
        weights=weights(name),
        workers=games,
    )
    app = r.attack / max(1, r.pieces)
    return {
        "name": name,
        "attack_per_piece": app,
        "attack": r.attack,
        "pieces": r.pieces,
        "topouts": r.topouts,
        "completed": r.completed,
        "spins": r.spins,
        "spin_lines": r.spin_lines,
        "tsd": r.t_spin_doubles,
        "tst": r.t_spin_triples,
    }


def solo_key(x):
    # Survival is primary gating; then attack efficiency and useful spin clears.
    return (-x["topouts"], x["completed"], x["attack_per_piece"], x["spin_lines"], x["tsd"], x["tst"])


def main():
    stage1 = [solo(name, games=2, pieces=80, seed_base=91001, seed_step=97) for name in CANDIDATES]
    baseline1 = next(x for x in stage1 if x["name"] == "baseline")
    survivors = sorted((x for x in stage1 if x["name"] != "baseline"), key=solo_key, reverse=True)[:3]

    stage2_names = ["baseline"] + [x["name"] for x in survivors]
    stage2 = [solo(name, games=3, pieces=150, seed_base=381001, seed_step=131) for name in stage2_names]
    baseline2 = next(x for x in stage2 if x["name"] == "baseline")
    eligible = [
        x for x in stage2 if x["name"] != "baseline"
        and x["topouts"] <= baseline2["topouts"]
        and x["completed"] >= baseline2["completed"]
        and x["attack_per_piece"] > baseline2["attack_per_piece"] * 1.02
    ]
    finalists = sorted(eligible, key=solo_key, reverse=True)[:2]

    versus = []
    for item in finalists:
        v = run_versus_benchmark(
            games=4,
            max_turns=60,
            seed_base=771001,
            seed_step=173,
            player_weights=weights(item["name"]),
            ai_weights=weights("baseline"),
        )
        versus.append({
            "name": item["name"],
            "wins": v.player_wins,
            "losses": v.ai_wins,
            "draws": v.draws,
            "attack": v.player_mean_attack,
            "baseline_attack": v.ai_mean_attack,
            "sent": v.player_mean_sent,
            "baseline_sent": v.ai_mean_sent,
            "cancel": sum(g.player_canceled for g in v.per_game) / v.games,
            "baseline_cancel": sum(g.ai_canceled for g in v.per_game) / v.games,
            "received": sum(g.player_received for g in v.per_game) / v.games,
            "baseline_received": sum(g.ai_received for g in v.per_game) / v.games,
            "max_b2b": sum(g.player_max_b2b for g in v.per_game) / v.games,
            "baseline_max_b2b": sum(g.ai_max_b2b for g in v.per_game) / v.games,
        })

    # Require fresh-seed gain and no clear mirrored-versus regression.
    winner = None
    for item in finalists:
        v = next(x for x in versus if x["name"] == item["name"])
        if v["losses"] <= v["wins"] + 1 and v["sent"] >= v["baseline_sent"] * 0.98:
            winner = item["name"]
            break

    print("TOURNAMENT_RESULT=" + json.dumps({
        "candidate_count": len(CANDIDATES) - 1,
        "stage1_baseline": baseline1,
        "stage1": stage1,
        "survivors": [x["name"] for x in survivors],
        "stage2": stage2,
        "finalists": [x["name"] for x in finalists],
        "versus": versus,
        "winner": winner,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
