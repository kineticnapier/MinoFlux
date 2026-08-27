from __future__ import annotations

import json
from dataclasses import replace

import minoflux_ai.heuristic as heuristic
from minoflux_ai.versus_benchmark import run_versus_benchmark

BASE_SCORE = heuristic.score_features
BASE_WEIGHTS = heuristic.DEFAULT_WEIGHTS
CANDIDATE_WEIGHTS = replace(BASE_WEIGHTS, perfect_clear=BASE_WEIGHTS.perfect_clear + 1e-9)


def patched_score(features, weights=BASE_WEIGHTS):
    score = BASE_SCORE(features, BASE_WEIGHTS)
    if weights.perfect_clear != BASE_WEIGHTS.perfect_clear:
        b = features.board
        score += 0.90 * b.t_spin_slots / (1.0 + b.holes + b.max_height / 6.0)
    return score


def main():
    heuristic.score_features = patched_score
    r = run_versus_benchmark(
        games=6,
        max_turns=180,
        seed_base=1510001,
        seed_step=130363,
        player_weights=CANDIDATE_WEIGHTS,
        ai_weights=BASE_WEIGHTS,
    )
    per = r.per_game
    payload = {
        "candidate_wins": r.player_wins,
        "baseline_wins": r.ai_wins,
        "draws": r.draws,
        "candidate_attack": r.player_mean_attack,
        "baseline_attack": r.ai_mean_attack,
        "candidate_sent": r.player_mean_sent,
        "baseline_sent": r.ai_mean_sent,
        "candidate_cancel": sum(x.player_canceled for x in per) / len(per),
        "baseline_cancel": sum(x.ai_canceled for x in per) / len(per),
        "candidate_received": sum(x.player_received for x in per) / len(per),
        "baseline_received": sum(x.ai_received for x in per) / len(per),
        "candidate_b2b": sum(x.player_max_b2b for x in per) / len(per),
        "baseline_b2b": sum(x.ai_max_b2b for x in per) / len(per),
        "candidate_topouts": sum(x.winner == "ai" for x in per),
        "baseline_topouts": sum(x.winner == "player" for x in per),
    }
    print("RESULT=" + json.dumps(payload, sort_keys=True))

if __name__ == "__main__":
    main()
