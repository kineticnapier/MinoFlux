from __future__ import annotations

import json

import self_improve_tournament as t
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_search import VersusSearchConfig

FAST_PLACEMENT = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=1,
    discount=0.90,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
)
t.DEFAULT_SEARCH_CONFIG = FAST_PLACEMENT

_ORIGINAL_VERSUS = t.run_versus_benchmark
FAST_VERSUS = VersusSearchConfig(
    placement_search=FAST_PLACEMENT,
    candidate_width=6,
    opponent_reply_width=1,
)


def _fast_versus_benchmark(*args, **kwargs):
    kwargs["player_config"] = FAST_VERSUS
    kwargs["ai_config"] = FAST_VERSUS
    return _ORIGINAL_VERSUS(*args, **kwargs)


t.run_versus_benchmark = _fast_versus_benchmark


def main() -> None:
    names = list(t.CANDIDATES)
    short = t.solo_stage(names, games=2, max_pieces=100, seed_base=810001, seed_step=83)
    contenders = [x for x in t.rank_solo(short) if x["name"] != "baseline"][:3]
    top_names = [x["name"] for x in contenders]

    fresh = t.solo_stage(["baseline", *top_names], games=3, max_pieces=220, seed_base=880003, seed_step=101)
    baseline_fresh = next(x for x in fresh if x["name"] == "baseline")
    fresh_advancers: list[str] = []
    for x in t.rank_solo(fresh):
        if x["name"] == "baseline":
            continue
        safe = x["topouts"] <= baseline_fresh["topouts"]
        completion_ok = x["completed"] >= baseline_fresh["completed"]
        firepower = x["attack_per_piece"] >= baseline_fresh["attack_per_piece"] * 1.01
        spin_or_b2b = (
            x["tsd"] + x["tst"] > baseline_fresh["tsd"] + baseline_fresh["tst"]
            or x["mean_max_b2b"] > baseline_fresh["mean_max_b2b"]
        )
        if safe and completion_ok and (firepower or spin_or_b2b):
            fresh_advancers.append(x["name"])
        if len(fresh_advancers) >= 2:
            break

    versus_results = []
    for name in fresh_advancers:
        candidate = t.weights_for(name)
        result = _ORIGINAL_VERSUS(
            games=8,
            max_turns=100,
            seed_base=910003,
            seed_step=97,
            player_weights=candidate,
            ai_weights=t.DEFAULT_WEIGHTS,
            player_config=FAST_VERSUS,
            ai_config=FAST_VERSUS,
            garbage_cap=8,
        )
        per = result.per_game
        versus_results.append({
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
        })

    report = {
        "format": "minoflux_self_improve_tournament_v1",
        "candidate_count": len(t.CANDIDATES) - 1,
        "screen_search": FAST_PLACEMENT.to_dict(),
        "versus_search": FAST_VERSUS.to_dict(),
        "short": short,
        "short_top3": top_names,
        "fresh": fresh,
        "versus": versus_results,
    }
    print("TOURNAMENT_RESULT=" + json.dumps(report, separators=(",", ":"), sort_keys=True))
    with open("tournament-result.json", "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
