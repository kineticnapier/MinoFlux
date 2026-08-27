from __future__ import annotations

from dataclasses import replace
from heapq import nlargest
import json

from minoflux_ai import heuristic, search
from minoflux_ai.benchmark import run_heuristic_game
from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
from minoflux_ai.versus_benchmark import run_versus_benchmark

ORIGINAL_RANK = heuristic.rank_placements
BASE_GAME_OVER = DEFAULT_WEIGHTS.game_over


def t_distance(game, limit=5):
    if game.current == "T":
        return 0
    for i, piece in enumerate(tuple(game.queue)[:limit], 1):
        if piece == "T":
            return i
    return None


def before_features(game):
    return extract_board_features(game.board)


def adjustment(name, game, ev):
    f = ev.features
    b = f.board
    before = before_features(game)
    td = t_distance(game)
    created = max(0, f.t_spin_slot_delta)
    destroyed = max(0, -f.t_spin_slot_delta)
    high = max(0, b.max_height - 8)
    very_high = max(0, b.max_height - 11)
    clean = 1.0 if f.new_holes == 0 else 0.0
    difficult = 1.0 if (f.spin_lines > 0 or f.lines == 4) else 0.0
    t_near = 0.0 if td is None or td > 4 else 1.0 / (td + 1)

    if name == "t_ready_create":
        return 0.55 * created * t_near
    if name == "t_ready_clean_create":
        return 0.75 * created * t_near * clean
    if name == "t_ready_safe_create":
        return 0.70 * created * t_near * clean * max(0.0, (14 - b.max_height) / 14)
    if name == "slot_survivability":
        return 0.22 * b.t_spin_slots * max(0.0, (14 - b.max_height) / 14)
    if name == "slot_density":
        return 0.28 * b.t_spin_slots / (1 + b.holes)
    if name == "b2b_ready_create":
        return (0.55 if game.back_to_back else 0.15) * created * max(t_near, 0.25)
    if name == "b2b_difficult_clear":
        return 0.55 * difficult * (1.0 if game.back_to_back else 0.25)
    if name == "attack_under_pressure":
        return 0.055 * f.attack * high
    if name == "clear_under_pressure":
        return 0.075 * f.lines * very_high
    if name == "clean_attack":
        return 0.18 * f.attack * clean
    if name == "efficient_attack":
        return 0.16 * max(0, f.attack - f.lines)
    if name == "safe_slot_attack":
        return 0.45 * created * t_near * clean + 0.08 * f.attack * high
    if name == "repair_then_attack":
        removed = max(0, before.holes - b.holes)
        return 0.22 * removed + 0.10 * f.attack * (1 if removed else 0)
    if name == "t_window_integrity":
        # Unlike the previously rejected pure preservation penalty, only protect a
        # slot when the placement is also clean and a T is immediately available.
        return -0.22 * destroyed * clean * (1.0 if td in (0, 1) else 0.0) + 0.35 * created * t_near
    return 0.0


def key(ev):
    return (
        ev.score,
        ev.features.attack,
        ev.features.spin_lines,
        ev.features.lines,
        -ev.features.board.holes,
        -ev.features.board.max_height,
        -ev.placement.rotation,
        -ev.placement.x,
    )


def install_candidate(name):
    def patched(game, weights=DEFAULT_WEIGHTS, *, placements=None, limit=None):
        ranked = ORIGINAL_RANK(game, weights, placements=placements, limit=None)
        is_candidate = weights.game_over < BASE_GAME_OVER - 0.5
        if is_candidate and name != "baseline":
            ranked = tuple(
                PlacementEvaluation(ev.placement, ev.score + adjustment(name, game, ev), ev.features)
                for ev in ranked
            )
            ranked = tuple(sorted(ranked, key=key, reverse=True))
        if limit is None:
            return ranked
        count = max(0, int(limit))
        if count == 0:
            return ()
        return tuple(nlargest(count, ranked, key=key)) if count < len(ranked) else ranked

    heuristic.rank_placements = patched
    search.rank_placements = patched


def summarize(name, seeds, pieces):
    marker = replace(DEFAULT_WEIGHTS, game_over=BASE_GAME_OVER - 1.0)
    games = [run_heuristic_game(seed, pieces, marker) for seed in seeds]
    total_pieces = sum(g.pieces for g in games)
    return {
        "name": name,
        "games": len(games),
        "pieces": total_pieces,
        "attack": sum(g.attack for g in games),
        "attackPerPiece": sum(g.attack for g in games) / max(1, total_pieces),
        "topouts": sum(g.topout for g in games),
        "completed": sum(g.completed for g in games),
        "spins": sum(g.spins for g in games),
        "spinLines": sum(g.spin_lines for g in games),
        "tsd": sum(g.t_spin_doubles for g in games),
        "tst": sum(g.t_spin_triples for g in games),
    }


def solo(name, seeds, pieces):
    install_candidate(name)
    return summarize(name, seeds, pieces)


def score_for_screen(row):
    # Survival is a hard gate; attack efficiency dominates among safe candidates.
    if row["topouts"]:
        return row["attackPerPiece"] - 0.25 * row["topouts"]
    return row["attackPerPiece"] + 0.002 * row["spinLines"] + 0.003 * row["tsd"]


def versus(name, seed_base):
    install_candidate(name)
    marker = replace(DEFAULT_WEIGHTS, game_over=BASE_GAME_OVER - 1.0)
    result = run_versus_benchmark(
        games=4,
        max_turns=70,
        seed_base=seed_base,
        seed_step=97,
        player_weights=marker,
        ai_weights=DEFAULT_WEIGHTS,
    )
    rows = result.per_game
    return {
        "name": name,
        "wins": result.player_wins,
        "losses": result.ai_wins,
        "draws": result.draws,
        "attack": result.player_mean_attack,
        "baselineAttack": result.ai_mean_attack,
        "sent": result.player_mean_sent,
        "baselineSent": result.ai_mean_sent,
        "canceled": sum(r.player_canceled for r in rows) / len(rows),
        "baselineCanceled": sum(r.ai_canceled for r in rows) / len(rows),
        "received": sum(r.player_received for r in rows) / len(rows),
        "baselineReceived": sum(r.ai_received for r in rows) / len(rows),
        "maxB2B": sum(r.player_max_b2b for r in rows) / len(rows),
        "baselineMaxB2B": sum(r.ai_max_b2b for r in rows) / len(rows),
    }


def main():
    candidates = [
        "baseline",
        "t_ready_create",
        "t_ready_clean_create",
        "t_ready_safe_create",
        "slot_survivability",
        "slot_density",
        "b2b_ready_create",
        "b2b_difficult_clear",
        "attack_under_pressure",
        "clear_under_pressure",
        "clean_attack",
        "efficient_attack",
        "safe_slot_attack",
        "repair_then_attack",
        "t_window_integrity",
    ]
    short_seeds = (26082701, 26082798)
    short = [solo(name, short_seeds, 90) for name in candidates]
    baseline = short[0]
    survivors = sorted(short[1:], key=score_for_screen, reverse=True)[:3]

    fresh_seeds = (8317001, 8317138, 8317275)
    fresh_names = ["baseline"] + [r["name"] for r in survivors]
    fresh = [solo(name, fresh_seeds, 150) for name in fresh_names]
    fresh_base = fresh[0]
    safe_fresh = [
        r for r in fresh[1:]
        if r["topouts"] <= fresh_base["topouts"]
        and r["completed"] >= fresh_base["completed"]
        and r["attackPerPiece"] > fresh_base["attackPerPiece"] * 1.01
    ]
    finalists = sorted(safe_fresh, key=score_for_screen, reverse=True)[:2]
    versus_rows = [versus(r["name"], 940001 + i * 1000) for i, r in enumerate(finalists)]

    accepted = []
    for row in versus_rows:
        if row["wins"] >= row["losses"] and row["sent"] >= row["baselineSent"]:
            accepted.append(row["name"])

    payload = {
        "candidateCount": len(candidates) - 1,
        "short": short,
        "survivors": [r["name"] for r in survivors],
        "fresh": fresh,
        "finalists": [r["name"] for r in finalists],
        "versus": versus_rows,
        "accepted": accepted,
    }
    print("TOURNAMENT_RESULT=" + json.dumps(payload, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
