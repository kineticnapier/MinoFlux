from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json

import minoflux_ai.heuristic as heuristic
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.versus_benchmark import run_versus_benchmark


CANDIDATES = (
    "slot_supply_window",
    "slot_preserve_until_t",
    "slot_consume_when_t",
    "b2b_preserve",
    "b2b_extend",
    "danger_attack",
    "danger_holes",
    "downstack_holes_removed",
    "downstack_depth_removed",
    "height_drop_clear",
    "hold_t_slot_reserve",
    "t_current_conversion",
    "slot_bumpiness_quality",
    "safe_attack_efficiency",
)


def _next_t_distance(game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 5:
            break
    return 7


def _extra_score(name: str, game, features) -> float:
    before = extract_board_features(game.board)
    after = features.board
    tdist = _next_t_distance(game)
    availability = max(0.0, (6.0 - tdist) / 6.0)
    low_clean = after.t_spin_slots / (1.0 + after.holes + after.max_height / 6.0)

    if name == "slot_supply_window":
        return 0.9 * availability * after.t_spin_slots / (1.0 + after.holes)
    if name == "slot_preserve_until_t":
        return 0.55 * max(0.0, (tdist - 1) / 5.0) * low_clean
    if name == "slot_consume_when_t":
        return 1.6 * features.spin_lines if game.current == "T" else 0.0
    if name == "b2b_preserve":
        if game.back_to_back and features.lines > 0 and features.spin is None and features.lines < 4:
            return -2.0
        return 0.0
    if name == "b2b_extend":
        difficult = features.spin is not None or features.lines == 4
        return 1.0 if game.back_to_back and difficult and features.lines > 0 else 0.0
    if name == "danger_attack":
        danger = max(0.0, (after.max_height - 10) / 6.0)
        return danger * (0.35 * features.attack + 0.12 * features.lines)
    if name == "danger_holes":
        danger = max(0.0, (after.max_height - 9) / 7.0)
        return -0.28 * danger * after.holes
    if name == "downstack_holes_removed":
        return 1.4 * max(0, before.holes - after.holes)
    if name == "downstack_depth_removed":
        return 0.32 * max(0, before.hole_depth - after.hole_depth)
    if name == "height_drop_clear":
        if features.lines <= 0:
            return 0.0
        return 0.08 * max(0, before.aggregate_height - after.aggregate_height)
    if name == "hold_t_slot_reserve":
        return 0.65 * low_clean if game.hold_piece == "T" and game.current != "T" else 0.0
    if name == "t_current_conversion":
        if game.current != "T":
            return 0.0
        return 0.9 * features.spin_lines + 0.12 * features.attack
    if name == "slot_bumpiness_quality":
        return 0.85 * after.t_spin_slots / (1.0 + after.holes + after.bumpiness / 4.0)
    if name == "safe_attack_efficiency":
        safety = 1.0 + after.holes + after.max_height / 10.0
        return 0.32 * features.attack / safety
    raise ValueError(name)


def _install_candidate(name: str):
    marker = replace(DEFAULT_WEIGHTS)
    marker_id = id(marker)
    original_context = heuristic._context_score

    def patched_context(game, features, weights):
        base = original_context(game, features, weights)
        if id(weights) == marker_id:
            return base + _extra_score(name, game, features)
        return base

    heuristic._context_score = patched_context
    return marker


def _solo(name: str, games: int, pieces: int, seed_base: int, seed_step: int) -> dict[str, object]:
    marker = _install_candidate(name) if name != "baseline" else DEFAULT_WEIGHTS
    result = run_heuristic_benchmark(
        games=games,
        max_pieces=pieces,
        seed_base=seed_base,
        seed_step=seed_step,
        weights=marker,
        workers=1,
    )
    t_spins = result.spins
    app = result.attack / result.pieces if result.pieces else 0.0
    return {
        "name": name,
        "pieces": result.pieces,
        "attack": result.attack,
        "attackPerPiece": app,
        "topouts": result.topouts,
        "completed": result.completed,
        "tSpins": t_spins,
        "spinLines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def _screen_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    games = 2
    return (
        -float(row["topouts"]),
        float(row["completed"]) / games,
        float(row["attackPerPiece"]),
        float(row["tsd"]) + 1.5 * float(row["tst"]),
        float(row["spinLines"]),
    )


def _fresh_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
    games = 3
    return (
        -float(row["topouts"]),
        float(row["completed"]) / games,
        float(row["attackPerPiece"]),
        float(row["tsd"]) + 1.5 * float(row["tst"]),
        float(row["spinLines"]),
    )


def _parallel_solo(names, games, pieces, seed_base, seed_step):
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, len(names))) as pool:
        futures = {
            pool.submit(_solo, name, games, pieces, seed_base, seed_step): name
            for name in names
        }
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: str(row["name"]))
    return rows


def _versus(name: str) -> dict[str, object]:
    marker = _install_candidate(name)
    result = run_versus_benchmark(
        games=6,
        max_turns=90,
        seed_base=910_003,
        seed_step=97,
        player_weights=marker,
        ai_weights=DEFAULT_WEIGHTS,
    )
    games = result.per_game
    return {
        "name": name,
        "wins": result.player_wins,
        "losses": result.ai_wins,
        "draws": result.draws,
        "attack": result.player_mean_attack,
        "baselineAttack": result.ai_mean_attack,
        "sent": result.player_mean_sent,
        "baselineSent": result.ai_mean_sent,
        "cancel": sum(g.player_canceled for g in games) / len(games),
        "baselineCancel": sum(g.ai_canceled for g in games) / len(games),
        "received": sum(g.player_received for g in games) / len(games),
        "baselineReceived": sum(g.ai_received for g in games) / len(games),
        "maxB2B": sum(g.player_max_b2b for g in games) / len(games),
        "baselineMaxB2B": sum(g.ai_max_b2b for g in games) / len(games),
        "topouts": result.ai_wins,
        "baselineTopouts": result.player_wins,
    }


def main() -> None:
    names = ("baseline", *CANDIDATES)
    short = _parallel_solo(names, games=2, pieces=140, seed_base=310_001, seed_step=73)
    baseline_short = next(row for row in short if row["name"] == "baseline")
    survivors = sorted(
        (row for row in short if row["name"] != "baseline"),
        key=_screen_key,
        reverse=True,
    )[:3]
    survivor_names = [str(row["name"]) for row in survivors]

    fresh = _parallel_solo(
        ("baseline", *survivor_names),
        games=3,
        pieces=220,
        seed_base=730_019,
        seed_step=101,
    )
    baseline_fresh = next(row for row in fresh if row["name"] == "baseline")
    fresh_candidates = sorted(
        (row for row in fresh if row["name"] != "baseline"),
        key=_fresh_key,
        reverse=True,
    )

    finalists = []
    for row in fresh_candidates:
        if row["topouts"] > baseline_fresh["topouts"]:
            continue
        if row["completed"] < baseline_fresh["completed"]:
            continue
        if row["attackPerPiece"] + 1e-12 < baseline_fresh["attackPerPiece"]:
            continue
        finalists.append(str(row["name"]))
        if len(finalists) == 2:
            break

    versus = [_versus(name) for name in finalists]
    payload = {
        "candidateCount": len(CANDIDATES),
        "short": short,
        "shortBaseline": baseline_short,
        "survivors": survivor_names,
        "fresh": fresh,
        "finalists": finalists,
        "versus": versus,
    }
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
