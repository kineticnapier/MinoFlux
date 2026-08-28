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
    "t_arrival_clean_conversion",
    "t_arrival_height_conversion",
    "t_arrival_hole_repair",
    "t_arrival_no_waste",
    "t_conversion_low_clean_chain",
    "t_conversion_preserve_next_slot",
    "t_conversion_b2b_ready",
    "t_conversion_surge_release",
    "height_recovery",
    "aggregate_recovery",
    "danger_clean_clear",
    "hold_bring_t_ready",
    "hold_avoid_stash_ready_t",
    "hold_rescue_high_stack",
)


def _extra_score(name: str, game, features) -> float:
    before = extract_board_features(game.board)
    after = features.board
    low_clean = after.t_spin_slots / (1.0 + after.holes + after.max_height / 6.0)
    height_drop = max(0, before.max_height - after.max_height)
    aggregate_drop = max(0, before.aggregate_height - after.aggregate_height)
    holes_removed = max(0, before.holes - after.holes)
    slot_loss = max(0, before.t_spin_slots - after.t_spin_slots)
    spin_conversion = game.current == "T" and features.spin_lines > 0
    danger = max(0.0, (before.max_height - 10.0) / 6.0)

    if name == "t_arrival_clean_conversion":
        if game.current != "T":
            return 0.0
        if spin_conversion:
            safety = 1.0 / (1.0 + after.holes + after.max_height / 8.0)
            return 1.00 * features.spin_lines * safety + 0.20 * features.attack * safety
        return -0.55 * slot_loss - 0.22 * features.new_holes

    if name == "t_arrival_height_conversion":
        if game.current != "T":
            return 0.0
        if spin_conversion:
            return 0.52 * features.spin_lines + 0.22 * height_drop + 0.08 * features.attack
        return -0.45 * slot_loss

    if name == "t_arrival_hole_repair":
        if game.current != "T":
            return 0.0
        if spin_conversion:
            return 0.48 * features.spin_lines + 0.42 * holes_removed + 0.08 * features.attack
        return 0.30 * holes_removed - 0.40 * slot_loss

    if name == "t_arrival_no_waste":
        if game.current != "T" or before.t_spin_slots <= 0:
            return 0.0
        if spin_conversion:
            return 0.65 * features.spin_lines + 0.10 * features.attack
        wasted = 1.0 if features.attack == 0 and features.lines == 0 else 0.0
        return -0.70 * slot_loss - 0.35 * wasted - 0.18 * features.new_holes

    if name == "t_conversion_low_clean_chain":
        if not spin_conversion:
            return 0.0
        return 0.45 * features.spin_lines + 0.55 * low_clean + 0.10 * features.attack

    if name == "t_conversion_preserve_next_slot":
        if not spin_conversion:
            return 0.0
        return 0.50 * features.spin_lines + 0.38 * after.t_spin_slots / (1.0 + after.holes) + 0.08 * features.attack

    if name == "t_conversion_b2b_ready":
        if not spin_conversion:
            return 0.0
        chain = max(0, game.b2b_chain)
        active = 1.0 if game.back_to_back else 0.0
        return 0.42 * features.spin_lines + 0.18 * features.attack + 0.24 * active + 0.08 * chain

    if name == "t_conversion_surge_release":
        if not spin_conversion:
            return 0.0
        charge = max(0, game.surge_charge)
        return 0.44 * features.spin_lines + 0.12 * features.attack + 0.07 * charge

    if name == "height_recovery":
        if before.max_height < 11:
            return 0.0
        return 0.26 * height_drop * (1.0 + danger) + 0.08 * features.attack - 0.18 * features.new_holes

    if name == "aggregate_recovery":
        if before.max_height < 10:
            return 0.0
        return 0.030 * aggregate_drop * (1.0 + danger) + 0.08 * features.attack - 0.15 * features.new_holes

    if name == "danger_clean_clear":
        if before.max_height < 11 or features.lines <= 0:
            return 0.0
        clean = 1.0 / (1.0 + features.new_holes + after.holes / 3.0)
        return danger * clean * (0.24 * features.attack + 0.16 * features.lines + 0.18 * height_drop)

    if name == "hold_bring_t_ready":
        if not game.hold_used or game.current != "T" or before.t_spin_slots <= 0:
            return 0.0
        if spin_conversion:
            return 0.62 * features.spin_lines + 0.14 * features.attack
        return -0.55 * slot_loss - 0.22 * features.new_holes

    if name == "hold_avoid_stash_ready_t":
        if not game.hold_used or game.hold_piece != "T" or before.t_spin_slots <= 0:
            return 0.0
        safety_need = max(0.0, (before.max_height - 13.0) / 4.0)
        return -0.48 * before.t_spin_slots + 0.20 * safety_need * features.attack + 0.15 * height_drop

    if name == "hold_rescue_high_stack":
        if not game.hold_used or before.max_height < 12:
            return 0.0
        return 0.18 * height_drop + 0.16 * holes_removed + 0.10 * features.attack - 0.18 * features.new_holes

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
    app = result.attack / result.pieces if result.pieces else 0.0
    return {
        "name": name,
        "pieces": result.pieces,
        "attack": result.attack,
        "attackPerPiece": app,
        "topouts": result.topouts,
        "completed": result.completed,
        "tSpins": result.spins,
        "spinLines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def _key(row: dict[str, object], games: int) -> tuple[float, float, float, float, float]:
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
        max_turns=110,
        seed_base=1_430_021,
        seed_step=109,
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
    short = _parallel_solo(names, games=2, pieces=160, seed_base=510_011, seed_step=83)
    survivors = sorted(
        (row for row in short if row["name"] != "baseline"),
        key=lambda row: _key(row, 2),
        reverse=True,
    )[:3]
    survivor_names = [str(row["name"]) for row in survivors]

    fresh = _parallel_solo(
        ("baseline", *survivor_names),
        games=3,
        pieces=260,
        seed_base=940_033,
        seed_step=107,
    )
    baseline_fresh = next(row for row in fresh if row["name"] == "baseline")
    fresh_candidates = sorted(
        (row for row in fresh if row["name"] != "baseline"),
        key=lambda row: _key(row, 3),
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
        "survivors": survivor_names,
        "fresh": fresh,
        "finalists": finalists,
        "versus": versus,
    }
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
