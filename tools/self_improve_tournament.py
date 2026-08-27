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
    "slot_timing_exact",
    "slot_demand_balance",
    "t_arrival_conversion",
    "setup_no_new_holes",
    "setup_headroom",
    "b2b_charge_difficult",
    "b2b_charge_break",
    "combo_safe_extend",
    "downstack_efficiency",
    "hole_depth_rescue",
    "hold_t_ready",
    "hold_i_rescue",
    "danger_clear_urgency",
    "spin_safety_efficiency",
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


def _upcoming_t_count(game, horizon: int = 6) -> int:
    count = int(game.current == "T")
    for index, piece in enumerate(game.queue):
        if index >= horizon - 1:
            break
        count += int(piece == "T")
    count += int(game.hold_piece == "T")
    return count


def _extra_score(name: str, game, features) -> float:
    before = extract_board_features(game.board)
    after = features.board
    tdist = _next_t_distance(game)
    low_clean = after.t_spin_slots / (1.0 + after.holes + after.max_height / 6.0)
    danger = max(0.0, min(1.5, (after.max_height - 8.0) / 8.0))
    holes_removed = max(0, before.holes - after.holes)
    depth_removed = max(0, before.hole_depth - after.hole_depth)
    difficult = features.spin is not None or features.lines == 4

    if name == "slot_timing_exact":
        if after.t_spin_slots <= 0:
            return 0.0
        closeness = max(0.0, (4.0 - tdist) / 4.0)
        far_penalty = max(0.0, (tdist - 3.0) / 4.0)
        return 1.10 * low_clean * closeness - 0.30 * after.t_spin_slots * far_penalty
    if name == "slot_demand_balance":
        demand = _upcoming_t_count(game)
        matched = min(after.t_spin_slots, demand)
        excess = max(0, after.t_spin_slots - demand)
        return 0.70 * matched / (1.0 + after.holes) - 0.32 * excess
    if name == "t_arrival_conversion":
        if game.current != "T":
            return 0.0
        slot_loss = max(0, before.t_spin_slots - after.t_spin_slots)
        converted = features.spin_lines > 0
        return (1.20 * features.spin_lines + 0.20 * features.attack) if converted else -0.80 * slot_loss
    if name == "setup_no_new_holes":
        if after.t_spin_slots <= 0:
            return 0.0
        return -0.65 * features.new_holes * (1.0 + min(2, after.t_spin_slots)) + 0.30 * low_clean
    if name == "setup_headroom":
        headroom = max(0.0, (16.0 - after.max_height) / 16.0)
        return 1.00 * after.t_spin_slots * headroom * headroom / (1.0 + after.holes)
    if name == "b2b_charge_difficult":
        if not difficult or features.lines <= 0:
            return 0.0
        charge = max(0, game.b2b_chain) + game.surge_charge / 4.0
        return 0.24 * (1.0 + charge) + 0.12 * features.attack
    if name == "b2b_charge_break":
        if not game.back_to_back or features.lines <= 0 or difficult:
            return 0.0
        charge = max(1, game.b2b_chain) + game.surge_charge / 3.0
        return -0.42 * charge
    if name == "combo_safe_extend":
        if game.combo < 0 or features.lines <= 0:
            return 0.0
        safety = 1.0 / (1.0 + after.holes + max(0, after.max_height - 10) / 4.0)
        return 0.34 * (game.combo + 1) * safety + 0.08 * features.attack
    if name == "downstack_efficiency":
        if features.lines <= 0:
            return 0.0
        return 0.95 * holes_removed + 0.12 * depth_removed + 0.08 * features.attack * holes_removed
    if name == "hole_depth_rescue":
        return 0.24 * depth_removed * (1.0 + danger) + 0.55 * holes_removed * danger
    if name == "hold_t_ready":
        if game.hold_piece != "T" or game.current == "T":
            return 0.0
        return 0.70 * low_clean - 0.22 * features.new_holes - 0.10 * danger
    if name == "hold_i_rescue":
        if game.hold_piece != "I" or after.max_height < 10:
            return 0.0
        return 0.30 * features.lines + 0.18 * features.attack + 0.30 * holes_removed - 0.10 * features.new_holes
    if name == "danger_clear_urgency":
        if danger <= 0:
            return 0.0
        return danger * (0.26 * features.attack + 0.20 * features.lines + 0.35 * holes_removed - 0.18 * features.new_holes)
    if name == "spin_safety_efficiency":
        if features.spin_lines <= 0:
            return 0.0
        safety = 1.0 / (1.0 + after.holes + after.max_height / 10.0)
        return 0.65 * features.spin_lines * safety + 0.16 * features.attack * safety
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
        max_turns=100,
        seed_base=1_210_003,
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
    short = _parallel_solo(names, games=2, pieces=150, seed_base=410_009, seed_step=79)
    survivors = sorted(
        (row for row in short if row["name"] != "baseline"),
        key=lambda row: _key(row, 2),
        reverse=True,
    )[:3]
    survivor_names = [str(row["name"]) for row in survivors]

    fresh = _parallel_solo(
        ("baseline", *survivor_names),
        games=3,
        pieces=240,
        seed_base=830_027,
        seed_step=103,
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
