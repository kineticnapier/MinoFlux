from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from itertools import islice
import json

import minoflux_ai.heuristic as heuristic
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.versus_benchmark import run_versus_benchmark

CANDIDATES = (
    "slot_supply_pressure",
    "slot_lifetime_quality",
    "t_conversion_next_chain",
    "t_conversion_cleanup",
    "setup_clean_exchange",
    "attack_setup_preservation",
    "surface_slot_stability",
    "slot_depth_quality",
    "b2b_spin_start_quality",
    "b2b_chain_cleanliness",
    "surge_charge_quality",
    "combo_difficult_bridge",
    "nonlinear_hole_depth",
    "setup_without_greed",
)


def _queue(game, count):
    return tuple(islice(game.queue, count))


def _next_t_distance(game):
    if game.current == "T":
        return 0
    for index, piece in enumerate(_queue(game, 7)):
        if piece == "T":
            return index + 1
    return 8


def _t_supply(game):
    return (1 if game.current == "T" else 0) + sum(1 for piece in _queue(game, 7) if piece == "T") + (1 if game.hold_piece == "T" else 0)


def _is_difficult(features):
    return features.spin_lines > 0 or features.lines == 4


def _extra_score(name, game, features):
    before = extract_board_features(game.board)
    after = features.board
    slots = after.t_spin_slots
    delta = features.t_spin_slot_delta
    tdist = _next_t_distance(game)
    supply = _t_supply(game)
    urgency = max(0.0, (7.0 - tdist) / 7.0)
    clean = 1.0 / (1.0 + after.holes + after.max_height / 7.0)
    stable = 1.0 / (1.0 + after.bumpiness / 8.0 + after.wells / 5.0)
    holes_removed = max(0, before.holes - after.holes)
    height_drop = max(0, before.max_height - after.max_height)
    depth_drop = max(0, before.hole_depth - after.hole_depth)
    spin = game.current == "T" and features.spin_lines > 0
    difficult = _is_difficult(features)

    if name == "slot_supply_pressure":
        usable = min(slots, supply)
        shortage = max(0, slots - supply)
        idle_supply = max(0, supply - slots)
        return urgency * (0.22 * usable - 0.30 * shortage) - (1.0 - urgency) * 0.10 * idle_supply
    if name == "slot_lifetime_quality":
        return (0.28 * slots * urgency * clean) - (0.16 * slots * (1.0 - urgency) * (1.0 - clean))
    if name == "t_conversion_next_chain":
        if not spin:
            return 0.0
        next_supply = sum(1 for piece in _queue(game, 7) if piece == "T") + (1 if game.hold_piece == "T" else 0)
        return 0.30 * features.spin_lines + 0.24 * min(slots, next_supply) * stable - 0.15 * max(0, slots - next_supply)
    if name == "t_conversion_cleanup":
        if not spin:
            return 0.0
        return 0.22 * features.spin_lines + 0.16 * holes_removed + 0.10 * depth_drop + 0.10 * height_drop - 0.12 * features.new_holes
    if name == "setup_clean_exchange":
        if delta <= 0:
            return 0.0
        return delta * (0.30 * clean + 0.16 * stable + 0.08 * height_drop) - 0.24 * features.new_holes
    if name == "attack_setup_preservation":
        destroyed = max(0, -delta)
        preserved_attack = features.attack if destroyed == 0 else 0
        return 0.14 * preserved_attack + 0.10 * max(0, delta) - 0.28 * destroyed * max(1, features.attack)
    if name == "surface_slot_stability":
        return 0.34 * slots * stable * clean
    if name == "slot_depth_quality":
        depth_factor = 1.0 / (1.0 + after.hole_depth / 10.0)
        return 0.32 * slots * depth_factor - 0.10 * max(0, after.hole_depth - before.hole_depth)
    if name == "b2b_spin_start_quality":
        if game.back_to_back or not difficult:
            return 0.0
        return 0.20 * features.attack + 0.18 * features.spin_lines + 0.14 * holes_removed + 0.08 * height_drop
    if name == "b2b_chain_cleanliness":
        if not game.back_to_back:
            return 0.0
        if difficult:
            return (0.16 + 0.035 * min(game.b2b_chain, 6)) * features.attack * clean + 0.12 * features.spin_lines
        if features.lines > 0:
            return -0.24 * min(game.b2b_chain, 6)
        return 0.0
    if name == "surge_charge_quality":
        if not game.back_to_back:
            return 0.0
        if difficult:
            return 0.035 * game.surge_charge * (features.attack + features.spin_lines) * clean
        if features.lines > 0 and game.surge_charge > 0:
            return -0.04 * game.surge_charge
        return 0.0
    if name == "combo_difficult_bridge":
        if game.combo < 0:
            return 0.0
        if difficult:
            return 0.14 * (game.combo + 1) + 0.12 * features.attack
        if features.lines > 0 and game.back_to_back:
            return -0.16 * (game.combo + 1)
        return 0.0
    if name == "nonlinear_hole_depth":
        excess = max(0.0, after.hole_depth - 6.0)
        repaired = max(0, before.hole_depth - after.hole_depth)
        return -0.012 * excess * excess + 0.08 * repaired
    if name == "setup_without_greed":
        created = max(0, delta)
        if created == 0:
            return 0.0
        greed = max(0, features.attack - 2)
        return 0.24 * created * clean + 0.10 * created * urgency - 0.06 * greed * features.new_holes
    raise ValueError(name)


def _install(name):
    marker = replace(DEFAULT_WEIGHTS)
    marker_id = id(marker)
    original = heuristic._context_score

    def patched(game, features, weights):
        base = original(game, features, weights)
        return base + (_extra_score(name, game, features) if id(weights) == marker_id else 0.0)

    heuristic._context_score = patched
    return marker


def _solo(name, games, pieces, seed_base, seed_step):
    weights = DEFAULT_WEIGHTS if name == "baseline" else _install(name)
    result = run_heuristic_benchmark(
        games=games,
        max_pieces=pieces,
        seed_base=seed_base,
        seed_step=seed_step,
        weights=weights,
        workers=1,
    )
    return {
        "name": name,
        "pieces": result.pieces,
        "attack": result.attack,
        "attackPerPiece": result.attack / result.pieces if result.pieces else 0.0,
        "topouts": result.topouts,
        "completed": result.completed,
        "tSpins": result.spins,
        "spinLines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def _key(row, games):
    return (
        -row["topouts"],
        row["completed"] / games,
        row["attackPerPiece"],
        row["tsd"] + 1.5 * row["tst"],
        row["spinLines"],
    )


def _parallel(names, games, pieces, seed_base, seed_step):
    rows = []
    with ProcessPoolExecutor(max_workers=min(8, len(names))) as pool:
        futures = [pool.submit(_solo, name, games, pieces, seed_base, seed_step) for name in names]
        for future in as_completed(futures):
            rows.append(future.result())
    return sorted(rows, key=lambda row: row["name"])


def _versus(name):
    weights = _install(name)
    result = run_versus_benchmark(
        games=8,
        max_turns=120,
        seed_base=2_280_037,
        seed_step=127,
        player_weights=weights,
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
        "cancel": sum(item.player_canceled for item in games) / len(games),
        "baselineCancel": sum(item.ai_canceled for item in games) / len(games),
        "received": sum(item.player_received for item in games) / len(games),
        "baselineReceived": sum(item.ai_received for item in games) / len(games),
        "maxB2B": sum(item.player_max_b2b for item in games) / len(games),
        "baselineMaxB2B": sum(item.ai_max_b2b for item in games) / len(games),
        "topouts": result.ai_wins,
        "baselineTopouts": result.player_wins,
    }


def main():
    names = ("baseline", *CANDIDATES)
    short = _parallel(names, 2, 180, 730_019, 97)
    survivors = [
        row["name"]
        for row in sorted(
            (item for item in short if item["name"] != "baseline"),
            key=lambda item: _key(item, 2),
            reverse=True,
        )[:3]
    ]
    fresh = _parallel(("baseline", *survivors), 3, 300, 1_330_031, 131)
    baseline = next(item for item in fresh if item["name"] == "baseline")
    ranked = sorted(
        (item for item in fresh if item["name"] != "baseline"),
        key=lambda item: _key(item, 3),
        reverse=True,
    )
    finalists = []
    for item in ranked:
        safe = item["topouts"] <= baseline["topouts"] and item["completed"] >= baseline["completed"]
        stronger = item["attackPerPiece"] >= baseline["attackPerPiece"] * 1.01
        if safe and stronger:
            finalists.append(item["name"])
            if len(finalists) == 2:
                break
    payload = {
        "candidateCount": len(CANDIDATES),
        "short": short,
        "survivors": survivors,
        "fresh": fresh,
        "finalists": finalists,
        "versus": [_versus(name) for name in finalists],
    }
    print("RESULT_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
