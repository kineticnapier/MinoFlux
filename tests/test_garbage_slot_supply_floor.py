from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import (
    DEFAULT_WEIGHTS,
    MODEL_FORMAT,
    HeuristicWeights,
    evaluate_placement,
    load_weights,
    save_weights,
)
from minoflux_ai.search import clone_game
from minoflux_engine import Game


def _t_supply(game: Game) -> int:
    supply = int(game.current == "T") + int(game.hold_piece == "T")
    for index, piece in enumerate(game.queue):
        if index >= 6:
            break
        supply += int(piece == "T")
    return supply


def test_garbage_slot_floor_matches_four_center_probes() -> None:
    game = Game(891_001)
    placement = game.legal_placements()[0]
    evaluation = evaluate_placement(game, placement)

    slot_counts = []
    for hole in (3, 4, 5, 6):
        probe = clone_game(game)
        probe.place(placement)
        probe.add_garbage(4, hole)
        slot_counts.append(extract_board_features(probe.board).t_spin_slots)

    assert evaluation.features.garbage_t_spin_slot_floor == min(slot_counts)


def test_garbage_slot_supply_floor_score_matches_formula() -> None:
    game = Game(891_002)
    placement = game.legal_placements()[0]
    disabled = replace(DEFAULT_WEIGHTS, garbage_slot_supply_floor=0.0)

    without = evaluate_placement(game, placement, disabled)
    with_feature = evaluate_placement(game, placement, DEFAULT_WEIGHTS)
    floor = with_feature.features.garbage_t_spin_slot_floor
    supply = _t_supply(game)
    expected = 0.30 * min(floor, supply) - 0.22 * max(0, floor - supply)

    assert with_feature.score - without.score == pytest.approx(expected)


def test_garbage_slot_supply_floor_does_not_mutate_game() -> None:
    game = Game(891_003)
    before = game.snapshot(queue_size=7)
    placement = game.legal_placements()[0]

    evaluate_placement(game, placement)

    assert game.snapshot(queue_size=7) == before


def test_old_model_defaults_garbage_slot_supply_floor(tmp_path: Path) -> None:
    path = tmp_path / "old-model.json"
    path.write_text(
        json.dumps({"format": MODEL_FORMAT, "weights": {"attack": 0.91}}),
        encoding="utf-8",
    )

    weights = load_weights(path)

    assert weights.attack == 0.91
    assert weights.garbage_slot_supply_floor == DEFAULT_WEIGHTS.garbage_slot_supply_floor


def test_garbage_slot_supply_floor_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    weights = HeuristicWeights(garbage_slot_supply_floor=0.42)

    save_weights(path, weights)

    assert load_weights(path) == weights
