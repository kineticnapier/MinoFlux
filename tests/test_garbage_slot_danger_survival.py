from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from minoflux_ai.heuristic import (
    DEFAULT_WEIGHTS,
    MODEL_FORMAT,
    HeuristicWeights,
    evaluate_placement,
    load_weights,
    save_weights,
)
from minoflux_engine import Game


def _t_supply(game: Game) -> int:
    supply = int(game.current == "T") + int(game.hold_piece == "T")
    for index, piece in enumerate(game.queue):
        if index >= 6:
            break
        supply += int(piece == "T")
    return supply


def test_garbage_slot_danger_survival_score_matches_formula() -> None:
    game = Game(892_001)
    placement = game.legal_placements()[0]
    disabled = replace(DEFAULT_WEIGHTS, garbage_slot_danger_survival=0.0)

    without = evaluate_placement(game, placement, disabled)
    with_feature = evaluate_placement(game, placement, DEFAULT_WEIGHTS)
    features = with_feature.features
    supply = _t_supply(game)
    floor = features.garbage_t_spin_slot_floor
    matched = min(floor, supply)
    unmatched = max(0, floor - supply)
    danger = max(0.0, -features.center_garbage_worst_case)
    fragile_slots = max(0, features.board.t_spin_slots - floor)
    expected = 0.34 * matched - 0.28 * unmatched - 0.05 * danger * fragile_slots

    assert with_feature.score - without.score == pytest.approx(expected)


def test_garbage_slot_danger_survival_does_not_mutate_game() -> None:
    game = Game(892_002)
    before = game.snapshot(queue_size=7)
    placement = game.legal_placements()[0]

    evaluate_placement(game, placement)

    assert game.snapshot(queue_size=7) == before


def test_old_model_defaults_garbage_slot_danger_survival(tmp_path: Path) -> None:
    path = tmp_path / "old-model.json"
    path.write_text(
        json.dumps({"format": MODEL_FORMAT, "weights": {"attack": 0.91}}),
        encoding="utf-8",
    )

    weights = load_weights(path)

    assert weights.attack == 0.91
    assert (
        weights.garbage_slot_danger_survival
        == DEFAULT_WEIGHTS.garbage_slot_danger_survival
    )


def test_garbage_slot_danger_survival_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    weights = HeuristicWeights(garbage_slot_danger_survival=0.42)

    save_weights(path, weights)

    assert load_weights(path) == weights
