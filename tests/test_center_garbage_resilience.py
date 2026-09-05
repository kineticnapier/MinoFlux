from __future__ import annotations

import json
from pathlib import Path

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


def test_center_garbage_resilience_matches_four_line_center_probe() -> None:
    game = Game(730_001)
    placement = game.legal_placements()[0]
    evaluation = evaluate_placement(game, placement)

    probe = clone_game(game)
    probe.place(placement)
    probe.add_garbage(4, 4)
    stressed = extract_board_features(probe.board)
    expected = -(
        0.075 * stressed.holes
        + 0.018 * stressed.hole_depth
        + 0.025 * stressed.max_height
    )

    assert evaluation.features.center_garbage_resilience == expected


def test_center_garbage_resilience_evaluation_does_not_mutate_game() -> None:
    game = Game(730_002)
    before = game.snapshot(queue_size=7)
    placement = game.legal_placements()[0]

    evaluate_placement(game, placement)

    assert game.snapshot(queue_size=7) == before


def test_old_model_defaults_center_garbage_resilience(tmp_path: Path) -> None:
    path = tmp_path / "old-model.json"
    path.write_text(
        json.dumps({"format": MODEL_FORMAT, "weights": {"attack": 0.91}}),
        encoding="utf-8",
    )

    weights = load_weights(path)

    assert weights.attack == 0.91
    assert weights.center_garbage_resilience == DEFAULT_WEIGHTS.center_garbage_resilience


def test_center_garbage_resilience_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    weights = HeuristicWeights(center_garbage_resilience=0.42)

    save_weights(path, weights)

    assert load_weights(path) == weights
