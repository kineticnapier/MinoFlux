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


def _danger(board) -> float:
    features = extract_board_features(board)
    return (
        2.2 * features.holes
        + 0.26 * features.hole_depth
        + 0.72 * features.max_height
        + 0.08 * features.bumpiness
    )


def test_center_garbage_worst_case_matches_four_center_probes() -> None:
    game = Game(890_001)
    placement = game.legal_placements()[0]
    evaluation = evaluate_placement(game, placement)

    dangers = []
    for hole in (3, 4, 5, 6):
        probe = clone_game(game)
        probe.place(placement)
        probe.add_garbage(4, hole)
        dangers.append(_danger(probe.board))

    assert evaluation.features.center_garbage_worst_case == -0.15 * max(dangers)


def test_center_garbage_worst_case_does_not_mutate_game() -> None:
    game = Game(890_002)
    before = game.snapshot(queue_size=7)
    placement = game.legal_placements()[0]

    evaluate_placement(game, placement)

    assert game.snapshot(queue_size=7) == before


def test_old_model_defaults_center_garbage_worst_case(tmp_path: Path) -> None:
    path = tmp_path / "old-model.json"
    path.write_text(
        json.dumps({"format": MODEL_FORMAT, "weights": {"attack": 0.91}}),
        encoding="utf-8",
    )

    weights = load_weights(path)

    assert weights.attack == 0.91
    assert weights.center_garbage_worst_case == DEFAULT_WEIGHTS.center_garbage_worst_case


def test_center_garbage_worst_case_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    weights = HeuristicWeights(center_garbage_worst_case=0.42)

    save_weights(path, weights)

    assert load_weights(path) == weights
