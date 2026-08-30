from __future__ import annotations

import random

from minoflux_ai.bitboard import board_row_masks
from minoflux_ai.features import extract_board_features, extract_board_features_from_masks
from minoflux_ai.reachability import reachable_placements
from minoflux_engine import Game


def _placement_signature(placement):
    return (
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
        placement.cells,
        placement.last_move_was_rotation,
        placement.rotation_kick_index,
        placement.rotation_from,
        placement.rotation_to,
    )


def test_mask_feature_extractor_matches_board_api() -> None:
    rng = random.Random(712367)
    for _ in range(30):
        board = [
            ["G" if rng.random() < 0.28 else None for _x in range(10)]
            for _y in range(24)
        ]
        expected = extract_board_features(board)
        actual = extract_board_features_from_masks(
            board_row_masks(board),
            width=10,
        )
        assert actual == expected


def test_pathless_reachability_keeps_geometry_and_spin_metadata() -> None:
    for seed in (11, 29, 47, 83):
        game = Game(seed)
        for _ in range(6):
            with_paths = reachable_placements(game, include_paths=True)
            pathless = reachable_placements(game, include_paths=False)
            assert with_paths
            assert {_placement_signature(item) for item in pathless} == {
                _placement_signature(item) for item in with_paths
            }
            assert all(not item.path for item in pathless)
            game.place(with_paths[len(with_paths) // 2])
            if game.game_over:
                break
