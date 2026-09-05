from __future__ import annotations

import random

import pytest

from minoflux_ai.features import (
    extract_board_features_from_masks,
    extract_teacher_board_features,
    extract_teacher_board_features_from_masks,
)


def _terms(features):
    return features.max_height, features.holes, features.hole_depth, features.bumpiness


def _cell_reference(rows, width):
    heights = []
    holes = 0
    depth = 0
    for x in range(width):
        column_height = 0
        filled_above = 0
        for y, row in enumerate(rows):
            if row & (1 << x):
                if not filled_above:
                    column_height = len(rows) - y
                filled_above += 1
            elif filled_above:
                holes += 1
                depth += filled_above
        heights.append(column_height)
    return (
        max(heights, default=0),
        holes,
        depth,
        sum(abs(a - b) for a, b in zip(heights, heights[1:])),
    )


def test_teacher_metrics_exhaustive_small_boards():
    width = 3
    for packed in range(1 << 9):
        rows = tuple((packed >> (y * width)) & 7 for y in range(3))
        actual = extract_teacher_board_features_from_masks(rows, width=width)
        assert _terms(actual) == _cell_reference(rows, width)
        assert _terms(actual) == _terms(extract_board_features_from_masks(rows, width=width))


def test_teacher_metrics_random_boards_and_board_wrapper():
    rng = random.Random(81237)
    for _ in range(400):
        width = rng.randrange(1, 13)
        height = rng.randrange(1, 25)
        rows = tuple(rng.getrandbits(width) for _ in range(height))
        board = [["T" if row & (1 << x) else None for x in range(width)] for row in rows]
        actual = extract_teacher_board_features_from_masks(rows, width=width)
        assert _terms(actual) == _cell_reference(rows, width)
        assert _terms(actual) == _terms(extract_board_features_from_masks(rows, width=width))
        assert extract_teacher_board_features(board) == actual
        assert (actual.aggregate_height, actual.wells, actual.t_spin_slots, actual.occupied_cells) == (0, 0, 0, 0)


@pytest.mark.parametrize("rows,width", [((), 10), ((0, 0), 0), ((-1, 1024, 2047), 10)])
def test_teacher_metrics_empty_and_mask_normalization(rows, width):
    assert _terms(extract_teacher_board_features_from_masks(rows, width=width)) == _terms(
        extract_board_features_from_masks(rows, width=width)
    )


def test_teacher_metrics_empty_and_ragged_board():
    assert _terms(extract_teacher_board_features([])) == (0, 0, 0, 0)
    with pytest.raises(ValueError, match="equal width"):
        extract_teacher_board_features([[None], [None, None]])
