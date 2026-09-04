from __future__ import annotations

from itertools import product
import random

from minoflux_ai.features import (
    extract_board_features_from_masks,
    max_height_and_holes_from_masks,
)


def _reference_max_height_and_holes(
    rows: tuple[int, ...],
    *,
    width: int,
) -> tuple[int, int]:
    """Straightforward per-column definition, independent of the fast path."""

    height = len(rows)
    row_limit = (1 << width) - 1
    normalized = tuple(int(row) & row_limit for row in rows)
    max_height = 0
    holes = 0
    for x in range(width):
        occupied = [y for y, row in enumerate(normalized) if row & (1 << x)]
        if not occupied:
            continue
        top = occupied[0]
        max_height = max(max_height, height - top)
        holes += sum(not (normalized[y] & (1 << x)) for y in range(top, height))
    return max_height, holes


def test_fast_board_metrics_cover_curated_edge_cases() -> None:
    cases = (
        ((), 10),
        ((0,) * 24, 10),
        ((0,) * 23 + (0b1111111111,), 10),
        ((0b1, 0, 0b1, 0), 1),
        ((0b1010, 0b0010, 0b1000, 0), 4),
        ((0b1111, 0, 0, 0b1111), 4),
        # Bits beyond width are deliberately ignored by both implementations.
        ((0b1_0001, 0b1_0000, 0b0010, 0), 4),
    )
    for rows, width in cases:
        assert max_height_and_holes_from_masks(rows, width=width) == (
            _reference_max_height_and_holes(rows, width=width)
        )


def test_fast_board_metrics_match_reference_exhaustively_on_small_boards() -> None:
    width = 4
    for rows in product(range(1 << width), repeat=4):
        assert max_height_and_holes_from_masks(rows, width=width) == (
            _reference_max_height_and_holes(rows, width=width)
        )


def test_fast_board_metrics_match_full_features_on_random_standard_boards() -> None:
    rng = random.Random(0x4D_46)
    width = 10
    for _ in range(5_000):
        rows = tuple(rng.randrange(1 << (width + 3)) for _ in range(24))
        fast = max_height_and_holes_from_masks(rows, width=width)
        full = extract_board_features_from_masks(rows, width=width)
        assert fast == (full.max_height, full.holes)
