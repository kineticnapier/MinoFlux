from __future__ import annotations

import random
import unittest

from minoflux_ai.features import (
    count_t_spin_slots_from_masks,
    extract_board_features_from_masks,
)


def _reference_occupied_or_wall(
    rows: tuple[int, ...], x: int, y: int, width: int
) -> bool:
    return (
        x < 0
        or x >= width
        or y < 0
        or y >= len(rows)
        or bool(rows[y] & (1 << x))
    )


def _reference_empty(rows: tuple[int, ...], x: int, y: int, width: int) -> bool:
    return (
        0 <= x < width
        and 0 <= y < len(rows)
        and not (rows[y] & (1 << x))
    )


def _reference_count_t_spin_slots_from_masks(
    rows: tuple[int, ...], *, width: int, start_y: int = 0
) -> int:
    if not rows:
        return 0
    height = len(rows)
    orientations = (
        ((0, -1), (-1, 0), (0, 0), (1, 0)),
        ((0, -1), (0, 0), (1, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((0, -1), (-1, 0), (0, 0), (0, 1)),
    )
    slots = 0
    for pivot_y in range(max(0, int(start_y)), height):
        row_mask = rows[pivot_y]
        for pivot_x in range(width):
            if row_mask & (1 << pivot_x):
                continue
            corners = (
                _reference_occupied_or_wall(rows, pivot_x - 1, pivot_y - 1, width),
                _reference_occupied_or_wall(rows, pivot_x + 1, pivot_y - 1, width),
                _reference_occupied_or_wall(rows, pivot_x - 1, pivot_y + 1, width),
                _reference_occupied_or_wall(rows, pivot_x + 1, pivot_y + 1, width),
            )
            if sum(corners) < 3:
                continue
            if any(
                all(
                    _reference_empty(rows, pivot_x + dx, pivot_y + dy, width)
                    for dx, dy in cells
                )
                for cells in orientations
            ):
                slots += 1
    return slots


def _rows_for_local_pattern(
    *, width: int, height: int, pivot_x: int, pivot_y: int, pattern: int
) -> tuple[int, ...]:
    rows = [0] * height
    neighbors = (
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    for bit, (dx, dy) in enumerate(neighbors):
        if not (pattern & (1 << bit)):
            continue
        x = pivot_x + dx
        y = pivot_y + dy
        if 0 <= x < width and 0 <= y < height:
            rows[y] |= 1 << x
    return tuple(rows)


class TSpinSlotEquivalenceTests(unittest.TestCase):
    def assert_matches_reference(
        self, rows: tuple[int, ...], *, width: int, start_y: int = 0
    ) -> None:
        self.assertEqual(
            count_t_spin_slots_from_masks(rows, width=width, start_y=start_y),
            _reference_count_t_spin_slots_from_masks(
                rows, width=width, start_y=start_y
            ),
        )

    def test_all_interior_eight_neighbor_patterns_match_reference(self) -> None:
        width = 5
        height = 5
        for pattern in range(1 << 8):
            with self.subTest(pattern=pattern):
                rows = _rows_for_local_pattern(
                    width=width,
                    height=height,
                    pivot_x=2,
                    pivot_y=2,
                    pattern=pattern,
                )
                self.assert_matches_reference(rows, width=width)

    def test_all_edge_and_corner_local_patterns_match_reference(self) -> None:
        width = 5
        height = 5
        pivots = (
            (0, 0),
            (2, 0),
            (4, 0),
            (0, 2),
            (4, 2),
            (0, 4),
            (2, 4),
            (4, 4),
        )
        for pivot_x, pivot_y in pivots:
            for pattern in range(1 << 8):
                with self.subTest(
                    pivot=(pivot_x, pivot_y), pattern=pattern
                ):
                    rows = _rows_for_local_pattern(
                        width=width,
                        height=height,
                        pivot_x=pivot_x,
                        pivot_y=pivot_y,
                        pattern=pattern,
                    )
                    self.assert_matches_reference(rows, width=width)

    def test_random_boards_and_start_y_values_match_reference(self) -> None:
        rng = random.Random(20260908)
        for width, height in ((4, 4), (10, 8), (10, 24)):
            row_limit = (1 << width) - 1
            for case in range(200):
                rows = tuple(rng.getrandbits(width) & row_limit for _ in range(height))
                for start_y in (0, height // 2, height - 1, height, -3):
                    with self.subTest(
                        width=width,
                        height=height,
                        case=case,
                        start_y=start_y,
                    ):
                        self.assert_matches_reference(
                            rows,
                            width=width,
                            start_y=start_y,
                        )

    def test_extract_board_features_uses_reference_slot_count(self) -> None:
        rng = random.Random(20260908 ^ 0x5A17)
        width = 10
        height = 24
        for case in range(300):
            rows = tuple(rng.getrandbits(width) for _ in range(height))
            features = extract_board_features_from_masks(rows, width=width)
            slot_start_y = (
                height
                if features.max_height == 0
                else max(0, height - features.max_height - 1)
            )
            with self.subTest(case=case):
                self.assertEqual(
                    features.t_spin_slots,
                    _reference_count_t_spin_slots_from_masks(
                        rows,
                        width=width,
                        start_y=slot_start_y,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
