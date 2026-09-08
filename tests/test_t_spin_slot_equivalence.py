from __future__ import annotations

import random
import unittest

from minoflux_ai.features import (
    BoardFeatures,
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
            empty_cardinals = (
                int(_reference_empty(rows, pivot_x, pivot_y - 1, width))
                + int(_reference_empty(rows, pivot_x - 1, pivot_y, width))
                + int(_reference_empty(rows, pivot_x + 1, pivot_y, width))
                + int(_reference_empty(rows, pivot_x, pivot_y + 1, width))
            )
            if empty_cardinals >= 3:
                slots += 1
    return slots


def _reference_board_features_from_masks(
    rows: tuple[int, ...], *, width: int
) -> BoardFeatures:
    height = len(rows)
    row_limit = (1 << width) - 1
    normalized = tuple(row & row_limit for row in rows)
    heights: list[int] = []
    holes = 0
    hole_depth = 0
    occupied_cells = 0
    for x in range(width):
        top = height
        seen_block = False
        blocks_above = 0
        for y, row in enumerate(normalized):
            occupied = bool(row & (1 << x))
            if occupied:
                occupied_cells += 1
                if top == height:
                    top = y
                seen_block = True
                blocks_above += 1
            elif seen_block:
                holes += 1
                hole_depth += blocks_above
        heights.append(height - top)

    wells = 0
    for x in range(width):
        depth = 0
        for row in normalized:
            if row & (1 << x):
                depth = 0
                continue
            left_filled = x == 0 or bool(row & (1 << (x - 1)))
            right_filled = x == width - 1 or bool(row & (1 << (x + 1)))
            if left_filled and right_filled:
                depth += 1
                wells += depth
            else:
                depth = 0

    max_height = max(heights, default=0)
    slot_start_y = height if max_height == 0 else max(0, height - max_height - 1)
    return BoardFeatures(
        aggregate_height=sum(heights),
        max_height=max_height,
        holes=holes,
        hole_depth=hole_depth,
        bumpiness=sum(
            abs(left - right) for left, right in zip(heights, heights[1:])
        ),
        wells=wells,
        t_spin_slots=_reference_count_t_spin_slots_from_masks(
            normalized,
            width=width,
            start_y=slot_start_y,
        ),
        occupied_cells=occupied_cells,
    )


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

    def test_exhaustive_small_boards_and_start_y_boundaries_match_reference(
        self,
    ) -> None:
        comparisons = 0
        for width in range(1, 5):
            row_limit = (1 << width) - 1
            for height in range(1, 5):
                start_values = set(
                    (-3, 0, height // 2, height - 1, height, height + 2)
                )
                for board_bits in range(1 << (width * height)):
                    rows = tuple(
                        (board_bits >> (y * width)) & row_limit
                        for y in range(height)
                    )
                    for start_y in start_values:
                        self.assert_matches_reference(
                            rows,
                            width=width,
                            start_y=start_y,
                        )
                        comparisons += 1
        self.assertEqual(comparisons, 449_324)

    def test_width_outside_bits_are_ignored(self) -> None:
        rng = random.Random(20260908)
        comparisons = 0
        for width in range(1, 21):
            row_limit = (1 << width) - 1
            for case in range(20):
                height = 2 * case + 1
                outside = ((1 << (1 + case % 6)) - 1) << width
                rows = tuple(
                    (rng.getrandbits(width) & row_limit) | outside
                    for _ in range(height)
                )
                start_y = (-3, 0, height // 2, height - 1, height + 2)[case % 5]
                actual = count_t_spin_slots_from_masks(
                    rows,
                    width=width,
                    start_y=start_y,
                )
                masked = count_t_spin_slots_from_masks(
                    tuple(row & row_limit for row in rows),
                    width=width,
                    start_y=start_y,
                )
                self.assertEqual(actual, masked)
                self.assert_matches_reference(rows, width=width, start_y=start_y)
                comparisons += 1
        self.assertEqual(comparisons, 400)

    def test_20_000_random_boards_match_reference(self) -> None:
        rng = random.Random(20260908 ^ 0xA11CE)
        start_groups = 5
        dimension_pairs = 20 * 40
        for case in range(20_000):
            width = case % 20 + 1
            height = (case // 20) % 40 + 1
            row_limit = (1 << width) - 1
            outside = 1 << (width + case % 7)
            rows = tuple(
                (rng.getrandbits(width) & row_limit) | outside for _ in range(height)
            )
            start_group = (case // dimension_pairs) % start_groups
            start_y = (-5, 0, height // 2, height - 1, height + 3)[start_group]
            self.assert_matches_reference(rows, width=width, start_y=start_y)

    def test_800_full_board_feature_values_match_reference(self) -> None:
        rng = random.Random(20260908 ^ 0x5A17)
        for case in range(800):
            width = case % 20 + 1
            height = (case // 20) % 40 + 1
            outside = 1 << (width + case % 7)
            rows = tuple(rng.getrandbits(width) | outside for _ in range(height))
            self.assertEqual(
                extract_board_features_from_masks(rows, width=width),
                _reference_board_features_from_masks(rows, width=width),
            )


if __name__ == "__main__":
    unittest.main()
