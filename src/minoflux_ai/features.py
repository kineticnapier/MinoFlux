from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

from .bitboard import board_row_masks

Board = Sequence[Sequence[object | None]]


@dataclass(frozen=True, slots=True)
class BoardFeatures:
    aggregate_height: int
    max_height: int
    holes: int
    hole_depth: int
    bumpiness: int
    wells: int
    t_spin_slots: int
    occupied_cells: int

    @property
    def t_spin_slot_density(self) -> float:
        """Reward T-spin cavities less when they coexist with buried holes."""
        return self.t_spin_slots / (1 + self.holes)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _column_masks(rows: Sequence[int], width: int) -> tuple[int, ...]:
    columns = [0] * width
    for y, row_mask in enumerate(rows):
        mask = int(row_mask)
        while mask:
            bit = mask & -mask
            columns[bit.bit_length() - 1] |= 1 << y
            mask ^= bit
    return tuple(columns)


def _column_heights_from_masks(rows: Sequence[int], width: int) -> tuple[int, ...]:
    height = len(rows)
    result: list[int] = []
    for bits in _column_masks(rows, width):
        if not bits:
            result.append(0)
            continue
        top = (bits & -bits).bit_length() - 1
        result.append(height - top)
    return tuple(result)


def column_heights(board: Board) -> tuple[int, ...]:
    if not board:
        return ()
    width = len(board[0])
    return _column_heights_from_masks(board_row_masks(board), width)


def _occupied_or_wall_masks(rows: Sequence[int], x: int, y: int, width: int) -> bool:
    return (
        x < 0
        or x >= width
        or y < 0
        or y >= len(rows)
        or bool(rows[y] & (1 << x))
    )


def _empty_masks(rows: Sequence[int], x: int, y: int, width: int) -> bool:
    return (
        0 <= x < width
        and 0 <= y < len(rows)
        and not (rows[y] & (1 << x))
    )


def count_t_spin_slots_from_masks(
    rows: Sequence[int],
    *,
    width: int,
    start_y: int = 0,
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
                _occupied_or_wall_masks(rows, pivot_x - 1, pivot_y - 1, width),
                _occupied_or_wall_masks(rows, pivot_x + 1, pivot_y - 1, width),
                _occupied_or_wall_masks(rows, pivot_x - 1, pivot_y + 1, width),
                _occupied_or_wall_masks(rows, pivot_x + 1, pivot_y + 1, width),
            )
            if sum(corners) < 3:
                continue
            if any(
                all(
                    _empty_masks(rows, pivot_x + dx, pivot_y + dy, width)
                    for dx, dy in cells
                )
                for cells in orientations
            ):
                slots += 1
    return slots


def count_t_spin_slots(board: Board, *, start_y: int = 0) -> int:
    if not board:
        return 0
    width = len(board[0])
    return count_t_spin_slots_from_masks(
        board_row_masks(board),
        width=width,
        start_y=start_y,
    )


def _triangular_runs(bits: int) -> int:
    total = 0
    while bits:
        start_bit = bits & -bits
        start = start_bit.bit_length() - 1
        shifted = bits >> start
        run = 0
        while shifted & 1:
            run += 1
            shifted >>= 1
        total += run * (run + 1) // 2
        bits &= ~(((1 << run) - 1) << start)
    return total


def extract_board_features_from_masks(
    rows: Sequence[int],
    *,
    width: int,
) -> BoardFeatures:
    """Extract exact stack features from compact occupancy rows."""

    if not rows:
        return BoardFeatures(0, 0, 0, 0, 0, 0, 0, 0)
    height = len(rows)
    row_limit = (1 << width) - 1
    normalized = tuple(int(row) & row_limit for row in rows)
    columns = _column_masks(normalized, width)
    full_height_mask = (1 << height) - 1

    heights: list[int] = []
    holes = 0
    hole_depth = 0
    for bits in columns:
        if not bits:
            heights.append(0)
            continue
        top = (bits & -bits).bit_length() - 1
        heights.append(height - top)
        below_top = full_height_mask ^ ((1 << top) - 1)
        hole_bits = below_top & ~bits & full_height_mask
        holes += hole_bits.bit_count()
        pending = hole_bits
        while pending:
            bit = pending & -pending
            y = bit.bit_length() - 1
            hole_depth += (bits & ((1 << y) - 1)).bit_count()
            pending ^= bit

    bumpiness = sum(abs(left - right) for left, right in zip(heights, heights[1:]))

    wells = 0
    for x, bits in enumerate(columns):
        left = full_height_mask if x == 0 else columns[x - 1]
        right = full_height_mask if x == width - 1 else columns[x + 1]
        well_cells = (~bits) & left & right & full_height_mask
        wells += _triangular_runs(well_cells)

    max_height = max(heights, default=0)
    slot_start_y = height if max_height == 0 else max(0, height - max_height - 1)
    return BoardFeatures(
        aggregate_height=sum(heights),
        max_height=max_height,
        holes=holes,
        hole_depth=hole_depth,
        bumpiness=bumpiness,
        wells=wells,
        t_spin_slots=count_t_spin_slots_from_masks(
            normalized,
            width=width,
            start_y=slot_start_y,
        ),
        occupied_cells=sum(mask.bit_count() for mask in normalized),
    )


def extract_board_features(board: Board) -> BoardFeatures:
    """Extract deterministic stack features from a board after line clears."""

    if not board:
        return BoardFeatures(0, 0, 0, 0, 0, 0, 0, 0)
    width = len(board[0])
    if any(len(row) != width for row in board):
        raise ValueError("Board rows must have equal width")
    return extract_board_features_from_masks(
        board_row_masks(board),
        width=width,
    )
