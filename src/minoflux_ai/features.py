from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

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
    danger_excess: int = 0
    deep_holes: int = 0
    t_spin_clear_potential: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def column_heights(board: Board) -> tuple[int, ...]:
    if not board:
        return ()
    height = len(board)
    width = len(board[0])
    result: list[int] = []
    for x in range(width):
        top = height
        for y, row in enumerate(board):
            if row[x] is not None:
                top = y
                break
        result.append(height - top)
    return tuple(result)


def _occupied_or_wall(board: Board, x: int, y: int) -> bool:
    height = len(board)
    width = len(board[0]) if board else 0
    return x < 0 or x >= width or y < 0 or y >= height or board[y][x] is not None


def _empty(board: Board, x: int, y: int) -> bool:
    height = len(board)
    width = len(board[0]) if board else 0
    return 0 <= x < width and 0 <= y < height and board[y][x] is None


def _t_spin_slot_pivots(board: Board, *, start_y: int = 0) -> tuple[tuple[int, int], ...]:
    if not board:
        return ()
    height = len(board)
    width = len(board[0])
    orientations = (
        ((0, -1), (-1, 0), (0, 0), (1, 0)),
        ((0, -1), (0, 0), (1, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((0, -1), (-1, 0), (0, 0), (0, 1)),
    )
    pivots: list[tuple[int, int]] = []
    for pivot_y in range(max(0, int(start_y)), height):
        for pivot_x in range(width):
            if board[pivot_y][pivot_x] is not None:
                continue
            corners = (
                _occupied_or_wall(board, pivot_x - 1, pivot_y - 1),
                _occupied_or_wall(board, pivot_x + 1, pivot_y - 1),
                _occupied_or_wall(board, pivot_x - 1, pivot_y + 1),
                _occupied_or_wall(board, pivot_x + 1, pivot_y + 1),
            )
            if sum(corners) < 3:
                continue
            if any(
                all(_empty(board, pivot_x + dx, pivot_y + dy) for dx, dy in cells)
                for cells in orientations
            ):
                pivots.append((pivot_x, pivot_y))
    return tuple(pivots)


def count_t_spin_slots(board: Board, *, start_y: int = 0) -> int:
    """Count geometric T-spin slots using the Guideline three-corner rule."""
    return len(_t_spin_slot_pivots(board, start_y=start_y))


def extract_board_features(board: Board) -> BoardFeatures:
    """Extract deterministic stack features from a board after line clears."""
    if not board:
        return BoardFeatures(0, 0, 0, 0, 0, 0, 0, 0)
    height = len(board)
    width = len(board[0])
    if any(len(row) != width for row in board):
        raise ValueError("Board rows must have equal width")

    heights = column_heights(board)
    holes = 0
    hole_depth = 0
    deep_holes = 0
    occupied_cells = 0

    for x in range(width):
        seen_block = False
        blocks_above = 0
        for y in range(height):
            occupied = board[y][x] is not None
            if occupied:
                occupied_cells += 1
                seen_block = True
                blocks_above += 1
            elif seen_block:
                holes += 1
                hole_depth += blocks_above
                if blocks_above >= 3:
                    deep_holes += 1

    bumpiness = sum(abs(left - right) for left, right in zip(heights, heights[1:]))

    wells = 0
    for x in range(width):
        depth = 0
        for y in range(height):
            if board[y][x] is not None:
                depth = 0
                continue
            left_filled = x == 0 or board[y][x - 1] is not None
            right_filled = x == width - 1 or board[y][x + 1] is not None
            if left_filled and right_filled:
                depth += 1
                wells += depth
            else:
                depth = 0

    max_height = max(heights, default=0)
    slot_start_y = height if max_height == 0 else max(0, height - max_height - 1)
    pivots = _t_spin_slot_pivots(board, start_y=slot_start_y)
    row_fill = [sum(cell is not None for cell in row) for row in board]
    t_spin_clear_potential = 0
    for _, pivot_y in pivots:
        for row_y in (pivot_y - 1, pivot_y, pivot_y + 1):
            if 0 <= row_y < height and row_fill[row_y] >= max(1, width - 3):
                t_spin_clear_potential += 1
    danger_excess = sum(max(0, value - 10) ** 2 for value in heights)
    return BoardFeatures(
        aggregate_height=sum(heights),
        max_height=max_height,
        holes=holes,
        hole_depth=hole_depth,
        bumpiness=bumpiness,
        wells=wells,
        t_spin_slots=len(pivots),
        occupied_cells=occupied_cells,
        danger_excess=danger_excess,
        deep_holes=deep_holes,
        t_spin_clear_potential=t_spin_clear_potential,
    )
