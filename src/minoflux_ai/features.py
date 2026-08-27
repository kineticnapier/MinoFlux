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
    occupied_cells: int

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


def extract_board_features(board: Board) -> BoardFeatures:
    """Extract deterministic stack features from a board after line clears."""
    if not board:
        return BoardFeatures(0, 0, 0, 0, 0, 0, 0)
    height = len(board)
    width = len(board[0])
    if any(len(row) != width for row in board):
        raise ValueError("Board rows must have equal width")

    heights = column_heights(board)
    holes = 0
    hole_depth = 0
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

    return BoardFeatures(
        aggregate_height=sum(heights),
        max_height=max(heights, default=0),
        holes=holes,
        hole_depth=hole_depth,
        bumpiness=bumpiness,
        wells=wells,
        occupied_cells=occupied_cells,
    )
