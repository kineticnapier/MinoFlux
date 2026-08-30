from __future__ import annotations

from typing import Sequence

from minoflux_engine import Placement
from minoflux_engine.pieces import BOARD_HEIGHT, BOARD_WIDTH, SHAPES

FULL_ROW_MASK = (1 << BOARD_WIDTH) - 1
ROW_OCCUPANCY_BYTES = tuple(
    bytes(1 if mask & (1 << x) else 0 for x in range(BOARD_WIDTH))
    for mask in range(FULL_ROW_MASK + 1)
)

# Compact per-rotation row masks and bounds for collision checks.
SHAPE_ROW_MASKS: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {}
SHAPE_BOUNDS: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
for _piece, _rotations in SHAPES.items():
    _piece_rows: list[tuple[tuple[int, int], ...]] = []
    _piece_bounds: list[tuple[int, int, int, int]] = []
    for _shape in _rotations:
        _by_y: dict[int, int] = {}
        _xs: list[int] = []
        _ys: list[int] = []
        for _dx, _dy in _shape:
            _by_y[_dy] = _by_y.get(_dy, 0) | (1 << _dx)
            _xs.append(_dx)
            _ys.append(_dy)
        _piece_rows.append(tuple(sorted(_by_y.items())))
        _piece_bounds.append((min(_xs), max(_xs), min(_ys), max(_ys)))
    SHAPE_ROW_MASKS[_piece] = tuple(_piece_rows)
    SHAPE_BOUNDS[_piece] = tuple(_piece_bounds)

_FRONT_CORNERS: dict[int, tuple[int, int]] = {
    0: (0, 1),
    1: (1, 3),
    2: (2, 3),
    3: (0, 2),
}


def board_row_masks(board: Sequence[Sequence[object | None]]) -> tuple[int, ...]:
    """Encode board occupancy as one integer bit mask per row."""

    rows: list[int] = []
    for row in board:
        mask = 0
        for x, cell in enumerate(row):
            if cell is not None:
                mask |= 1 << x
        rows.append(mask)
    return tuple(rows)


def collides_row_masks(
    rows: Sequence[int],
    piece: str,
    x: int,
    y: int,
    rotation: int,
    *,
    width: int = BOARD_WIDTH,
) -> bool:
    """Collision test against compact occupancy rows without cell tuples.

    SRS anchors can be negative even when every occupied mino is on-board (for
    example a vertical I at x=-2). Shift row masks right for those anchors rather
    than attempting an invalid negative left shift.
    """

    rotation %= 4
    min_x, max_x, _min_y, _max_y = SHAPE_BOUNDS[piece][rotation]
    if x + min_x < 0 or x + max_x >= width:
        return True
    height = len(rows)
    for dy, relative_mask in SHAPE_ROW_MASKS[piece][rotation]:
        row_y = y + dy
        if row_y >= height:
            return True
        if row_y < 0:
            continue
        shifted_mask = relative_mask << x if x >= 0 else relative_mask >> (-x)
        if rows[row_y] & shifted_mask:
            return True
    return False


def placement_cells(piece: str, x: int, y: int, rotation: int) -> tuple[tuple[int, int], ...]:
    return tuple((x + dx, y + dy) for dx, dy in SHAPES[piece][rotation % 4])


def place_and_clear_row_masks(
    rows: Sequence[int],
    placement: Placement,
    *,
    width: int = BOARD_WIDTH,
) -> tuple[tuple[int, ...], int, bool]:
    """Apply one already-legal placement to occupancy rows and clear full lines."""

    result = list(rows)
    height = len(result)
    topped_out = False
    for cell_x, cell_y in placement.cells:
        if cell_y < 0 or cell_y >= height:
            topped_out = True
            continue
        result[cell_y] |= 1 << cell_x

    full = FULL_ROW_MASK if width == BOARD_WIDTH else (1 << width) - 1
    kept = [mask for mask in result if mask != full]
    lines = height - len(kept)
    if lines:
        result = [0] * lines + kept
    return tuple(result), lines, topped_out


def hidden_rows_occupied(rows: Sequence[int], hidden_rows: int) -> bool:
    return any(rows[: max(0, int(hidden_rows))])


def _occupied_or_wall(rows: Sequence[int], x: int, y: int, width: int) -> bool:
    if x < 0 or x >= width or y < 0 or y >= len(rows):
        return True
    return bool(rows[y] & (1 << x))


def classify_t_spin_row_masks(
    rows: Sequence[int],
    *,
    piece: str,
    x: int,
    y: int,
    rotation: int,
    last_move_was_rotation: bool,
    rotation_kick_index: int | None,
    width: int = BOARD_WIDTH,
) -> str | None:
    """Bitboard equivalent of the engine Guideline three-corner classifier."""

    if piece != "T" or not last_move_was_rotation:
        return None
    pivot_x, pivot_y = x + 1, y + 1
    corners = (
        _occupied_or_wall(rows, pivot_x - 1, pivot_y - 1, width),
        _occupied_or_wall(rows, pivot_x + 1, pivot_y - 1, width),
        _occupied_or_wall(rows, pivot_x - 1, pivot_y + 1, width),
        _occupied_or_wall(rows, pivot_x + 1, pivot_y + 1, width),
    )
    if sum(corners) < 3:
        return None
    front = _FRONT_CORNERS[rotation % 4]
    if corners[front[0]] and corners[front[1]]:
        return "full"
    if rotation_kick_index == 4:
        return "full"
    return "mini"


def cells_key(cells: Sequence[tuple[int, int]], *, width: int = BOARD_WIDTH) -> int:
    """Pack a non-negative placement geometry into a hashable integer."""

    key = 0
    for x, y in cells:
        if y < 0:
            return -1
        key |= 1 << (y * width + x)
    return key
