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

# Reachability uses a small anchor margin around the board. Pre-shift every
# standard-width shape row for that complete range so hot collision checks do
# not repeatedly perform bounds arithmetic and bit shifts.
_SHIFT_X_MIN = -4
_SHIFT_X_MAX = BOARD_WIDTH + 3
SHIFTED_SHAPE_ROW_MASKS: dict[
    str,
    tuple[tuple[tuple[tuple[int, int], ...] | None, ...], ...],
] = {}
for _piece, _rotations in SHAPE_ROW_MASKS.items():
    _piece_shifted: list[tuple[tuple[tuple[int, int], ...] | None, ...]] = []
    for _rotation, _rows in enumerate(_rotations):
        _min_x, _max_x, _min_y, _max_y = SHAPE_BOUNDS[_piece][_rotation]
        _anchors: list[tuple[tuple[int, int], ...] | None] = []
        for _x in range(_SHIFT_X_MIN, _SHIFT_X_MAX + 1):
            if _x + _min_x < 0 or _x + _max_x >= BOARD_WIDTH:
                _anchors.append(None)
                continue
            _anchors.append(
                tuple(
                    (
                        _dy,
                        _relative_mask << _x
                        if _x >= 0
                        else _relative_mask >> (-_x),
                    )
                    for _dy, _relative_mask in _rows
                )
            )
        _piece_shifted.append(tuple(_anchors))
    SHIFTED_SHAPE_ROW_MASKS[_piece] = tuple(_piece_shifted)

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


def _shifted_rows_collide(
    rows: Sequence[int],
    shifted_rows: tuple[tuple[int, int], ...],
    y: int,
    height: int,
) -> bool:
    """Unrolled 1-4 row collision check used by the standard 10-column hot path."""

    dy0, mask0 = shifted_rows[0]
    row0 = y + dy0
    if row0 >= height or (row0 >= 0 and rows[row0] & mask0):
        return True
    count = len(shifted_rows)
    if count == 1:
        return False

    dy1, mask1 = shifted_rows[1]
    row1 = y + dy1
    if row1 >= height or (row1 >= 0 and rows[row1] & mask1):
        return True
    if count == 2:
        return False

    dy2, mask2 = shifted_rows[2]
    row2 = y + dy2
    if row2 >= height or (row2 >= 0 and rows[row2] & mask2):
        return True
    if count == 3:
        return False

    dy3, mask3 = shifted_rows[3]
    row3 = y + dy3
    return row3 >= height or (row3 >= 0 and bool(rows[row3] & mask3))


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
    example a vertical I at x=-2). Standard-width checks use pre-shifted row
    masks; arbitrary widths keep the generic reference path.
    """

    rotation %= 4
    height = len(rows)
    if width == BOARD_WIDTH and _SHIFT_X_MIN <= x <= _SHIFT_X_MAX:
        shifted_rows = SHIFTED_SHAPE_ROW_MASKS[piece][rotation][x - _SHIFT_X_MIN]
        if shifted_rows is None:
            return True
        return _shifted_rows_collide(rows, shifted_rows, y, height)

    min_x, max_x, _min_y, _max_y = SHAPE_BOUNDS[piece][rotation]
    if x + min_x < 0 or x + max_x >= width:
        return True
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