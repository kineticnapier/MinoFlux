from __future__ import annotations

from typing import Sequence

T_SPIN = "T_SPIN"
T_SPIN_MINI = "T_SPIN_MINI"
T_SPIN_SINGLE = "T_SPIN_SINGLE"
T_SPIN_MINI_SINGLE = "T_SPIN_MINI_SINGLE"
T_SPIN_DOUBLE = "T_SPIN_DOUBLE"
T_SPIN_TRIPLE = "T_SPIN_TRIPLE"

T_SPIN_EVENTS = (
    T_SPIN,
    T_SPIN_MINI,
    T_SPIN_SINGLE,
    T_SPIN_MINI_SINGLE,
    T_SPIN_DOUBLE,
    T_SPIN_TRIPLE,
)

_FRONT_CORNERS: dict[int, tuple[int, int]] = {
    0: (0, 1),
    1: (1, 3),
    2: (2, 3),
    3: (0, 2),
}


def _occupied(board: Sequence[Sequence[object | None]], x: int, y: int) -> bool:
    if not board:
        return True
    height = len(board)
    width = len(board[0])
    if x < 0 or x >= width or y < 0 or y >= height:
        return True
    return board[y][x] is not None


def classify_t_spin(
    board: Sequence[Sequence[object | None]],
    *,
    piece: str,
    x: int,
    y: int,
    rotation: int,
    last_move_was_rotation: bool,
    rotation_kick_index: int | None,
) -> str | None:
    """Return ``full`` or ``mini`` from the Guideline three-corner rule."""

    if piece != "T" or not last_move_was_rotation:
        return None
    pivot_x, pivot_y = x + 1, y + 1
    corners = (
        _occupied(board, pivot_x - 1, pivot_y - 1),
        _occupied(board, pivot_x + 1, pivot_y - 1),
        _occupied(board, pivot_x - 1, pivot_y + 1),
        _occupied(board, pivot_x + 1, pivot_y + 1),
    )
    if sum(corners) < 3:
        return None
    front = _FRONT_CORNERS[rotation % 4]
    if corners[front[0]] and corners[front[1]]:
        return "full"
    if rotation_kick_index == 4:
        return "full"
    return "mini"


def t_spin_event(kind: str | None, lines: int) -> str | None:
    if kind is None:
        return None
    count = max(0, int(lines))
    if kind == "mini":
        if count == 0:
            return T_SPIN_MINI
        if count == 1:
            return T_SPIN_MINI_SINGLE
        return T_SPIN_DOUBLE if count == 2 else T_SPIN_TRIPLE
    if count == 0:
        return T_SPIN
    if count == 1:
        return T_SPIN_SINGLE
    if count == 2:
        return T_SPIN_DOUBLE
    return T_SPIN_TRIPLE


def is_t_spin(event: str | None) -> bool:
    return event in T_SPIN_EVENTS


def is_difficult_clear(lines: int, event: str | None) -> bool:
    return int(lines) == 4 or (is_t_spin(event) and int(lines) > 0)


def base_attack(lines: int, event: str | None) -> int:
    count = max(0, int(lines))
    if event == T_SPIN_MINI_SINGLE:
        return 0
    if event == T_SPIN_SINGLE:
        return 2
    if event == T_SPIN_DOUBLE:
        return 4
    if event == T_SPIN_TRIPLE:
        return 6
    return {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}.get(count, 0)


def base_score(lines: int, event: str | None) -> int:
    count = max(0, int(lines))
    if event == T_SPIN_MINI:
        return 100
    if event == T_SPIN_MINI_SINGLE:
        return 200
    if event == T_SPIN:
        return 400
    if event == T_SPIN_SINGLE:
        return 800
    if event == T_SPIN_DOUBLE:
        return 1200
    if event == T_SPIN_TRIPLE:
        return 1600
    return {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}.get(count, 1200)
