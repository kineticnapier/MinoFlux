from __future__ import annotations

from collections import deque
import random

BOARD_WIDTH = 10
VISIBLE_HEIGHT = 20
HIDDEN_ROWS = 4
BOARD_HEIGHT = VISIBLE_HEIGHT + HIDDEN_ROWS
PIECE_NAMES = ("I", "O", "T", "S", "Z", "J", "L")

SHAPES: dict[str, tuple[tuple[tuple[int, int], ...], ...]] = {
    "I": (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((2, 0), (2, 1), (2, 2), (2, 3)),
        ((0, 2), (1, 2), (2, 2), (3, 2)),
        ((1, 0), (1, 1), (1, 2), (1, 3)),
    ),
    "O": (((1, 0), (2, 0), (1, 1), (2, 1)),) * 4,
    "T": (
        ((1, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (1, 2)),
        ((1, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "S": (
        ((1, 0), (2, 0), (0, 1), (1, 1)),
        ((1, 0), (1, 1), (2, 1), (2, 2)),
        ((1, 1), (2, 1), (0, 2), (1, 2)),
        ((0, 0), (0, 1), (1, 1), (1, 2)),
    ),
    "Z": (
        ((0, 0), (1, 0), (1, 1), (2, 1)),
        ((2, 0), (1, 1), (2, 1), (1, 2)),
        ((0, 1), (1, 1), (1, 2), (2, 2)),
        ((1, 0), (0, 1), (1, 1), (0, 2)),
    ),
    "J": (
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (2, 0), (1, 1), (1, 2)),
        ((0, 1), (1, 1), (2, 1), (2, 2)),
        ((1, 0), (1, 1), (0, 2), (1, 2)),
    ),
    "L": (
        ((2, 0), (0, 1), (1, 1), (2, 1)),
        ((1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 1), (1, 1), (2, 1), (0, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ),
}

KICK_TESTS: tuple[tuple[int, int], ...] = (
    (0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0),
    (0, -1), (-1, -1), (1, -1), (0, -2),
)
LINE_SCORES = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}
LINE_ATTACK = {0: 0, 1: 0, 2: 1, 3: 2, 4: 4}


class SevenBag:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._queue: deque[str] = deque()

    def pop(self) -> str:
        if not self._queue:
            bag = list(PIECE_NAMES)
            self._rng.shuffle(bag)
            self._queue.extend(bag)
        return self._queue.popleft()
