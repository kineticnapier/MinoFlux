from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Placement:
    piece: str
    x: int
    y: int
    rotation: int
    cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class LockResult:
    lines: int
    attack: int
    spin: str | None
    perfect_clear: bool
    combo: int
    back_to_back: bool
    game_over: bool


@dataclass(frozen=True, slots=True)
class GameSnapshot:
    board: tuple[tuple[str | None, ...], ...]
    current: str
    x: int
    y: int
    rotation: int
    ghost_y: int
    hold: str | None
    hold_used: bool
    queue: tuple[str, ...]
    score: int
    lines: int
    attack: int
    combo: int
    back_to_back: bool
    pieces_placed: int
    game_over: bool
    paused: bool
    grounded: bool
    lock_elapsed_ms: float
    lock_delay_ms: float
    lock_resets: int
    lock_reset_limit: int
    last_lock: LockResult | None
