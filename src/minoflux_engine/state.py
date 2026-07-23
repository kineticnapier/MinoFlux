from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Placement:
    piece: str
    x: int
    y: int
    rotation: int
    cells: tuple[tuple[int, int], ...]
    path: tuple[str, ...] = ()
    last_move_was_rotation: bool = False
    rotation_kick_index: int | None = None
    rotation_from: int | None = None
    rotation_to: int | None = None


@dataclass(frozen=True, slots=True)
class LockResult:
    lines: int
    attack: int
    spin: str | None
    perfect_clear: bool
    combo: int
    back_to_back: bool
    game_over: bool
    b2b_chain: int = 0
    surge_charge: int = 0
    surge_released: int = 0
    attack_packets: tuple[int, ...] = ()


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
    b2b_chain: int
    surge_charge: int
    pieces_placed: int
    game_over: bool
    paused: bool
    grounded: bool
    lock_elapsed_ms: float
    lock_delay_ms: float
    lock_resets: int
    lock_reset_limit: int
    last_lock: LockResult | None
