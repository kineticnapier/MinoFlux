from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class RepeatBatch:
    count: int = 0
    instant: bool = False


class RepeatTimer:
    """Frame-rate-independent held-key repeat timer."""

    def __init__(self) -> None:
        self.held = False
        self.next_at = math.inf
        self._instant_emitted = False

    def press(self, now: float, initial_delay_ms: int) -> None:
        self.held = True
        self.next_at = now + max(0, initial_delay_ms) / 1000.0
        self._instant_emitted = False

    def release(self) -> None:
        self.held = False
        self.next_at = math.inf
        self._instant_emitted = False

    def poll(self, now: float, interval_ms: int, max_count: int = 64) -> RepeatBatch:
        epsilon = 1e-9
        if not self.held or now + epsilon < self.next_at:
            return RepeatBatch()
        interval_ms = max(0, int(interval_ms))
        if interval_ms == 0:
            if self._instant_emitted:
                return RepeatBatch()
            self._instant_emitted = True
            self.next_at = math.inf
            return RepeatBatch(instant=True)

        interval = interval_ms / 1000.0
        count = min(max_count, 1 + int(((now - self.next_at) + epsilon) // interval))
        self.next_at += count * interval
        return RepeatBatch(count=count)


class HandlingController:
    """Tracks DAS/ARR horizontal input and held soft drop."""

    def __init__(self) -> None:
        self._held_horizontal: set[int] = set()
        self._active_horizontal = 0
        self._horizontal = RepeatTimer()
        self._soft_drop = RepeatTimer()

    @property
    def active_horizontal(self) -> int:
        return self._active_horizontal

    def press_horizontal(self, direction: int, now: float, das_ms: int) -> None:
        direction = -1 if direction < 0 else 1
        self._held_horizontal.add(direction)
        self._active_horizontal = direction
        self._horizontal.press(now, das_ms)

    def release_horizontal(self, direction: int, now: float, das_ms: int) -> None:
        direction = -1 if direction < 0 else 1
        self._held_horizontal.discard(direction)
        if self._active_horizontal != direction:
            return
        opposite = -direction
        if opposite in self._held_horizontal:
            self._active_horizontal = opposite
            self._horizontal.press(now, das_ms)
        else:
            self._active_horizontal = 0
            self._horizontal.release()

    def poll_horizontal(self, now: float, arr_ms: int) -> tuple[int, RepeatBatch]:
        if self._active_horizontal == 0:
            return 0, RepeatBatch()
        return self._active_horizontal, self._horizontal.poll(now, arr_ms)

    def press_soft_drop(self, now: float, soft_drop_ms: int) -> None:
        self._soft_drop.press(now, soft_drop_ms)

    def release_soft_drop(self) -> None:
        self._soft_drop.release()

    def poll_soft_drop(self, now: float, soft_drop_ms: int) -> RepeatBatch:
        return self._soft_drop.poll(now, soft_drop_ms)

    def clear(self) -> None:
        self._held_horizontal.clear()
        self._active_horizontal = 0
        self._horizontal.release()
        self._soft_drop.release()
