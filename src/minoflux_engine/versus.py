from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import random
from typing import Literal

from .game import Game
from .state import LockResult

SideName = Literal["player", "ai"]


@dataclass(slots=True)
class GarbagePacket:
    lines: int
    hole: int

    def normalized(self, width: int) -> "GarbagePacket":
        return GarbagePacket(max(0, int(self.lines)), max(0, min(width - 1, int(self.hole))))


@dataclass(slots=True)
class GarbageQueue:
    width: int = 10
    packets: deque[GarbagePacket] = field(default_factory=deque)

    @property
    def pending_lines(self) -> int:
        return sum(packet.lines for packet in self.packets)

    def clear(self) -> None:
        self.packets.clear()

    def enqueue(self, lines: int, hole: int) -> None:
        packet = GarbagePacket(lines, hole).normalized(self.width)
        if packet.lines > 0:
            self.packets.append(packet)

    def cancel(self, lines: int) -> tuple[int, int]:
        """Cancel FIFO garbage and return ``(remaining_attack, canceled)``."""

        remaining = max(0, int(lines))
        canceled = 0
        while remaining > 0 and self.packets:
            packet = self.packets[0]
            amount = min(remaining, packet.lines)
            remaining -= amount
            canceled += amount
            packet.lines -= amount
            if packet.lines <= 0:
                self.packets.popleft()
        return remaining, canceled

    def take(self, limit: int) -> tuple[GarbagePacket, ...]:
        remaining = max(0, int(limit))
        result: list[GarbagePacket] = []
        while remaining > 0 and self.packets:
            packet = self.packets[0]
            amount = min(remaining, packet.lines)
            result.append(GarbagePacket(amount, packet.hole))
            remaining -= amount
            packet.lines -= amount
            if packet.lines <= 0:
                self.packets.popleft()
        return tuple(result)


@dataclass(slots=True)
class VersusSide:
    game: Game
    pending: GarbageQueue
    sent: int = 0
    received: int = 0
    canceled: int = 0
    garbage_applied: int = 0


@dataclass(slots=True)
class VersusResolution:
    side: SideName
    sent_packets: tuple[int, ...]
    sent_lines: int
    canceled_lines: int
    garbage_applied: int
    winner: SideName | Literal["draw"] | None


class VersusMatch:
    """Local one-versus-one garbage exchange using attack packet semantics."""

    def __init__(self, seed: int | None = None, *, garbage_cap: int = 8) -> None:
        self.seed = 1 if seed is None else int(seed)
        self.garbage_cap = max(1, int(garbage_cap))
        self._garbage_rng = random.Random(self.seed ^ 0x6D696E6F)
        self.player = VersusSide(Game(self.seed), GarbageQueue(Game.width))
        self.ai = VersusSide(Game(self.seed + 1), GarbageQueue(Game.width))
        self.winner: SideName | Literal["draw"] | None = None

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        self._garbage_rng = random.Random(self.seed ^ 0x6D696E6F)
        self.player = VersusSide(Game(self.seed), GarbageQueue(Game.width))
        self.ai = VersusSide(Game(self.seed + 1), GarbageQueue(Game.width))
        self.winner = None

    def side(self, name: SideName) -> VersusSide:
        return self.player if name == "player" else self.ai

    def opponent(self, name: SideName) -> VersusSide:
        return self.ai if name == "player" else self.player

    def _next_hole(self, previous: int | None = None) -> int:
        hole = self._garbage_rng.randrange(Game.width)
        if previous is not None and Game.width > 1 and hole == previous:
            hole = (hole + 1 + self._garbage_rng.randrange(Game.width - 1)) % Game.width
        return hole

    def _apply_pending(self, side: VersusSide) -> int:
        applied = 0
        for packet in side.pending.take(self.garbage_cap):
            side.game.add_garbage(packet.lines, packet.hole)
            applied += packet.lines
            if side.game.game_over:
                break
        side.garbage_applied += applied
        return applied

    def _update_winner(self) -> None:
        player_dead = self.player.game.game_over
        ai_dead = self.ai.game.game_over
        if player_dead and ai_dead:
            self.winner = "draw"
        elif player_dead:
            self.winner = "ai"
        elif ai_dead:
            self.winner = "player"

    def resolve_lock(self, name: SideName, result: LockResult) -> VersusResolution:
        if self.winner is not None:
            return VersusResolution(name, (), 0, 0, 0, self.winner)

        attacker = self.side(name)
        defender = self.opponent(name)
        sent_packets: list[int] = []
        canceled_total = 0
        previous_hole: int | None = None

        packets = result.attack_packets or ((result.attack,) if result.attack > 0 else ())
        for attack_packet in packets:
            remaining, canceled = attacker.pending.cancel(attack_packet)
            canceled_total += canceled
            attacker.canceled += canceled
            if remaining <= 0:
                continue
            hole = self._next_hole(previous_hole)
            previous_hole = hole
            defender.pending.enqueue(remaining, hole)
            attacker.sent += remaining
            defender.received += remaining
            sent_packets.append(remaining)

        applied = 0
        # TETR.IO-style garbage blocking: a line clear blocks queued garbage for
        # this placement. A no-clear placement accepts up to the per-turn cap.
        if result.lines == 0 and not attacker.game.game_over:
            applied = self._apply_pending(attacker)

        self._update_winner()
        return VersusResolution(
            side=name,
            sent_packets=tuple(sent_packets),
            sent_lines=sum(sent_packets),
            canceled_lines=canceled_total,
            garbage_applied=applied,
            winner=self.winner,
        )
