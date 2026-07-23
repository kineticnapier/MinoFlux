from __future__ import annotations

from collections import deque

from .b2b import resolve_b2b_charging, split_surge
from .pieces import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    HIDDEN_ROWS,
    SHAPES,
    VISIBLE_HEIGHT,
    SevenBag,
    kick_tests,
)
from .spin import base_attack, base_score, classify_t_spin, is_difficult_clear, t_spin_event
from .state import GameSnapshot, LockResult, Placement

DEFAULT_LOCK_DELAY_MS = 500.0
DEFAULT_LOCK_RESET_LIMIT = 15
GARBAGE_CELL = "G"


class Game:
    """Dependency-free Tetris-like engine shared by play and learning."""

    width = BOARD_WIDTH
    visible_height = VISIBLE_HEIGHT
    hidden_rows = HIDDEN_ROWS
    height = BOARD_HEIGHT

    def __init__(
        self,
        seed: int | None = None,
        *,
        lock_delay_ms: float = DEFAULT_LOCK_DELAY_MS,
        lock_reset_limit: int = DEFAULT_LOCK_RESET_LIMIT,
    ) -> None:
        self.seed = seed
        self.lock_delay_ms = max(0.0, float(lock_delay_ms))
        self.lock_reset_limit = max(0, int(lock_reset_limit))
        self._bag = SevenBag(seed)
        self.board: list[list[str | None]] = []
        self.queue: deque[str] = deque()
        self.current = "T"
        self.x = 3
        self.y = 1
        self.rotation = 0
        self.hold_piece: str | None = None
        self.hold_used = False
        self.last_action: str | None = None
        self.last_move_was_rotation = False
        self.last_rotation_kick_index: int | None = None
        self.last_rotation_from: int | None = None
        self.last_rotation_to: int | None = None
        self.score = 0
        self.lines = 0
        self.attack = 0
        self.combo = -1
        self.back_to_back = False
        self.b2b_chain = 0
        self.surge_charge = 0
        self.pieces_placed = 0
        self.game_over = False
        self.paused = False
        self.lock_elapsed_ms = 0.0
        self.lock_resets = 0
        self.last_lock: LockResult | None = None
        self.reset(seed)

    def _clear_rotation_metadata(self) -> None:
        self.last_move_was_rotation = False
        self.last_rotation_kick_index = None
        self.last_rotation_from = None
        self.last_rotation_to = None

    def reset(self, seed: int | None = None) -> GameSnapshot:
        if seed is not None or self.seed is None:
            self.seed = seed
        self._bag = SevenBag(self.seed)
        self.board = [[None] * self.width for _ in range(self.height)]
        self.queue = deque()
        self._fill_queue(7)
        self.current = self.queue.popleft()
        self._fill_queue(7)
        self.x, self.y, self.rotation = 3, 1, 0
        self.hold_piece = None
        self.hold_used = False
        self.last_action = None
        self._clear_rotation_metadata()
        self.score = self.lines = self.attack = self.pieces_placed = 0
        self.combo = -1
        self.back_to_back = False
        self.b2b_chain = 0
        self.surge_charge = 0
        self.game_over = self.paused = False
        self.last_lock = None
        self._reset_lock_state()
        self.game_over = self._collides(self.current, self.x, self.y, self.rotation)
        return self.snapshot()

    def _reset_lock_state(self) -> None:
        self.lock_elapsed_ms = 0.0
        self.lock_resets = 0

    def _fill_queue(self, minimum: int) -> None:
        while len(self.queue) < minimum:
            self.queue.append(self._bag.pop())

    def cells(
        self,
        piece: str | None = None,
        x: int | None = None,
        y: int | None = None,
        rotation: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        name = piece or self.current
        px = self.x if x is None else x
        py = self.y if y is None else y
        rot = self.rotation if rotation is None else rotation % 4
        return tuple((px + dx, py + dy) for dx, dy in SHAPES[name][rot])

    def _collides(self, piece: str, x: int, y: int, rotation: int) -> bool:
        for cell_x, cell_y in self.cells(piece, x, y, rotation):
            if cell_x < 0 or cell_x >= self.width or cell_y >= self.height:
                return True
            if cell_y >= 0 and self.board[cell_y][cell_x] is not None:
                return True
        return False

    def is_grounded(self) -> bool:
        return self._collides(self.current, self.x, self.y + 1, self.rotation)

    def _spawn(self, piece: str | None = None) -> bool:
        if piece is None:
            self._fill_queue(7)
            piece = self.queue.popleft()
            self._fill_queue(7)
        self.current = piece
        self.x, self.y, self.rotation = 3, 1, 0
        self.hold_used = False
        self.last_action = None
        self._clear_rotation_metadata()
        self._reset_lock_state()
        self.game_over = self._collides(self.current, self.x, self.y, self.rotation)
        return not self.game_over

    def _reset_lock_after_manipulation(self, was_grounded: bool) -> None:
        if not (was_grounded or self.is_grounded()):
            return
        if self.lock_resets >= self.lock_reset_limit:
            return
        self.lock_elapsed_ms = 0.0
        self.lock_resets += 1

    def _reset_lock_after_descent(self) -> None:
        if self.lock_resets < self.lock_reset_limit:
            self.lock_elapsed_ms = 0.0

    def move(self, dx: int, dy: int = 0) -> bool:
        if self.game_over or self.paused:
            return False
        dx, dy = int(dx), int(dy)
        was_grounded = self.is_grounded()
        target_x, target_y = self.x + dx, self.y + dy
        if self._collides(self.current, target_x, target_y, self.rotation):
            return False
        self.x, self.y = target_x, target_y
        self.last_action = "move" if dx else "soft_drop"
        self._clear_rotation_metadata()
        if dy > 0:
            self._reset_lock_after_descent()
        elif dx:
            self._reset_lock_after_manipulation(was_grounded)
        return True

    def move_left(self) -> bool:
        return self.move(-1)

    def move_right(self) -> bool:
        return self.move(1)

    def soft_drop(self) -> bool:
        moved = self.move(0, 1)
        if moved:
            self.score += 1
        return moved

    def rotate(self, direction: int = 1) -> bool:
        if direction not in (-2, -1, 1, 2):
            raise ValueError("Rotation direction must be -2, -1, 1, or 2")
        if self.game_over or self.paused or self.current == "O":
            return False
        source_rotation = self.rotation
        target_rotation = (source_rotation + direction) % 4
        was_grounded = self.is_grounded()
        for kick_index, (kick_x, kick_y) in enumerate(
            kick_tests(self.current, source_rotation, target_rotation)
        ):
            target_x, target_y = self.x + kick_x, self.y + kick_y
            if not self._collides(self.current, target_x, target_y, target_rotation):
                self.x, self.y, self.rotation = target_x, target_y, target_rotation
                self.last_action = "rotate"
                self.last_move_was_rotation = True
                self.last_rotation_kick_index = kick_index
                self.last_rotation_from = source_rotation
                self.last_rotation_to = target_rotation
                self._reset_lock_after_manipulation(was_grounded)
                return True
        return False

    def rotate_cw(self) -> bool:
        return self.rotate(1)

    def rotate_ccw(self) -> bool:
        return self.rotate(-1)

    def rotate_180(self) -> bool:
        return self.rotate(2)

    def hold(self) -> bool:
        if self.game_over or self.paused or self.hold_used:
            return False
        outgoing, incoming = self.current, self.hold_piece
        self.hold_piece = outgoing
        if incoming is None:
            self._fill_queue(7)
            incoming = self.queue.popleft()
            self._fill_queue(7)
        self.current = incoming
        self.x, self.y, self.rotation = 3, 1, 0
        self.hold_used = True
        self.last_action = "hold"
        self._clear_rotation_metadata()
        self._reset_lock_state()
        self.game_over = self._collides(self.current, self.x, self.y, self.rotation)
        return not self.game_over

    def ghost_y(self) -> int:
        target = self.y
        while not self._collides(self.current, self.x, target + 1, self.rotation):
            target += 1
        return target

    def hard_drop(self) -> LockResult:
        if self.game_over or self.paused:
            return self.last_lock or LockResult(0, 0, None, False, self.combo, self.back_to_back, self.game_over)
        target = self.ghost_y()
        self.score += max(0, target - self.y) * 2
        self.y = target
        self.last_action = "hard_drop"
        return self._lock_piece()

    def gravity_step(self) -> LockResult | None:
        if self.game_over or self.paused:
            return None
        if self.move(0, 1):
            self.last_action = "gravity"
        return None

    def advance_time(self, delta_ms: float) -> LockResult | None:
        if self.game_over or self.paused:
            return None
        delta = max(0.0, float(delta_ms))
        if not self.is_grounded():
            return None
        self.lock_elapsed_ms += delta
        if self.lock_elapsed_ms + 1e-9 >= self.lock_delay_ms:
            return self._lock_piece()
        return None

    def _detect_spin_kind(self) -> str | None:
        return classify_t_spin(
            self.board,
            piece=self.current,
            x=self.x,
            y=self.y,
            rotation=self.rotation,
            last_move_was_rotation=self.last_move_was_rotation,
            rotation_kick_index=self.last_rotation_kick_index,
        )

    def _lock_piece(self) -> LockResult:
        spin_kind = self._detect_spin_kind()
        topped_out = False
        for cell_x, cell_y in self.cells():
            if cell_y < 0:
                topped_out = True
            elif 0 <= cell_y < self.height:
                self.board[cell_y][cell_x] = self.current

        full_rows = [index for index, row in enumerate(self.board) if all(cell is not None for cell in row)]
        lines = len(full_rows)
        for index in reversed(full_rows):
            del self.board[index]
        for _ in full_rows:
            self.board.insert(0, [None] * self.width)

        spin = t_spin_event(spin_kind, lines)
        perfect_clear = all(cell is None for row in self.board for cell in row)
        difficult = is_difficult_clear(lines, spin)
        b2b = resolve_b2b_charging(
            active=self.back_to_back,
            chain=self.b2b_chain,
            difficult=difficult,
            lines=lines,
            perfect_clear=perfect_clear and lines > 0,
        )

        sent = base_attack(lines, spin) + b2b.attack_bonus
        if lines:
            self.combo += 1
            if self.combo > 0:
                sent += min(4, self.combo // 2 + 1)
        else:
            self.combo = -1
        if perfect_clear and lines:
            sent += 10

        attack_packets = ((sent,) if sent > 0 else ()) + split_surge(b2b.released)
        total_sent = sum(attack_packets)

        level = self.lines // 10 + 1
        gained = base_score(lines, spin)
        if b2b.attack_bonus:
            gained = int(gained * 1.5)
        self.score += gained * level
        self.lines += lines
        self.attack += total_sent
        self.back_to_back = b2b.active
        self.b2b_chain = b2b.chain
        self.surge_charge = b2b.charge

        self.pieces_placed += 1
        hidden_occupied = any(cell is not None for row in self.board[: self.hidden_rows] for cell in row)
        self.game_over = topped_out or hidden_occupied
        if not self.game_over:
            self._spawn()
        else:
            self.lock_elapsed_ms = 0.0

        self.last_lock = LockResult(
            lines=lines,
            attack=total_sent,
            spin=spin,
            perfect_clear=perfect_clear,
            combo=self.combo,
            back_to_back=self.back_to_back,
            game_over=self.game_over,
            b2b_chain=self.b2b_chain,
            surge_charge=self.surge_charge,
            surge_released=b2b.released,
            attack_packets=attack_packets,
        )
        return self.last_lock

    def add_garbage(self, lines: int, hole: int) -> bool:
        """Raise solid garbage lines and return whether the game remains alive."""

        if self.game_over:
            return False
        count = max(0, int(lines))
        gap = max(0, min(self.width - 1, int(hole)))
        topped_out = False
        for _ in range(count):
            removed = self.board.pop(0)
            if any(cell is not None for cell in removed):
                topped_out = True
            row: list[str | None] = [GARBAGE_CELL] * self.width
            row[gap] = None
            self.board.append(row)
        hidden_occupied = any(cell is not None for row in self.board[: self.hidden_rows] for cell in row)
        active_collision = self._collides(self.current, self.x, self.y, self.rotation)
        self.game_over = topped_out or hidden_occupied or active_collision
        return not self.game_over

    def legal_placements(self) -> tuple[Placement, ...]:
        if self.game_over:
            return ()
        placements: list[Placement] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        rotation_count = 1 if self.current == "O" else 4
        for rotation in range(rotation_count):
            shape = SHAPES[self.current][rotation]
            min_x = min(dx for dx, _ in shape)
            max_x = max(dx for dx, _ in shape)
            for x in range(-min_x, self.width - max_x):
                y = -4
                while not self._collides(self.current, x, y + 1, rotation):
                    y += 1
                cells = self.cells(self.current, x, y, rotation)
                if any(cell_y < 0 for _, cell_y in cells):
                    continue
                key = tuple(sorted(cells))
                if key in seen:
                    continue
                seen.add(key)
                placements.append(Placement(self.current, x, y, rotation, cells))
        return tuple(placements)

    def place(self, placement: Placement) -> LockResult:
        if self.game_over or self.paused:
            raise RuntimeError("Cannot place a piece while the game is stopped")
        if placement.piece != self.current:
            raise ValueError(f"Placement is for {placement.piece}, current piece is {self.current}")
        if self._collides(self.current, placement.x, placement.y, placement.rotation):
            raise ValueError("Placement collides with the board")
        if not self._collides(self.current, placement.x, placement.y + 1, placement.rotation):
            raise ValueError("Placement is not resting on the stack")
        self.x, self.y, self.rotation = placement.x, placement.y, placement.rotation
        self.last_move_was_rotation = bool(placement.last_move_was_rotation)
        self.last_rotation_kick_index = placement.rotation_kick_index
        self.last_rotation_from = placement.rotation_from
        self.last_rotation_to = placement.rotation_to
        self.last_action = "placement"
        return self._lock_piece()

    def snapshot(self, queue_size: int = 5) -> GameSnapshot:
        return GameSnapshot(
            board=tuple(tuple(row) for row in self.board),
            current=self.current,
            x=self.x,
            y=self.y,
            rotation=self.rotation,
            ghost_y=self.ghost_y() if not self.game_over else self.y,
            hold=self.hold_piece,
            hold_used=self.hold_used,
            queue=tuple(list(self.queue)[:queue_size]),
            score=self.score,
            lines=self.lines,
            attack=self.attack,
            combo=self.combo,
            back_to_back=self.back_to_back,
            b2b_chain=self.b2b_chain,
            surge_charge=self.surge_charge,
            pieces_placed=self.pieces_placed,
            game_over=self.game_over,
            paused=self.paused,
            grounded=self.is_grounded() if not self.game_over else False,
            lock_elapsed_ms=self.lock_elapsed_ms,
            lock_delay_ms=self.lock_delay_ms,
            lock_resets=self.lock_resets,
            lock_reset_limit=self.lock_reset_limit,
            last_lock=self.last_lock,
        )
