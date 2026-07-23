from __future__ import annotations

from collections import deque

from .pieces import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    HIDDEN_ROWS,
    LINE_ATTACK,
    LINE_SCORES,
    SHAPES,
    VISIBLE_HEIGHT,
    SevenBag,
    kick_tests,
)
from .state import GameSnapshot, LockResult, Placement

DEFAULT_LOCK_DELAY_MS = 500.0
DEFAULT_LOCK_RESET_LIMIT = 15


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
        self.score = 0
        self.lines = 0
        self.attack = 0
        self.combo = -1
        self.back_to_back = False
        self.pieces_placed = 0
        self.game_over = False
        self.paused = False
        self.lock_elapsed_ms = 0.0
        self.lock_resets = 0
        self.last_lock: LockResult | None = None
        self.reset(seed)

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
        self.last_move_was_rotation = False
        self.score = self.lines = self.attack = self.pieces_placed = 0
        self.combo = -1
        self.back_to_back = self.game_over = self.paused = False
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
        self.last_move_was_rotation = False
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
        self.last_move_was_rotation = False
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
        for kick_x, kick_y in kick_tests(self.current, source_rotation, target_rotation):
            target_x, target_y = self.x + kick_x, self.y + kick_y
            if not self._collides(self.current, target_x, target_y, target_rotation):
                self.x, self.y, self.rotation = target_x, target_y, target_rotation
                self.last_action = "rotate"
                self.last_move_was_rotation = True
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
        self.last_move_was_rotation = False
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
        """Advance lock delay without depending on a wall-clock implementation."""
        if self.game_over or self.paused:
            return None
        delta = max(0.0, float(delta_ms))
        if not self.is_grounded():
            return None
        self.lock_elapsed_ms += delta
        if self.lock_elapsed_ms + 1e-9 >= self.lock_delay_ms:
            return self._lock_piece()
        return None

    def _detect_spin(self) -> str | None:
        if not self.last_move_was_rotation or self.current == "O":
            return None
        if self.current == "T":
            pivot_x, pivot_y = self.x + 1, self.y + 1
            occupied = 0
            for cx, cy in (
                (pivot_x - 1, pivot_y - 1),
                (pivot_x + 1, pivot_y - 1),
                (pivot_x - 1, pivot_y + 1),
                (pivot_x + 1, pivot_y + 1),
            ):
                if cx < 0 or cx >= self.width or cy < 0 or cy >= self.height:
                    occupied += 1
                elif self.board[cy][cx] is not None:
                    occupied += 1
            if occupied >= 3:
                return "T"
        blocked = (
            self._collides(self.current, self.x - 1, self.y, self.rotation)
            and self._collides(self.current, self.x + 1, self.y, self.rotation)
            and self._collides(self.current, self.x, self.y + 1, self.rotation)
        )
        return self.current if blocked else None

    def _lock_piece(self) -> LockResult:
        spin = self._detect_spin()
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

        perfect_clear = all(cell is None for row in self.board for cell in row)
        difficult = lines == 4 or (spin is not None and lines > 0)
        sent = LINE_ATTACK.get(lines, 0)
        if spin is not None and lines:
            sent += max(1, lines)
        if difficult and self.back_to_back:
            sent += 1
        if lines:
            self.combo += 1
            if self.combo > 0:
                sent += min(4, self.combo // 2 + 1)
        else:
            self.combo = -1
        if perfect_clear and lines:
            sent += 10

        level = self.lines // 10 + 1
        gained = LINE_SCORES.get(lines, 1200)
        if spin is not None and lines:
            gained += 400 * lines
        if difficult and self.back_to_back:
            gained = int(gained * 1.5)
        self.score += gained * level
        self.lines += lines
        self.attack += sent
        if difficult:
            self.back_to_back = True
        elif lines:
            self.back_to_back = False

        self.pieces_placed += 1
        hidden_occupied = any(cell is not None for row in self.board[: self.hidden_rows] for cell in row)
        self.game_over = topped_out or hidden_occupied
        if not self.game_over:
            self._spawn()
        else:
            self.lock_elapsed_ms = 0.0

        self.last_lock = LockResult(
            lines=lines,
            attack=sent,
            spin=spin,
            perfect_clear=perfect_clear,
            combo=self.combo,
            back_to_back=self.back_to_back,
            game_over=self.game_over,
        )
        return self.last_lock

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
        old_rotation = self.rotation
        self.x, self.y, self.rotation = placement.x, placement.y, placement.rotation
        self.last_move_was_rotation = placement.rotation != old_rotation
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
