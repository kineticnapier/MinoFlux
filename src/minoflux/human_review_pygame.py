from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

from minoflux_ai.human_review import append_human_label, load_review_queue, review_key
from minoflux_engine import Game, HIDDEN_ROWS

from .game import (
    COLORS,
    Palette,
    _draw_piece_preview,
    _key_codes,
    _move_horizontal,
    _soft_drop,
)
from .handling import HandlingController
from .settings import load_settings


def _row_strings(section: Mapping[str, object]) -> list[str]:
    display = section.get("displayRows")
    if isinstance(display, Sequence) and not isinstance(display, (str, bytes)):
        rows = [str(row)[:10].ljust(10, ".") for row in display]
        if rows:
            return rows[-20:]

    packed = section.get("rows")
    if isinstance(packed, Sequence) and not isinstance(packed, (str, bytes)):
        rows: list[str] = []
        for raw_mask in packed[-20:]:
            mask = int(raw_mask)
            rows.append("".join("G" if mask & (1 << x) else "." for x in range(10)))
        if rows:
            return rows
    return ["." * 10 for _ in range(20)]


def _restore_game(record: Mapping[str, object]) -> Game:
    source = record.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("Review position has no source state")

    game = Game(int(record.get("seed", 0)))
    visible = _row_strings(source)
    game.board = [[None] * game.width for _ in range(HIDDEN_ROWS)]
    for row in visible[-game.visible_height :]:
        game.board.append([None if cell == "." else cell for cell in row[: game.width]])
    while len(game.board) < game.height:
        game.board.insert(0, [None] * game.width)
    if len(game.board) > game.height:
        game.board = game.board[-game.height :]

    game.current = str(source.get("current", game.current))
    hold = source.get("hold")
    game.hold_piece = None if hold in (None, "", "-") else str(hold)
    next_pieces = source.get("next", ())
    if isinstance(next_pieces, Sequence) and not isinstance(next_pieces, (str, bytes)):
        game.queue = deque(str(piece) for piece in next_pieces)
        game._fill_queue(7)
    game.x, game.y, game.rotation = 3, 1, 0
    game.hold_used = False
    game.last_action = None
    game._clear_rotation_metadata()
    game._reset_lock_state()
    game.score = 0
    game.lines = 0
    game.attack = 0
    game.combo = int(source.get("combo", -1))
    game.back_to_back = bool(source.get("b2b", False))
    game.b2b_chain = int(source.get("b2bChain", 0))
    game.surge_charge = int(source.get("surgeCharge", 0))
    game.pieces_placed = int(record.get("pieceIndex", 0))
    game.paused = False
    game.last_lock = None
    game.game_over = game._collides(game.current, game.x, game.y, game.rotation)
    return game


def _find_candidate_index(
    record: Mapping[str, object],
    *,
    use_hold: bool,
    piece: str,
    x: int,
    y: int,
    rotation: int,
    result_game: Game | None = None,
) -> int | None:
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        move = candidate.get("move")
        if not isinstance(move, Mapping):
            continue
        if (
            bool(move.get("hold")) == bool(use_hold)
            and str(move.get("piece")) == piece
            and int(move.get("x", 0)) == int(x)
            and int(move.get("y", 0)) == int(y)
            and int(move.get("rotation", 0)) % 4 == int(rotation) % 4
        ):
            return index

    if result_game is None:
        return None
    result_rows = [
        "".join(cell if cell is not None else "." for cell in row)
        for row in result_game.board[HIDDEN_ROWS:]
    ]
    matching: list[int] = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        if _row_strings(candidate) == result_rows:
            matching.append(index)
    return matching[0] if len(matching) == 1 else None


def _reviewed_keys(path: Path) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, Mapping):
                keys.add(review_key(value))
    return keys


def _draw_review(
    pygame,
    screen,
    game: Game,
    record: Mapping[str, object],
    sample_index: int,
    total: int,
    message: str,
    settings,
    acceptable_count: int,
) -> None:
    palette = Palette()
    cell = 30
    board_x, board_y = 215, 55
    font = pygame.font.Font(None, 30)
    small = pygame.font.Font(None, 21)
    screen.fill(palette.background)

    pygame.draw.rect(
        screen,
        palette.panel,
        (board_x - 5, board_y - 5, game.width * cell + 10, game.visible_height * cell + 10),
        border_radius=6,
    )
    for y in range(game.visible_height):
        for x in range(game.width):
            rect = pygame.Rect(board_x + x * cell, board_y + y * cell, cell, cell)
            pygame.draw.rect(screen, palette.grid, rect, 1)
            piece = game.board[y + HIDDEN_ROWS][x]
            if piece:
                pygame.draw.rect(
                    screen,
                    COLORS.get(piece, palette.muted),
                    rect.inflate(-3, -3),
                    border_radius=3,
                )

    if not game.game_over:
        ghost_y = game.ghost_y()
        for x, y in game.cells(y=ghost_y):
            visible_y = y - HIDDEN_ROWS
            if visible_y >= 0:
                pygame.draw.rect(
                    screen,
                    palette.ghost,
                    (board_x + x * cell + 5, board_y + visible_y * cell + 5, cell - 10, cell - 10),
                    2,
                    border_radius=3,
                )
        for x, y in game.cells():
            visible_y = y - HIDDEN_ROWS
            if visible_y >= 0:
                pygame.draw.rect(
                    screen,
                    COLORS.get(game.current, palette.muted),
                    (board_x + x * cell + 2, board_y + visible_y * cell + 2, cell - 4, cell - 4),
                    border_radius=3,
                )

    screen.blit(font.render("HOLD", True, palette.text), (28, 55))
    _draw_piece_preview(pygame, screen, game.hold_piece, (42, 95), 24)
    screen.blit(font.render("NEXT", True, palette.text), (550, 55))
    for index, piece in enumerate(list(game.queue)[:5]):
        _draw_piece_preview(pygame, screen, piece, (565, 95 + index * 82), 20)

    source = record.get("source")
    source_map = source if isinstance(source, Mapping) else {}
    reasons = record.get("reasons", ())
    reason_text = ", ".join(str(value) for value in reasons) if isinstance(reasons, Sequence) else ""
    info = [
        f"Review {sample_index + 1}/{total}",
        f"Seed {record.get('seed')}",
        f"Piece {record.get('pieceIndex')}",
        f"Flag {reason_text}",
        f"Combo {source_map.get('combo', 0)}",
        f"B2B {'ON' if source_map.get('b2b') else 'OFF'}",
        f"Also-good marked {acceptable_count}",
        f"DAS {settings.das_ms}",
        f"ARR {settings.arr_ms}",
        f"SDS {settings.soft_drop_ms}",
    ]
    for index, line in enumerate(info):
        screen.blit(small.render(line, True, palette.text), (18, 205 + index * 23))

    screen.blit(
        small.render("Play exactly ONE piece with your normal controls.", True, palette.selected),
        (18, 510),
    )
    screen.blit(
        small.render("Hard drop = save primary teacher move", True, palette.selected),
        (18, 535),
    )
    screen.blit(
        small.render("Shift + Hard drop = mark this move as ALSO acceptable", True, palette.selected),
        (18, 560),
    )
    screen.blit(small.render("R/restart = reset position", True, palette.muted), (18, 595))
    screen.blit(small.render("N = skip   Backspace = previous   Esc = quit", True, palette.muted), (18, 620))
    if message:
        screen.blit(small.render(message, True, palette.selected), (18, 660))


def launch_human_review_app(queue_path: str | Path, output_path: str | Path) -> int:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit(
            "Pygame is required for neural review. Install the review extra or pygame-ce."
        ) from error

    records = load_review_queue(queue_path)
    output = Path(output_path)
    reviewed = _reviewed_keys(output)
    pending = [record for record in records if review_key(record) not in reviewed]
    if not pending:
        print(f"No pending human-review positions. Labels: {output}")
        return 0

    pygame.init()
    screen = pygame.display.set_mode((760, 700))
    pygame.display.set_caption("MinoFlux Neural Human Review")
    clock = pygame.time.Clock()
    settings = load_settings()
    key_codes = _key_codes(pygame, settings)
    handling = HandlingController()

    sample_index = 0
    game = _restore_game(pending[sample_index])
    used_hold = False
    acceptable_indices: set[int] = set()
    message = ""
    running = True

    def reset_position(*, clear_marks: bool = False) -> None:
        nonlocal game, used_hold, message
        game = _restore_game(pending[sample_index])
        used_hold = False
        handling.clear()
        if clear_marks:
            acceptable_indices.clear()
        message = "Position reset."

    def move_sample(delta: int) -> None:
        nonlocal sample_index, game, used_hold, message
        sample_index = max(0, min(len(pending), sample_index + delta))
        used_hold = False
        acceptable_indices.clear()
        handling.clear()
        if sample_index < len(pending):
            game = _restore_game(pending[sample_index])
        message = ""

    while running:
        now = time.monotonic()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue
            if event.type == pygame.WINDOWFOCUSLOST:
                handling.clear()
                continue
            if event.type == pygame.KEYDOWN:
                if getattr(event, "repeat", False):
                    continue
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                if sample_index >= len(pending):
                    continue
                if event.key == pygame.K_n:
                    move_sample(1)
                    continue
                if event.key == pygame.K_BACKSPACE:
                    move_sample(-1)
                    continue
                if event.key == pygame.K_r:
                    reset_position(clear_marks=True)
                    continue

                action = key_codes.get(event.key)
                if action == "restart":
                    reset_position(clear_marks=True)
                elif action == "left":
                    game.move_left()
                    handling.press_horizontal(-1, now, settings.das_ms)
                elif action == "right":
                    game.move_right()
                    handling.press_horizontal(1, now, settings.das_ms)
                elif action == "soft_drop":
                    game.soft_drop()
                    handling.press_soft_drop(now, settings.soft_drop_ms)
                elif action == "rotate_cw":
                    game.rotate_cw()
                elif action == "rotate_ccw":
                    game.rotate_ccw()
                elif action == "rotate_180":
                    game.rotate_180()
                elif action == "hold":
                    if game.hold():
                        used_hold = True
                elif action == "hard_drop":
                    record = pending[sample_index]
                    piece = game.current
                    x = game.x
                    y = game.ghost_y()
                    rotation = game.rotation
                    game.hard_drop()
                    candidate_index = _find_candidate_index(
                        record,
                        use_hold=used_hold,
                        piece=piece,
                        x=x,
                        y=y,
                        rotation=rotation,
                        result_game=game,
                    )
                    if candidate_index is None:
                        message = "Placement is outside this review candidate set; reset and retry."
                        game = _restore_game(record)
                        used_hold = False
                        handling.clear()
                    elif event.mod & pygame.KMOD_SHIFT:
                        acceptable_indices.add(candidate_index)
                        game = _restore_game(record)
                        used_hold = False
                        handling.clear()
                        message = f"Marked candidate {candidate_index + 1} as also acceptable."
                    else:
                        added = append_human_label(
                            output,
                            record,
                            candidate_index,
                            sorted(acceptable_indices),
                        )
                        message = "Saved teacher move." if added else "Already labeled; existing label kept."
                        move_sample(1)

            elif event.type == pygame.KEYUP and sample_index < len(pending):
                action = key_codes.get(event.key)
                if action == "left":
                    handling.release_horizontal(-1, now, settings.das_ms)
                elif action == "right":
                    handling.release_horizontal(1, now, settings.das_ms)
                elif action == "soft_drop":
                    handling.release_soft_drop()

        if sample_index < len(pending):
            direction, horizontal_batch = handling.poll_horizontal(now, settings.arr_ms)
            if direction:
                _move_horizontal(game, direction, horizontal_batch)
            _soft_drop(game, handling.poll_soft_drop(now, settings.soft_drop_ms))
            _draw_review(
                pygame,
                screen,
                game,
                pending[sample_index],
                sample_index,
                len(pending),
                message,
                settings,
                len(acceptable_indices),
            )
        else:
            palette = Palette()
            screen.fill(palette.background)
            font = pygame.font.Font(None, 34)
            small = pygame.font.Font(None, 23)
            done = font.render(f"Review complete - {len(pending)} positions", True, palette.selected)
            screen.blit(done, done.get_rect(center=(380, 320)))
            labels = small.render(f"Labels: {output}", True, palette.text)
            screen.blit(labels, labels.get_rect(center=(380, 365)))
            screen.blit(small.render("Esc to close", True, palette.muted), (330, 410))

        pygame.display.flip()
        clock.tick(120)

    pygame.quit()
    return 0
