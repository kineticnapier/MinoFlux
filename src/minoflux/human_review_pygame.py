from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from minoflux_ai.human_review import append_human_label, load_review_queue, review_key

from .game import COLORS, Palette, _draw_piece_preview


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


def _candidate_order(record: Mapping[str, object]) -> tuple[int, ...]:
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return ()
    order = list(range(len(candidates)))
    seed = int(record.get("seed", 0))
    piece_index = int(record.get("pieceIndex", 0))
    random.Random((seed << 17) ^ piece_index ^ 0x4D494E4F).shuffle(order)
    return tuple(order)


def _draw_board(
    pygame,
    screen,
    rows: Sequence[str],
    origin: tuple[int, int],
    cell: int,
    palette: Palette,
    *,
    label: str,
    font,
) -> None:
    board_x, board_y = origin
    screen.blit(font.render(label, True, palette.text), (board_x, board_y - 34))
    pygame.draw.rect(
        screen,
        palette.panel,
        (board_x - 5, board_y - 5, 10 * cell + 10, 20 * cell + 10),
        border_radius=6,
    )
    visible = list(rows)[-20:]
    if len(visible) < 20:
        visible = ["." * 10] * (20 - len(visible)) + visible
    for y, row in enumerate(visible):
        for x in range(10):
            rect = pygame.Rect(board_x + x * cell, board_y + y * cell, cell, cell)
            pygame.draw.rect(screen, palette.grid, rect, 1)
            piece = row[x] if x < len(row) else "."
            if piece != ".":
                color = COLORS.get(piece, palette.muted)
                pygame.draw.rect(screen, color, rect.inflate(-3, -3), border_radius=3)


def _move_text(candidate: Mapping[str, object]) -> str:
    move = candidate.get("move")
    if not isinstance(move, Mapping):
        return "unknown move"
    prefix = "HOLD -> " if bool(move.get("hold")) else ""
    return (
        f"{prefix}{move.get('piece', '?')}  "
        f"x={int(move.get('x', 0))} y={int(move.get('y', 0))} "
        f"r={int(move.get('rotation', 0))}"
    )


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
    width, height = 1180, 720
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MinoFlux Neural Human Review")
    clock = pygame.time.Clock()
    palette = Palette()
    title_font = pygame.font.Font(None, 34)
    font = pygame.font.Font(None, 26)
    small = pygame.font.Font(None, 21)

    sample_index = 0
    candidate_cursor = 0
    message = ""
    running = True

    prev_rect = pygame.Rect(500, 640, 135, 42)
    choose_rect = pygame.Rect(645, 640, 170, 42)
    next_rect = pygame.Rect(825, 640, 135, 42)
    skip_rect = pygame.Rect(970, 640, 135, 42)

    def current_state() -> tuple[dict[str, object], tuple[int, ...], int, Mapping[str, object]]:
        nonlocal candidate_cursor
        record = pending[sample_index]
        order = _candidate_order(record)
        if not order:
            raise ValueError("Review position has no candidates")
        candidate_cursor %= len(order)
        candidates = record.get("candidates")
        assert isinstance(candidates, Sequence)
        original_index = order[candidate_cursor]
        candidate = candidates[original_index]
        if not isinstance(candidate, Mapping):
            raise ValueError("Review candidate must be an object")
        return record, order, original_index, candidate

    def advance_sample(delta: int = 1) -> None:
        nonlocal sample_index, candidate_cursor
        sample_index = max(0, min(len(pending), sample_index + delta))
        candidate_cursor = 0

    def choose_current() -> None:
        nonlocal message
        if sample_index >= len(pending):
            return
        record, _, original_index, _ = current_state()
        added = append_human_label(output, record, original_index)
        message = "Saved human label." if added else "Already labeled; kept existing label."
        advance_sample(1)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif sample_index < len(pending):
                    record, order, _, _ = current_state()
                    del record
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        candidate_cursor = (candidate_cursor - 1) % len(order)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        candidate_cursor = (candidate_cursor + 1) % len(order)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        choose_current()
                    elif event.key in (pygame.K_n, pygame.K_DOWN):
                        message = "Skipped position."
                        advance_sample(1)
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_UP):
                        message = ""
                        advance_sample(-1)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if sample_index >= len(pending):
                    continue
                _, order, _, _ = current_state()
                if prev_rect.collidepoint(event.pos):
                    candidate_cursor = (candidate_cursor - 1) % len(order)
                elif next_rect.collidepoint(event.pos):
                    candidate_cursor = (candidate_cursor + 1) % len(order)
                elif choose_rect.collidepoint(event.pos):
                    choose_current()
                elif skip_rect.collidepoint(event.pos):
                    message = "Skipped position."
                    advance_sample(1)

        screen.fill(palette.background)
        screen.blit(title_font.render("MinoFlux Human Review", True, palette.text), (34, 18))

        if sample_index >= len(pending):
            done = title_font.render(
                f"Review complete - {len(pending)} positions handled",
                True,
                palette.selected,
            )
            screen.blit(done, done.get_rect(center=(width // 2, height // 2 - 20)))
            screen.blit(
                font.render(f"Labels: {output}", True, palette.text),
                (220, height // 2 + 28),
            )
            screen.blit(
                small.render("Esc: close", True, palette.muted),
                (width // 2 - 40, height // 2 + 72),
            )
            pygame.display.flip()
            clock.tick(60)
            continue

        record, order, _, candidate = current_state()
        source = record.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("Review position has no source state")

        screen.blit(
            small.render(
                f"Position {sample_index + 1}/{len(pending)}   seed {record.get('seed')}   piece {record.get('pieceIndex')}",
                True,
                palette.muted,
            ),
            (36, 54),
        )
        reasons = record.get("reasons", ())
        reason_text = ", ".join(str(value) for value in reasons) if isinstance(reasons, Sequence) else ""
        screen.blit(small.render(f"Flagged: {reason_text}", True, palette.muted), (36, 76))

        _draw_board(
            pygame,
            screen,
            _row_strings(source),
            (54, 116),
            27,
            palette,
            label="CURRENT",
            font=font,
        )
        _draw_board(
            pygame,
            screen,
            _row_strings(candidate),
            (772, 116),
            27,
            palette,
            label=f"CANDIDATE {candidate_cursor + 1}/{len(order)}",
            font=font,
        )

        info_x = 372
        screen.blit(font.render("STATE", True, palette.text), (info_x, 118))
        state_lines = [
            f"Current  {source.get('current', '-')}",
            f"Hold     {source.get('hold') or '-'}",
            f"Combo    {source.get('combo', 0)}",
            f"B2B      {'ON' if source.get('b2b') else 'OFF'}",
            f"B2B chain {source.get('b2bChain', 0)}",
        ]
        for index, line in enumerate(state_lines):
            screen.blit(small.render(line, True, palette.text), (info_x, 154 + index * 25))

        next_pieces = source.get("next", ())
        screen.blit(font.render("NEXT", True, palette.text), (info_x, 300))
        if isinstance(next_pieces, Sequence):
            for index, piece in enumerate(next_pieces[:5]):
                piece_name = str(piece)
                _draw_piece_preview(pygame, screen, piece_name, (info_x + 8, 334 + index * 48), 15)

        screen.blit(font.render("YOUR CHOICE", True, palette.text), (info_x, 585))
        screen.blit(small.render(_move_text(candidate), True, palette.selected), (info_x, 614))

        for rect, text in (
            (prev_rect, "< Candidate"),
            (choose_rect, "CHOOSE (Enter)"),
            (next_rect, "Candidate >"),
            (skip_rect, "Skip (N)"),
        ):
            pygame.draw.rect(screen, palette.panel, rect, border_radius=7)
            pygame.draw.rect(screen, palette.grid, rect, 1, border_radius=7)
            rendered = small.render(text, True, palette.text)
            screen.blit(rendered, rendered.get_rect(center=rect.center))

        if message:
            screen.blit(small.render(message, True, palette.selected), (36, 684))
        controls = "Left/Right or A/D: candidate   Enter/Space: teach   N/Down: skip   Up/Backspace: previous   Esc: quit"
        screen.blit(small.render(controls, True, palette.muted), (36, 660))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0
