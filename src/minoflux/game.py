from __future__ import annotations

from dataclasses import dataclass
import time

from minoflux_engine import Game, HIDDEN_ROWS
from minoflux_engine.pieces import SHAPES

from .handling import HandlingController, RepeatBatch
from .settings import ACTION_LABELS, DEFAULT_BINDINGS, GameSettings, load_settings, save_settings


@dataclass(frozen=True)
class Palette:
    background: tuple[int, int, int] = (20, 22, 29)
    panel: tuple[int, int, int] = (31, 34, 44)
    grid: tuple[int, int, int] = (53, 57, 71)
    ghost: tuple[int, int, int] = (90, 94, 108)
    text: tuple[int, int, int] = (232, 234, 241)
    muted: tuple[int, int, int] = (160, 164, 178)
    selected: tuple[int, int, int] = (67, 201, 224)
    overlay: tuple[int, int, int, int] = (9, 11, 17, 238)


COLORS = {
    "I": (67, 201, 224),
    "O": (240, 210, 75),
    "T": (174, 91, 214),
    "S": (86, 194, 96),
    "Z": (222, 76, 83),
    "J": (72, 111, 214),
    "L": (236, 145, 63),
}

SETTING_ROWS: tuple[tuple[str, str, int, int, int], ...] = (
    ("DAS", "das_ms", 10, 0, 1000),
    ("ARR", "arr_ms", 1, 0, 500),
    ("SDS", "soft_drop_ms", 1, 0, 500),
)
BINDING_ACTIONS = tuple(DEFAULT_BINDINGS)


def _draw_piece_preview(pygame, surface, piece: str | None, origin: tuple[int, int], cell: int) -> None:
    if not piece:
        return
    shape = SHAPES[piece][0]
    min_x = min(x for x, _ in shape)
    min_y = min(y for _, y in shape)
    color = COLORS[piece]
    for x, y in shape:
        rect = pygame.Rect(
            origin[0] + (x - min_x) * cell,
            origin[1] + (y - min_y) * cell,
            cell - 2,
            cell - 2,
        )
        pygame.draw.rect(surface, color, rect, border_radius=3)


def _key_codes(pygame, settings: GameSettings) -> dict[int, str]:
    result: dict[int, str] = {}
    for action, key_name in settings.bindings.items():
        try:
            result[pygame.key.key_code(key_name)] = action
        except ValueError:
            continue
    return result


def _move_horizontal(game: Game, direction: int, batch: RepeatBatch) -> None:
    mover = game.move_left if direction < 0 else game.move_right
    if batch.instant:
        while mover():
            pass
        return
    for _ in range(batch.count):
        if not mover():
            break


def _soft_drop(game: Game, batch: RepeatBatch) -> None:
    if batch.instant:
        while game.soft_drop():
            pass
        return
    for _ in range(batch.count):
        if not game.soft_drop():
            break


def _draw_settings(
    pygame,
    screen,
    font,
    small,
    palette: Palette,
    settings: GameSettings,
    selected: int,
    rebinding: str | None,
) -> None:
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    overlay.fill(palette.overlay)
    screen.blit(overlay, (0, 0))
    screen.blit(font.render("INPUT SETTINGS", True, palette.text), (150, 32))
    screen.blit(
        small.render("F1 / Esc: close    Enter: rebind    Backspace: defaults", True, palette.muted),
        (92, 64),
    )

    rows: list[tuple[str, str]] = [
        ("DAS", f"{settings.das_ms} ms"),
        ("ARR", "instant" if settings.arr_ms == 0 else f"{settings.arr_ms} ms"),
        ("SDS", "instant" if settings.soft_drop_ms == 0 else f"{settings.soft_drop_ms} ms/cell"),
    ]
    rows.extend((ACTION_LABELS[action], settings.bindings[action]) for action in BINDING_ACTIONS)

    for index, (label, value) in enumerate(rows):
        y = 105 + index * 34
        color = palette.selected if index == selected else palette.text
        prefix = "> " if index == selected else "  "
        screen.blit(small.render(prefix + label, True, color), (120, y))
        screen.blit(small.render(value, True, color), (420, y))

    if rebinding:
        prompt = font.render(f"Press a key for {ACTION_LABELS[rebinding]}", True, palette.selected)
        pygame.draw.rect(screen, palette.panel, (75, 565, 500, 70), border_radius=8)
        screen.blit(prompt, prompt.get_rect(center=(325, 600)))


def _adjust_setting(settings: GameSettings, selected: int, direction: int) -> bool:
    if selected >= len(SETTING_ROWS):
        return False
    _, attribute, step, minimum, maximum = SETTING_ROWS[selected]
    value = int(getattr(settings, attribute)) + step * direction
    setattr(settings, attribute, max(minimum, min(maximum, value)))
    return True


def main() -> None:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Pygame is not installed. Run: uv sync --extra game") from error

    pygame.init()
    cell = 30
    board_x, board_y = 180, 30
    width, height = 650, 690
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("MinoFlux")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)
    small = pygame.font.Font(None, 23)
    palette = Palette()
    game = Game()
    settings = load_settings()
    key_codes = _key_codes(pygame, settings)
    handling = HandlingController()
    running = True
    paused = False
    settings_open = False
    settings_selected = 0
    rebinding: str | None = None
    gravity_interval = 0.75
    last_gravity = time.monotonic()

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

                if rebinding is not None:
                    if event.key == pygame.K_ESCAPE:
                        rebinding = None
                    else:
                        new_key = pygame.key.name(event.key).lower()
                        previous_key = settings.bindings[rebinding]
                        for action, key_name in settings.bindings.items():
                            if action != rebinding and key_name == new_key:
                                settings.bindings[action] = previous_key
                                break
                        settings.bindings[rebinding] = new_key
                        settings.normalize()
                        save_settings(settings)
                        key_codes = _key_codes(pygame, settings)
                        rebinding = None
                    continue

                if settings_open:
                    row_count = len(SETTING_ROWS) + len(BINDING_ACTIONS)
                    if event.key in (pygame.K_ESCAPE, pygame.K_F1):
                        settings_open = False
                        save_settings(settings)
                    elif event.key == pygame.K_UP:
                        settings_selected = (settings_selected - 1) % row_count
                    elif event.key == pygame.K_DOWN:
                        settings_selected = (settings_selected + 1) % row_count
                    elif event.key == pygame.K_LEFT and _adjust_setting(settings, settings_selected, -1):
                        save_settings(settings)
                    elif event.key == pygame.K_RIGHT and _adjust_setting(settings, settings_selected, 1):
                        save_settings(settings)
                    elif event.key == pygame.K_RETURN and settings_selected >= len(SETTING_ROWS):
                        rebinding = BINDING_ACTIONS[settings_selected - len(SETTING_ROWS)]
                    elif event.key == pygame.K_BACKSPACE:
                        settings = GameSettings()
                        save_settings(settings)
                        key_codes = _key_codes(pygame, settings)
                    continue

                if event.key == pygame.K_F1:
                    settings_open = True
                    handling.clear()
                    continue
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue

                action = key_codes.get(event.key)
                if action == "restart":
                    game.reset()
                    paused = False
                    game.paused = False
                    handling.clear()
                    last_gravity = now
                elif action == "pause":
                    paused = not paused
                    game.paused = paused
                    handling.clear()
                elif not paused and not game.game_over:
                    if action == "left":
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
                        game.hold()
                    elif action == "hard_drop":
                        game.hard_drop()
                        last_gravity = now

            elif event.type == pygame.KEYUP:
                action = key_codes.get(event.key)
                if action == "left":
                    handling.release_horizontal(-1, now, settings.das_ms)
                elif action == "right":
                    handling.release_horizontal(1, now, settings.das_ms)
                elif action == "soft_drop":
                    handling.release_soft_drop()

        now = time.monotonic()
        if not settings_open and not paused and not game.game_over:
            direction, horizontal_batch = handling.poll_horizontal(now, settings.arr_ms)
            if direction:
                _move_horizontal(game, direction, horizontal_batch)
            _soft_drop(game, handling.poll_soft_drop(now, settings.soft_drop_ms))

            if now - last_gravity >= gravity_interval:
                game.gravity_step()
                last_gravity = now
        else:
            last_gravity = now

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
                    pygame.draw.rect(screen, COLORS[piece], rect.inflate(-3, -3), border_radius=3)

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
                        COLORS[game.current],
                        (board_x + x * cell + 2, board_y + visible_y * cell + 2, cell - 4, cell - 4),
                        border_radius=3,
                    )

        screen.blit(font.render("HOLD", True, palette.text), (30, 35))
        _draw_piece_preview(pygame, screen, game.hold_piece, (45, 75), 24)
        screen.blit(font.render("NEXT", True, palette.text), (510, 35))
        for index, piece in enumerate(list(game.queue)[:5]):
            _draw_piece_preview(pygame, screen, piece, (520, 75 + index * 88), 20)

        stats = [
            f"Score {game.score}",
            f"Lines {game.lines}",
            f"Attack {game.attack}",
            f"Pieces {game.pieces_placed}",
            f"Combo {max(0, game.combo)}",
            f"B2B {'ON' if game.back_to_back else 'OFF'}",
            f"DAS {settings.das_ms}",
            f"ARR {settings.arr_ms}",
            f"SDS {settings.soft_drop_ms}",
        ]
        for index, line in enumerate(stats):
            screen.blit(small.render(line, True, palette.text), (30, 230 + index * 27))

        controls = "F1 settings    held movement enabled"
        screen.blit(small.render(controls, True, palette.text), (24, 650))
        if paused or game.game_over:
            label = "PAUSED" if paused else "GAME OVER — restart key"
            overlay = font.render(label, True, palette.text)
            screen.blit(
                overlay,
                overlay.get_rect(
                    center=(
                        board_x + game.width * cell // 2,
                        board_y + game.visible_height * cell // 2,
                    )
                ),
            )
        if settings_open:
            _draw_settings(
                pygame,
                screen,
                font,
                small,
                palette,
                settings,
                settings_selected,
                rebinding,
            )

        pygame.display.flip()
        clock.tick(120)

    save_settings(settings)
    pygame.quit()


if __name__ == "__main__":
    main()
