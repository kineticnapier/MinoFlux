from __future__ import annotations

from dataclasses import dataclass
import time

from minoflux_engine import Game, HIDDEN_ROWS
from minoflux_engine.pieces import SHAPES


@dataclass(frozen=True)
class Palette:
    background: tuple[int, int, int] = (20, 22, 29)
    panel: tuple[int, int, int] = (31, 34, 44)
    grid: tuple[int, int, int] = (53, 57, 71)
    ghost: tuple[int, int, int] = (90, 94, 108)
    text: tuple[int, int, int] = (232, 234, 241)


COLORS = {
    "I": (67, 201, 224),
    "O": (240, 210, 75),
    "T": (174, 91, 214),
    "S": (86, 194, 96),
    "Z": (222, 76, 83),
    "J": (72, 111, 214),
    "L": (236, 145, 63),
}


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


def main() -> None:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Pygame is not installed. Run: python -m pip install -e '.[game]'") from error

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
    running = True
    paused = False
    gravity_interval = 0.75
    last_gravity = time.monotonic()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    game.reset()
                    paused = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    game.paused = paused
                elif not paused and not game.game_over:
                    if event.key == pygame.K_LEFT:
                        game.move_left()
                    elif event.key == pygame.K_RIGHT:
                        game.move_right()
                    elif event.key == pygame.K_DOWN:
                        game.soft_drop()
                    elif event.key in (pygame.K_x, pygame.K_UP):
                        game.rotate_cw()
                    elif event.key == pygame.K_z:
                        game.rotate_ccw()
                    elif event.key == pygame.K_a:
                        game.rotate_180()
                    elif event.key in (pygame.K_c, pygame.K_LSHIFT, pygame.K_RSHIFT):
                        game.hold()
                    elif event.key == pygame.K_SPACE:
                        game.hard_drop()

        now = time.monotonic()
        if not paused and not game.game_over and now - last_gravity >= gravity_interval:
            game.gravity_step()
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
        ]
        for index, line in enumerate(stats):
            screen.blit(small.render(line, True, palette.text), (30, 250 + index * 30))

        controls = "←→ move  ↓ soft  Z/X rotate  A 180  C hold  Space drop"
        screen.blit(small.render(controls, True, palette.text), (24, 650))
        if paused or game.game_over:
            label = "PAUSED" if paused else "GAME OVER — R to restart"
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

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
