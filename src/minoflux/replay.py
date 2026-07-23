from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from minoflux_ai import Replay, apply_replay_step, load_replay
from minoflux_engine import Game, HIDDEN_ROWS

from .game import COLORS, Palette, _draw_piece_preview


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="minoflux-replay", description="Replay a MinoFlux benchmark game")
    parser.add_argument("path", help="Path to a MinoFlux replay JSON file")
    parser.add_argument("--interval-ms", type=int, default=250, help="Milliseconds between placements")
    return parser


def _rebuild(replay: Replay, step_count: int) -> Game:
    game = Game(replay.seed)
    for step in replay.steps[: max(0, min(len(replay.steps), step_count))]:
        apply_replay_step(game, step)
    return game


def _draw_board(pygame, screen, game: Game, replay: Replay, step_index: int, interval_ms: int, paused: bool, error: str | None) -> None:
    palette = Palette()
    cell = 30
    board_x, board_y = 180, 30
    font = pygame.font.Font(None, 30)
    small = pygame.font.Font(None, 23)
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

    holds = sum(step.hold for step in replay.steps[:step_index])
    stats = [
        f"Replay {step_index}/{len(replay.steps)}",
        f"Seed {replay.seed}",
        f"Score {game.score}",
        f"Lines {game.lines}",
        f"Attack {game.attack}",
        f"Holds {holds}",
        f"Speed {interval_ms} ms",
        "DONE" if step_index >= len(replay.steps) else ("PAUSED" if paused else "PLAYING"),
    ]
    for index, line in enumerate(stats):
        screen.blit(small.render(line, True, palette.text), (30, 225 + index * 28))

    controls = "Space pause  Left/Right step  Up/Down speed  Home/End  Esc quit"
    screen.blit(small.render(controls, True, palette.text), (20, 650))
    if error:
        overlay = font.render(error, True, palette.text)
        screen.blit(overlay, overlay.get_rect(center=(330, 610)))


def play_replay(path: str | Path, interval_ms: int = 250) -> int:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Pygame is not installed. Run: uv sync --extra game") from error

    replay = load_replay(path)
    pygame.init()
    screen = pygame.display.set_mode((650, 690))
    pygame.display.set_caption(f"MinoFlux Replay — {Path(path).name}")
    clock = pygame.time.Clock()
    interval = max(30, int(interval_ms))
    game = Game(replay.seed)
    step_index = 0
    paused = False
    accumulator = 0.0
    error_message: str | None = None
    running = True

    while running:
        delta_ms = clock.tick(120)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_RIGHT and step_index < len(replay.steps):
                    try:
                        apply_replay_step(game, replay.steps[step_index])
                        step_index += 1
                        error_message = None
                    except ValueError as error:
                        error_message = str(error)
                        paused = True
                elif event.key == pygame.K_LEFT and step_index > 0:
                    step_index -= 1
                    game = _rebuild(replay, step_index)
                    error_message = None
                elif event.key == pygame.K_HOME:
                    step_index = 0
                    game = Game(replay.seed)
                    error_message = None
                elif event.key == pygame.K_END:
                    step_index = len(replay.steps)
                    game = _rebuild(replay, step_index)
                    error_message = None
                elif event.key == pygame.K_UP:
                    interval = max(30, int(interval / 1.25))
                elif event.key == pygame.K_DOWN:
                    interval = min(2000, int(interval * 1.25))

        if not paused and error_message is None and step_index < len(replay.steps):
            accumulator += delta_ms
            while accumulator >= interval and step_index < len(replay.steps):
                accumulator -= interval
                try:
                    apply_replay_step(game, replay.steps[step_index])
                    step_index += 1
                except ValueError as error:
                    error_message = str(error)
                    paused = True
                    break
        else:
            accumulator = 0.0

        _draw_board(pygame, screen, game, replay, step_index, interval, paused, error_message)
        pygame.display.flip()

    pygame.quit()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return play_replay(args.path, args.interval_ms)


if __name__ == "__main__":
    raise SystemExit(main())
