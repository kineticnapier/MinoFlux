from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    SearchChoice,
    SearchConfig,
    apply_search_action,
    choose_search_action,
    clone_game,
    load_weights,
)
from minoflux_engine import Game, HIDDEN_ROWS, LockResult, VersusMatch, VersusSide
from minoflux_engine.pieces import SHAPES

from .game import COLORS, Palette
from .handling import HandlingController, RepeatBatch
from .settings import GameSettings, load_settings

CHAMPION_MODEL = Path("data/models/champion-cem.json")
RECOVERED_MODEL = Path("presets/recovered-attack-20260723.json")
GARBAGE_COLOR = (105, 109, 120)


@dataclass(frozen=True, slots=True)
class AIJob:
    generation: int
    signature: tuple[object, ...]
    choice: SearchChoice | None


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="minoflux-versus", description="Play MinoFlux against the heuristic AI")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ai-pps", type=float, default=1.0, help="AI placements per second")
    parser.add_argument("--ai-model", help="Path to a minoflux_heuristic_v1 model")
    parser.add_argument("--ai-lookahead", type=int, default=0)
    parser.add_argument("--ai-beam", type=int, default=2)
    parser.add_argument("--garbage-cap", type=int, default=8)
    return parser


def _load_ai_weights(path: str | None):
    if path:
        target = Path(path)
        return load_weights(target), str(target)
    for target in (CHAMPION_MODEL, RECOVERED_MODEL):
        if target.is_file():
            return load_weights(target), str(target)
    return DEFAULT_WEIGHTS, "Built-in weights"


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


def _signature(game: Game) -> tuple[object, ...]:
    return (
        game.pieces_placed,
        game.current,
        game.hold_piece,
        tuple(game.queue),
        tuple(tuple(row) for row in game.board),
        game.back_to_back,
        game.b2b_chain,
        game.surge_charge,
        game.game_over,
    )


def _run_ai_job(
    generation: int,
    game: Game,
    weights,
    config: SearchConfig,
) -> AIJob:
    signature = _signature(game)
    choice = choose_search_action(game, weights, config)
    return AIJob(generation, signature, choice)


def _piece_color(piece: str) -> tuple[int, int, int]:
    return COLORS.get(piece, GARBAGE_COLOR)


def _draw_preview(pygame, surface, piece: str | None, origin: tuple[int, int], cell: int) -> None:
    if not piece:
        return
    shape = SHAPES[piece][0]
    min_x = min(x for x, _ in shape)
    min_y = min(y for _, y in shape)
    for x, y in shape:
        pygame.draw.rect(
            surface,
            _piece_color(piece),
            pygame.Rect(
                origin[0] + (x - min_x) * cell,
                origin[1] + (y - min_y) * cell,
                cell - 2,
                cell - 2,
            ),
            border_radius=3,
        )


def _draw_side(
    pygame,
    screen,
    side: VersusSide,
    *,
    panel_x: int,
    title: str,
    palette: Palette,
    font,
    small,
    elapsed: float,
) -> None:
    game = side.game
    cell = 26
    hold_x = panel_x + 10
    board_x = panel_x + 112
    next_x = board_x + game.width * cell + 24
    board_y = 60

    pygame.draw.rect(screen, palette.panel, (panel_x, 18, 650, 672), border_radius=10)
    screen.blit(font.render(title, True, palette.text), (panel_x + 16, 25))

    screen.blit(small.render("HOLD", True, palette.text), (hold_x, board_y))
    _draw_preview(pygame, screen, game.hold_piece, (hold_x + 8, board_y + 38), 20)

    pygame.draw.rect(
        screen,
        palette.background,
        (board_x - 5, board_y - 5, game.width * cell + 10, game.visible_height * cell + 10),
        border_radius=6,
    )
    for y in range(game.visible_height):
        for x in range(game.width):
            rect = pygame.Rect(board_x + x * cell, board_y + y * cell, cell, cell)
            pygame.draw.rect(screen, palette.grid, rect, 1)
            piece = game.board[y + HIDDEN_ROWS][x]
            if piece:
                pygame.draw.rect(screen, _piece_color(piece), rect.inflate(-3, -3), border_radius=3)

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
                    _piece_color(game.current),
                    (board_x + x * cell + 2, board_y + visible_y * cell + 2, cell - 4, cell - 4),
                    border_radius=3,
                )

    pending = min(20, side.pending.pending_lines)
    if pending:
        bar_height = pending * cell
        pygame.draw.rect(
            screen,
            (220, 85, 85),
            (board_x - 14, board_y + game.visible_height * cell - bar_height, 8, bar_height),
            border_radius=3,
        )

    screen.blit(small.render("NEXT", True, palette.text), (next_x, board_y))
    for index, piece in enumerate(list(game.queue)[:5]):
        _draw_preview(pygame, screen, piece, (next_x + 4, board_y + 36 + index * 76), 17)

    minutes = max(elapsed / 60.0, 1e-9)
    stats = (
        f"Attack {game.attack}  APM {game.attack / minutes:.1f}",
        f"Pieces {game.pieces_placed}  Lines {game.lines}",
        f"B2B x{game.b2b_chain}  Surge {game.surge_charge}",
        f"Pending {side.pending.pending_lines}  Sent {side.sent}",
        f"Canceled {side.canceled}  Received {side.received}",
    )
    for index, text in enumerate(stats):
        screen.blit(small.render(text, True, palette.text), (panel_x + 14, 596 + index * 20))


def _resolve_player_lock(match: VersusMatch, result: LockResult | None) -> None:
    if result is not None:
        match.resolve_lock("player", result)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Pygame is not installed. Run: uv sync --extra game") from error

    pygame.init()
    screen = pygame.display.set_mode((1360, 720))
    pygame.display.set_caption("MinoFlux — Versus AI")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 31)
    small = pygame.font.Font(None, 21)
    palette = Palette()
    settings = load_settings()
    key_codes = _key_codes(pygame, settings)
    handling = HandlingController()
    weights, model_name = _load_ai_weights(args.ai_model)
    ai_config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=args.ai_lookahead,
        beam_width=args.ai_beam,
        discount=0.9,
        srs_reachable=True,
        allow_180=False,
    ).normalized()
    match = VersusMatch(args.seed, garbage_cap=args.garbage_cap)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minoflux-ai")
    ai_future: Future[AIJob] | None = None
    generation = 0
    ai_pps = min(8.0, max(0.1, float(args.ai_pps)))
    ai_next_at = time.monotonic()
    started_at = time.monotonic()
    last_frame = started_at
    last_gravity = started_at
    gravity_interval = 0.75
    paused = False
    running = True

    def restart() -> None:
        nonlocal generation, ai_future, ai_next_at, started_at, last_gravity, paused
        generation += 1
        if ai_future is not None:
            ai_future.cancel()
        ai_future = None
        match.reset(args.seed + generation * 1009)
        handling.clear()
        paused = False
        now = time.monotonic()
        ai_next_at = now
        started_at = now
        last_gravity = now

    try:
        while running:
            now = time.monotonic()
            delta_ms = max(0.0, (now - last_frame) * 1000.0)
            last_frame = now
            player = match.player.game
            skip_lock_advance = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if event.type == pygame.WINDOWFOCUSLOST:
                    handling.clear()
                    skip_lock_advance = True
                    continue
                if event.type == pygame.KEYDOWN:
                    if getattr(event, "repeat", False):
                        continue
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        continue
                    if event.key == pygame.K_LEFTBRACKET:
                        ai_pps = max(0.1, round(ai_pps - 0.1, 1))
                        continue
                    if event.key == pygame.K_RIGHTBRACKET:
                        ai_pps = min(8.0, round(ai_pps + 0.1, 1))
                        continue
                    action = key_codes.get(event.key)
                    if action == "restart":
                        restart()
                    elif action == "pause":
                        paused = not paused
                        handling.clear()
                    elif not paused and match.winner is None and not player.game_over:
                        if action == "left":
                            player.move_left()
                            handling.press_horizontal(-1, now, settings.das_ms)
                        elif action == "right":
                            player.move_right()
                            handling.press_horizontal(1, now, settings.das_ms)
                        elif action == "soft_drop":
                            player.soft_drop()
                            handling.press_soft_drop(now, settings.soft_drop_ms)
                        elif action == "rotate_cw":
                            player.rotate_cw()
                        elif action == "rotate_ccw":
                            player.rotate_ccw()
                        elif action == "rotate_180":
                            player.rotate_180()
                        elif action == "hold":
                            if player.hold():
                                skip_lock_advance = True
                                last_gravity = now
                        elif action == "hard_drop":
                            before = player.pieces_placed
                            result = player.hard_drop()
                            if player.pieces_placed > before:
                                _resolve_player_lock(match, result)
                            skip_lock_advance = True
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
            if not paused and match.winner is None and not player.game_over:
                direction, horizontal_batch = handling.poll_horizontal(now, settings.arr_ms)
                if direction:
                    _move_horizontal(player, direction, horizontal_batch)
                _soft_drop(player, handling.poll_soft_drop(now, settings.soft_drop_ms))
                if now - last_gravity >= gravity_interval:
                    player.gravity_step()
                    last_gravity = now
                if not skip_lock_advance:
                    result = player.advance_time(delta_ms)
                    _resolve_player_lock(match, result)
            else:
                last_gravity = now

            ai_game = match.ai.game
            if not paused and match.winner is None and not ai_game.game_over:
                if ai_future is None and now >= ai_next_at:
                    preview = clone_game(ai_game)
                    ai_future = executor.submit(_run_ai_job, generation, preview, weights, ai_config)
                elif ai_future is not None and ai_future.done():
                    try:
                        job = ai_future.result()
                    except Exception:
                        job = AIJob(generation, (), None)
                    ai_future = None
                    if (
                        job.generation == generation
                        and job.signature == _signature(ai_game)
                        and job.choice is not None
                    ):
                        result = apply_search_action(ai_game, job.choice.action)
                        match.resolve_lock("ai", result)
                    ai_next_at = now + 1.0 / ai_pps

            screen.fill(palette.background)
            elapsed = max(0.0, now - started_at)
            _draw_side(
                pygame,
                screen,
                match.player,
                panel_x=18,
                title="PLAYER",
                palette=palette,
                font=font,
                small=small,
                elapsed=elapsed,
            )
            _draw_side(
                pygame,
                screen,
                match.ai,
                panel_x=692,
                title=f"AI  {ai_pps:.1f} PPS",
                palette=palette,
                font=font,
                small=small,
                elapsed=elapsed,
            )

            footer = f"[ / ] AI speed    Pause/Restart use your bindings    Model: {model_name}"
            screen.blit(small.render(footer, True, palette.muted), (24, 696))
            if paused or match.winner is not None:
                if paused:
                    label = "PAUSED"
                elif match.winner == "player":
                    label = "PLAYER WINS — press restart"
                elif match.winner == "ai":
                    label = "AI WINS — press restart"
                else:
                    label = "DRAW — press restart"
                overlay = font.render(label, True, palette.text)
                screen.blit(overlay, overlay.get_rect(center=(680, 360)))

            pygame.display.flip()
            clock.tick(120)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
