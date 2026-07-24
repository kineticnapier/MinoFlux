from __future__ import annotations

from argparse import ArgumentParser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

from minoflux_ai import SearchConfig, apply_search_action
from minoflux_ai.versus_search import (
    VersusChoice,
    VersusSearchConfig,
    choose_versus_action,
    clone_versus_match,
)
from minoflux_engine import VersusMatch

from .game import Palette
from .handling import HandlingController
from .settings import load_settings
from .versus import (
    _draw_side,
    _key_codes,
    _load_ai_weights,
    _move_horizontal,
    _resolve_player_lock,
    _soft_drop,
)


@dataclass(frozen=True, slots=True)
class AIJob:
    generation: int
    signature: tuple[object, ...]
    choice: VersusChoice | None


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="minoflux-versus", description="Play MinoFlux against the opponent-aware AI")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ai-pps", type=float, default=1.0)
    parser.add_argument("--ai-model")
    parser.add_argument("--ai-lookahead", type=int, default=0)
    parser.add_argument("--ai-beam", type=int, default=2)
    parser.add_argument("--ai-candidates", type=int, default=8)
    parser.add_argument("--ai-reply-width", type=int, default=2)
    parser.add_argument("--garbage-cap", type=int, default=8)
    return parser


def _game_signature(game) -> tuple[object, ...]:
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


def _match_signature(match: VersusMatch) -> tuple[object, ...]:
    return (
        _game_signature(match.player.game),
        tuple((packet.lines, packet.hole) for packet in match.player.pending.packets),
        _game_signature(match.ai.game),
        tuple((packet.lines, packet.hole) for packet in match.ai.pending.packets),
        match.winner,
    )


def _run_ai_job(generation: int, match: VersusMatch, weights, config: VersusSearchConfig) -> AIJob:
    signature = _match_signature(match)
    choice = choose_versus_action(match, "ai", weights, config)
    return AIJob(generation, signature, choice)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Pygame is not installed. Run: uv sync --extra game") from error

    pygame.init()
    screen = pygame.display.set_mode((1360, 720))
    pygame.display.set_caption("MinoFlux — Versus-aware AI")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 31)
    small = pygame.font.Font(None, 21)
    palette = Palette()
    settings = load_settings()
    key_codes = _key_codes(pygame, settings)
    handling = HandlingController()
    weights, model_name = _load_ai_weights(args.ai_model)
    placement_config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=args.ai_lookahead,
        beam_width=args.ai_beam,
        discount=0.9,
        srs_reachable=True,
        allow_180=False,
    ).normalized()
    versus_config = VersusSearchConfig(
        placement_search=placement_config,
        candidate_width=args.ai_candidates,
        opponent_reply_width=args.ai_reply_width,
    ).normalized()
    match = VersusMatch(args.seed, garbage_cap=args.garbage_cap)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="minoflux-versus-ai")
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
                    _resolve_player_lock(match, player.advance_time(delta_ms))
            else:
                last_gravity = now

            ai_game = match.ai.game
            if not paused and match.winner is None and not ai_game.game_over:
                if ai_future is None and now >= ai_next_at:
                    preview = clone_versus_match(match)
                    ai_future = executor.submit(_run_ai_job, generation, preview, weights, versus_config)
                elif ai_future is not None and ai_future.done():
                    try:
                        job = ai_future.result()
                    except Exception:
                        job = AIJob(generation, (), None)
                    ai_future = None
                    if (
                        job.generation == generation
                        and job.signature == _match_signature(match)
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
                title=f"VERSUS AI  {ai_pps:.1f} PPS",
                palette=palette,
                font=font,
                small=small,
                elapsed=elapsed,
            )
            footer = (
                f"[ / ] AI speed    replies {versus_config.opponent_reply_width}    "
                f"candidates {versus_config.candidate_width}    Model: {model_name}"
            )
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
