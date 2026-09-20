from __future__ import annotations

import os
import random
import unittest
from unittest.mock import patch

from minoflux_ai.reachability import (
    ReachabilityProfile,
    clear_reachability_cache,
    collect_reachability_profile,
)
from minoflux_ai.reachability_native import (
    native_pathless_available,
    reachable_placements_pathless_native,
    reachable_placements_pathless_python,
)
from minoflux_ai.search import SearchConfig, _branch_groups
from minoflux_engine import Game


_COUNTER_FIELDS = (
    "calls",
    "bfs_nodes",
    "collision_checks",
    "collision_evaluations",
    "collision_cache_hits",
    "kick_checks",
    "landing_queries",
    "landing_cache_hits",
    "representative_nodes",
    "representative_duplicate_skips",
    "placements",
    "path_calls",
    "cache_hits",
    "cache_misses",
)


def _empty_game(piece: str = "T") -> Game:
    game = Game(1)
    game.board = [[None] * game.width for _ in range(game.height)]
    game.current = piece
    game.x = 3
    game.y = 0
    game.rotation = 0
    game.game_over = False
    game.paused = False
    return game


def _profiled(callable_):
    clear_reachability_cache()
    profile = ReachabilityProfile()
    with collect_reachability_profile(profile):
        placements = callable_()
    return placements, profile


@unittest.skipUnless(native_pathless_available(), "native reachability extension unavailable")
class NativeReachabilityDifferentialTests(unittest.TestCase):
    def assert_native_matches_python(
        self,
        game: Game,
        *,
        allow_180: bool = False,
        max_nodes: int = 8_000,
    ) -> None:
        python_placements, python_profile = _profiled(
            lambda: reachable_placements_pathless_python(
                game,
                allow_180=allow_180,
                max_nodes=max_nodes,
            )
        )
        native_placements, native_profile = _profiled(
            lambda: reachable_placements_pathless_native(
                game,
                allow_180=allow_180,
                max_nodes=max_nodes,
            )
        )
        self.assertEqual(native_placements, python_placements)
        for field in _COUNTER_FIELDS:
            self.assertEqual(
                getattr(native_profile, field),
                getattr(python_profile, field),
                field,
            )

    def test_empty_board_all_piece_types(self) -> None:
        for piece in "IJLOSTZ":
            with self.subTest(piece=piece):
                self.assert_native_matches_python(_empty_game(piece))

    def test_wall_and_floor_kick_dense_boundaries(self) -> None:
        for piece in ("I", "J", "L", "S", "T", "Z"):
            for x in (-2, -1, 0, 7, 8, 9):
                for y in (0, 19, 20, 21, 22):
                    game = _empty_game(piece)
                    game.x = x
                    game.y = y
                    game.rotation = (x + y) & 3
                    with self.subTest(piece=piece, x=x, y=y):
                        self.assert_native_matches_python(game)

    def test_t_spin_metadata_board(self) -> None:
        game = _empty_game("T")
        game.x, game.y, game.rotation = 3, 19, 0
        for x in range(game.width):
            game.board[23][x] = "J"
        for x in (3, 4, 5):
            game.board[23][x] = None
        game.board[21][3] = "J"
        game.board[21][5] = "J"
        game.board[22][3] = "J"
        game.board[22][5] = "J"
        self.assert_native_matches_python(game)

    def test_allow_180_both_modes(self) -> None:
        game = _empty_game("T")
        for allow_180 in (False, True):
            with self.subTest(allow_180=allow_180):
                self.assert_native_matches_python(game, allow_180=allow_180)

    def test_node_limit_boundaries(self) -> None:
        game = _empty_game("T")
        for max_nodes in (1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127):
            with self.subTest(max_nodes=max_nodes):
                self.assert_native_matches_python(game, max_nodes=max_nodes)

    def test_deterministic_random_boards(self) -> None:
        rng = random.Random(20260905)
        for case in range(80):
            piece = rng.choice(tuple("IJLOSTZ"))
            game = _empty_game(piece)
            filled_from = rng.randrange(10, game.height + 1)
            for y in range(filled_from, game.height):
                for x in range(game.width):
                    if rng.random() < 0.42:
                        game.board[y][x] = "J"
            game.x = rng.randrange(1, 6)
            game.y = rng.randrange(-2, 4)
            game.rotation = rng.randrange(4)
            with self.subTest(case=case, piece=piece):
                self.assert_native_matches_python(
                    game,
                    allow_180=bool(case & 1),
                    max_nodes=(32, 128, 512, 8_000)[case & 3],
                )

    def test_hold_and_direct_branch_order_matches_python_fallback(self) -> None:
        config = SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=4,
            srs_reachable=True,
            allow_180=False,
            reachability_node_limit=8_000,
        ).normalized()
        for seed in (1, 7, 97, 8100001):
            native_game = Game(seed)
            clear_reachability_cache()
            native_direct, native_hold_game, native_hold = _branch_groups(
                native_game,
                config,
                include_paths=False,
            )
            with patch.dict(os.environ, {"MINOFLUX_DISABLE_NATIVE_REACHABILITY": "1"}):
                python_game = Game(seed)
                clear_reachability_cache()
                python_direct, python_hold_game, python_hold = _branch_groups(
                    python_game,
                    config,
                    include_paths=False,
                )
            self.assertEqual(native_direct, python_direct)
            self.assertEqual(native_hold, python_hold)
            self.assertEqual(native_hold_game is None, python_hold_game is None)


if __name__ == "__main__":
    unittest.main()
