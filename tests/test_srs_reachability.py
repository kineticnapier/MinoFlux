from __future__ import annotations

import unittest

from minoflux_ai import (
    REPLAY_FORMAT,
    SearchConfig,
    apply_replay_step,
    rank_search_actions,
    reachable_placements,
    record_heuristic_game,
)
from minoflux_engine import (
    Game,
    Placement,
    T_SPIN_DOUBLE,
    T_SPIN_MINI_SINGLE,
    T_SPIN_SINGLE,
    T_SPIN_TRIPLE,
)
from minoflux_engine.spin import t_spin_event


class ReachabilityTests(unittest.TestCase):
    def test_empty_board_paths_replay_with_engine_srs(self) -> None:
        source = Game(123)
        placements = reachable_placements(source)
        self.assertTrue(placements)
        for placement in placements:
            game = Game(123)
            self.assertEqual(game.current, placement.piece)
            for command in placement.path[:-1]:
                if command == "left":
                    self.assertTrue(game.move_left())
                elif command == "right":
                    self.assertTrue(game.move_right())
                elif command == "down":
                    self.assertTrue(game.soft_drop())
                elif command == "cw":
                    self.assertTrue(game.rotate_cw())
                elif command == "ccw":
                    self.assertTrue(game.rotate_ccw())
                elif command == "180":
                    self.assertTrue(game.rotate_180())
                else:
                    self.fail(f"Unknown command {command}")
            self.assertEqual(game.ghost_y(), placement.y)
            self.assertEqual(game.x, placement.x)
            self.assertEqual(game.rotation, placement.rotation)

    def test_srs_search_records_paths(self) -> None:
        game = Game(5)
        ranked = rank_search_actions(
            game,
            config=SearchConfig(
                allow_hold=True,
                lookahead_pieces=0,
                beam_width=4,
                srs_reachable=True,
            ),
            limit=8,
        )
        self.assertTrue(ranked)
        self.assertTrue(all(action.placement.path[-1] == "hard_drop" for action, _ in ranked))


class TSpinTests(unittest.TestCase):
    @staticmethod
    def _single_board(*, mini: bool) -> Game:
        game = Game(1)
        game.board = [[None] * game.width for _ in range(game.height)]
        game.current = "T"
        game.x, game.y, game.rotation = 3, 21, 0
        game.board[22] = ["J"] * game.width
        for x in (3, 4, 5):
            game.board[22][x] = None
        if mini:
            game.board[21][3] = "J"
            game.board[23][3] = "J"
            game.board[23][5] = "J"
        else:
            game.board[21][3] = "J"
            game.board[21][5] = "J"
            game.board[23][3] = "J"
        return game

    def test_full_t_spin_single(self) -> None:
        game = self._single_board(mini=False)
        cells = game.cells("T", 3, 21, 0)
        result = game.place(
            Placement(
                "T", 3, 21, 0, cells,
                path=("cw", "ccw", "hard_drop"),
                last_move_was_rotation=True,
                rotation_kick_index=0,
                rotation_from=1,
                rotation_to=0,
            )
        )
        self.assertEqual(result.spin, T_SPIN_SINGLE)
        self.assertEqual(result.lines, 1)
        self.assertEqual(result.attack, 2)

    def test_t_spin_mini_single(self) -> None:
        game = self._single_board(mini=True)
        cells = game.cells("T", 3, 21, 0)
        result = game.place(
            Placement(
                "T", 3, 21, 0, cells,
                path=("ccw", "cw", "hard_drop"),
                last_move_was_rotation=True,
                rotation_kick_index=0,
                rotation_from=3,
                rotation_to=0,
            )
        )
        self.assertEqual(result.spin, T_SPIN_MINI_SINGLE)
        self.assertEqual(result.lines, 1)

    def test_fifth_kick_upgrades_mini(self) -> None:
        game = self._single_board(mini=True)
        cells = game.cells("T", 3, 21, 0)
        result = game.place(
            Placement(
                "T", 3, 21, 0, cells,
                last_move_was_rotation=True,
                rotation_kick_index=4,
                rotation_from=3,
                rotation_to=0,
            )
        )
        self.assertEqual(result.spin, T_SPIN_SINGLE)

    def test_line_event_names(self) -> None:
        self.assertEqual(t_spin_event("full", 2), T_SPIN_DOUBLE)
        self.assertEqual(t_spin_event("full", 3), T_SPIN_TRIPLE)


class ReplayV3Tests(unittest.TestCase):
    def test_replay_contains_route_metadata_and_rebuilds(self) -> None:
        summary, replay = record_heuristic_game(
            7,
            max_pieces=12,
            search_config=SearchConfig(lookahead_pieces=0, beam_width=2, srs_reachable=True),
        )
        self.assertEqual(replay.to_dict()["format"], REPLAY_FORMAT)
        self.assertTrue(any(step.path for step in replay.steps))
        game = Game(replay.seed)
        for step in replay.steps:
            apply_replay_step(game, step, strict=True)
        self.assertEqual(game.pieces_placed, summary.pieces)
        self.assertEqual(game.attack, summary.attack)


if __name__ == "__main__":
    unittest.main()
