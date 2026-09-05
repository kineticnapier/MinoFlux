from __future__ import annotations

import unittest

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    FESTIVAL_SAFE_CONFIG,
    FESTIVAL_SAFE_SCORER,
    SearchConfig,
    choose_search_action,
)
from minoflux_ai.bitboard import placement_cells, place_and_clear_row_masks, board_row_masks
from minoflux_ai.festival_safe import _line_clear_score, score_festival_safe_placement
from minoflux_ai.reachability import reachable_placements
from minoflux_engine import BOARD_HEIGHT, BOARD_WIDTH, Game, Placement


def _placement(piece: str, x: int, y: int, rotation: int) -> Placement:
    return Placement(
        piece=piece,
        x=x,
        y=y,
        rotation=rotation,
        cells=placement_cells(piece, x, y, rotation),
    )


def _empty_board() -> list[list[str | None]]:
    return [[None] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]


class FestivalSafeTests(unittest.TestCase):
    def test_default_weights_are_untouched(self) -> None:
        self.assertEqual(DEFAULT_WEIGHTS, type(DEFAULT_WEIGHTS)())

    def test_empty_board_scores_every_exact_srs_candidate_without_mutation(self) -> None:
        game = Game(12345)
        before = game.snapshot()
        placements = reachable_placements(game, include_paths=False)
        values = FESTIVAL_SAFE_SCORER.score_placements(game, placements)
        self.assertTrue(placements)
        self.assertEqual(len(values), len(placements))
        self.assertEqual(game.snapshot(), before)

    def test_quad_completion_is_selected(self) -> None:
        game = Game(1)
        game.board = _empty_board()
        for y in range(BOARD_HEIGHT - 4, BOARD_HEIGHT):
            for x in range(BOARD_WIDTH - 1):
                game.board[y][x] = "G"
        game.current = "I"
        game.x, game.y, game.rotation = 3, 1, 0
        game.hold_used = True
        game.game_over = False

        choice = choose_search_action(
            game,
            DEFAULT_WEIGHTS,
            SearchConfig(
                allow_hold=False,
                lookahead_pieces=0,
                beam_width=1,
                srs_reachable=True,
            ),
            scorer=FESTIVAL_SAFE_SCORER,
        )
        self.assertIsNotNone(choice)
        assert choice is not None
        after_rows, lines, topped_out = place_and_clear_row_masks(
            board_row_masks(game.board),
            choice.action.placement,
            width=BOARD_WIDTH,
        )
        self.assertFalse(topped_out)
        self.assertEqual(lines, 4)
        self.assertFalse(any(after_rows[-4:]))

    def test_healthy_board_preserves_right_well_instead_of_weak_double(self) -> None:
        game = Game(2)
        game.board = _empty_board()
        for y in range(BOARD_HEIGHT - 2, BOARD_HEIGHT):
            for x in range(BOARD_WIDTH - 1):
                game.board[y][x] = "J"
        game.current = "I"

        vertical_in_well = _placement("I", 7, BOARD_HEIGHT - 4, 1)
        horizontal_on_stack = _placement("I", 0, BOARD_HEIGHT - 4, 0)
        weak_clear = score_festival_safe_placement(game, vertical_in_well)
        preserve = score_festival_safe_placement(game, horizontal_on_stack)
        self.assertGreater(preserve, weak_clear)

    def test_danger_mode_allows_single_double_triple(self) -> None:
        for lines in (1, 2, 3):
            self.assertLess(
                _line_clear_score(lines, danger=False, garbage=False),
                0.0,
            )
            self.assertGreater(
                _line_clear_score(lines, danger=True, garbage=False),
                0.0,
            )

    def test_garbage_mode_prefers_digging_over_quad_discipline(self) -> None:
        for lines in (1, 2, 3):
            self.assertGreater(
                _line_clear_score(lines, danger=False, garbage=True),
                0.0,
            )
        self.assertLess(
            FESTIVAL_SAFE_CONFIG.garbage_well_scale,
            0.25,
        )

    def test_new_holes_are_punished_more_than_existing_holes(self) -> None:
        self.assertLess(FESTIVAL_SAFE_CONFIG.new_holes, FESTIVAL_SAFE_CONFIG.holes)
        self.assertGreater(FESTIVAL_SAFE_CONFIG.removed_holes, 0.0)

    def test_topout_is_overwhelmingly_worse_than_safe_placement(self) -> None:
        game = Game(3)
        game.board = _empty_board()
        game.current = "O"
        topout = _placement("O", 3, 0, 0)
        safe = _placement("O", 3, BOARD_HEIGHT - 2, 0)
        self.assertEqual(
            score_festival_safe_placement(game, topout),
            FESTIVAL_SAFE_CONFIG.topout,
        )
        self.assertGreater(score_festival_safe_placement(game, safe), FESTIVAL_SAFE_CONFIG.topout)

    def test_high_stack_has_strong_nonlinear_penalty(self) -> None:
        self.assertLess(FESTIVAL_SAFE_CONFIG.danger_height_quadratic, 0.0)
        self.assertLess(FESTIVAL_SAFE_CONFIG.critical_height_penalty, 0.0)
        self.assertLess(FESTIVAL_SAFE_CONFIG.max_height, 0.0)


if __name__ == "__main__":
    unittest.main()
