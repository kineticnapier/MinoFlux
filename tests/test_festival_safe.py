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
from minoflux_ai.festival_safe import (
    _garbage_excavation_metrics,
    _line_clear_score,
    score_festival_safe_placement,
)
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


def _garbage_base(*, gap: int = 4, lines: int = 4) -> list[list[str | None]]:
    board = _empty_board()
    for y in range(BOARD_HEIGHT - lines, BOARD_HEIGHT):
        for x in range(BOARD_WIDTH):
            if x != gap:
                board[y][x] = "G"
    return board


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
        self.assertLess(FESTIVAL_SAFE_CONFIG.garbage_well_scale, 0.25)
        self.assertLess(FESTIVAL_SAFE_CONFIG.garbage_channel_blockers, 0.0)
        self.assertLess(FESTIVAL_SAFE_CONFIG.garbage_channel_debt, 0.0)
        self.assertLess(FESTIVAL_SAFE_CONFIG.garbage_clear_debt, 0.0)
        self.assertLess(FESTIVAL_SAFE_CONFIG.garbage_surface_spread, 0.0)
        self.assertGreater(FESTIVAL_SAFE_CONFIG.garbage_rows_removed, 0.0)

    def test_garbage_excavation_metrics_track_a_buried_channel(self) -> None:
        board = _garbage_base()
        gap = 4

        open_metrics = _garbage_excavation_metrics(board)
        self.assertEqual(open_metrics.rows, 4)
        self.assertEqual(open_metrics.cells, 36)
        self.assertEqual(open_metrics.channel_blockers, 0)
        self.assertEqual(open_metrics.channel_debt, 0)
        self.assertEqual(open_metrics.clear_debt, 0)
        self.assertEqual(open_metrics.stack_height, 0)

        board[BOARD_HEIGHT - 5][gap] = "T"
        buried_metrics = _garbage_excavation_metrics(board)
        self.assertEqual(buried_metrics.channel_blockers, 1)
        self.assertEqual(buried_metrics.channel_debt, 9)
        self.assertEqual(buried_metrics.clear_debt, 9)
        self.assertEqual(buried_metrics.cover_cells, 1)
        self.assertEqual(buried_metrics.stack_height, 1)

    def test_channel_block_in_nearly_complete_row_is_cheaper_than_sparse_peak(self) -> None:
        gap = 4
        sparse = _garbage_base(gap=gap)
        sparse[BOARD_HEIGHT - 5][gap] = "T"

        nearly_clear = _garbage_base(gap=gap)
        row = nearly_clear[BOARD_HEIGHT - 5]
        for x in range(BOARD_WIDTH - 1):
            row[x] = "T"
        # Keep exactly one non-channel cell empty so this remains a live row.
        row[BOARD_WIDTH - 1] = None

        sparse_metrics = _garbage_excavation_metrics(sparse)
        clear_metrics = _garbage_excavation_metrics(nearly_clear)
        self.assertEqual(sparse_metrics.channel_blockers, 1)
        self.assertEqual(clear_metrics.channel_blockers, 1)
        self.assertGreater(sparse_metrics.channel_debt, clear_metrics.channel_debt)
        self.assertGreater(sparse_metrics.clear_debt, clear_metrics.clear_debt)

    def test_garbage_surface_metrics_distinguish_flat_layer_from_mountain(self) -> None:
        gap = 4
        flat = _garbage_base(gap=gap)
        flat_row = flat[BOARD_HEIGHT - 5]
        for x in range(BOARD_WIDTH):
            if x not in (gap, BOARD_WIDTH - 1):
                flat_row[x] = "J"

        mountain = _garbage_base(gap=gap)
        for x in (2, 3, 4, 5, 6):
            mountain[BOARD_HEIGHT - 5][x] = "J"
        for x in (3, 4, 5):
            mountain[BOARD_HEIGHT - 6][x] = "L"

        flat_metrics = _garbage_excavation_metrics(flat)
        mountain_metrics = _garbage_excavation_metrics(mountain)
        self.assertGreater(mountain_metrics.clear_debt, flat_metrics.clear_debt)
        self.assertGreater(mountain_metrics.surface_spread, flat_metrics.surface_spread)
        self.assertGreater(mountain_metrics.surface_variation, flat_metrics.surface_variation)
        self.assertGreater(mountain_metrics.stack_height, flat_metrics.stack_height)

    def test_garbage_mode_avoids_covering_the_access_channel_when_rows_are_sparse(self) -> None:
        game = Game(4)
        game.board = _garbage_base()
        game.current = "O"

        clear_channel = _placement("O", 0, BOARD_HEIGHT - 6, 0)
        cover_channel = _placement("O", 2, BOARD_HEIGHT - 6, 0)
        self.assertGreater(
            score_festival_safe_placement(game, clear_channel),
            score_festival_safe_placement(game, cover_channel),
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
