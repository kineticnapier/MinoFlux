from __future__ import annotations

import unittest

from minoflux_ai import (
    SearchConfig,
    apply_search_action,
    choose_search_action,
    rank_search_actions,
    run_heuristic_benchmark,
)
from minoflux_engine import Game


class SearchTests(unittest.TestCase):
    def test_hold_and_direct_candidates_are_generated(self) -> None:
        game = Game(seed=10)
        current = game.current
        next_piece = game.queue[0]
        actions = rank_search_actions(
            game,
            config=SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=4),
        )
        self.assertTrue(any(not action.use_hold and action.placement.piece == current for action, _ in actions))
        self.assertTrue(any(action.use_hold and action.placement.piece == next_piece for action, _ in actions))
        self.assertEqual(game.current, current)
        self.assertIsNone(game.hold_piece)

    def test_hold_can_be_disabled(self) -> None:
        game = Game(seed=11)
        actions = rank_search_actions(
            game,
            config=SearchConfig(allow_hold=False, lookahead_pieces=0, beam_width=4),
        )
        self.assertTrue(actions)
        self.assertFalse(any(action.use_hold for action, _ in actions))

    def test_one_piece_lookahead_is_deterministic(self) -> None:
        config = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4, discount=0.9)
        first = choose_search_action(Game(seed=12), config=config)
        second = choose_search_action(Game(seed=12), config=config)
        self.assertIsNotNone(first)
        self.assertEqual(first, second)
        assert first is not None
        self.assertGreaterEqual(len(first.path), 2)

    def test_selected_action_can_be_applied(self) -> None:
        game = Game(seed=13)
        choice = choose_search_action(
            game,
            config=SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=4),
        )
        self.assertIsNotNone(choice)
        assert choice is not None
        result = apply_search_action(game, choice.action)
        self.assertEqual(game.pieces_placed, 1)
        self.assertEqual(result, game.last_lock)

    def test_benchmark_records_search_configuration(self) -> None:
        config = SearchConfig(allow_hold=True, lookahead_pieces=1, beam_width=3, discount=0.8)
        result = run_heuristic_benchmark(
            games=1,
            max_pieces=5,
            seed_base=14,
            search_config=config,
            record_best_replay=True,
        )
        self.assertEqual(result.search_config, config.normalized())
        self.assertIsNotNone(result.best_replay)
        assert result.best_replay is not None
        self.assertEqual(result.best_replay.search_config, config.normalized().to_dict())


if __name__ == "__main__":
    unittest.main()
