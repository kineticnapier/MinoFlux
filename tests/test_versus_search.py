from __future__ import annotations

import unittest

from minoflux_ai import (
    SearchConfig,
    VersusSearchConfig,
    choose_versus_action,
    clone_versus_match,
    run_versus_benchmark,
)
from minoflux_engine import VersusMatch


class _CountingScorer:
    def __init__(self) -> None:
        self.calls = 0

    def score_many(self, game, evaluations):
        self.calls += 1
        return tuple(float(index) for index, _evaluation in enumerate(evaluations))


class _GroupedCountingScorer:
    def __init__(self) -> None:
        self.group_calls = 0
        self.placement_calls = 0

    @staticmethod
    def _values(placements):
        return tuple(float(index) for index, _placement in enumerate(placements))

    def score_placement_groups(self, groups):
        self.group_calls += 1
        return tuple(self._values(placements) for _game, placements in groups)

    def score_placements(self, game, placements):
        self.placement_calls += 1
        return self._values(placements)

    def score_many(self, game, evaluations):
        self.placement_calls += 1
        return tuple(float(index) for index, _evaluation in enumerate(evaluations))


class _BatchStateScorer:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def score_matches(self, entries):
        self.batch_sizes.append(len(entries))
        return (0.0,) * len(entries)

    def score_match(self, match, root_side, to_move=None):
        raise AssertionError("batch hook should be used")


class VersusSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = VersusSearchConfig(
            placement_search=SearchConfig(
                allow_hold=False,
                lookahead_pieces=0,
                beam_width=1,
                srs_reachable=False,
            ),
            candidate_width=3,
            opponent_reply_width=1,
        )

    def test_clone_preserves_full_match_without_aliasing(self) -> None:
        match = VersusMatch(123)
        match.player.pending.enqueue(4, 2)
        cloned = clone_versus_match(match)
        cloned.player.pending.cancel(2)
        cloned.ai.game.board[-1][0] = "G"
        self.assertEqual(match.player.pending.pending_lines, 4)
        self.assertIsNone(match.ai.game.board[-1][0])
        self.assertEqual(cloned.player.pending.pending_lines, 2)

    def test_choice_reads_pending_and_does_not_mutate_match(self) -> None:
        match = VersusMatch(7)
        match.ai.pending.enqueue(5, 4)
        before_board = tuple(tuple(row) for row in match.ai.game.board)
        before_pending = match.ai.pending.pending_lines
        choice = choose_versus_action(match, "ai", config=self.config)
        self.assertIsNotNone(choice)
        self.assertEqual(tuple(tuple(row) for row in match.ai.game.board), before_board)
        self.assertEqual(match.ai.pending.pending_lines, before_pending)
        assert choice is not None
        self.assertEqual(choice.action.placement.piece, match.ai.game.current)

    def test_neural_scorer_is_used_for_root_and_reply_candidates(self) -> None:
        match = VersusMatch(71)
        root = _CountingScorer()
        reply = _CountingScorer()
        choice = choose_versus_action(
            match,
            "ai",
            config=self.config,
            scorer=root,
            opponent_scorer=reply,
        )
        self.assertIsNotNone(choice)
        self.assertGreater(root.calls, 0)
        self.assertGreater(reply.calls, 0)

    def test_grouped_reply_scorer_batches_all_root_candidates(self) -> None:
        match = VersusMatch(71)
        config = VersusSearchConfig(
            placement_search=SearchConfig(
                allow_hold=False,
                lookahead_pieces=0,
                beam_width=1,
                srs_reachable=False,
            ),
            candidate_width=3,
            opponent_reply_width=2,
        )
        root = _GroupedCountingScorer()
        reply = _GroupedCountingScorer()
        state = _BatchStateScorer()
        choice = choose_versus_action(
            match,
            "ai",
            config=config,
            scorer=root,
            opponent_scorer=reply,
            state_scorer=state,
        )
        self.assertIsNotNone(choice)
        self.assertEqual(root.group_calls, 1)
        self.assertEqual(reply.group_calls, 1)
        self.assertEqual(reply.placement_calls, 0)
        self.assertEqual(state.batch_sizes, [3, 6])

    def test_small_headless_benchmark_is_deterministic(self) -> None:
        first = run_versus_benchmark(
            1,
            max_turns=8,
            seed_base=99,
            player_config=self.config,
            ai_config=self.config,
        )
        second = run_versus_benchmark(
            1,
            max_turns=8,
            seed_base=99,
            player_config=self.config,
            ai_config=self.config,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.games, 1)
        self.assertLessEqual(first.per_game[0].turns, 8)
        result = first.to_dict()
        self.assertIn("playerSentPerPiece", result)
        self.assertIn("playerMeanCanceled", result)


if __name__ == "__main__":
    unittest.main()
