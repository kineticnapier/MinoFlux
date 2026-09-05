from __future__ import annotations

import unittest

from minoflux_ai.heuristic import PlacementEvaluation
from minoflux_ai.neural import NeuralValueConfig, encode_game_state
from minoflux_ai.search import SearchConfig, choose_search_action, rank_search_actions
from minoflux_engine import Game


class _RightmostScorer:
    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def score_many(
        self,
        game: Game,
        evaluations: list[PlacementEvaluation] | tuple[PlacementEvaluation, ...],
    ) -> tuple[float, ...]:
        self.calls += 1
        self.batch_sizes.append(len(evaluations))
        return tuple(float(evaluation.placement.x) for evaluation in evaluations)


class _BrokenScorer:
    def score_many(
        self,
        game: Game,
        evaluations: list[PlacementEvaluation] | tuple[PlacementEvaluation, ...],
    ) -> tuple[float, ...]:
        return (0.0,)


class NeuralValueScaffoldTests(unittest.TestCase):
    def test_state_encoder_has_stable_shapes(self) -> None:
        game = Game(seed=41)
        placement = game.legal_placements()[0]
        game.place(placement)
        config = NeuralValueConfig()
        encoded = encode_game_state(game, config)

        self.assertEqual(len(encoded.board), 24 * 10)
        self.assertEqual(len(encoded.context), config.context_size)
        self.assertEqual(config.context_size, 59)
        self.assertTrue(all(value in (0.0, 1.0) for value in encoded.board))
        self.assertGreater(sum(encoded.board), 0.0)

    def test_custom_scorer_sees_whole_branch_before_pruning(self) -> None:
        game = Game(seed=42)
        scorer = _RightmostScorer()
        config = SearchConfig(
            allow_hold=False,
            lookahead_pieces=0,
            beam_width=4,
            srs_reachable=False,
        )
        legal = game.legal_placements()
        choice = choose_search_action(game, config=config, scorer=scorer)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.action.placement.x, max(placement.x for placement in legal))
        self.assertEqual(scorer.calls, 1)
        self.assertEqual(scorer.batch_sizes, [len(legal)])
        self.assertEqual(choice.score, float(choice.action.placement.x))

    def test_custom_scorer_works_through_lookahead(self) -> None:
        scorer = _RightmostScorer()
        choice = choose_search_action(
            Game(seed=43),
            config=SearchConfig(
                allow_hold=True,
                lookahead_pieces=1,
                beam_width=2,
                srs_reachable=False,
            ),
            scorer=scorer,
        )
        self.assertIsNotNone(choice)
        self.assertGreater(scorer.calls, 1)

    def test_scorer_must_return_one_value_per_candidate(self) -> None:
        with self.assertRaisesRegex(ValueError, "Search scorer returned"):
            rank_search_actions(
                Game(seed=44),
                config=SearchConfig(
                    allow_hold=False,
                    lookahead_pieces=0,
                    beam_width=4,
                    srs_reachable=False,
                ),
                limit=2,
                scorer=_BrokenScorer(),
            )


if __name__ == "__main__":
    unittest.main()
