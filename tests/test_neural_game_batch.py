from __future__ import annotations

from minoflux_ai.search import SearchConfig, choose_search_action, choose_search_actions_batch
from minoflux_engine import Game


class _PlacementBatchScorer:
    def __init__(self) -> None:
        self.group_calls = 0

    def score_placements(self, game, placements):
        return tuple(
            float(placement.x)
            + 0.01 * float(placement.rotation)
            + 0.0001 * float(placement.y)
            for placement in placements
        )

    def score_placement_groups(self, groups):
        self.group_calls += 1
        return tuple(
            self.score_placements(game, placements)
            for game, placements in groups
        )

    def score_many(self, game, evaluations):
        return self.score_placements(
            game,
            tuple(evaluation.placement for evaluation in evaluations),
        )


def test_multi_game_batch_matches_individual_root_choices() -> None:
    config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        srs_reachable=False,
    )
    seeds = (7001, 7002, 7003, 7004)

    expected = []
    for seed in seeds:
        scorer = _PlacementBatchScorer()
        expected.append(choose_search_action(Game(seed), config=config, scorer=scorer))

    scorer = _PlacementBatchScorer()
    actual = choose_search_actions_batch(
        tuple(Game(seed) for seed in seeds),
        config=config,
        scorer=scorer,
    )

    assert scorer.group_calls == 1
    assert len(actual) == len(expected)
    for single, batched in zip(expected, actual):
        assert single is not None
        assert batched is not None
        assert batched.action.to_dict() == single.action.to_dict()
        assert batched.score == single.score
