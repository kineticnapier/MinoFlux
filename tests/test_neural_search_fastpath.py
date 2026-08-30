from __future__ import annotations

from unittest.mock import patch

from minoflux_ai import DEFAULT_WEIGHTS
from minoflux_ai.search import SearchConfig, rank_search_actions
from minoflux_engine import Game


class _FastIndexScorer:
    def __init__(self) -> None:
        self.placement_calls = 0
        self.evaluation_calls = 0
        self.placements_seen = 0

    def score_placements(self, game, placements):
        self.placement_calls += 1
        self.placements_seen += len(placements)
        return tuple(float(index) for index, _placement in enumerate(placements))

    def score_many(self, game, evaluations):
        self.evaluation_calls += 1
        raise AssertionError("placement-only fast path should avoid score_many")


class _FastTieScorer:
    def score_placements(self, game, placements):
        return (1.0,) * len(placements)

    def score_many(self, game, evaluations):
        raise AssertionError("placement-only fast path should avoid score_many")


class _SlowTieScorer:
    def score_many(self, game, evaluations):
        return (1.0,) * len(evaluations)


def test_fast_scorer_prunes_before_expensive_heuristic_features() -> None:
    import minoflux_ai.search as search_module

    game = Game(12345)
    config = SearchConfig(
        allow_hold=False,
        lookahead_pieces=0,
        beam_width=4,
        srs_reachable=False,
    )
    scorer = _FastIndexScorer()
    feature_batch_sizes: list[int] = []
    original = search_module.rank_placements

    def recording_rank(game, weights=DEFAULT_WEIGHTS, *, placements=None, limit=None):
        materialized = tuple(placements) if placements is not None else None
        if materialized is not None:
            feature_batch_sizes.append(len(materialized))
        return original(game, weights, placements=materialized, limit=limit)

    with patch.object(search_module, "rank_placements", side_effect=recording_rank):
        ranked = rank_search_actions(
            game,
            DEFAULT_WEIGHTS,
            config,
            limit=1,
            scorer=scorer,
        )

    assert ranked
    assert scorer.placement_calls == 1
    assert scorer.evaluation_calls == 0
    assert scorer.placements_seen > 1
    assert feature_batch_sizes == [1]


def test_fast_scorer_keeps_existing_tie_break_order() -> None:
    config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        srs_reachable=False,
    )
    fast = rank_search_actions(
        Game(777),
        DEFAULT_WEIGHTS,
        config,
        limit=1,
        scorer=_FastTieScorer(),
    )
    slow = rank_search_actions(
        Game(777),
        DEFAULT_WEIGHTS,
        config,
        limit=1,
        scorer=_SlowTieScorer(),
    )

    assert fast
    assert slow
    assert fast[0][0].to_dict() == slow[0][0].to_dict()
    assert fast[0][1].score == slow[0][1].score
