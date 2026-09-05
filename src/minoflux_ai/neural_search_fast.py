from __future__ import annotations

from heapq import nlargest
import time
from typing import Sequence

from minoflux_engine import Game

from . import search as _search
from .neural_fast import native_neural_fast_path_available
from .reachability_native import (
    NativePlacementRecords,
    native_pathless_available,
    reachable_placement_records_native,
)


_ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH = _search.choose_search_actions_batch


def _scorer_config(scorer):
    config = getattr(scorer, "config", None)
    if config is not None:
        return config
    evaluator = getattr(scorer, "evaluator", None)
    return getattr(evaluator, "config", None)


def _native_record_scorer(scorer):
    direct = getattr(scorer, "score_native_record_groups", None)
    if callable(direct):
        return direct

    evaluator = getattr(scorer, "evaluator", None)
    method = getattr(evaluator, "score_native_record_groups", None)
    if not callable(method):
        return None

    # The evaluate --profile wrapper intentionally records scorer totals. Keep
    # those counters identical while bypassing its Placement-only method.
    if not all(
        hasattr(scorer, name)
        for name in ("seconds", "calls", "groups", "states")
    ):
        return method

    def profiled(groups):
        started = time.perf_counter()
        try:
            return method(groups)
        finally:
            scorer.seconds += time.perf_counter() - started
            scorer.calls += 1
            scorer.groups += len(groups)
            scorer.states += sum(len(batch) for _game, batch in groups)

    return profiled


def _native_board_supported(game: Game) -> bool:
    return (
        0 < game.width <= 64
        and 0 < game.height
        and game.width * game.height <= 256
    )


def _rank_native_record_branches(
    branches: Sequence[
        tuple[bool, Game, NativePlacementRecords, Sequence[float]]
    ],
    limit: int,
):
    total = 0
    score_values: list[float] = []
    for _use_hold, _game, batch, values in branches:
        if len(values) != len(batch):
            raise ValueError(
                f"Search scorer returned {len(values)} values for {len(batch)} placements"
            )
        total += len(batch)
        score_values.extend(float(value) for value in values)
    if total == 0:
        return ()

    count = max(0, int(limit))
    if count == 0:
        return ()
    cutoff = (
        nlargest(count, score_values)[-1]
        if count < len(score_values)
        else float("-inf")
    )

    materialized: list[
        tuple[bool, Game, tuple[object, ...], tuple[float, ...]]
    ] = []
    for use_hold, branch_game, batch, values in branches:
        placements = []
        kept_values = []
        for index, value in enumerate(values):
            score = float(value)
            if score >= cutoff:
                placements.append(batch.materialize(index))
                kept_values.append(score)
        if placements:
            materialized.append(
                (use_hold, branch_game, tuple(placements), tuple(kept_values))
            )

    return _search._rank_precomputed_actions(tuple(materialized), count)


def choose_search_actions_batch(
    games: Sequence[Game],
    weights=_search.DEFAULT_WEIGHTS,
    config=_search.DEFAULT_SEARCH_CONFIG,
    *,
    scorer=None,
):
    """Native-record lookahead=0 scorer with the existing search as fallback."""

    cfg = config.normalized()
    score_records = _native_record_scorer(scorer) if scorer is not None else None
    scorer_config = _scorer_config(scorer) if scorer is not None else None
    if (
        cfg.lookahead_pieces != 0
        or not cfg.srs_reachable
        or not callable(score_records)
        or scorer_config is None
        or not native_pathless_available()
        or not native_neural_fast_path_available()
        or any(not _native_board_supported(game) for game in games)
    ):
        return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
            games,
            weights,
            cfg,
            scorer=scorer,
        )

    queue_length = scorer_config.queue_length
    prepared = []
    groups: list[tuple[Game, NativePlacementRecords]] = []
    group_keys: list[tuple[int, bool]] = []

    for index, game in enumerate(games):
        if game.game_over:
            prepared.append((NativePlacementRecords.empty(game.current), None, None))
            continue

        # Short queues are rare near synthetic/test boundaries. Preserve the
        # existing generic fallback instead of changing its semantics.
        if len(game.queue) < queue_length + 1:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        held = _search._held_search_game(game) if cfg.allow_hold else None
        if held is not None and len(held.queue) < queue_length + 1:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        direct = reachable_placement_records_native(
            game,
            allow_180=cfg.allow_180,
            max_nodes=cfg.reachability_node_limit,
        )
        if direct is None:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )
        held_records = (
            reachable_placement_records_native(
                held,
                allow_180=cfg.allow_180,
                max_nodes=cfg.reachability_node_limit,
                rows=direct.rows,
            )
            if held is not None
            else None
        )
        if held is not None and held_records is None:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        prepared.append((direct, held, held_records))
        if direct:
            groups.append((game, direct))
            group_keys.append((index, False))
        if held is not None and held_records:
            groups.append((held, held_records))
            group_keys.append((index, True))

    grouped_values = score_records(tuple(groups)) if groups else ()
    if len(grouped_values) != len(groups):
        raise ValueError("Search scorer returned the wrong number of placement groups")
    values_by_key = {
        key: values
        for key, values in zip(group_keys, grouped_values)
    }

    choices = []
    for index, game in enumerate(games):
        direct, held, held_records = prepared[index]
        branches = []
        if direct:
            branches.append((False, game, direct, values_by_key[(index, False)]))
        if held is not None and held_records:
            branches.append(
                (True, held, held_records, values_by_key[(index, True)])
            )
        ranked = _rank_native_record_branches(tuple(branches), 1)
        if not ranked:
            choices.append(None)
            continue
        action, evaluation = ranked[0]
        choices.append(
            _search.SearchChoice(
                action,
                evaluation.score,
                evaluation,
                (action,),
            )
        )
    return tuple(choices)


def install_neural_search_fast_path() -> None:
    _search.choose_search_actions_batch = choose_search_actions_batch
