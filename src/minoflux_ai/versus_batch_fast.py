from __future__ import annotations

import os
from typing import Literal, Sequence

from minoflux_engine import Game

from .heuristic import HeuristicWeights, PlacementEvaluation
from .search import SearchAction, SearchConfig, SearchScorer
from . import versus_search as _versus

_DISABLE_ENV = "MINOFLUX_DISABLE_VERSUS_BATCH_FUSION"
_ORIGINAL_CHOOSE_VERSUS_ACTIONS_BATCH = _versus.choose_versus_actions_batch

_Stage = Literal["root", "reply"]
_RankItem = tuple[
    int,
    _Stage,
    Game,
    HeuristicWeights,
    SearchConfig,
    int,
    SearchScorer | None,
]


def versus_batch_fusion_enabled() -> bool:
    disabled = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return disabled not in {"1", "true", "yes", "on"}


def _rank_mixed_request_games(
    items: Sequence[_RankItem],
    output_count: int,
    *,
    profile=None,
) -> tuple[tuple[tuple[SearchAction, PlacementEvaluation], ...], ...]:
    """Rank root/reply games with one placement forward per compatible scorer.

    Candidate limits remain per request. Exact reachability and existing neural
    tie-breaking are unchanged; only compatible score-placement groups are
    concatenated before the model forward.
    """

    output: list[tuple[tuple[SearchAction, PlacementEvaluation], ...]] = [
        () for _ in range(output_count)
    ]
    groups: dict[
        tuple[int, HeuristicWeights, SearchConfig],
        list[_RankItem],
    ] = {}
    scorers: dict[tuple[int, HeuristicWeights, SearchConfig], SearchScorer | None] = {}
    for item in items:
        output_index, _stage, _game, weights, config, _limit, scorer = item
        if output_index < 0 or output_index >= output_count:
            raise IndexError(output_index)
        cfg = config.normalized()
        key = (id(scorer), weights, cfg)
        groups.setdefault(key, []).append(
            (
                output_index,
                item[1],
                item[2],
                weights,
                cfg,
                item[5],
                scorer,
            )
        )
        scorers[key] = scorer

    for key, members in groups.items():
        _scorer_id, weights, config = key
        scorer = scorers[key]
        score_groups = getattr(scorer, "score_placement_groups", None) if scorer is not None else None
        score_placements = getattr(scorer, "score_placements", None) if scorer is not None else None
        if not callable(score_groups) or not callable(score_placements):
            for output_index, stage, game, _weights, _config, limit, _scorer in members:
                started = _versus.profile_timer_start(profile)
                output[output_index] = _versus.rank_search_actions(
                    game,
                    weights,
                    config,
                    limit=limit,
                    scorer=scorer,
                    _profile=profile,
                    _ranking_only=True,
                )
                _versus.record_profile_elapsed(
                    profile,
                    "root_placement_generation" if stage == "root" else "opponent_placement_generation",
                    started,
                )
            continue

        prepared: list[
            tuple[
                int,
                _Stage,
                int,
                Game,
                tuple[object, ...],
                Game | None,
                tuple[object, ...],
            ]
        ] = []
        score_inputs: list[tuple[Game, Sequence[object]]] = []
        score_keys: list[tuple[int, bool]] = []

        for member_index, (
            output_index,
            stage,
            game,
            _weights,
            _config,
            limit,
            _scorer,
        ) in enumerate(members):
            started = _versus.profile_timer_start(profile)
            direct, held, held_placements = _versus._branch_groups(
                game,
                config,
                include_paths=True,
                _profile=profile,
            )
            _versus.record_profile_elapsed(
                profile,
                "root_placement_generation" if stage == "root" else "opponent_placement_generation",
                started,
            )
            prepared.append(
                (
                    output_index,
                    stage,
                    max(0, int(limit)),
                    game,
                    direct,
                    held,
                    held_placements,
                )
            )
            if direct:
                score_inputs.append((game, direct))
                score_keys.append((member_index, False))
            if held is not None and held_placements:
                score_inputs.append((held, held_placements))
                score_keys.append((member_index, True))

        scoring_started = _versus.profile_timer_start(profile)
        grouped_values = score_groups(tuple(score_inputs)) if score_inputs else ()
        _versus.record_profile_elapsed(
            profile,
            "neural_placement_scoring",
            scoring_started,
            calls=len(score_inputs),
        )
        if len(grouped_values) != len(score_inputs):
            raise ValueError("Search scorer returned the wrong number of placement groups")
        values_by_key = {
            group_key: values
            for group_key, values in zip(score_keys, grouped_values)
        }

        for member_index, (
            output_index,
            _stage,
            limit,
            game,
            direct,
            held,
            held_placements,
        ) in enumerate(prepared):
            branches: list[
                tuple[bool, Game, Sequence[object], Sequence[float]]
            ] = []
            if direct:
                branches.append(
                    (False, game, direct, values_by_key[(member_index, False)])
                )
            if held is not None and held_placements:
                branches.append(
                    (True, held, held_placements, values_by_key[(member_index, True)])
                )
            output[output_index] = _versus._rank_precomputed_actions(
                tuple(branches),
                limit,
                _profile=profile,
                _ranking_only=True,
            )

    return tuple(output)


def _state_values_combined(
    requests: Sequence[_versus.VersusSearchRequest],
    root_entries: Sequence[
        Sequence[tuple[object, _versus.SideName, _versus.SideName | None]]
    ],
    reply_entries: Sequence[
        Sequence[tuple[object, _versus.SideName, _versus.SideName | None]]
    ],
) -> tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]:
    """Evaluate root and reply match states in one forward per state scorer."""

    output: list[tuple[tuple[float, ...], tuple[float, ...]]] = [
        ((), ()) for _ in requests
    ]
    groups: dict[
        int,
        tuple[
            _versus.VersusStateScorer,
            list[tuple[int, int, int]],
            list[tuple[object, _versus.SideName, _versus.SideName | None]],
        ],
    ] = {}

    for request_index, (request, roots, replies) in enumerate(
        zip(requests, root_entries, reply_entries)
    ):
        root_count = len(roots)
        reply_count = len(replies)
        scorer = request.state_scorer
        if scorer is None:
            output[request_index] = (
                (0.0,) * root_count,
                (0.0,) * reply_count,
            )
            continue
        key = id(scorer)
        group = groups.get(key)
        if group is None:
            group = (scorer, [], [])
            groups[key] = group
        _group_scorer, slices, flat_entries = group
        slices.append((request_index, root_count, reply_count))
        flat_entries.extend(roots)
        flat_entries.extend(replies)

    for scorer, slices, flat_entries in groups.values():
        values = _versus._state_values(scorer, tuple(flat_entries), _stage=None)
        offset = 0
        for request_index, root_count, reply_count in slices:
            root_end = offset + root_count
            reply_end = root_end + reply_count
            output[request_index] = (
                values[offset:root_end],
                values[root_end:reply_end],
            )
            offset = reply_end
        if offset != len(values):
            raise AssertionError("Combined versus value scorer slice mismatch")
    return tuple(output)


def choose_versus_actions_batch(
    requests: Sequence[_versus.VersusSearchRequest],
) -> tuple[_versus.VersusChoice | None, ...]:
    """Exact versus batch search with fused root/reply neural forward waves."""

    if not versus_batch_fusion_enabled() or len(requests) <= 1:
        return _ORIGINAL_CHOOSE_VERSUS_ACTIONS_BATCH(requests)
    if not requests:
        return ()

    profile = _versus.active_versus_profile()
    search_started = _versus.profile_timer_start(profile)
    metric_cache: _versus._BoardMetricsCache = {}
    normalized = tuple(
        _versus.VersusSearchRequest(
            match=request.match,
            side_name=request.side_name,
            heuristic_weights=request.heuristic_weights,
            config=request.config.normalized(),
            versus_weights=request.versus_weights,
            scorer=request.scorer,
            opponent_scorer=request.opponent_scorer,
            opponent_heuristic_weights=request.opponent_heuristic_weights,
            state_scorer=request.state_scorer,
        )
        for request in requests
    )
    count = len(normalized)

    rank_items: list[_RankItem] = []
    for request_index, request in enumerate(normalized):
        rank_items.append(
            (
                request_index,
                "root",
                _versus._side(request.match, request.side_name).game,
                request.heuristic_weights,
                request.config.placement_search,
                request.config.candidate_width,
                request.scorer,
            )
        )
        if request.config.opponent_reply_width > 0:
            opponent_name = _versus._opponent_name(request.side_name)
            opponent_game = _versus._side(request.match, opponent_name).game
            if not opponent_game.game_over:
                reply_weights = (
                    request.heuristic_weights
                    if request.opponent_heuristic_weights is None
                    else request.opponent_heuristic_weights
                )
                rank_items.append(
                    (
                        count + request_index,
                        "reply",
                        opponent_game,
                        reply_weights,
                        request.config.placement_search,
                        request.config.opponent_reply_width,
                        request.opponent_scorer,
                    )
                )

    ranked = _rank_mixed_request_games(
        tuple(rank_items),
        count * 2,
        profile=profile,
    )
    root_ranked = ranked[:count]
    shared_replies = ranked[count:]

    root_candidates: list[
        list[
            tuple[
                SearchAction,
                PlacementEvaluation,
                object,
                object,
            ]
        ]
    ] = []
    root_entries: list[
        list[tuple[object, _versus.SideName, _versus.SideName | None]]
    ] = []
    reply_root_indices: list[list[int]] = []

    for request, request_ranked in zip(normalized, root_ranked):
        opponent_name = _versus._opponent_name(request.side_name)
        candidates = []
        entries = []
        live_indices = []
        for action, evaluation in request_ranked:
            after, resolution = _versus._simulate_action(
                request.match,
                request.side_name,
                action,
                _stage="root",
                _profile=profile,
            )
            candidate_index = len(candidates)
            candidates.append((action, evaluation, after, resolution))
            entries.append((after, request.side_name, opponent_name))
            if (
                request.config.opponent_reply_width > 0
                and after.winner is None
                and not _versus._side(after, opponent_name).game.game_over
            ):
                live_indices.append(candidate_index)
        root_candidates.append(candidates)
        root_entries.append(entries)
        reply_root_indices.append(live_indices)

    reply_candidates: list[list[tuple[int, SearchAction, PlacementEvaluation, object, object]]] = []
    reply_entries: list[
        list[tuple[object, _versus.SideName, _versus.SideName | None]]
    ] = []
    for request, live_indices, candidates, replies in zip(
        normalized,
        reply_root_indices,
        root_candidates,
        shared_replies,
    ):
        opponent_name = _versus._opponent_name(request.side_name)
        request_replies = []
        request_entries = []
        for root_index in live_indices:
            after = candidates[root_index][2]
            for reply, reply_evaluation in replies:
                replied, resolution = _versus._simulate_action(
                    after,
                    opponent_name,
                    reply,
                    _stage="reply",
                    _profile=profile,
                )
                request_replies.append(
                    (root_index, reply, reply_evaluation, replied, resolution)
                )
                request_entries.append((replied, request.side_name, request.side_name))
        reply_candidates.append(request_replies)
        reply_entries.append(request_entries)

    combined_values = _state_values_combined(
        normalized,
        root_entries,
        reply_entries,
    )

    base_scores: list[list[float]] = []
    for request, candidates, (root_values, _reply_values) in zip(
        normalized,
        root_candidates,
        combined_values,
    ):
        base_scores.append(
            [
                _versus.score_versus_state(
                    after,
                    request.side_name,
                    weights=request.versus_weights,
                    resolution=resolution,
                    solo_score=evaluation.score,
                    state_value=state_value,
                    path_length=len(action.placement.path),
                    action_side=request.side_name,
                    _metric_cache=metric_cache,
                    _profile=profile,
                )
                for (action, evaluation, after, resolution), state_value in zip(
                    candidates,
                    root_values,
                )
            ]
        )

    aggregation_started = _versus.profile_timer_start(profile)
    choices: list[_versus.VersusChoice | None] = []
    for request, candidates, scores, replies, (_root_values, reply_values) in zip(
        normalized,
        root_candidates,
        base_scores,
        reply_candidates,
        combined_values,
    ):
        opponent_name = _versus._opponent_name(request.side_name)
        worst_by_root: dict[int, tuple[float, SearchAction]] = {}
        for (
            root_index,
            reply,
            reply_evaluation,
            replied,
            reply_resolution,
        ), state_value in zip(replies, reply_values):
            reply_score = _versus.score_versus_state(
                replied,
                request.side_name,
                weights=request.versus_weights,
                resolution=reply_resolution,
                solo_score=-reply_evaluation.score,
                state_value=state_value,
                path_length=len(reply.placement.path),
                action_side=opponent_name,
                _metric_cache=metric_cache,
                _profile=profile,
            )
            previous = worst_by_root.get(root_index)
            if previous is None or reply_score < previous[0]:
                worst_by_root[root_index] = (reply_score, reply)

        best: _versus.VersusChoice | None = None
        for index, ((action, evaluation, _after, resolution), base_score) in enumerate(
            zip(candidates, scores)
        ):
            worst = worst_by_root.get(index)
            choice = _versus.VersusChoice(
                action=action,
                score=base_score if worst is None else worst[0],
                immediate=evaluation,
                resolution=resolution,
                opponent_reply=None if worst is None else worst[1],
            )
            if best is None or (
                choice.score,
                choice.resolution.sent_lines,
                choice.resolution.canceled_lines,
                -len(choice.action.placement.path),
                -int(choice.action.use_hold),
            ) > (
                best.score,
                best.resolution.sent_lines,
                best.resolution.canceled_lines,
                -len(best.action.placement.path),
                -int(best.action.use_hold),
            ):
                best = choice
        choices.append(
            _versus._materialize_selected_immediate(
                request.match,
                request.side_name,
                best,
                request.scorer,
            )
        )

    _versus.record_profile_elapsed(
        profile,
        "python_aggregation_tie_breaking",
        aggregation_started,
    )
    _versus.record_profile_elapsed(
        profile,
        "total_search",
        search_started,
        calls=len(normalized),
    )
    return tuple(choices)


def install_versus_batch_fast_path() -> None:
    _versus.choose_versus_actions_batch = choose_versus_actions_batch
