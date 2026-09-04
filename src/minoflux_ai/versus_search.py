from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field
import random
from typing import Literal, Protocol, Sequence

from minoflux_engine import Game, GarbagePacket, GarbageQueue, Placement, VersusMatch, VersusResolution, VersusSide

from .features import max_height_and_holes
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights, PlacementEvaluation
from .search import (
    DEFAULT_SEARCH_CONFIG,
    SearchAction,
    SearchConfig,
    SearchScorer,
    _branch_groups,
    _rank_precomputed_actions,
    apply_search_action,
    clone_game,
    rank_search_actions,
)

SideName = Literal["player", "ai"]


class VersusStateScorer(Protocol):
    """Root-side value scorer for complete versus match states."""

    def score_match(
        self,
        match: VersusMatch,
        root_side: SideName,
        to_move: SideName | None = None,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class VersusWeights:
    solo_evaluation: float = 1.0
    state_value: float = 24.0
    sent_lines: float = 8.0
    canceled_lines: float = 6.0
    garbage_applied: float = -8.0
    own_pending: float = -5.0
    opponent_pending: float = 3.0
    own_max_height: float = -2.4
    opponent_max_height: float = 1.4
    own_holes: float = -5.0
    opponent_holes: float = 1.5
    own_b2b: float = 0.8
    opponent_b2b: float = -0.5
    own_surge: float = 1.4
    opponent_surge: float = -0.8
    input_cost: float = -0.025
    win: float = 1_000_000.0
    loss: float = -1_000_000.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VersusSearchConfig:
    placement_search: SearchConfig = field(default_factory=lambda: DEFAULT_SEARCH_CONFIG)
    candidate_width: int = 8
    opponent_reply_width: int = 2

    def normalized(self) -> "VersusSearchConfig":
        return VersusSearchConfig(
            placement_search=self.placement_search.normalized(),
            candidate_width=min(64, max(1, int(self.candidate_width))),
            opponent_reply_width=min(16, max(0, int(self.opponent_reply_width))),
        )

    def to_dict(self) -> dict[str, object]:
        cfg = self.normalized()
        return {
            "placementSearch": cfg.placement_search.to_dict(),
            "candidateWidth": cfg.candidate_width,
            "opponentReplyWidth": cfg.opponent_reply_width,
        }


DEFAULT_VERSUS_WEIGHTS = VersusWeights()
DEFAULT_VERSUS_SEARCH_CONFIG = VersusSearchConfig()


@dataclass(frozen=True, slots=True)
class VersusChoice:
    action: SearchAction
    score: float
    immediate: PlacementEvaluation
    resolution: VersusResolution
    opponent_reply: SearchAction | None = None


@dataclass(frozen=True, slots=True)
class VersusSearchRequest:
    """One independent versus decision in a cross-game search batch."""

    match: VersusMatch
    side_name: SideName
    heuristic_weights: HeuristicWeights = DEFAULT_WEIGHTS
    config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG
    versus_weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS
    scorer: SearchScorer | None = None
    opponent_scorer: SearchScorer | None = None
    opponent_heuristic_weights: HeuristicWeights | None = None
    state_scorer: VersusStateScorer | None = None


def _clone_queue(queue: GarbageQueue) -> GarbageQueue:
    cloned = GarbageQueue(queue.width)
    cloned.packets.extend(GarbagePacket(packet.lines, packet.hole) for packet in queue.packets)
    return cloned


def _clone_side(side: VersusSide, *, copy_game: bool = True) -> VersusSide:
    return VersusSide(
        game=clone_game(side.game) if copy_game else side.game,
        pending=_clone_queue(side.pending),
        sent=side.sent,
        received=side.received,
        canceled=side.canceled,
        garbage_applied=side.garbage_applied,
    )


def clone_versus_match(match: VersusMatch) -> VersusMatch:
    cloned = copy(match)
    cloned.player = _clone_side(match.player)
    cloned.ai = _clone_side(match.ai)
    cloned._garbage_rng = random.Random()
    cloned._garbage_rng.setstate(match._garbage_rng.getstate())
    return cloned


def _clone_versus_match_for_action(
    match: VersusMatch,
    side_name: SideName,
) -> VersusMatch:
    """Copy only the Game that one simulated lock can mutate.

    ``resolve_lock`` may alter both garbage queues, but it only changes the acting
    side's Game (placement and pending-garbage application). The opponent Game is
    read-only for this one-ply simulation and can safely be shared.
    """

    cloned = copy(match)
    cloned.player = _clone_side(match.player, copy_game=side_name == "player")
    cloned.ai = _clone_side(match.ai, copy_game=side_name == "ai")
    cloned._garbage_rng = random.Random()
    cloned._garbage_rng.setstate(match._garbage_rng.getstate())
    return cloned


def _side(match: VersusMatch, name: SideName) -> VersusSide:
    return match.player if name == "player" else match.ai


def _opponent_name(name: SideName) -> SideName:
    return "ai" if name == "player" else "player"


def score_versus_state(
    match: VersusMatch,
    root_side: SideName,
    *,
    weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS,
    resolution: VersusResolution | None = None,
    solo_score: float = 0.0,
    state_value: float = 0.0,
    path_length: int = 0,
    action_side: SideName | None = None,
) -> float:
    own = _side(match, root_side)
    opponent = _side(match, _opponent_name(root_side))
    if match.winner == root_side:
        return weights.win
    if match.winner == _opponent_name(root_side):
        return weights.loss
    if match.winner == "draw":
        return 0.0

    own_max_height, own_holes = max_height_and_holes(own.game.board)
    opponent_max_height, opponent_holes = max_height_and_holes(opponent.game.board)
    input_direction = 1.0 if action_side in (None, root_side) else -1.0
    score = (
        solo_score * weights.solo_evaluation
        + state_value * weights.state_value
        + own.pending.pending_lines * weights.own_pending
        + opponent.pending.pending_lines * weights.opponent_pending
        + own_max_height * weights.own_max_height
        + opponent_max_height * weights.opponent_max_height
        + own_holes * weights.own_holes
        + opponent_holes * weights.opponent_holes
        + own.game.b2b_chain * weights.own_b2b
        + opponent.game.b2b_chain * weights.opponent_b2b
        + own.game.surge_charge * weights.own_surge
        + opponent.game.surge_charge * weights.opponent_surge
        + input_direction * max(0, int(path_length)) * weights.input_cost
    )
    if resolution is not None:
        direction = 1.0 if resolution.side == root_side else -1.0
        score += direction * resolution.sent_lines * weights.sent_lines
        score += direction * resolution.canceled_lines * weights.canceled_lines
        score += direction * resolution.garbage_applied * weights.garbage_applied
    return score


def _simulate_action(
    match: VersusMatch,
    side_name: SideName,
    action: SearchAction,
) -> tuple[VersusMatch, VersusResolution]:
    simulated = _clone_versus_match_for_action(match, side_name)
    result = apply_search_action(_side(simulated, side_name).game, action)
    resolution = simulated.resolve_lock(side_name, result)
    return simulated, resolution


def _state_values(
    scorer: VersusStateScorer | None,
    entries: Sequence[tuple[VersusMatch, SideName, SideName | None]],
) -> tuple[float, ...]:
    if not entries:
        return ()
    if scorer is None:
        return (0.0,) * len(entries)
    score_matches = getattr(scorer, "score_matches", None)
    if callable(score_matches):
        values = tuple(float(value) for value in score_matches(entries))
        if len(values) != len(entries):
            raise ValueError("Versus state scorer returned the wrong number of values")
        return values
    return tuple(float(scorer.score_match(match, root_side, to_move)) for match, root_side, to_move in entries)


def _rank_search_actions_batch(
    games: Sequence[Game],
    weights: HeuristicWeights,
    config: SearchConfig,
    *,
    limit: int,
    scorer: SearchScorer | None,
) -> tuple[tuple[tuple[SearchAction, PlacementEvaluation], ...], ...]:
    """Rank immediate actions for many games with one grouped neural forward pass.

    This preserves the existing per-game ranking semantics and falls back to the
    serial path for heuristic or scorers that do not expose placement-group
    batching.
    """

    cfg = config.normalized()
    score_groups = getattr(scorer, "score_placement_groups", None) if scorer is not None else None
    score_placements = getattr(scorer, "score_placements", None) if scorer is not None else None
    if not callable(score_groups) or not callable(score_placements):
        return tuple(
            rank_search_actions(
                game,
                weights,
                cfg,
                limit=limit,
                scorer=scorer,
            )
            for game in games
        )

    prepared: list[tuple[tuple[Placement, ...], Game | None, tuple[Placement, ...]]] = []
    groups: list[tuple[Game, Sequence[Placement]]] = []
    group_keys: list[tuple[int, bool]] = []
    for index, game in enumerate(games):
        direct, held, held_placements = _branch_groups(game, cfg, include_paths=True)
        prepared.append((direct, held, held_placements))
        if direct:
            groups.append((game, direct))
            group_keys.append((index, False))
        if held is not None and held_placements:
            groups.append((held, held_placements))
            group_keys.append((index, True))

    grouped_values = score_groups(tuple(groups)) if groups else ()
    if len(grouped_values) != len(groups):
        raise ValueError("Search scorer returned the wrong number of placement groups")
    values_by_key = {
        key: values
        for key, values in zip(group_keys, grouped_values)
    }

    ranked_groups: list[tuple[tuple[SearchAction, PlacementEvaluation], ...]] = []
    for index, game in enumerate(games):
        direct, held, held_placements = prepared[index]
        branches: list[tuple[bool, Game, Sequence[Placement], Sequence[float]]] = []
        if direct:
            branches.append((False, game, direct, values_by_key[(index, False)]))
        if held is not None and held_placements:
            branches.append((True, held, held_placements, values_by_key[(index, True)]))
        ranked_groups.append(_rank_precomputed_actions(tuple(branches), limit))
    return tuple(ranked_groups)


def choose_versus_action(
    match: VersusMatch,
    side_name: SideName,
    heuristic_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG,
    versus_weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS,
    *,
    scorer: SearchScorer | None = None,
    opponent_scorer: SearchScorer | None = None,
    opponent_heuristic_weights: HeuristicWeights | None = None,
    state_scorer: VersusStateScorer | None = None,
) -> VersusChoice | None:
    """Choose a versus action with neural candidate, reply, and state-value scoring.

    ``opponent_scorer=None`` deliberately means heuristic opponent replies. Callers
    doing neural self-play should explicitly pass the same neural scorer for both
    ``scorer`` and ``opponent_scorer``.
    """

    cfg = config.normalized()
    own = _side(match, side_name)
    ranked = rank_search_actions(
        own.game,
        heuristic_weights,
        cfg.placement_search,
        limit=cfg.candidate_width,
        scorer=scorer,
    )
    if not ranked:
        return None

    opponent_name = _opponent_name(side_name)
    reply_scorer = opponent_scorer
    reply_weights = heuristic_weights if opponent_heuristic_weights is None else opponent_heuristic_weights

    root_candidates: list[
        tuple[SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]
    ] = []
    for action, evaluation in ranked:
        after, resolution = _simulate_action(match, side_name, action)
        root_candidates.append((action, evaluation, after, resolution))
    root_state_values = _state_values(
        state_scorer,
        tuple((after, side_name, opponent_name) for _action, _evaluation, after, _resolution in root_candidates),
    )

    base_scores = [
        score_versus_state(
            after,
            side_name,
            weights=versus_weights,
            resolution=resolution,
            solo_score=evaluation.score,
            state_value=root_state_value,
            path_length=len(action.placement.path),
            action_side=side_name,
        )
        for (action, evaluation, after, resolution), root_state_value in zip(
            root_candidates,
            root_state_values,
        )
    ]

    reply_root_indices: list[int] = []
    if cfg.opponent_reply_width > 0:
        for index, (_action, _evaluation, after, _resolution) in enumerate(root_candidates):
            if after.winner is None and not _side(after, opponent_name).game.game_over:
                reply_root_indices.append(index)

    # A root action can change the opponent's pending garbage, but not the
    # opponent Game itself. Garbage is only applied after the opponent locks.
    # Therefore legal reply placements and solo-neural reply ranking are
    # identical for every non-terminal root candidate and only need computing once.
    shared_replies: tuple[tuple[SearchAction, PlacementEvaluation], ...] = ()
    if reply_root_indices:
        reply_game = _side(root_candidates[reply_root_indices[0]][2], opponent_name).game
        shared_replies = rank_search_actions(
            reply_game,
            reply_weights,
            cfg.placement_search,
            limit=cfg.opponent_reply_width,
            scorer=reply_scorer,
        )

    flat_reply_candidates: list[
        tuple[int, SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]
    ] = []
    for root_index in reply_root_indices:
        if not shared_replies:
            continue
        after = root_candidates[root_index][2]
        for reply, reply_evaluation in shared_replies:
            replied, reply_resolution = _simulate_action(after, opponent_name, reply)
            flat_reply_candidates.append(
                (root_index, reply, reply_evaluation, replied, reply_resolution)
            )

    reply_state_values = _state_values(
        state_scorer,
        tuple(
            (replied, side_name, side_name)
            for _root_index, _reply, _reply_evaluation, replied, _reply_resolution in flat_reply_candidates
        ),
    )

    worst_by_root: dict[int, tuple[float, SearchAction]] = {}
    for (
        root_index,
        reply,
        reply_evaluation,
        replied,
        reply_resolution,
    ), reply_state_value in zip(flat_reply_candidates, reply_state_values):
        reply_score = score_versus_state(
            replied,
            side_name,
            weights=versus_weights,
            resolution=reply_resolution,
            solo_score=-reply_evaluation.score,
            state_value=reply_state_value,
            path_length=len(reply.placement.path),
            action_side=opponent_name,
        )
        previous = worst_by_root.get(root_index)
        if previous is None or reply_score < previous[0]:
            worst_by_root[root_index] = (reply_score, reply)

    best: VersusChoice | None = None
    for index, ((action, evaluation, _after, resolution), base_score) in enumerate(
        zip(root_candidates, base_scores)
    ):
        worst = worst_by_root.get(index)
        score = base_score if worst is None else worst[0]
        reply_action = None if worst is None else worst[1]
        choice = VersusChoice(
            action=action,
            score=score,
            immediate=evaluation,
            resolution=resolution,
            opponent_reply=reply_action,
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
    return best


def _rank_request_games(
    items: Sequence[
        tuple[
            int,
            Game,
            HeuristicWeights,
            SearchConfig,
            int,
            SearchScorer | None,
        ]
    ],
    output_count: int,
) -> tuple[tuple[tuple[SearchAction, PlacementEvaluation], ...], ...]:
    """Batch compatible placement-ranking requests without mixing evaluators."""

    output: list[tuple[tuple[SearchAction, PlacementEvaluation], ...]] = [
        () for _ in range(output_count)
    ]
    groups: dict[
        tuple[int, HeuristicWeights, SearchConfig, int],
        list[tuple[int, Game]],
    ] = {}
    scorers: dict[tuple[int, HeuristicWeights, SearchConfig, int], SearchScorer | None] = {}
    for output_index, game, weights, config, limit, scorer in items:
        cfg = config.normalized()
        key = (id(scorer), weights, cfg, int(limit))
        groups.setdefault(key, []).append((output_index, game))
        scorers[key] = scorer

    for key, members in groups.items():
        _scorer_id, weights, config, limit = key
        ranked = _rank_search_actions_batch(
            tuple(game for _index, game in members),
            weights,
            config,
            limit=limit,
            scorer=scorers[key],
        )
        for (output_index, _game), actions in zip(members, ranked):
            output[output_index] = actions
    return tuple(output)


def _state_values_by_request(
    requests: Sequence[VersusSearchRequest],
    entries_by_request: Sequence[
        Sequence[tuple[VersusMatch, SideName, SideName | None]]
    ],
) -> tuple[tuple[float, ...], ...]:
    """Combine match-value requests that share the same evaluator instance."""

    output: list[tuple[float, ...]] = [() for _ in requests]
    groups: dict[
        int,
        tuple[
            VersusStateScorer,
            list[tuple[int, int]],
            list[tuple[VersusMatch, SideName, SideName | None]],
        ],
    ] = {}
    for request_index, (request, entries) in enumerate(zip(requests, entries_by_request)):
        if not entries:
            continue
        scorer = request.state_scorer
        if scorer is None:
            output[request_index] = (0.0,) * len(entries)
            continue
        key = id(scorer)
        group = groups.get(key)
        if group is None:
            group = (scorer, [], [])
            groups[key] = group
        _group_scorer, slices, flat_entries = group
        slices.append((request_index, len(entries)))
        flat_entries.extend(entries)

    for scorer, slices, flat_entries in groups.values():
        values = _state_values(scorer, tuple(flat_entries))
        offset = 0
        for request_index, size in slices:
            output[request_index] = values[offset : offset + size]
            offset += size
    return tuple(output)


def choose_versus_actions_batch(
    requests: Sequence[VersusSearchRequest],
) -> tuple[VersusChoice | None, ...]:
    """Choose actions for several matches with four cross-game neural batches.

    Legal placement generation and all tie-breaking stay per game. Only compatible
    solo/root/reply value forwards are concatenated, so evaluator instances and
    policies are never mixed.
    """

    if not requests:
        return ()
    if len(requests) == 1:
        request = requests[0]
        return (
            choose_versus_action(
                request.match,
                request.side_name,
                request.heuristic_weights,
                request.config,
                request.versus_weights,
                scorer=request.scorer,
                opponent_scorer=request.opponent_scorer,
                opponent_heuristic_weights=request.opponent_heuristic_weights,
                state_scorer=request.state_scorer,
            ),
        )

    normalized = tuple(
        VersusSearchRequest(
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

    root_ranked = _rank_request_games(
        tuple(
            (
                index,
                _side(request.match, request.side_name).game,
                request.heuristic_weights,
                request.config.placement_search,
                request.config.candidate_width,
                request.scorer,
            )
            for index, request in enumerate(normalized)
        ),
        len(normalized),
    )

    root_candidates: list[
        list[tuple[SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]]
    ] = []
    root_entries: list[list[tuple[VersusMatch, SideName, SideName | None]]] = []
    reply_root_indices: list[list[int]] = []
    for request, ranked in zip(normalized, root_ranked):
        opponent_name = _opponent_name(request.side_name)
        candidates: list[
            tuple[SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]
        ] = []
        live_indices: list[int] = []
        for action, evaluation in ranked:
            after, resolution = _simulate_action(request.match, request.side_name, action)
            candidate_index = len(candidates)
            candidates.append((action, evaluation, after, resolution))
            if (
                request.config.opponent_reply_width > 0
                and after.winner is None
                and not _side(after, opponent_name).game.game_over
            ):
                live_indices.append(candidate_index)
        root_candidates.append(candidates)
        root_entries.append(
            [(after, request.side_name, opponent_name) for _action, _evaluation, after, _resolution in candidates]
        )
        reply_root_indices.append(live_indices)

    root_state_values = _state_values_by_request(normalized, root_entries)
    base_scores: list[list[float]] = []
    for request, candidates, values in zip(normalized, root_candidates, root_state_values):
        base_scores.append(
            [
                score_versus_state(
                    after,
                    request.side_name,
                    weights=request.versus_weights,
                    resolution=resolution,
                    solo_score=evaluation.score,
                    state_value=state_value,
                    path_length=len(action.placement.path),
                    action_side=request.side_name,
                )
                for (action, evaluation, after, resolution), state_value in zip(candidates, values)
            ]
        )

    reply_rank_items: list[
        tuple[int, Game, HeuristicWeights, SearchConfig, int, SearchScorer | None]
    ] = []
    for request_index, (request, live_indices, candidates) in enumerate(
        zip(normalized, reply_root_indices, root_candidates)
    ):
        if not live_indices:
            continue
        opponent_name = _opponent_name(request.side_name)
        reply_weights = (
            request.heuristic_weights
            if request.opponent_heuristic_weights is None
            else request.opponent_heuristic_weights
        )
        reply_game = _side(candidates[live_indices[0]][2], opponent_name).game
        reply_rank_items.append(
            (
                request_index,
                reply_game,
                reply_weights,
                request.config.placement_search,
                request.config.opponent_reply_width,
                request.opponent_scorer,
            )
        )
    shared_replies = _rank_request_games(
        tuple(reply_rank_items),
        len(normalized),
    )

    reply_candidates: list[
        list[tuple[int, SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]]
    ] = []
    reply_entries: list[list[tuple[VersusMatch, SideName, SideName | None]]] = []
    for request, live_indices, candidates, replies in zip(
        normalized,
        reply_root_indices,
        root_candidates,
        shared_replies,
    ):
        opponent_name = _opponent_name(request.side_name)
        request_replies: list[
            tuple[int, SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]
        ] = []
        for root_index in live_indices:
            after = candidates[root_index][2]
            for reply, reply_evaluation in replies:
                replied, resolution = _simulate_action(after, opponent_name, reply)
                request_replies.append(
                    (root_index, reply, reply_evaluation, replied, resolution)
                )
        reply_candidates.append(request_replies)
        reply_entries.append(
            [
                (replied, request.side_name, request.side_name)
                for _root_index, _reply, _evaluation, replied, _resolution in request_replies
            ]
        )

    reply_state_values = _state_values_by_request(normalized, reply_entries)
    choices: list[VersusChoice | None] = []
    for request, candidates, scores, replies, reply_values in zip(
        normalized,
        root_candidates,
        base_scores,
        reply_candidates,
        reply_state_values,
    ):
        opponent_name = _opponent_name(request.side_name)
        worst_by_root: dict[int, tuple[float, SearchAction]] = {}
        for (
            root_index,
            reply,
            reply_evaluation,
            replied,
            reply_resolution,
        ), state_value in zip(replies, reply_values):
            reply_score = score_versus_state(
                replied,
                request.side_name,
                weights=request.versus_weights,
                resolution=reply_resolution,
                solo_score=-reply_evaluation.score,
                state_value=state_value,
                path_length=len(reply.placement.path),
                action_side=opponent_name,
            )
            previous = worst_by_root.get(root_index)
            if previous is None or reply_score < previous[0]:
                worst_by_root[root_index] = (reply_score, reply)

        best: VersusChoice | None = None
        for index, ((action, evaluation, _after, resolution), base_score) in enumerate(
            zip(candidates, scores)
        ):
            worst = worst_by_root.get(index)
            choice = VersusChoice(
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
        choices.append(best)
    return tuple(choices)
