from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import asdict, dataclass
from heapq import nlargest
import random
from typing import Protocol, Sequence

from minoflux_engine import Game, LockResult, Placement
from minoflux_engine.b2b import resolve_b2b_charging
from minoflux_engine.spin import base_attack, is_difficult_clear, t_spin_event

from .bitboard import (
    board_row_masks,
    classify_t_spin_row_masks,
    hidden_rows_occupied,
    place_and_clear_row_masks,
)
from .features import extract_board_features_from_masks
from .heuristic import (
    DEFAULT_WEIGHTS,
    HeuristicWeights,
    PlacementEvaluation,
    PlacementFeatures,
    rank_placements,
)
from .reachability import reachable_placements


def clone_game(game: Game) -> Game:
    cloned = copy(game)
    cloned.board = [row.copy() for row in game.board]
    cloned.queue = deque(game.queue)
    cloned._bag = copy(game._bag)
    cloned._bag._queue = deque(game._bag._queue)
    cloned_rng = random.Random()
    cloned_rng.setstate(game._bag._rng.getstate())
    cloned._bag._rng = cloned_rng
    return cloned


def _held_search_game(game: Game) -> Game | None:
    """Create a read-only Hold branch without copying board/RNG state."""

    if game.hold_used or game.game_over or game.paused:
        return None
    held = copy(game)
    held.board = game.board
    held.queue = deque(game.queue)
    outgoing = game.current
    incoming = game.hold_piece
    held.hold_piece = outgoing
    if incoming is None:
        if not held.queue:
            return None
        incoming = held.queue.popleft()
    held.current = incoming
    held.x, held.y, held.rotation = 3, 1, 0
    held.hold_used = True
    held.last_action = "hold"
    held.last_move_was_rotation = False
    held.last_rotation_kick_index = None
    held.last_rotation_from = None
    held.last_rotation_to = None
    held.lock_elapsed_ms = 0.0
    held.lock_resets = 0
    held.game_over = held._collides(held.current, held.x, held.y, held.rotation)
    return None if held.game_over else held


@dataclass(frozen=True, slots=True)
class SearchConfig:
    allow_hold: bool = True
    lookahead_pieces: int = 1
    beam_width: int = 4
    discount: float = 0.90
    srs_reachable: bool = True
    allow_180: bool = False
    reachability_node_limit: int = 8_000

    def normalized(self) -> "SearchConfig":
        return SearchConfig(
            allow_hold=bool(self.allow_hold),
            lookahead_pieces=min(3, max(0, int(self.lookahead_pieces))),
            beam_width=min(128, max(1, int(self.beam_width))),
            discount=min(1.0, max(0.0, float(self.discount))),
            srs_reachable=bool(self.srs_reachable),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())


DEFAULT_SEARCH_CONFIG = SearchConfig()
DIRECT_SEARCH_CONFIG = SearchConfig(
    allow_hold=False,
    lookahead_pieces=0,
    beam_width=1,
    srs_reachable=False,
)


@dataclass(frozen=True, slots=True)
class SearchAction:
    use_hold: bool
    placement: Placement

    def to_dict(self) -> dict[str, object]:
        return {
            "hold": self.use_hold,
            "piece": self.placement.piece,
            "x": self.placement.x,
            "y": self.placement.y,
            "rotation": self.placement.rotation,
            "path": list(self.placement.path),
            "lastMoveWasRotation": self.placement.last_move_was_rotation,
            "rotationKickIndex": self.placement.rotation_kick_index,
        }


@dataclass(frozen=True, slots=True)
class SearchChoice:
    action: SearchAction
    score: float
    immediate: PlacementEvaluation
    path: tuple[SearchAction, ...]


@dataclass(slots=True)
class _BeamNode:
    game: Game
    score: float
    path: tuple[SearchAction, ...]
    first_evaluation: PlacementEvaluation


class SearchScorer(Protocol):
    """Batch scorer for replacing heuristic leaf values without replacing move generation."""

    def score_many(
        self,
        game: Game,
        evaluations: Sequence[PlacementEvaluation],
    ) -> Sequence[float]: ...


def _evaluation_key(item: PlacementEvaluation) -> tuple[float, int, int, int, int, int, int, int]:
    return (
        item.score,
        item.features.attack,
        item.features.spin_lines,
        item.features.lines,
        -item.features.board.holes,
        -item.features.board.max_height,
        -item.placement.rotation,
        -item.placement.x,
    )


def _candidate_key(item: tuple[SearchAction, PlacementEvaluation]) -> tuple[float, int, int, int, int, int, int, int]:
    action, evaluation = item
    features = evaluation.features
    return (
        evaluation.score,
        features.attack,
        features.spin_lines,
        features.lines,
        -features.board.holes,
        -features.board.max_height,
        -int(action.use_hold),
        -action.placement.rotation * 100 - action.placement.x,
    )


def _neural_metadata_evaluation(
    game: Game,
    placement: Placement,
    score: float,
    *,
    source_rows: Sequence[int] | None = None,
    before_features=None,
) -> PlacementEvaluation:
    """Compute only metadata actually used to break equal neural scores.

    Champion-only garbage stress features are deliberately skipped: once a neural
    value replaces the heuristic score they cannot affect ordering. The normal
    board/line/spin/attack metadata remains exact for telemetry and tie-breaking.
    """

    rows = tuple(source_rows) if source_rows is not None else board_row_masks(game.board)
    spin_kind = classify_t_spin_row_masks(
        rows,
        piece=placement.piece,
        x=placement.x,
        y=placement.y,
        rotation=placement.rotation,
        last_move_was_rotation=placement.last_move_was_rotation,
        rotation_kick_index=placement.rotation_kick_index,
        width=game.width,
    )
    after_rows, lines, topped_out = place_and_clear_row_masks(
        rows,
        placement,
        width=game.width,
    )
    spin = t_spin_event(spin_kind, lines)
    perfect_clear = not any(after_rows)
    difficult = is_difficult_clear(lines, spin)
    b2b = resolve_b2b_charging(
        active=game.back_to_back,
        chain=game.b2b_chain,
        difficult=difficult,
        lines=lines,
        perfect_clear=perfect_clear and lines > 0,
    )
    combo = game.combo + 1 if lines else -1
    attack = base_attack(lines, spin) + b2b.attack_bonus + b2b.released
    if lines and combo > 0:
        attack += min(4, combo // 2 + 1)
    if perfect_clear and lines:
        attack += 10

    before = before_features or extract_board_features_from_masks(rows, width=game.width)
    after = extract_board_features_from_masks(after_rows, width=game.width)
    features = PlacementFeatures(
        board=after,
        new_holes=max(0, after.holes - before.holes),
        lines=lines,
        attack=attack,
        spin_lines=lines if spin is not None else 0,
        perfect_clear=perfect_clear,
        game_over=topped_out or hidden_rows_occupied(after_rows, game.hidden_rows),
        spin=spin,
        t_spin_slot_delta=after.t_spin_slots - before.t_spin_slots,
    )
    return PlacementEvaluation(placement=placement, score=float(score), features=features)


def _rank_precomputed_actions(
    branches: Sequence[tuple[bool, Game, Sequence[Placement], Sequence[float]]],
    limit: int | None,
) -> tuple[tuple[SearchAction, PlacementEvaluation], ...]:
    """Globally prune direct/Hold neural candidates before feature extraction."""

    raw: list[tuple[bool, Game, Placement, float]] = []
    for use_hold, branch_game, placements, values in branches:
        if len(values) != len(placements):
            raise ValueError(
                f"Search scorer returned {len(values)} values for {len(placements)} placements"
            )
        raw.extend(
            (use_hold, branch_game, placement, float(value))
            for placement, value in zip(placements, values)
        )
    if not raw:
        return ()

    count = len(raw) if limit is None else max(0, int(limit))
    if count == 0:
        return ()
    if count < len(raw):
        cutoff = nlargest(count, (item[3] for item in raw))[-1]
        survivors = [item for item in raw if item[3] >= cutoff]
    else:
        survivors = raw

    board_cache: dict[int, tuple[tuple[int, ...], object]] = {}
    ranked: list[tuple[SearchAction, PlacementEvaluation]] = []
    for use_hold, branch_game, placement, score in survivors:
        board_key = id(branch_game.board)
        cached = board_cache.get(board_key)
        if cached is None:
            rows = board_row_masks(branch_game.board)
            before = extract_board_features_from_masks(rows, width=branch_game.width)
            board_cache[board_key] = (rows, before)
        else:
            rows, before = cached
        evaluation = _neural_metadata_evaluation(
            branch_game,
            placement,
            score,
            source_rows=rows,
            before_features=before,
        )
        ranked.append((SearchAction(use_hold, placement), evaluation))

    if count < len(ranked):
        return tuple(nlargest(count, ranked, key=_candidate_key))
    ranked.sort(key=_candidate_key, reverse=True)
    return tuple(ranked)


def _score_evaluations(
    game: Game,
    evaluations: Sequence[PlacementEvaluation],
    scorer: SearchScorer,
    limit: int | None,
) -> tuple[PlacementEvaluation, ...]:
    if not evaluations:
        return ()
    values = tuple(float(value) for value in scorer.score_many(game, evaluations))
    if len(values) != len(evaluations):
        raise ValueError(
            f"Search scorer returned {len(values)} values for {len(evaluations)} placements"
        )
    rescored = tuple(
        PlacementEvaluation(
            placement=evaluation.placement,
            score=score,
            features=evaluation.features,
        )
        for evaluation, score in zip(evaluations, values)
    )
    if limit is None:
        return tuple(sorted(rescored, key=_evaluation_key, reverse=True))
    count = max(0, int(limit))
    if count == 0:
        return ()
    if count < len(rescored):
        return tuple(nlargest(count, rescored, key=_evaluation_key))
    return tuple(sorted(rescored, key=_evaluation_key, reverse=True))


def _score_placements_before_features(
    game: Game,
    placements: Sequence[Placement],
    weights: HeuristicWeights,
    scorer: SearchScorer,
    limit: int | None,
    *,
    precomputed_values: Sequence[float] | None = None,
) -> tuple[PlacementEvaluation, ...] | None:
    """Placement-only fast path that never evaluates Champion garbage features."""

    del weights
    score_placements = getattr(scorer, "score_placements", None)
    if not callable(score_placements) and precomputed_values is None:
        return None
    if not placements:
        return ()
    values = tuple(
        float(value)
        for value in (
            precomputed_values
            if precomputed_values is not None
            else score_placements(game, placements)
        )
    )
    ranked = _rank_precomputed_actions(((False, game, placements, values),), limit)
    return tuple(evaluation for _action, evaluation in ranked)


def _placements_for_game(
    game: Game,
    config: SearchConfig,
    *,
    include_paths: bool = True,
) -> tuple[Placement, ...]:
    if not config.srs_reachable:
        return game.legal_placements()
    return reachable_placements(
        game,
        allow_180=config.allow_180,
        max_nodes=config.reachability_node_limit,
        include_paths=include_paths,
    )


def _rank_branch(
    game: Game,
    placements: Sequence[Placement],
    weights: HeuristicWeights,
    scorer: SearchScorer | None,
    branch_limit: int | None,
    *,
    precomputed_values: Sequence[float] | None = None,
) -> tuple[PlacementEvaluation, ...]:
    if scorer is not None:
        fast = _score_placements_before_features(
            game,
            placements,
            weights,
            scorer,
            branch_limit,
            precomputed_values=precomputed_values,
        )
        if fast is not None:
            return fast

    evaluations = rank_placements(
        game,
        weights,
        placements=placements,
        limit=branch_limit if scorer is None else None,
    )
    if scorer is not None:
        evaluations = _score_evaluations(game, evaluations, scorer, branch_limit)
    return evaluations


def _branch_groups(
    game: Game,
    cfg: SearchConfig,
    *,
    include_paths: bool = True,
) -> tuple[tuple[Placement, ...], Game | None, tuple[Placement, ...]]:
    direct = _placements_for_game(game, cfg, include_paths=include_paths)
    held = _held_search_game(game) if cfg.allow_hold else None
    held_placements = (
        _placements_for_game(held, cfg, include_paths=include_paths)
        if held is not None
        else ()
    )
    return direct, held, held_placements


def rank_search_actions(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    *,
    limit: int | None = None,
    scorer: SearchScorer | None = None,
) -> tuple[tuple[SearchAction, PlacementEvaluation], ...]:
    cfg = config.normalized()
    branch_limit = None if limit is None else max(1, int(limit))
    direct_placements, held, held_placements = _branch_groups(game, cfg, include_paths=True)

    score_groups = getattr(scorer, "score_placement_groups", None) if scorer is not None else None
    score_placements = getattr(scorer, "score_placements", None) if scorer is not None else None
    if callable(score_groups) and callable(score_placements):
        groups: list[tuple[Game, Sequence[Placement]]] = []
        branch_specs: list[tuple[bool, Game, Sequence[Placement]]] = []
        if direct_placements:
            groups.append((game, direct_placements))
            branch_specs.append((False, game, direct_placements))
        if held is not None and held_placements:
            groups.append((held, held_placements))
            branch_specs.append((True, held, held_placements))
        grouped_values = score_groups(tuple(groups)) if groups else ()
        if len(grouped_values) != len(branch_specs):
            raise ValueError("Search scorer returned the wrong number of placement groups")
        return _rank_precomputed_actions(
            tuple(
                (use_hold, branch_game, placements, values)
                for (use_hold, branch_game, placements), values in zip(branch_specs, grouped_values)
            ),
            limit,
        )

    direct_evaluations = _rank_branch(
        game,
        direct_placements,
        weights,
        scorer,
        branch_limit,
    )
    candidates: list[tuple[SearchAction, PlacementEvaluation]] = [
        (SearchAction(False, evaluation.placement), evaluation)
        for evaluation in direct_evaluations
    ]

    if held is not None:
        held_evaluations = _rank_branch(
            held,
            held_placements,
            weights,
            scorer,
            branch_limit,
        )
        candidates.extend(
            (SearchAction(True, evaluation.placement), evaluation)
            for evaluation in held_evaluations
        )

    if limit is not None:
        count = max(0, int(limit))
        if count == 0:
            return ()
        if count < len(candidates):
            return tuple(nlargest(count, candidates, key=_candidate_key))
    candidates.sort(key=_candidate_key, reverse=True)
    return tuple(candidates)


def apply_search_action(game: Game, action: SearchAction) -> LockResult:
    if action.use_hold and not game.hold():
        raise ValueError("Search action requested an unavailable Hold")
    if game.current != action.placement.piece:
        raise ValueError(
            f"Search action expected {action.placement.piece}, but engine produced {game.current}"
        )
    return game.place(action.placement)


def _node_key(node: _BeamNode) -> tuple[float, int, int, int, int, int]:
    first = node.path[0]
    return (
        node.score,
        node.game.attack,
        node.game.lines,
        node.game.pieces_placed,
        -int(first.use_hold),
        -first.placement.rotation * 100 - first.placement.x,
    )


def choose_search_action(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    *,
    scorer: SearchScorer | None = None,
) -> SearchChoice | None:
    cfg = config.normalized()
    root_limit = 1 if cfg.lookahead_pieces == 0 else cfg.beam_width
    ranked_root = rank_search_actions(
        game,
        weights,
        cfg,
        limit=root_limit,
        scorer=scorer,
    )
    if not ranked_root:
        return None

    if cfg.lookahead_pieces == 0:
        action, evaluation = ranked_root[0]
        return SearchChoice(action, evaluation.score, evaluation, (action,))

    frontier: list[_BeamNode] = []
    for action, evaluation in ranked_root:
        child = clone_game(game)
        apply_search_action(child, action)
        frontier.append(
            _BeamNode(
                game=child,
                score=evaluation.score,
                path=(action,),
                first_evaluation=evaluation,
            )
        )
    frontier.sort(key=_node_key, reverse=True)
    frontier = frontier[: cfg.beam_width]

    for depth in range(1, cfg.lookahead_pieces + 1):
        expanded: list[_BeamNode] = []
        future_weight = cfg.discount ** depth
        for node in frontier:
            if node.game.game_over:
                expanded.append(node)
                continue
            for action, evaluation in rank_search_actions(
                node.game,
                weights,
                cfg,
                limit=cfg.beam_width,
                scorer=scorer,
            ):
                child = clone_game(node.game)
                apply_search_action(child, action)
                expanded.append(
                    _BeamNode(
                        game=child,
                        score=node.score + future_weight * evaluation.score,
                        path=(*node.path, action),
                        first_evaluation=node.first_evaluation,
                    )
                )
        if not expanded:
            break
        if len(expanded) > cfg.beam_width:
            frontier = nlargest(cfg.beam_width, expanded, key=_node_key)
        else:
            expanded.sort(key=_node_key, reverse=True)
            frontier = expanded

    best = max(frontier, key=_node_key)
    return SearchChoice(
        action=best.path[0],
        score=best.score,
        immediate=best.first_evaluation,
        path=best.path,
    )


def choose_search_actions_batch(
    games: Sequence[Game],
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    *,
    scorer: SearchScorer | None = None,
) -> tuple[SearchChoice | None, ...]:
    """Choose lookahead=0 actions for many games with one neural forward pass."""

    cfg = config.normalized()
    score_groups = getattr(scorer, "score_placement_groups", None) if scorer is not None else None
    score_placements = getattr(scorer, "score_placements", None) if scorer is not None else None
    if cfg.lookahead_pieces != 0 or not callable(score_groups) or not callable(score_placements):
        return tuple(
            choose_search_action(game, weights, cfg, scorer=scorer)
            for game in games
        )

    prepared: list[tuple[tuple[Placement, ...], Game | None, tuple[Placement, ...]]] = []
    groups: list[tuple[Game, Sequence[Placement]]] = []
    group_keys: list[tuple[int, bool]] = []
    for index, game in enumerate(games):
        if game.game_over:
            prepared.append(((), None, ()))
            continue
        direct, held, held_placements = _branch_groups(game, cfg, include_paths=False)
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

    choices: list[SearchChoice | None] = []
    for index, game in enumerate(games):
        direct, held, held_placements = prepared[index]
        branches: list[tuple[bool, Game, Sequence[Placement], Sequence[float]]] = []
        if direct:
            branches.append((False, game, direct, values_by_key[(index, False)]))
        if held is not None and held_placements:
            branches.append((True, held, held_placements, values_by_key[(index, True)]))
        ranked = _rank_precomputed_actions(tuple(branches), 1)
        if not ranked:
            choices.append(None)
            continue
        action, evaluation = ranked[0]
        choices.append(SearchChoice(action, evaluation.score, evaluation, (action,)))
    return tuple(choices)
