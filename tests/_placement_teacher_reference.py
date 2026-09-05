"""Frozen clone/place/full-sort teacher oracle from self-improve 46316f0.

Keep independent from production teacher internals: this deliberately retains
full engine transitions, full paths, full features, and original summation order.
"""
from __future__ import annotations
from minoflux_engine import Game, LockResult
from minoflux_engine.spin import is_difficult_clear
from minoflux_ai.features import BoardFeatures, extract_board_features
from minoflux_ai.reachability import reachable_placements
from minoflux_ai.search import SearchAction, apply_search_action, clone_game
from minoflux_ai.placement_teacher import (
    PlacementTeacherConfig, PlacementTeacherWeights, PlacementTeacherBreakdown,
    PlacementTeacherScore, DEFAULT_PLACEMENT_TEACHER_CONFIG,
    DEFAULT_PLACEMENT_TEACHER_WEIGHTS,
)


def _action_key(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        int(action.use_hold),
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
        int(placement.last_move_was_rotation),
        -1 if placement.rotation_kick_index is None else placement.rotation_kick_index,
    )


def _ranking_key(score: PlacementTeacherScore) -> tuple[object, ...]:
    breakdown = score.breakdown
    action = score.action
    return (
        score.total,
        score.immediate,
        breakdown.attack,
        breakdown.lines,
        -breakdown.after_holes,
        -breakdown.after_max_height,
        -int(action.use_hold),
        -action.placement.rotation,
        -action.placement.x,
        tuple(action.placement.path),
    )


def _legal_actions(
    game: Game,
    config: PlacementTeacherConfig,
) -> tuple[SearchAction, ...]:
    """Enumerate exact-SRS root actions without heuristic filtering."""

    cfg = config.normalized()
    direct = reachable_placements(
        game,
        allow_180=cfg.allow_180,
        max_nodes=cfg.reachability_node_limit,
        include_paths=True,
    )
    actions: list[SearchAction] = [SearchAction(False, placement) for placement in direct]

    if cfg.allow_hold and not game.hold_used and not game.game_over and not game.paused:
        held = clone_game(game)
        if held.hold():
            held_placements = reachable_placements(
                held,
                allow_180=cfg.allow_180,
                max_nodes=cfg.reachability_node_limit,
                include_paths=True,
            )
            actions.extend(SearchAction(True, placement) for placement in held_placements)

    # Reachability already canonicalizes final placements. Keep this guard so a
    # future reachability implementation cannot silently duplicate teacher labels.
    unique: dict[tuple[object, ...], SearchAction] = {}
    for action in actions:
        unique.setdefault(_action_key(action), action)
    return tuple(unique.values())


def placement_teacher_transition_score(
    before: Game,
    after: Game,
    result: LockResult,
    weights: PlacementTeacherWeights = DEFAULT_PLACEMENT_TEACHER_WEIGHTS,
    config: PlacementTeacherConfig = DEFAULT_PLACEMENT_TEACHER_CONFIG,
    *,
    before_features: BoardFeatures | None = None,
) -> PlacementTeacherBreakdown:
    """Score one actual engine transition without mutating either game."""

    cfg = config.normalized()
    before_board = before_features or extract_board_features(before.board)
    after_board = extract_board_features(after.board)
    new_holes = max(0, after_board.holes - before_board.holes)
    difficult = is_difficult_clear(result.lines, result.spin)
    b2b_growth = max(0, int(after.b2b_chain) - int(before.b2b_chain))
    b2b_broken = bool(
        before.back_to_back
        and result.lines > 0
        and not after.back_to_back
    )
    high_stack = max(0, after_board.max_height - cfg.high_stack_height)
    combo = max(0, int(result.combo))
    spin_lines = int(result.lines) if result.spin is not None else 0

    return PlacementTeacherBreakdown(
        attack_reward=float(result.attack) * weights.attack,
        line_reward=float(result.lines) * weights.lines,
        difficult_reward=float(difficult) * weights.difficult_clear,
        spin_reward=float(spin_lines) * weights.spin_lines,
        b2b_growth_reward=float(b2b_growth) * weights.b2b_chain_growth,
        combo_reward=float(combo) * weights.combo,
        perfect_clear_reward=float(bool(result.perfect_clear)) * weights.perfect_clear,
        b2b_break_penalty=float(b2b_broken) * weights.b2b_break_penalty,
        new_holes_penalty=float(new_holes) * weights.new_holes_penalty,
        holes_penalty=float(after_board.holes) * weights.holes_penalty,
        hole_depth_penalty=float(after_board.hole_depth) * weights.hole_depth_penalty,
        max_height_penalty=float(after_board.max_height) * weights.max_height_penalty,
        high_stack_penalty=float(high_stack * high_stack) * weights.high_stack_penalty,
        bumpiness_penalty=float(after_board.bumpiness) * weights.bumpiness_penalty,
        topout_penalty=float(bool(result.game_over or after.game_over)) * weights.topout_penalty,
        before_holes=before_board.holes,
        after_holes=after_board.holes,
        after_hole_depth=after_board.hole_depth,
        after_max_height=after_board.max_height,
        after_bumpiness=after_board.bumpiness,
        new_holes=new_holes,
        difficult_clear=bool(difficult),
        b2b_broken=b2b_broken,
        b2b_chain_growth=b2b_growth,
        combo=combo,
        attack=int(result.attack),
        lines=int(result.lines),
        spin_lines=spin_lines,
        perfect_clear=bool(result.perfect_clear),
        game_over=bool(result.game_over or after.game_over),
    )


def _simulate_action(
    game: Game,
    action: SearchAction,
    weights: PlacementTeacherWeights,
    config: PlacementTeacherConfig,
    *,
    before_features: BoardFeatures | None = None,
) -> tuple[Game, PlacementTeacherBreakdown]:
    child = clone_game(game)
    result = apply_search_action(child, action)
    breakdown = placement_teacher_transition_score(
        game,
        child,
        result,
        weights,
        config,
        before_features=before_features,
    )
    return child, breakdown


def _future_value(
    game: Game,
    remaining_depth: int,
    weights: PlacementTeacherWeights,
    config: PlacementTeacherConfig,
) -> float:
    if remaining_depth <= 0 or game.game_over:
        return 0.0

    actions = _legal_actions(game, config)
    if not actions:
        return 0.0
    before_features = extract_board_features(game.board)
    immediate_nodes: list[tuple[float, SearchAction, Game]] = []
    for action in actions:
        child, breakdown = _simulate_action(
            game,
            action,
            weights,
            config,
            before_features=before_features,
        )
        immediate_nodes.append((breakdown.total, action, child))

    immediate_nodes.sort(
        key=lambda item: (
            item[0],
            -int(item[1].use_hold),
            -item[1].placement.rotation,
            -item[1].placement.x,
            tuple(item[1].placement.path),
        ),
        reverse=True,
    )
    frontier = immediate_nodes[: config.beam_width]

    if remaining_depth == 1:
        return frontier[0][0]

    best: float | None = None
    for immediate, _action, child in frontier:
        value = immediate
        if not child.game_over:
            value += config.discount * _future_value(
                child,
                remaining_depth - 1,
                weights,
                config,
            )
        if best is None or value > best:
            best = value
    return 0.0 if best is None else best


def rank_placement_teacher_actions(
    game: Game,
    weights: PlacementTeacherWeights = DEFAULT_PLACEMENT_TEACHER_WEIGHTS,
    config: PlacementTeacherConfig = DEFAULT_PLACEMENT_TEACHER_CONFIG,
) -> tuple[PlacementTeacherScore, ...]:
    """Score every legal root action with the non-heuristic offline teacher."""

    cfg = config.normalized()
    actions = _legal_actions(game, cfg)
    if not actions:
        return ()
    before_features = extract_board_features(game.board)
    scored: list[PlacementTeacherScore] = []
    for action in actions:
        child, breakdown = _simulate_action(
            game,
            action,
            weights,
            cfg,
            before_features=before_features,
        )
        future = 0.0
        if cfg.depth > 1 and not child.game_over:
            future = _future_value(child, cfg.depth - 1, weights, cfg)
        total = breakdown.total + cfg.discount * future
        scored.append(
            PlacementTeacherScore(
                action=action,
                immediate=breakdown.total,
                future_value=future,
                total=total,
                breakdown=breakdown,
            )
        )
    scored.sort(key=_ranking_key, reverse=True)
    return tuple(scored)
