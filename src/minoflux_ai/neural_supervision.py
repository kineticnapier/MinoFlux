from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from minoflux_engine import (
    Game,
    T_SPIN_DOUBLE,
    T_SPIN_MINI,
    T_SPIN_MINI_SINGLE,
    T_SPIN_SINGLE,
    T_SPIN_TRIPLE,
)

from .benchmark import HIGH_STACK_HEIGHT
from .cem import CLEAN_ATTACK_FITNESS
from .heuristic import HeuristicWeights, PlacementEvaluation
from .search import (
    SearchAction,
    SearchConfig,
    apply_search_action,
    choose_search_action,
    clone_game,
)


def teacher_score_action(
    game: Game,
    action: SearchAction,
    immediate: PlacementEvaluation,
    weights: HeuristicWeights,
    search_config: SearchConfig,
    *,
    lookahead_pieces: int,
    beam_width: int,
) -> float:
    """Score one selected root with a stronger lookahead teacher.

    Unlike the regular beam search, every selected root gets its own future
    search, so a diverse/random training candidate is not discarded merely
    because its immediate heuristic score missed the root beam.
    """

    cfg = search_config.normalized()
    depth = max(0, int(lookahead_pieces))
    if depth <= 0:
        return float(immediate.score)

    child = clone_game(game)
    apply_search_action(child, action)
    if child.game_over:
        return float(immediate.score)

    future_cfg = replace(
        cfg,
        lookahead_pieces=max(0, depth - 1),
        beam_width=max(1, int(beam_width)),
    ).normalized()
    future = choose_search_action(child, weights, future_cfg)
    if future is None:
        return float(immediate.score)
    return float(immediate.score) + cfg.discount * float(future.score)


def teacher_scores_for_actions(
    game: Game,
    ranked_actions: Sequence[tuple[SearchAction, PlacementEvaluation]],
    weights: HeuristicWeights,
    search_config: SearchConfig,
    *,
    lookahead_pieces: int,
    beam_width: int,
) -> tuple[float, ...]:
    return tuple(
        teacher_score_action(
            game,
            action,
            evaluation,
            weights,
            search_config,
            lookahead_pieces=lookahead_pieces,
            beam_width=beam_width,
        )
        for action, evaluation in ranked_actions
    )


def _spin_counts(spin: str | None) -> tuple[int, int, int, int, int]:
    return (
        int(spin == T_SPIN_MINI),
        int(spin == T_SPIN_MINI_SINGLE),
        int(spin == T_SPIN_SINGLE),
        int(spin == T_SPIN_DOUBLE),
        int(spin == T_SPIN_TRIPLE),
    )


def rollout_clean_attack_target(
    game: Game,
    action: SearchAction,
    immediate: PlacementEvaluation,
    weights: HeuristicWeights,
    search_config: SearchConfig,
    *,
    horizon: int,
) -> float:
    """Return a normalized long-horizon clean-attack value for one root action.

    The objective is the existing CLEAN_ATTACK_FITNESS applied to a single
    fixed-horizon rollout. Dividing by the requested horizon keeps target scale
    stable when the rollout length changes. Future moves use the supplied
    teacher search policy; the root action itself is always forced.
    """

    limit = max(1, int(horizon))
    cfg = search_config.normalized()
    rollout = clone_game(game)

    pieces = 0
    lines = 0
    attack = 0
    spins = 0
    spin_lines = 0
    t_spin_minis = 0
    t_spin_mini_singles = 0
    t_spin_singles = 0
    t_spin_doubles = 0
    t_spin_triples = 0
    perfect_clears = 0
    stack_samples = 0
    holes_sum = 0.0
    hole_depth_sum = 0.0
    bumpiness_sum = 0.0
    max_height_sum = 0.0
    high_stack_steps = 0

    def record_step(result, board_features) -> None:
        nonlocal pieces, lines, attack, spins, spin_lines
        nonlocal t_spin_minis, t_spin_mini_singles, t_spin_singles
        nonlocal t_spin_doubles, t_spin_triples, perfect_clears
        nonlocal stack_samples, holes_sum, hole_depth_sum, bumpiness_sum
        nonlocal max_height_sum, high_stack_steps

        pieces += 1
        lines += int(result.lines)
        attack += int(result.attack)
        if result.spin is not None:
            spins += 1
            spin_lines += int(result.lines)
        mini, mini_single, single, double, triple = _spin_counts(result.spin)
        t_spin_minis += mini
        t_spin_mini_singles += mini_single
        t_spin_singles += single
        t_spin_doubles += double
        t_spin_triples += triple
        perfect_clears += int(bool(result.perfect_clear))

        stack_samples += 1
        holes_sum += float(board_features.holes)
        hole_depth_sum += float(board_features.hole_depth)
        bumpiness_sum += float(board_features.bumpiness)
        max_height_sum += float(board_features.max_height)
        high_stack_steps += int(board_features.max_height >= HIGH_STACK_HEIGHT)

    root_result = apply_search_action(rollout, action)
    record_step(root_result, immediate.features.board)

    while not rollout.game_over and pieces < limit:
        choice = choose_search_action(rollout, weights, cfg)
        if choice is None:
            break
        result = apply_search_action(rollout, choice.action)
        record_step(result, choice.immediate.features.board)

    profile = CLEAN_ATTACK_FITNESS
    denominator = max(1, stack_samples)
    mean_holes = holes_sum / denominator
    mean_hole_depth = hole_depth_sum / denominator
    mean_bumpiness = bumpiness_sum / denominator
    mean_max_height = max_height_sum / denominator
    high_stack_fraction = high_stack_steps / denominator
    completed = not rollout.game_over and pieces >= limit

    value = (
        pieces * profile.pieces
        + lines * profile.lines
        + attack * profile.attack
        + spins * profile.spins
        + spin_lines * profile.spin_lines
        + t_spin_minis * profile.t_spin_minis
        + t_spin_mini_singles * profile.t_spin_mini_singles
        + t_spin_singles * profile.t_spin_singles
        + t_spin_doubles * profile.t_spin_doubles
        + t_spin_triples * profile.t_spin_triples
        + perfect_clears * profile.perfect_clears
        + int(completed) * limit * profile.completion_bonus
        - int(rollout.game_over) * limit * profile.topout_penalty
        - mean_holes * profile.mean_holes_penalty
        - mean_hole_depth * profile.mean_hole_depth_penalty
        - mean_bumpiness * profile.mean_bumpiness_penalty
        - mean_max_height * profile.mean_max_height_penalty
        - high_stack_fraction * limit * profile.high_stack_penalty
    )
    return float(value) / float(limit)
