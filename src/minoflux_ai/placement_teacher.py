from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
import json
import os
from pathlib import Path
import random
from typing import Callable, Iterable, Mapping, Sequence

from minoflux_engine import Game, LockResult
from minoflux_engine.b2b import resolve_b2b_charging
from minoflux_engine.spin import base_attack, is_difficult_clear, t_spin_event

from .bitboard import (
    board_row_masks,
    classify_t_spin_row_masks,
    collides_row_masks,
    hidden_rows_occupied,
    place_and_clear_row_masks,
)
from .features import (
    BoardFeatures,
    extract_teacher_board_features,
    extract_teacher_board_features_from_masks,
)
from .neural import NeuralValueConfig, encode_game_state
from .neural_dataset import NEURAL_DATASET_FORMAT, pack_board_rows
from .reachability import reachable_placements
from .search import SearchAction, _held_search_game, apply_search_action, clone_game


@dataclass(frozen=True, slots=True)
class PlacementTeacherWeights:
    """Explicit engine-outcome objective for the offline placement teacher.

    Attack remains the dominant positive term. Small B2B/combo terms reward
    preserving offensive structure without double-counting the attack bonuses
    that the engine already includes in ``LockResult.attack``.
    """

    attack: float = 8.0
    lines: float = 0.35
    difficult_clear: float = 1.0
    spin_lines: float = 0.75
    b2b_chain_growth: float = 0.35
    combo: float = 0.20
    perfect_clear: float = 24.0
    b2b_break_penalty: float = 2.5
    new_holes_penalty: float = 11.0
    holes_penalty: float = 2.25
    hole_depth_penalty: float = 0.55
    max_height_penalty: float = 0.35
    high_stack_penalty: float = 0.80
    bumpiness_penalty: float = 0.18
    topout_penalty: float = 10_000.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PlacementTeacherWeights":
        names = {item.name for item in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"Unknown placement-teacher weights: {sorted(unknown)}")
        defaults = cls().to_dict()
        defaults.update({key: float(value) for key, value in values.items()})
        return cls(**defaults)


DEFAULT_PLACEMENT_TEACHER_WEIGHTS = PlacementTeacherWeights()


@dataclass(frozen=True, slots=True)
class PlacementTeacherConfig:
    """Search settings for the exact, offline-only placement teacher.

    ``depth`` counts the root placement. depth=1 is an immediate objective;
    depth=2 evaluates one future placement, and so on. Every legal root action
    is scored. ``beam_width`` only limits recursively explored future actions.
    """

    depth: int = 2
    beam_width: int = 24
    discount: float = 0.92
    allow_hold: bool = True
    allow_180: bool = False
    reachability_node_limit: int = 8_000
    high_stack_height: int = 12

    def normalized(self) -> "PlacementTeacherConfig":
        return PlacementTeacherConfig(
            depth=min(3, max(1, int(self.depth))),
            beam_width=min(128, max(1, int(self.beam_width))),
            discount=min(1.0, max(0.0, float(self.discount))),
            allow_hold=bool(self.allow_hold),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
            high_stack_height=min(23, max(1, int(self.high_stack_height))),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())


DEFAULT_PLACEMENT_TEACHER_CONFIG = PlacementTeacherConfig()


@dataclass(frozen=True, slots=True)
class PlacementTeacherBreakdown:
    attack_reward: float
    line_reward: float
    difficult_reward: float
    spin_reward: float
    b2b_growth_reward: float
    combo_reward: float
    perfect_clear_reward: float
    b2b_break_penalty: float
    new_holes_penalty: float
    holes_penalty: float
    hole_depth_penalty: float
    max_height_penalty: float
    high_stack_penalty: float
    bumpiness_penalty: float
    topout_penalty: float
    before_holes: int
    after_holes: int
    after_hole_depth: int
    after_max_height: int
    after_bumpiness: int
    new_holes: int
    difficult_clear: bool
    b2b_broken: bool
    b2b_chain_growth: int
    combo: int
    attack: int
    lines: int
    spin_lines: int
    perfect_clear: bool
    game_over: bool

    @property
    def total(self) -> float:
        positive = (
            self.attack_reward
            + self.line_reward
            + self.difficult_reward
            + self.spin_reward
            + self.b2b_growth_reward
            + self.combo_reward
            + self.perfect_clear_reward
        )
        negative = (
            self.b2b_break_penalty
            + self.new_holes_penalty
            + self.holes_penalty
            + self.hole_depth_penalty
            + self.max_height_penalty
            + self.high_stack_penalty
            + self.bumpiness_penalty
            + self.topout_penalty
        )
        return positive - negative

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["immediateTotal"] = self.total
        return result


@dataclass(frozen=True, slots=True)
class PlacementTeacherScore:
    action: SearchAction
    immediate: float
    future_value: float
    total: float
    breakdown: PlacementTeacherBreakdown

    def to_dict(self) -> dict[str, object]:
        return {
            "action": self.action.to_dict(),
            "immediate": self.immediate,
            "futureValue": self.future_value,
            "total": self.total,
            "breakdown": self.breakdown.to_dict(),
        }


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
    *,
    include_paths: bool = True,
) -> tuple[SearchAction, ...]:
    """Enumerate exact-SRS root actions without heuristic filtering."""

    return _legal_actions_normalized(game, config.normalized(), include_paths=include_paths)


def _legal_actions_normalized(
    game: Game,
    cfg: PlacementTeacherConfig,
    *,
    include_paths: bool = True,
) -> tuple[SearchAction, ...]:
    direct = reachable_placements(
        game,
        allow_180=cfg.allow_180,
        max_nodes=cfg.reachability_node_limit,
        include_paths=include_paths,
    )
    actions: list[SearchAction] = [SearchAction(False, placement) for placement in direct]

    if cfg.allow_hold and not game.hold_used and not game.game_over and not game.paused:
        # The search view shares occupancy/RNG read-only. Unusual empty queues
        # (or custom engine subclasses) need the real engine's refill/hold.
        if type(game) is not Game or (game.hold_piece is None and not game.queue):
            held = clone_game(game)
            if not held.hold():
                held = None
        else:
            held = _held_search_game(game)
        if held is not None:
            held_placements = reachable_placements(
                held,
                allow_180=cfg.allow_180,
                max_nodes=cfg.reachability_node_limit,
                include_paths=include_paths,
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

    return _transition_score_normalized(
        before, after, result, weights, config.normalized(),
        before_features=before_features,
    )


def _transition_score_normalized(
    before: Game,
    after: Game,
    result: LockResult,
    weights: PlacementTeacherWeights,
    cfg: PlacementTeacherConfig,
    *,
    before_features: BoardFeatures | None = None,
) -> PlacementTeacherBreakdown:
    before_board = before_features or extract_teacher_board_features(before.board)
    after_board = extract_teacher_board_features(after.board)
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
    breakdown = _transition_score_normalized(
        game,
        child,
        result,
        weights,
        config,
        before_features=before_features,
    )
    return child, breakdown


def _leaf_action_score(
    game: Game,
    action: SearchAction,
    weights: PlacementTeacherWeights,
    config: PlacementTeacherConfig,
    *,
    source_rows: Sequence[int],
    before_features: BoardFeatures,
) -> float:
    """Exact immediate objective for an already-legal final-ply action.

    No child, bag, or RNG is needed: the next piece is already in the queue.
    Keep the engine fallback for custom Game implementations and short queues
    that require bag draws before the next spawn. Only occupancy and the lock
    outcome affect the teacher; score/telemetry/queue updates are unobserved.
    """

    next_index = int(action.use_hold and game.hold_piece is None)
    if type(game) is not Game or len(game.queue) <= next_index:
        return _simulate_action(
            game, action, weights, config, before_features=before_features,
        )[1].total

    placement = action.placement
    spin_kind = classify_t_spin_row_masks(
        source_rows,
        piece=placement.piece,
        x=placement.x,
        y=placement.y,
        rotation=placement.rotation,
        last_move_was_rotation=placement.last_move_was_rotation,
        rotation_kick_index=placement.rotation_kick_index,
        width=game.width,
    )
    after_rows, lines, topped_out = place_and_clear_row_masks(
        source_rows, placement, width=game.width,
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
    attack = base_attack(lines, spin) + b2b.attack_bonus
    combo = game.combo + 1 if lines else -1
    if lines and combo > 0:
        attack += min(4, combo // 2 + 1)
    if perfect_clear and lines:
        attack += 10
    # split_surge only partitions this integer; the engine's total is its sum.
    attack += b2b.released
    game_over = (
        topped_out
        or hidden_rows_occupied(after_rows, game.hidden_rows)
        or collides_row_masks(
            after_rows, game.queue[next_index], 3, 1, 0, width=game.width,
        )
    )
    board = extract_teacher_board_features_from_masks(after_rows, width=game.width)
    new_holes = max(0, board.holes - before_features.holes)
    b2b_growth = max(0, int(b2b.chain) - int(game.b2b_chain))
    b2b_broken = bool(game.back_to_back and lines > 0 and not b2b.active)
    high_stack = max(0, board.max_height - config.high_stack_height)
    spin_lines = int(lines) if spin is not None else 0

    # Preserve the exact float conversions, grouping and addition order of
    # PlacementTeacherBreakdown.total. Reassociation changes dataset labels.
    positive = (
        float(attack) * weights.attack
        + float(lines) * weights.lines
        + float(difficult) * weights.difficult_clear
        + float(spin_lines) * weights.spin_lines
        + float(b2b_growth) * weights.b2b_chain_growth
        + float(max(0, int(combo))) * weights.combo
        + float(perfect_clear) * weights.perfect_clear
    )
    negative = (
        float(b2b_broken) * weights.b2b_break_penalty
        + float(new_holes) * weights.new_holes_penalty
        + float(board.holes) * weights.holes_penalty
        + float(board.hole_depth) * weights.hole_depth_penalty
        + float(board.max_height) * weights.max_height_penalty
        + float(high_stack * high_stack) * weights.high_stack_penalty
        + float(board.bumpiness) * weights.bumpiness_penalty
        + float(bool(game_over)) * weights.topout_penalty
    )
    return positive - negative


def _future_value(
    game: Game,
    remaining_depth: int,
    weights: PlacementTeacherWeights,
    config: PlacementTeacherConfig,
) -> float:
    if remaining_depth <= 0 or game.game_over:
        return 0.0

    actions = _legal_actions_normalized(game, config, include_paths=remaining_depth > 1)
    if not actions:
        return 0.0
    source_rows = board_row_masks(game.board)
    before_features = extract_teacher_board_features_from_masks(source_rows, width=game.width)
    if remaining_depth == 1:
        # No child survives a leaf: evaluate all legal actions, retaining just
        # the scalar maximum. beam_width cannot affect this value.
        return max(
            _leaf_action_score(
                game, action, weights, config,
                source_rows=source_rows, before_features=before_features,
            )
            for action in actions
        )
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
    actions = _legal_actions_normalized(game, cfg)
    if not actions:
        return ()
    before_features = extract_teacher_board_features(game.board)
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


def choose_placement_teacher_action(
    game: Game,
    weights: PlacementTeacherWeights = DEFAULT_PLACEMENT_TEACHER_WEIGHTS,
    config: PlacementTeacherConfig = DEFAULT_PLACEMENT_TEACHER_CONFIG,
) -> PlacementTeacherScore | None:
    ranked = rank_placement_teacher_actions(game, weights, config)
    return ranked[0] if ranked else None


@dataclass(frozen=True, slots=True)
class PlacementV2DatasetConfig:
    games: int = 50
    max_pieces: int = 300
    seed_base: int = 8_000_001
    seed_step: int = 31
    max_candidates: int = 24
    hard_candidates: int = 8
    medium_candidates: int = 5
    random_candidates: int = 5
    bad_candidates: int = 5
    acceptable_margin: float = 0.50
    random_seed: int = 26_090_904
    teacher: PlacementTeacherConfig = PlacementTeacherConfig()
    neural: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "PlacementV2DatasetConfig":
        return PlacementV2DatasetConfig(
            games=max(1, int(self.games)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=max(1, int(self.seed_step)),
            max_candidates=max(1, int(self.max_candidates)),
            hard_candidates=max(0, int(self.hard_candidates)),
            medium_candidates=max(0, int(self.medium_candidates)),
            random_candidates=max(0, int(self.random_candidates)),
            bad_candidates=max(0, int(self.bad_candidates)),
            acceptable_margin=max(0.0, float(self.acceptable_margin)),
            random_seed=int(self.random_seed),
            teacher=self.teacher.normalized(),
            neural=self.neural.normalized(),
        )

    def to_dict(self) -> dict[str, object]:
        cfg = self.normalized()
        return {
            "games": cfg.games,
            "maxPieces": cfg.max_pieces,
            "seedBase": cfg.seed_base,
            "seedStep": cfg.seed_step,
            "maxCandidates": cfg.max_candidates,
            "hardCandidates": cfg.hard_candidates,
            "mediumCandidates": cfg.medium_candidates,
            "randomCandidates": cfg.random_candidates,
            "badCandidates": cfg.bad_candidates,
            "acceptableMargin": cfg.acceptable_margin,
            "randomSeed": cfg.random_seed,
            "teacher": cfg.teacher.to_dict(),
            "neural": asdict(cfg.neural),
        }


def _select_dataset_indices(
    ranked: Sequence[PlacementTeacherScore],
    config: PlacementV2DatasetConfig,
    rng: random.Random,
) -> tuple[tuple[int, str], ...]:
    cfg = config.normalized()
    n = len(ranked)
    if n <= cfg.max_candidates:
        return tuple((index, "expert" if index == 0 else "all") for index in range(n))

    cap = min(n, cfg.max_candidates)
    selected: dict[int, str] = {0: "expert"}

    # Keep every near-optimal positive first so acceptable actions do not vanish
    # merely because the negative-sampling cap was small.
    best = ranked[0].total
    for index, score in enumerate(ranked[1:], start=1):
        if len(selected) >= cap:
            break
        if best - score.total <= cfg.acceptable_margin + 1e-12:
            selected[index] = "acceptable"

    hard_pool = list(range(1, max(2, n // 3)))
    medium_pool = list(range(max(1, n // 3), max(2, (2 * n) // 3)))
    bad_pool = list(range(max(0, (2 * n) // 3), n))

    def take(pool: Sequence[int], count: int, bucket: str, *, randomize: bool) -> None:
        remaining = cap - len(selected)
        if remaining <= 0 or count <= 0:
            return
        available = [index for index in pool if index not in selected]
        wanted = min(remaining, count, len(available))
        if wanted <= 0:
            return
        choices = rng.sample(available, wanted) if randomize else available[:wanted]
        for index in choices:
            selected[index] = bucket

    take(hard_pool, cfg.hard_candidates, "hard", randomize=False)
    take(medium_pool, cfg.medium_candidates, "medium", randomize=True)
    take(tuple(reversed(bad_pool)), cfg.bad_candidates, "bad", randomize=False)
    take(tuple(range(n)), cfg.random_candidates, "random", randomize=True)

    if len(selected) < cap:
        for index in range(n):
            if index in selected:
                continue
            selected[index] = "fill"
            if len(selected) >= cap:
                break

    return tuple((index, selected[index]) for index in sorted(selected))


def _move_list(action: SearchAction) -> list[object]:
    placement = action.placement
    return [
        int(action.use_hold),
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
    ]


def _candidate_dict(
    game: Game,
    score: PlacementTeacherScore,
    bucket: str,
    neural_config: NeuralValueConfig,
) -> dict[str, object]:
    child = clone_game(game)
    apply_search_action(child, score.action)
    state = encode_game_state(child, neural_config)
    return {
        "rows": list(pack_board_rows(state.board, neural_config)),
        "context": list(state.context),
        "move": _move_list(score.action),
        "teacherScore": float(score.total),
        "samplingBucket": bucket,
        "teacherBreakdown": score.breakdown.to_dict(),
        "teacherImmediate": float(score.immediate),
        "teacherFutureValue": float(score.future_value),
    }


def _generate_game_records(
    config: PlacementV2DatasetConfig,
    weights: PlacementTeacherWeights,
    game_index: int,
) -> tuple[dict[str, object], ...]:
    cfg = config.normalized()
    seed = cfg.seed_base + game_index * cfg.seed_step
    game = Game(seed)
    records: list[dict[str, object]] = []

    while not game.game_over and game.pieces_placed < cfg.max_pieces:
        ranked = rank_placement_teacher_actions(game, weights, cfg.teacher)
        if not ranked:
            break
        state_rng = random.Random(
            cfg.random_seed
            ^ seed
            ^ ((game.pieces_placed + 1) * 0x9E3779B1)
        )
        selected = _select_dataset_indices(ranked, cfg, state_rng)
        if not selected:
            break

        old_to_new = {old_index: new_index for new_index, (old_index, _bucket) in enumerate(selected)}
        best_score = ranked[0].total
        positive_old = tuple(
            index
            for index, score in enumerate(ranked)
            if best_score - score.total <= cfg.acceptable_margin + 1e-12
            and index in old_to_new
        )
        expert_indices = tuple(sorted(old_to_new[index] for index in positive_old)) or (0,)
        candidates = [
            _candidate_dict(game, ranked[index], bucket, cfg.neural)
            for index, bucket in selected
        ]
        records.append(
            {
                "format": NEURAL_DATASET_FORMAT,
                "seed": seed,
                "pieceIndex": game.pieces_placed,
                "expertIndex": 0,
                "expertIndices": list(expert_indices),
                "candidates": candidates,
                "teacher": "placement-v2",
            }
        )
        apply_search_action(game, ranked[0].action)

    return tuple(records)


def _generate_game_task(
    task: tuple[PlacementV2DatasetConfig, PlacementTeacherWeights, int],
) -> tuple[dict[str, object], ...]:
    return _generate_game_records(*task)


def _resolve_workers(requested: int, games: int) -> int:
    if int(requested) > 0:
        return min(games, max(1, int(requested)))
    return min(games, max(1, (os.cpu_count() or 1) - 1))


def write_placement_v2_dataset(
    path: str | Path,
    config: PlacementV2DatasetConfig = PlacementV2DatasetConfig(),
    weights: PlacementTeacherWeights = DEFAULT_PLACEMENT_TEACHER_WEIGHTS,
    *,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, object]:
    """Write a training-compatible ranking JSONL using Placement Teacher v2."""

    cfg = config.normalized()
    resolved_workers = _resolve_workers(workers, cfg.games)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = 0
    candidates = 0

    tasks = ((cfg, weights, game_index) for game_index in range(cfg.games))
    if resolved_workers == 1:
        game_records: Iterable[tuple[dict[str, object], ...]] = (
            _generate_game_task(task) for task in tasks
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=resolved_workers)
        game_records = executor.map(_generate_game_task, tasks)

    try:
        with target.open("w", encoding="utf-8") as stream:
            for records in game_records:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    samples += 1
                    raw_candidates = record.get("candidates", ())
                    candidates += len(raw_candidates) if isinstance(raw_candidates, Sequence) else 0
                if progress is not None:
                    progress(samples, candidates)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    return {
        "format": NEURAL_DATASET_FORMAT,
        "teacher": "placement-v2",
        "path": str(target),
        "games": cfg.games,
        "samples": samples,
        "candidates": candidates,
        "workers": resolved_workers,
        "datasetConfig": cfg.to_dict(),
        "teacherWeights": weights.to_dict(),
    }


def load_placement_teacher_weights(path: str | Path) -> PlacementTeacherWeights:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Placement teacher weight file must contain a JSON object")
    return PlacementTeacherWeights.from_mapping(value)
