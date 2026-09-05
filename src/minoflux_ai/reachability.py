from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
import time
from typing import Iterator, Sequence

from minoflux_engine import Game, Placement
from minoflux_engine.pieces import SHAPES, kick_tests

from .bitboard import (
    board_row_masks,
    classify_t_spin_row_masks,
    placement_cells,
)

MOVE_LEFT = "left"
MOVE_RIGHT = "right"
MOVE_DOWN = "down"
ROTATE_CW = "cw"
ROTATE_CCW = "ccw"
ROTATE_180 = "180"
HARD_DROP = "hard_drop"

_CMD_NONE = -1
_CMD_LEFT = 0
_CMD_RIGHT = 1
_CMD_DOWN = 2
_CMD_CW = 3
_CMD_CCW = 4
_CMD_180 = 5
_COMMAND_NAMES = (MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, ROTATE_CW, ROTATE_CCW, ROTATE_180)

_NO_STATE = -1
_NO_LANDING = -1
_COLLISION_UNKNOWN = 0
_COLLISION_CLEAR = 1
_COLLISION_BLOCKED = 2
_X_MARGIN = 4
_Y_MIN = -4
_REACHABILITY_CACHE_MAXSIZE = 8_192

_ReachabilityCacheKey = tuple[
    tuple[int, ...],
    str,
    int,
    int,
    int,
    int,
    int,
    bool,
    int,
    bool,
]
_REACHABILITY_CACHE: OrderedDict[
    _ReachabilityCacheKey,
    tuple[Placement, ...],
] = OrderedDict()


@dataclass(slots=True)
class ReachabilityProfile:
    """Low-overhead aggregate timings for exact-SRS placement generation."""

    calls: int = 0
    total_seconds: float = 0.0
    board_mask_seconds: float = 0.0
    setup_seconds: float = 0.0
    bfs_seconds: float = 0.0
    rotation_seconds: float = 0.0
    landing_seconds: float = 0.0
    representative_seconds: float = 0.0
    placement_seconds: float = 0.0
    path_seconds: float = 0.0
    bfs_nodes: int = 0
    collision_checks: int = 0
    collision_evaluations: int = 0
    collision_cache_hits: int = 0
    kick_checks: int = 0
    landing_queries: int = 0
    landing_cache_hits: int = 0
    representative_nodes: int = 0
    representative_duplicate_skips: int = 0
    placements: int = 0
    path_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    def to_dict(self) -> dict[str, int | float]:
        accounted = (
            self.board_mask_seconds
            + self.setup_seconds
            + self.bfs_seconds
            + self.landing_seconds
            + self.representative_seconds
            + self.placement_seconds
        )
        return {
            "calls": self.calls,
            "totalSeconds": self.total_seconds,
            "boardMaskSeconds": self.board_mask_seconds,
            "setupSeconds": self.setup_seconds,
            "bfsSeconds": self.bfs_seconds,
            "rotationSeconds": self.rotation_seconds,
            "bfsOtherSeconds": max(0.0, self.bfs_seconds - self.rotation_seconds),
            "landingSeconds": self.landing_seconds,
            "representativeSeconds": self.representative_seconds,
            "placementSeconds": self.placement_seconds,
            "pathSeconds": self.path_seconds,
            "unaccountedSeconds": max(0.0, self.total_seconds - accounted),
            "bfsNodes": self.bfs_nodes,
            "collisionChecks": self.collision_checks,
            "collisionEvaluations": self.collision_evaluations,
            "collisionCacheHits": self.collision_cache_hits,
            "collisionCacheHitRate": self.collision_cache_hits / max(1, self.collision_checks),
            "kickChecks": self.kick_checks,
            "landingQueries": self.landing_queries,
            "landingCacheHits": self.landing_cache_hits,
            "representativeNodes": self.representative_nodes,
            "representativeDuplicateSkips": self.representative_duplicate_skips,
            "placements": self.placements,
            "pathCalls": self.path_calls,
            "cacheHits": self.cache_hits,
            "cacheMisses": self.cache_misses,
            "cacheHitRate": self.cache_hits
            / max(1, self.cache_hits + self.cache_misses),
            "bfsNodesPerCall": self.bfs_nodes / max(1, self.calls),
            "placementsPerCall": self.placements / max(1, self.calls),
            "landingCacheHitRate": self.landing_cache_hits / max(1, self.landing_queries),
        }


_ACTIVE_REACHABILITY_PROFILE: ContextVar[ReachabilityProfile | None] = ContextVar(
    "minoflux_reachability_profile",
    default=None,
)


@contextmanager
def collect_reachability_profile(
    profile: ReachabilityProfile | None = None,
) -> Iterator[ReachabilityProfile]:
    """Collect reachability timings in the current execution context."""

    active = profile if profile is not None else ReachabilityProfile()
    token = _ACTIVE_REACHABILITY_PROFILE.set(active)
    try:
        yield active
    finally:
        _ACTIVE_REACHABILITY_PROFILE.reset(token)


def clear_reachability_cache() -> None:
    """Clear cached exact-SRS results, primarily for benchmarks and tests."""

    _REACHABILITY_CACHE.clear()


def _cache_reachability_result(
    key: _ReachabilityCacheKey,
    result: tuple[Placement, ...],
) -> tuple[Placement, ...]:
    _REACHABILITY_CACHE[key] = result
    if len(_REACHABILITY_CACHE) > _REACHABILITY_CACHE_MAXSIZE:
        _REACHABILITY_CACHE.popitem(last=False)
    return result


def _state_layout_values(width: int, height: int) -> tuple[int, int, int, int, int]:
    x_min = -_X_MARGIN
    x_max = width + _X_MARGIN - 1
    x_count = x_max - x_min + 1
    y_min = _Y_MIN
    y_count = height - y_min
    state_count = x_count * y_count * 4
    return x_min, x_max, x_count, y_min, state_count


def _state_layout(game: Game) -> tuple[int, int, int, int, int]:
    return _state_layout_values(game.width, game.height)


def _pack_state(
    x: int,
    y: int,
    rotation: int,
    *,
    x_min: int,
    x_max: int,
    x_count: int,
    y_min: int,
    height: int,
) -> int:
    if x < x_min or x > x_max or y < y_min or y >= height:
        return _NO_STATE
    return ((((y - y_min) * x_count) + (x - x_min)) << 2) | (rotation & 3)


def _path(
    parents: list[int],
    commands: list[int],
    node_index: int,
) -> tuple[str, ...]:
    result: list[str] = []
    index = node_index
    while index >= 0:
        command = commands[index]
        if command != _CMD_NONE:
            result.append(_COMMAND_NAMES[command])
        index = parents[index]
    result.reverse()
    result.append(HARD_DROP)
    return tuple(result)


@cache
def _geometry_key(piece: str, x: int, y: int, rotation: int, width: int) -> int:
    """Canonical final-cell key retained for compatibility/tests."""

    key = 0
    for dx, dy in SHAPES[piece][rotation % 4]:
        cell_x, cell_y = x + dx, y + dy
        if cell_y < 0:
            return -1
        key |= 1 << (cell_y * width + cell_x)
    return key


@cache
def _rotation_options(
    piece: str,
    allow_180: bool,
    x_count: int,
) -> tuple[tuple[tuple[int, int, tuple[tuple[int, int, int, int], ...]], ...], ...]:
    """Precompute exact SRS targets, commands, kicks, and packed-state deltas."""

    if piece == "O":
        return ((), (), (), ())
    rotation_specs = (
        ((1, _CMD_CW), (-1, _CMD_CCW), (2, _CMD_180))
        if allow_180
        else ((1, _CMD_CW), (-1, _CMD_CCW))
    )
    return tuple(
        tuple(
            (
                target_rotation,
                command,
                tuple(
                    (
                        kick_index,
                        kick_x,
                        kick_y,
                        ((kick_y * x_count + kick_x) << 2) + target_rotation,
                    )
                    for kick_index, (kick_x, kick_y) in enumerate(
                        kick_tests(piece, source_rotation, target_rotation)
                    )
                ),
            )
            for direction, command in rotation_specs
            for target_rotation in ((source_rotation + direction) & 3,)
        )
        for source_rotation in range(4)
    )


@dataclass(frozen=True, slots=True)
class _StateTables:
    x_min: int
    x_max: int
    x_count: int
    y_min: int
    state_count: int
    state_x: tuple[int, ...]
    state_y: tuple[int, ...]
    left_state: tuple[int, ...]
    right_state: tuple[int, ...]
    down_state: tuple[int, ...]
    collision_mask: tuple[int, ...]
    geometry_mask: tuple[int, ...]


@cache
def _state_tables(piece: str, width: int, height: int) -> _StateTables:
    """Precompute board-independent packed-state geometry for one piece."""

    x_min, x_max, x_count, y_min, state_count = _state_layout_values(width, height)
    row_state_stride = x_count << 2

    state_x: list[int] = []
    state_y: list[int] = []
    left_state: list[int] = []
    right_state: list[int] = []
    down_state: list[int] = []
    collision_mask: list[int] = []
    geometry_mask: list[int] = []

    for y in range(y_min, height):
        for x in range(x_min, x_max + 1):
            for rotation in range(4):
                state_id = len(state_x)
                state_x.append(x)
                state_y.append(y)
                left_state.append(state_id - 4 if x > x_min else _NO_STATE)
                right_state.append(state_id + 4 if x < x_max else _NO_STATE)
                down_state.append(
                    state_id + row_state_stride if y + 1 < height else _NO_STATE
                )

                mask = 0
                blocked = False
                above_top = False
                for dx, dy in SHAPES[piece][rotation]:
                    cell_x = x + dx
                    cell_y = y + dy
                    if cell_x < 0 or cell_x >= width or cell_y >= height:
                        blocked = True
                        break
                    if cell_y < 0:
                        above_top = True
                        continue
                    mask |= 1 << (cell_y * width + cell_x)

                collision_mask.append(-1 if blocked else mask)
                geometry_mask.append(-1 if blocked or above_top else mask)

    if len(state_x) != state_count:
        raise AssertionError(f"Packed-state table mismatch: {len(state_x)} != {state_count}")

    return _StateTables(
        x_min=x_min,
        x_max=x_max,
        x_count=x_count,
        y_min=y_min,
        state_count=state_count,
        state_x=tuple(state_x),
        state_y=tuple(state_y),
        left_state=tuple(left_state),
        right_state=tuple(right_state),
        down_state=tuple(down_state),
        collision_mask=tuple(collision_mask),
        geometry_mask=tuple(geometry_mask),
    )


@cache
def _rotation_transitions(
    piece: str,
    allow_180: bool,
    width: int,
    height: int,
) -> tuple[tuple[tuple[int, tuple[tuple[int, int], ...]], ...], ...]:
    """Precompute all in-layout SRS kick target states for every packed state."""

    tables = _state_tables(piece, width, height)
    options = _rotation_options(piece, allow_180, tables.x_count)
    transitions: list[tuple[tuple[int, tuple[tuple[int, int], ...]], ...]] = []

    for state_id in range(tables.state_count):
        x = tables.state_x[state_id]
        y = tables.state_y[state_id]
        source_rotation = state_id & 3
        state_base = state_id & ~3
        groups: list[tuple[int, tuple[tuple[int, int], ...]]] = []

        for _target_rotation, command, kicks in options[source_rotation]:
            valid: list[tuple[int, int]] = []
            for kick_index, kick_x, kick_y, packed_delta in kicks:
                target_x = x + kick_x
                target_y = y + kick_y
                if target_y < tables.y_min or target_y >= height:
                    continue
                if target_x < tables.x_min or target_x > tables.x_max:
                    continue
                valid.append((state_base + packed_delta, kick_index))
            groups.append((command, tuple(valid)))
        transitions.append(tuple(groups))

    return tuple(transitions)


def _rows_to_board_bits(rows: Sequence[int], width: int) -> int:
    """Pack row masks into one Python integer so collision is one C-level AND."""

    result = 0
    shift = 0
    for row in rows:
        result |= int(row) << shift
        shift += width
    return result


def reachable_placements(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
    include_paths: bool = True,
) -> tuple[Placement, ...]:
    """Enumerate exact-SRS placements with packed-state BFS and cached geometry."""

    profile = _ACTIVE_REACHABILITY_PROFILE.get()
    profiling = profile is not None
    profile_started = time.perf_counter() if profiling else 0.0
    if profile is not None:
        profile.calls += 1

    if game.game_over or game.paused:
        if profile is not None:
            profile.total_seconds += time.perf_counter() - profile_started
        return ()

    board_mask_started = time.perf_counter() if profiling else 0.0
    rows = board_row_masks(game.board)
    if profile is not None:
        profile.board_mask_seconds += time.perf_counter() - board_mask_started

    normalized_allow_180 = bool(allow_180)
    normalized_max_nodes = max(1, int(max_nodes))
    normalized_include_paths = bool(include_paths)
    cache_key: _ReachabilityCacheKey = (
        rows,
        game.current,
        game.x,
        game.y,
        game.rotation & 3,
        game.width,
        game.height,
        normalized_allow_180,
        normalized_max_nodes,
        normalized_include_paths,
    )
    cached = _REACHABILITY_CACHE.pop(cache_key, None)
    if cached is not None:
        _REACHABILITY_CACHE[cache_key] = cached
        if profile is not None:
            profile.cache_hits += 1
            profile.placements += len(cached)
            profile.total_seconds += time.perf_counter() - profile_started
        return cached
    if profile is not None:
        profile.cache_misses += 1

    setup_started = time.perf_counter() if profiling else 0.0

    piece = game.current
    piece_is_t = piece == "T"
    width = game.width
    height = game.height
    tables = _state_tables(piece, width, height)
    x_min = tables.x_min
    x_max = tables.x_max
    x_count = tables.x_count
    y_min = tables.y_min
    packed_count = tables.state_count
    state_x = tables.state_x
    state_y = tables.state_y
    left_state = tables.left_state
    right_state = tables.right_state
    down_state = tables.down_state
    collision_masks = tables.collision_mask
    geometry_masks = tables.geometry_mask
    rotation_transitions = _rotation_transitions(
        piece,
        normalized_allow_180,
        width,
        height,
    )

    start_rotation = game.rotation & 3
    if (
        game.x < x_min
        or game.x > x_max
        or game.y < y_min
        or game.y >= height
    ):
        if profile is not None:
            profile.setup_seconds += time.perf_counter() - setup_started
            profile.total_seconds += time.perf_counter() - profile_started
        return _cache_reachability_result(cache_key, ())
    start_state = (
        ((((game.y - y_min) * x_count) + (game.x - x_min)) << 2)
        | start_rotation
    )

    board_bits = _rows_to_board_bits(rows, width)
    collision_cache = bytearray(packed_count)
    collision_checks = 0
    collision_evaluations = 0
    collision_cache_hits = 0
    kick_checks = 0

    if profile is not None:
        collision_checks += 1
        collision_evaluations += 1
    shape_mask = collision_masks[start_state]
    start_blocked = shape_mask < 0 or bool(board_bits & shape_mask)
    collision_cache[start_state] = (
        _COLLISION_BLOCKED if start_blocked else _COLLISION_CLEAR
    )
    if start_blocked:
        if profile is not None:
            profile.collision_checks += collision_checks
            profile.collision_evaluations += collision_evaluations
            profile.collision_cache_hits += collision_cache_hits
            profile.kick_checks += kick_checks
            profile.setup_seconds += time.perf_counter() - setup_started
            profile.total_seconds += time.perf_counter() - profile_started
        return _cache_reachability_result(cache_key, ())

    landing_state = [_NO_LANDING] * packed_count

    if profile is None:

        def compute_landing_state(state_id: int) -> int:
            trail: list[int] = [state_id]
            current_id = state_id
            while True:
                target_id = down_state[current_id]
                if target_id == _NO_STATE:
                    result = current_id
                    break
                cached_collision = collision_cache[target_id]
                if cached_collision == _COLLISION_UNKNOWN:
                    shape_mask = collision_masks[target_id]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_id] = (
                        _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                    )
                else:
                    blocked = cached_collision == _COLLISION_BLOCKED
                if blocked:
                    result = current_id
                    break
                current_id = target_id
                cached_landing = landing_state[current_id]
                if cached_landing != _NO_LANDING:
                    result = cached_landing
                    break
                trail.append(current_id)
            for cached_id in trail:
                landing_state[cached_id] = result
            return result

    else:

        def compute_landing_state(state_id: int) -> int:
            nonlocal collision_checks, collision_evaluations, collision_cache_hits
            trail: list[int] = [state_id]
            current_id = state_id
            while True:
                target_id = down_state[current_id]
                if target_id == _NO_STATE:
                    result = current_id
                    break
                collision_checks += 1
                cached_collision = collision_cache[target_id]
                if cached_collision == _COLLISION_UNKNOWN:
                    shape_mask = collision_masks[target_id]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_id] = (
                        _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                    )
                    collision_evaluations += 1
                else:
                    collision_cache_hits += 1
                    blocked = cached_collision == _COLLISION_BLOCKED
                if blocked:
                    result = current_id
                    break
                current_id = target_id
                cached_landing = landing_state[current_id]
                if cached_landing != _NO_LANDING:
                    profile.landing_cache_hits += 1
                    result = cached_landing
                    break
                trail.append(current_id)
            for cached_id in trail:
                landing_state[cached_id] = result
            return result

    node_states: list[int] = [start_state]
    parents: list[int] = [-1]
    commands: list[int] = [_CMD_NONE] if normalized_include_paths else []
    depths: list[int] = [0]
    kick_indices: list[int] = [-1]

    append_node_state = node_states.append
    append_parent = parents.append
    append_depth = depths.append
    append_kick_index = kick_indices.append

    frontier: list[int] = [0]
    append_frontier = frontier.append
    frontier_index = 0

    state_nodes = [_NO_STATE] * packed_count
    rotation_nodes = [_NO_STATE] * packed_count if piece_is_t else []
    state_nodes[start_state] = 0
    visited_state_ids: list[int] = [start_state]
    visited_rotation_ids: list[int] = []
    append_visited_state = visited_state_ids.append
    append_visited_rotation = visited_rotation_ids.append

    reachable_count = 1
    budget = normalized_max_nodes

    if profile is not None:
        profile.setup_seconds += time.perf_counter() - setup_started
    bfs_started = time.perf_counter() if profiling else 0.0

    while frontier_index < len(frontier) and reachable_count <= budget:
        node_index = frontier[frontier_index]
        frontier_index += 1
        if profile is not None:
            profile.bfs_nodes += 1

        state_id = node_states[node_index]
        depth = depths[node_index]
        new_depth = depth + 1

        target_state = left_state[state_id]
        if target_state != _NO_STATE and state_nodes[target_state] == _NO_STATE:
            if profile is not None:
                collision_checks += 1
            cached_collision = collision_cache[target_state]
            if cached_collision == _COLLISION_UNKNOWN:
                shape_mask = collision_masks[target_state]
                blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                collision_cache[target_state] = (
                    _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                )
                if profile is not None:
                    collision_evaluations += 1
            else:
                blocked = cached_collision == _COLLISION_BLOCKED
                if profile is not None:
                    collision_cache_hits += 1
            if not blocked:
                successor = len(node_states)
                append_node_state(target_state)
                append_parent(node_index)
                if normalized_include_paths:
                    commands.append(_CMD_LEFT)
                append_depth(new_depth)
                append_kick_index(-1)
                state_nodes[target_state] = successor
                append_visited_state(target_state)
                reachable_count += 1
                append_frontier(successor)

        target_state = right_state[state_id]
        if target_state != _NO_STATE and state_nodes[target_state] == _NO_STATE:
            if profile is not None:
                collision_checks += 1
            cached_collision = collision_cache[target_state]
            if cached_collision == _COLLISION_UNKNOWN:
                shape_mask = collision_masks[target_state]
                blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                collision_cache[target_state] = (
                    _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                )
                if profile is not None:
                    collision_evaluations += 1
            else:
                blocked = cached_collision == _COLLISION_BLOCKED
                if profile is not None:
                    collision_cache_hits += 1
            if not blocked:
                successor = len(node_states)
                append_node_state(target_state)
                append_parent(node_index)
                if normalized_include_paths:
                    commands.append(_CMD_RIGHT)
                append_depth(new_depth)
                append_kick_index(-1)
                state_nodes[target_state] = successor
                append_visited_state(target_state)
                reachable_count += 1
                append_frontier(successor)

        target_state = down_state[state_id]
        if target_state != _NO_STATE and state_nodes[target_state] == _NO_STATE:
            if profile is not None:
                collision_checks += 1
            cached_collision = collision_cache[target_state]
            if cached_collision == _COLLISION_UNKNOWN:
                shape_mask = collision_masks[target_state]
                blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                collision_cache[target_state] = (
                    _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                )
                if profile is not None:
                    collision_evaluations += 1
            else:
                blocked = cached_collision == _COLLISION_BLOCKED
                if profile is not None:
                    collision_cache_hits += 1
            if not blocked:
                successor = len(node_states)
                append_node_state(target_state)
                append_parent(node_index)
                if normalized_include_paths:
                    commands.append(_CMD_DOWN)
                append_depth(new_depth)
                append_kick_index(-1)
                state_nodes[target_state] = successor
                append_visited_state(target_state)
                reachable_count += 1
                append_frontier(successor)

        rotation_started = time.perf_counter() if profiling else 0.0
        for command, kicks in rotation_transitions[state_id]:
            successful_state = _NO_STATE
            successful_kick = -1
            for target_state, kick_index in kicks:
                if profile is not None:
                    kick_checks += 1
                    collision_checks += 1
                    cached_collision = collision_cache[target_state]
                    if cached_collision != _COLLISION_UNKNOWN:
                        collision_cache_hits += 1
                        if cached_collision == _COLLISION_BLOCKED:
                            continue
                    else:
                        shape_mask = collision_masks[target_state]
                        blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                        collision_cache[target_state] = (
                            _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                        )
                        collision_evaluations += 1
                        if blocked:
                            continue
                else:
                    cached_collision = collision_cache[target_state]
                    if cached_collision == _COLLISION_BLOCKED:
                        continue
                    if cached_collision == _COLLISION_UNKNOWN:
                        shape_mask = collision_masks[target_state]
                        blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                        collision_cache[target_state] = (
                            _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
                        )
                        if blocked:
                            continue
                successful_state = target_state
                successful_kick = kick_index
                break

            if successful_state == _NO_STATE:
                continue

            adds_geometry = state_nodes[successful_state] == _NO_STATE
            if piece_is_t:
                previous_rotation = rotation_nodes[successful_state]
                improves_rotation = (
                    previous_rotation == _NO_STATE
                    or new_depth < depths[previous_rotation]
                )
            else:
                previous_rotation = _NO_STATE
                improves_rotation = False

            # A separate rotation-ending representative can only change the
            # selected placement for T pieces, where it may carry T-spin state.
            # For every other piece the geometry BFS representative is already
            # minimum-depth, wins equal-depth ordering, and has identical lock
            # semantics, so materializing rotation-only nodes is wasted work.
            if not improves_rotation and not adds_geometry:
                continue

            successor = len(node_states)
            append_node_state(successful_state)
            append_parent(node_index)
            if normalized_include_paths:
                commands.append(command)
            append_depth(new_depth)
            append_kick_index(successful_kick)

            if improves_rotation:
                if previous_rotation == _NO_STATE:
                    append_visited_rotation(successful_state)
                rotation_nodes[successful_state] = successor

            if adds_geometry:
                state_nodes[successful_state] = successor
                append_visited_state(successful_state)
                reachable_count += 1
                append_frontier(successor)

        if profile is not None:
            profile.rotation_seconds += time.perf_counter() - rotation_started

        if reachable_count > budget:
            break

    if profile is not None:
        profile.bfs_seconds += time.perf_counter() - bfs_started

    # The old implementation scanned every packed state in numeric order. We only
    # visit states that were actually reached, and preserve that exact tie order
    # explicitly with order_rank.
    best: dict[
        int,
        tuple[
            tuple[int, int, int],
            int,
            int,
            int,
            int,
            int,
            bool,
            int,
            int,
            int,
        ],
    ] = {}
    spin_cache: dict[tuple[int, int, int, int], str | None] = {}

    def emit(state_id: int, node_index: int, order_rank: int) -> None:
        emit_started = time.perf_counter() if profiling else 0.0
        landing_elapsed = 0.0
        try:
            landing_started = time.perf_counter() if profiling else 0.0
            if profile is not None:
                profile.landing_queries += 1

            final_state = landing_state[state_id]
            if final_state != _NO_LANDING:
                if profile is not None:
                    profile.landing_cache_hits += 1
            else:
                final_state = compute_landing_state(state_id)

            if profile is not None:
                landing_elapsed = time.perf_counter() - landing_started
                profile.landing_seconds += landing_elapsed

            key = geometry_masks[final_state]
            if key < 0:
                return

            x = state_x[state_id]
            landing_y = state_y[final_state]
            rotation = state_id & 3
            kick_index = kick_indices[node_index]
            last_rotation = kick_index >= 0

            spin_kind: str | None = None
            if last_rotation and piece_is_t:
                spin_key = (x, landing_y, rotation, kick_index)
                if spin_key in spin_cache:
                    spin_kind = spin_cache[spin_key]
                else:
                    spin_kind = classify_t_spin_row_masks(
                        rows,
                        piece=piece,
                        x=x,
                        y=landing_y,
                        rotation=rotation,
                        last_move_was_rotation=True,
                        rotation_kick_index=kick_index,
                        width=width,
                    )
                    spin_cache[spin_key] = spin_kind

            preference = (
                int(spin_kind is not None),
                int(spin_kind == "full"),
                -depths[node_index],
            )
            previous = best.get(key)
            if (
                previous is None
                or preference > previous[0]
                or (preference == previous[0] and order_rank < previous[1])
            ):
                parent = parents[node_index]
                rotation_from = (
                    (node_states[parent] & 3)
                    if last_rotation and parent >= 0
                    else -1
                )
                best[key] = (
                    preference,
                    order_rank,
                    x,
                    landing_y,
                    rotation,
                    node_index,
                    last_rotation,
                    kick_index if last_rotation else -1,
                    rotation_from,
                    rotation if last_rotation else -1,
                )
        finally:
            if profile is not None:
                profile.representative_nodes += 1
                profile.representative_seconds += max(
                    0.0,
                    time.perf_counter() - emit_started - landing_elapsed,
                )

    for state_id in visited_state_ids:
        emit(state_id, state_nodes[state_id], state_id)

    rotation_phase_base = packed_count
    for state_id in visited_rotation_ids:
        node_index = rotation_nodes[state_id]
        if state_nodes[state_id] == node_index:
            if profile is not None:
                profile.representative_duplicate_skips += 1
            continue
        emit(state_id, node_index, rotation_phase_base + state_id)

    placement_started = time.perf_counter() if profiling else 0.0
    placements: list[Placement] = []

    for (
        _preference,
        _order_rank,
        x,
        landing_y,
        rotation,
        node_index,
        last_rotation,
        kick_index,
        rotation_from,
        rotation_to,
    ) in best.values():
        path: tuple[str, ...] = ()
        if normalized_include_paths:
            path_started = time.perf_counter() if profiling else 0.0
            path = _path(parents, commands, node_index)
            if profile is not None:
                profile.path_seconds += time.perf_counter() - path_started
                profile.path_calls += 1

        placements.append(
            Placement(
                piece=piece,
                x=x,
                y=landing_y,
                rotation=rotation,
                cells=placement_cells(piece, x, landing_y, rotation),
                path=path,
                last_move_was_rotation=last_rotation,
                rotation_kick_index=kick_index if kick_index >= 0 else None,
                rotation_from=rotation_from if rotation_from >= 0 else None,
                rotation_to=rotation_to if rotation_to >= 0 else None,
            )
        )

    placements.sort(key=lambda item: (item.rotation, item.x, item.y, len(item.path)))

    if profile is not None:
        profile.collision_checks += collision_checks
        profile.collision_evaluations += collision_evaluations
        profile.collision_cache_hits += collision_cache_hits
        profile.kick_checks += kick_checks
        profile.placement_seconds += time.perf_counter() - placement_started
        profile.placements += len(placements)
        profile.total_seconds += time.perf_counter() - profile_started

    result = tuple(placements)
    return _cache_reachability_result(cache_key, result)
