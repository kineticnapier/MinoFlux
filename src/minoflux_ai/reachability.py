from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache
import time
from typing import Iterator

from minoflux_engine import Game, Placement
from minoflux_engine.pieces import BOARD_WIDTH, SHAPES, kick_tests

from .bitboard import (
    SHIFTED_SHAPE_ROW_MASKS,
    board_row_masks,
    classify_t_spin_row_masks,
    collides_row_masks,
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
_NO_LANDING = -10_000
_COLLISION_UNKNOWN = 0
_COLLISION_CLEAR = 1
_COLLISION_BLOCKED = 2
_X_MARGIN = 4
_Y_MIN = -4


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


def _state_layout(game: Game) -> tuple[int, int, int, int, int]:
    x_min = -_X_MARGIN
    x_max = game.width + _X_MARGIN - 1
    x_count = x_max - x_min + 1
    y_min = _Y_MIN
    y_count = game.height - y_min
    state_count = x_count * y_count * 4
    return x_min, x_max, x_count, y_min, state_count


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
    """Canonical final-cell key, cached across repeated reachability searches."""

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


def reachable_placements(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
    include_paths: bool = True,
) -> tuple[Placement, ...]:
    """Enumerate exact-SRS placements using packed states and bitboard collision.

    ``include_paths=False`` skips materializing command paths while preserving the
    exact historical placement representative and rotation metadata.
    """

    profile = _ACTIVE_REACHABILITY_PROFILE.get()
    profile_started = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile.calls += 1

    if game.game_over or game.paused:
        if profile is not None:
            profile.total_seconds += time.perf_counter() - profile_started
        return ()

    board_mask_started = time.perf_counter() if profile is not None else 0.0
    rows = board_row_masks(game.board)
    if profile is not None:
        profile.board_mask_seconds += time.perf_counter() - board_mask_started
    setup_started = time.perf_counter() if profile is not None else 0.0

    piece = game.current
    piece_is_t = piece == "T"
    width = game.width
    height = game.height
    x_min, x_max, x_count, y_min, packed_count = _state_layout(game)
    row_state_stride = x_count << 2

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
        return ()
    start_state = (
        ((((game.y - y_min) * x_count) + (game.x - x_min)) << 2)
        | start_rotation
    )

    collision_cache = bytearray(packed_count)
    if width == BOARD_WIDTH:
        shifted_piece_rows = SHIFTED_SHAPE_ROW_MASKS[piece]

        def evaluate_collision(x: int, y: int, rotation: int) -> bool:
            shifted_rows = shifted_piece_rows[rotation & 3][x + _X_MARGIN]
            if shifted_rows is None:
                return True
            for dy, shifted_mask in shifted_rows:
                row_y = y + dy
                if row_y >= height:
                    return True
                if row_y >= 0 and rows[row_y] & shifted_mask:
                    return True
            return False

    else:

        def evaluate_collision(x: int, y: int, rotation: int) -> bool:
            return collides_row_masks(rows, piece, x, y, rotation, width=width)

    def collides_state(state_id: int, x: int, y: int, rotation: int) -> bool:
        if profile is not None:
            profile.collision_checks += 1
        if state_id < 0 or state_id >= packed_count:
            return True
        cached = collision_cache[state_id]
        if cached != _COLLISION_UNKNOWN:
            if profile is not None:
                profile.collision_cache_hits += 1
            return cached == _COLLISION_BLOCKED
        blocked = evaluate_collision(x, y, rotation)
        collision_cache[state_id] = (
            _COLLISION_BLOCKED if blocked else _COLLISION_CLEAR
        )
        if profile is not None:
            profile.collision_evaluations += 1
        return blocked

    if collides_state(start_state, game.x, game.y, start_rotation):
        if profile is not None:
            profile.setup_seconds += time.perf_counter() - setup_started
            profile.total_seconds += time.perf_counter() - profile_started
        return ()

    landing = [_NO_LANDING] * packed_count

    def compute_landing_from_state(
        state_id: int,
        x: int,
        y: int,
        rotation: int,
    ) -> int:
        trail: list[int] = [state_id]
        current_id = state_id
        current_y = y
        result = _NO_LANDING
        while True:
            target_y = current_y + 1
            target_id = current_id + row_state_stride
            if collides_state(target_id, x, target_y, rotation):
                result = current_y
                break
            current_y = target_y
            current_id = target_id
            cached = landing[current_id]
            if cached != _NO_LANDING:
                if profile is not None:
                    profile.landing_cache_hits += 1
                result = cached
                break
            trail.append(current_id)

        if result != _NO_LANDING:
            for cached_id in trail:
                landing[cached_id] = result
        return result

    xs: list[int] = [game.x]
    ys: list[int] = [game.y]
    rotations: list[int] = [start_rotation]
    state_ids: list[int] = [start_state]
    parents: list[int] = [-1]
    commands: list[int] = [_CMD_NONE]
    depths: list[int] = [0]
    last_rotations: list[bool] = [False]
    kick_indices: list[int] = [-1]
    rotation_froms: list[int] = [-1]
    rotation_tos: list[int] = [-1]

    append_x = xs.append
    append_y = ys.append
    append_rotation = rotations.append
    append_state_id = state_ids.append
    append_parent = parents.append
    append_command = commands.append
    append_depth = depths.append
    append_last_rotation = last_rotations.append
    append_kick_index = kick_indices.append
    append_rotation_from = rotation_froms.append
    append_rotation_to = rotation_tos.append

    frontier: list[int] = [0]
    append_frontier = frontier.append
    frontier_index = 0
    state_nodes = [_NO_STATE] * packed_count
    rotation_nodes = [_NO_STATE] * packed_count
    state_nodes[start_state] = 0
    reachable_count = 1
    budget = max(1, int(max_nodes))
    rotation_options = _rotation_options(piece, bool(allow_180), x_count)

    if profile is not None:
        profile.setup_seconds += time.perf_counter() - setup_started
    bfs_started = time.perf_counter() if profile is not None else 0.0

    while frontier_index < len(frontier) and reachable_count <= budget:
        node_index = frontier[frontier_index]
        frontier_index += 1
        if profile is not None:
            profile.bfs_nodes += 1
        x = xs[node_index]
        y = ys[node_index]
        rotation = rotations[node_index]
        state_id = state_ids[node_index]
        depth = depths[node_index]
        new_depth = depth + 1
        state_base = state_id & ~3

        if x > x_min:
            target_x = x - 1
            target_state = state_id - 4
            if (
                state_nodes[target_state] == _NO_STATE
                and not collides_state(target_state, target_x, y, rotation)
            ):
                successor = len(xs)
                append_x(target_x)
                append_y(y)
                append_rotation(rotation)
                append_state_id(target_state)
                append_parent(node_index)
                append_command(_CMD_LEFT)
                append_depth(new_depth)
                append_last_rotation(False)
                append_kick_index(-1)
                append_rotation_from(-1)
                append_rotation_to(-1)
                state_nodes[target_state] = successor
                reachable_count += 1
                append_frontier(successor)

        if x < x_max:
            target_x = x + 1
            target_state = state_id + 4
            if (
                state_nodes[target_state] == _NO_STATE
                and not collides_state(target_state, target_x, y, rotation)
            ):
                successor = len(xs)
                append_x(target_x)
                append_y(y)
                append_rotation(rotation)
                append_state_id(target_state)
                append_parent(node_index)
                append_command(_CMD_RIGHT)
                append_depth(new_depth)
                append_last_rotation(False)
                append_kick_index(-1)
                append_rotation_from(-1)
                append_rotation_to(-1)
                state_nodes[target_state] = successor
                reachable_count += 1
                append_frontier(successor)

        if y + 1 < height:
            target_y = y + 1
            target_state = state_id + row_state_stride
            if (
                state_nodes[target_state] == _NO_STATE
                and not collides_state(target_state, x, target_y, rotation)
            ):
                successor = len(xs)
                append_x(x)
                append_y(target_y)
                append_rotation(rotation)
                append_state_id(target_state)
                append_parent(node_index)
                append_command(_CMD_DOWN)
                append_depth(new_depth)
                append_last_rotation(False)
                append_kick_index(-1)
                append_rotation_from(-1)
                append_rotation_to(-1)
                state_nodes[target_state] = successor
                reachable_count += 1
                append_frontier(successor)

        rotation_started = time.perf_counter() if profile is not None else 0.0
        for target_rotation, command, kicks in rotation_options[rotation]:
            successful_kick = -1
            successful_state = _NO_STATE
            target_x = x
            target_y = y
            for kick_index, kick_x, kick_y, packed_delta in kicks:
                if profile is not None:
                    profile.kick_checks += 1
                target_x = x + kick_x
                target_y = y + kick_y
                if target_y < y_min:
                    continue
                if target_x < x_min or target_x > x_max or target_y >= height:
                    continue
                target_state = state_base + packed_delta
                if collides_state(
                    target_state,
                    target_x,
                    target_y,
                    target_rotation,
                ):
                    continue
                successful_kick = kick_index
                successful_state = target_state
                break
            if successful_kick < 0:
                continue

            previous_rotation = rotation_nodes[successful_state]
            improves_rotation = (
                previous_rotation == _NO_STATE
                or new_depth < depths[previous_rotation]
            )
            adds_geometry = state_nodes[successful_state] == _NO_STATE
            if not improves_rotation and not adds_geometry:
                continue

            successor = len(xs)
            append_x(target_x)
            append_y(target_y)
            append_rotation(target_rotation)
            append_state_id(successful_state)
            append_parent(node_index)
            append_command(command)
            append_depth(new_depth)
            append_last_rotation(True)
            append_kick_index(successful_kick)
            append_rotation_from(rotation)
            append_rotation_to(target_rotation)
            if improves_rotation:
                rotation_nodes[successful_state] = successor
            if adds_geometry:
                state_nodes[successful_state] = successor
                reachable_count += 1
                append_frontier(successor)
        if profile is not None:
            profile.rotation_seconds += time.perf_counter() - rotation_started

        if reachable_count > budget:
            break

    if profile is not None:
        profile.bfs_seconds += time.perf_counter() - bfs_started

    best: dict[int, tuple[tuple[int, int, int], int, int, int, int, bool, int, int, int]] = {}

    def emit(state_id: int, node_index: int) -> None:
        emit_started = time.perf_counter() if profile is not None else 0.0
        landing_elapsed = 0.0
        try:
            x = xs[node_index]
            y = ys[node_index]
            rotation = rotations[node_index]
            landing_started = time.perf_counter() if profile is not None else 0.0
            if profile is not None:
                profile.landing_queries += 1
            landing_y = landing[state_id]
            if landing_y != _NO_LANDING:
                if profile is not None:
                    profile.landing_cache_hits += 1
            else:
                landing_y = compute_landing_from_state(state_id, x, y, rotation)
            if profile is not None:
                landing_elapsed = time.perf_counter() - landing_started
                profile.landing_seconds += landing_elapsed
            if landing_y == _NO_LANDING:
                return
            key = _geometry_key(piece, x, landing_y, rotation, width)
            if key < 0:
                return
            last_rotation = last_rotations[node_index]
            spin_kind = (
                classify_t_spin_row_masks(
                    rows,
                    piece=piece,
                    x=x,
                    y=landing_y,
                    rotation=rotation,
                    last_move_was_rotation=True,
                    rotation_kick_index=(
                        kick_indices[node_index] if kick_indices[node_index] >= 0 else None
                    ),
                    width=width,
                )
                if last_rotation and piece_is_t
                else None
            )
            preference = (
                int(spin_kind is not None),
                int(spin_kind == "full"),
                -depths[node_index],
            )
            previous = best.get(key)
            if previous is None or preference > previous[0]:
                best[key] = (
                    preference,
                    x,
                    landing_y,
                    rotation,
                    node_index,
                    last_rotation,
                    kick_indices[node_index] if last_rotation else -1,
                    rotation_froms[node_index] if last_rotation else -1,
                    rotation_tos[node_index] if last_rotation else -1,
                )
        finally:
            if profile is not None:
                profile.representative_nodes += 1
                profile.representative_seconds += max(
                    0.0,
                    time.perf_counter() - emit_started - landing_elapsed,
                )

    for state_id, node_index in enumerate(state_nodes):
        if node_index != _NO_STATE:
            emit(state_id, node_index)

    for state_id, node_index in enumerate(rotation_nodes):
        if node_index == _NO_STATE:
            continue
        if state_nodes[state_id] == node_index:
            if profile is not None:
                profile.representative_duplicate_skips += 1
            continue
        emit(state_id, node_index)

    placement_started = time.perf_counter() if profile is not None else 0.0
    placements: list[Placement] = []
    for (
        _preference,
        x,
        landing_y,
        rotation,
        node_index,
        last_rotation,
        kick_index,
        rotation_from,
        rotation_to,
    ) in best.values():
        placements.append(
            Placement(
                piece=piece,
                x=x,
                y=landing_y,
                rotation=rotation,
                cells=placement_cells(piece, x, landing_y, rotation),
                path=_path(parents, commands, node_index) if include_paths else (),
                last_move_was_rotation=last_rotation,
                rotation_kick_index=kick_index if kick_index >= 0 else None,
                rotation_from=rotation_from if rotation_from >= 0 else None,
                rotation_to=rotation_to if rotation_to >= 0 else None,
            )
        )

    placements.sort(key=lambda item: (item.rotation, item.x, item.y, len(item.path)))
    if profile is not None:
        profile.placement_seconds += time.perf_counter() - placement_started
        profile.placements += len(placements)
        profile.total_seconds += time.perf_counter() - profile_started
    return tuple(placements)
