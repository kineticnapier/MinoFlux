from __future__ import annotations

import time

from minoflux_engine import Game, Placement

from .bitboard import (
    board_row_masks,
    classify_t_spin_row_masks,
    placement_cells,
)
from . import reachability as _reference


_REFERENCE_REACHABLE_PLACEMENTS = _reference.reachable_placements


def reachable_placements_pathless(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
) -> tuple[Placement, ...]:
    """Enumerate exact-SRS placements without path reconstruction bookkeeping."""

    profile = _reference._ACTIVE_REACHABILITY_PROFILE.get()
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
    cache_key = (
        rows,
        game.current,
        game.x,
        game.y,
        game.rotation & 3,
        game.width,
        game.height,
        normalized_allow_180,
        normalized_max_nodes,
        False,
    )
    cached = _reference._REACHABILITY_CACHE.pop(cache_key, None)
    if cached is not None:
        _reference._REACHABILITY_CACHE[cache_key] = cached
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
    tables = _reference._state_tables(piece, width, height)
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
    rotation_transitions = _reference._rotation_transitions(
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
        return _reference._cache_reachability_result(cache_key, ())

    start_state = (
        ((((game.y - y_min) * x_count) + (game.x - x_min)) << 2)
        | start_rotation
    )

    no_state = _reference._NO_STATE
    no_landing = _reference._NO_LANDING
    collision_unknown = _reference._COLLISION_UNKNOWN
    collision_clear = _reference._COLLISION_CLEAR
    collision_blocked = _reference._COLLISION_BLOCKED
    kick_index_bits = _reference._KICK_INDEX_BITS
    kick_index_mask = _reference._KICK_INDEX_MASK

    board_bits = _reference._rows_to_board_bits(rows, width)
    collision_cache = bytearray(packed_count)
    collision_checks = 0
    collision_evaluations = 0
    collision_cache_hits = 0
    kick_checks = 0

    if profile is not None:
        collision_checks = 1
        collision_evaluations = 1
    shape_mask = collision_masks[start_state]
    start_blocked = shape_mask < 0 or bool(board_bits & shape_mask)
    collision_cache[start_state] = (
        collision_blocked if start_blocked else collision_clear
    )
    if start_blocked:
        if profile is not None:
            profile.collision_checks += collision_checks
            profile.collision_evaluations += collision_evaluations
            profile.collision_cache_hits += collision_cache_hits
            profile.kick_checks += kick_checks
            profile.setup_seconds += time.perf_counter() - setup_started
            profile.total_seconds += time.perf_counter() - profile_started
        return _reference._cache_reachability_result(cache_key, ())

    landing_state = [no_landing] * packed_count

    if profile is None:

        def compute_landing_state(state_id: int) -> int:
            trail = [state_id]
            current_id = state_id
            while True:
                target_id = down_state[current_id]
                if target_id == no_state:
                    result = current_id
                    break
                cached_collision = collision_cache[target_id]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_id]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_id] = (
                        collision_blocked if blocked else collision_clear
                    )
                else:
                    blocked = cached_collision == collision_blocked
                if blocked:
                    result = current_id
                    break
                current_id = target_id
                cached_landing = landing_state[current_id]
                if cached_landing != no_landing:
                    result = cached_landing
                    break
                trail.append(current_id)
            for cached_id in trail:
                landing_state[cached_id] = result
            return result

    else:

        def compute_landing_state(state_id: int) -> int:
            nonlocal collision_checks, collision_evaluations, collision_cache_hits
            trail = [state_id]
            current_id = state_id
            while True:
                target_id = down_state[current_id]
                if target_id == no_state:
                    result = current_id
                    break
                collision_checks += 1
                cached_collision = collision_cache[target_id]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_id]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_id] = (
                        collision_blocked if blocked else collision_clear
                    )
                    collision_evaluations += 1
                else:
                    collision_cache_hits += 1
                    blocked = cached_collision == collision_blocked
                if blocked:
                    result = current_id
                    break
                current_id = target_id
                cached_landing = landing_state[current_id]
                if cached_landing != no_landing:
                    profile.landing_cache_hits += 1
                    result = cached_landing
                    break
                trail.append(current_id)
            for cached_id in trail:
                landing_state[cached_id] = result
            return result

    state_depths = [no_state] * packed_count
    state_kick_infos = [-1] * packed_count
    state_depths[start_state] = 0

    rotation_depths = [no_state] * packed_count if piece_is_t else []
    rotation_kick_infos = [-1] * packed_count if piece_is_t else []
    rotation_is_geometry = bytearray(packed_count) if piece_is_t else bytearray()

    frontier = [start_state]
    append_frontier = frontier.append
    frontier_index = 0

    visited_state_ids = [start_state]
    visited_rotation_ids: list[int] = []
    append_visited_state = visited_state_ids.append
    append_visited_rotation = visited_rotation_ids.append

    reachable_count = 1
    budget = normalized_max_nodes

    if profile is not None:
        profile.setup_seconds += time.perf_counter() - setup_started
    bfs_started = time.perf_counter() if profiling else 0.0

    if profile is None:
        while frontier_index < len(frontier) and reachable_count <= budget:
            state_id = frontier[frontier_index]
            frontier_index += 1
            new_depth = state_depths[state_id] + 1

            target_state = left_state[state_id]
            if target_state != no_state and state_depths[target_state] == no_state:
                cached_collision = collision_cache[target_state]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_state]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_state] = (
                        collision_blocked if blocked else collision_clear
                    )
                else:
                    blocked = cached_collision == collision_blocked
                if not blocked:
                    state_depths[target_state] = new_depth
                    state_kick_infos[target_state] = -1
                    append_visited_state(target_state)
                    reachable_count += 1
                    append_frontier(target_state)

            target_state = right_state[state_id]
            if target_state != no_state and state_depths[target_state] == no_state:
                cached_collision = collision_cache[target_state]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_state]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_state] = (
                        collision_blocked if blocked else collision_clear
                    )
                else:
                    blocked = cached_collision == collision_blocked
                if not blocked:
                    state_depths[target_state] = new_depth
                    state_kick_infos[target_state] = -1
                    append_visited_state(target_state)
                    reachable_count += 1
                    append_frontier(target_state)

            target_state = down_state[state_id]
            if target_state != no_state and state_depths[target_state] == no_state:
                cached_collision = collision_cache[target_state]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_state]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_state] = (
                        collision_blocked if blocked else collision_clear
                    )
                else:
                    blocked = cached_collision == collision_blocked
                if not blocked:
                    state_depths[target_state] = new_depth
                    state_kick_infos[target_state] = -1
                    append_visited_state(target_state)
                    reachable_count += 1
                    append_frontier(target_state)

            for _command, kicks in rotation_transitions[state_id]:
                successful_state = no_state
                successful_kick = -1
                for target_state, kick_index in kicks:
                    cached_collision = collision_cache[target_state]
                    if cached_collision == collision_blocked:
                        continue
                    if cached_collision == collision_unknown:
                        shape_mask = collision_masks[target_state]
                        blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                        collision_cache[target_state] = (
                            collision_blocked if blocked else collision_clear
                        )
                        if blocked:
                            continue
                    successful_state = target_state
                    successful_kick = kick_index
                    break

                if successful_state == no_state:
                    continue

                adds_geometry = state_depths[successful_state] == no_state
                if piece_is_t:
                    previous_rotation_depth = rotation_depths[successful_state]
                    improves_rotation = (
                        previous_rotation_depth == no_state
                        or new_depth < previous_rotation_depth
                    )
                else:
                    previous_rotation_depth = no_state
                    improves_rotation = False

                if not improves_rotation and not adds_geometry:
                    continue

                rotation_info = (
                    ((state_id & 3) << kick_index_bits) | successful_kick
                )

                if improves_rotation:
                    if previous_rotation_depth == no_state:
                        append_visited_rotation(successful_state)
                    rotation_depths[successful_state] = new_depth
                    rotation_kick_infos[successful_state] = rotation_info
                    rotation_is_geometry[successful_state] = int(adds_geometry)

                if adds_geometry:
                    state_depths[successful_state] = new_depth
                    state_kick_infos[successful_state] = rotation_info
                    append_visited_state(successful_state)
                    reachable_count += 1
                    append_frontier(successful_state)

            if reachable_count > budget:
                break
    else:
        while frontier_index < len(frontier) and reachable_count <= budget:
            state_id = frontier[frontier_index]
            frontier_index += 1
            profile.bfs_nodes += 1
            new_depth = state_depths[state_id] + 1

            target_state = left_state[state_id]
            if target_state != no_state and state_depths[target_state] == no_state:
                collision_checks += 1
                cached_collision = collision_cache[target_state]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_state]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_state] = (
                        collision_blocked if blocked else collision_clear
                    )
                    collision_evaluations += 1
                else:
                    blocked = cached_collision == collision_blocked
                    collision_cache_hits += 1
                if not blocked:
                    state_depths[target_state] = new_depth
                    state_kick_infos[target_state] = -1
                    append_visited_state(target_state)
                    reachable_count += 1
                    append_frontier(target_state)

            target_state = right_state[state_id]
            if target_state != no_state and state_depths[target_state] == no_state:
                collision_checks += 1
                cached_collision = collision_cache[target_state]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_state]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_state] = (
                        collision_blocked if blocked else collision_clear
                    )
                    collision_evaluations += 1
                else:
                    blocked = cached_collision == collision_blocked
                    collision_cache_hits += 1
                if not blocked:
                    state_depths[target_state] = new_depth
                    state_kick_infos[target_state] = -1
                    append_visited_state(target_state)
                    reachable_count += 1
                    append_frontier(target_state)

            target_state = down_state[state_id]
            if target_state != no_state and state_depths[target_state] == no_state:
                collision_checks += 1
                cached_collision = collision_cache[target_state]
                if cached_collision == collision_unknown:
                    shape_mask = collision_masks[target_state]
                    blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                    collision_cache[target_state] = (
                        collision_blocked if blocked else collision_clear
                    )
                    collision_evaluations += 1
                else:
                    blocked = cached_collision == collision_blocked
                    collision_cache_hits += 1
                if not blocked:
                    state_depths[target_state] = new_depth
                    state_kick_infos[target_state] = -1
                    append_visited_state(target_state)
                    reachable_count += 1
                    append_frontier(target_state)

            rotation_started = time.perf_counter()
            for _command, kicks in rotation_transitions[state_id]:
                successful_state = no_state
                successful_kick = -1
                for target_state, kick_index in kicks:
                    kick_checks += 1
                    collision_checks += 1
                    cached_collision = collision_cache[target_state]
                    if cached_collision != collision_unknown:
                        collision_cache_hits += 1
                        if cached_collision == collision_blocked:
                            continue
                    else:
                        shape_mask = collision_masks[target_state]
                        blocked = shape_mask < 0 or bool(board_bits & shape_mask)
                        collision_cache[target_state] = (
                            collision_blocked if blocked else collision_clear
                        )
                        collision_evaluations += 1
                        if blocked:
                            continue
                    successful_state = target_state
                    successful_kick = kick_index
                    break

                if successful_state == no_state:
                    continue

                adds_geometry = state_depths[successful_state] == no_state
                if piece_is_t:
                    previous_rotation_depth = rotation_depths[successful_state]
                    improves_rotation = (
                        previous_rotation_depth == no_state
                        or new_depth < previous_rotation_depth
                    )
                else:
                    previous_rotation_depth = no_state
                    improves_rotation = False

                if not improves_rotation and not adds_geometry:
                    continue

                rotation_info = (
                    ((state_id & 3) << kick_index_bits) | successful_kick
                )

                if improves_rotation:
                    if previous_rotation_depth == no_state:
                        append_visited_rotation(successful_state)
                    rotation_depths[successful_state] = new_depth
                    rotation_kick_infos[successful_state] = rotation_info
                    rotation_is_geometry[successful_state] = int(adds_geometry)

                if adds_geometry:
                    state_depths[successful_state] = new_depth
                    state_kick_infos[successful_state] = rotation_info
                    append_visited_state(successful_state)
                    reachable_count += 1
                    append_frontier(successful_state)

            profile.rotation_seconds += time.perf_counter() - rotation_started

            if reachable_count > budget:
                break

        profile.bfs_seconds += time.perf_counter() - bfs_started

    best: dict[
        int,
        tuple[tuple[int, int, int], int, int, int, int, bool, int, int, int],
    ] = {}

    if not piece_is_t:
        if profile is None:
            for state_id in visited_state_ids:
                final_state = landing_state[state_id]
                if final_state == no_landing:
                    final_state = compute_landing_state(state_id)

                key = geometry_masks[final_state]
                if key < 0:
                    continue

                negative_depth = -state_depths[state_id]
                previous = best.get(key)
                if previous is not None and (
                    negative_depth < previous[0][2]
                    or (
                        negative_depth == previous[0][2]
                        and state_id >= previous[1]
                    )
                ):
                    continue

                x = state_x[state_id]
                landing_y = state_y[final_state]
                rotation = state_id & 3
                rotation_info = state_kick_infos[state_id]
                last_rotation = rotation_info >= 0
                kick_index = (
                    rotation_info & kick_index_mask if last_rotation else -1
                )
                rotation_from = (
                    rotation_info >> kick_index_bits if last_rotation else -1
                )
                best[key] = (
                    (0, 0, negative_depth),
                    state_id,
                    x,
                    landing_y,
                    rotation,
                    last_rotation,
                    kick_index,
                    rotation_from,
                    rotation if last_rotation else -1,
                )
        else:
            for state_id in visited_state_ids:
                emit_started = time.perf_counter()
                landing_started = time.perf_counter()
                profile.landing_queries += 1

                final_state = landing_state[state_id]
                if final_state != no_landing:
                    profile.landing_cache_hits += 1
                else:
                    final_state = compute_landing_state(state_id)

                landing_elapsed = time.perf_counter() - landing_started
                profile.landing_seconds += landing_elapsed
                key = geometry_masks[final_state]
                if key >= 0:
                    negative_depth = -state_depths[state_id]
                    previous = best.get(key)
                    if previous is None or negative_depth > previous[0][2] or (
                        negative_depth == previous[0][2]
                        and state_id < previous[1]
                    ):
                        x = state_x[state_id]
                        landing_y = state_y[final_state]
                        rotation = state_id & 3
                        rotation_info = state_kick_infos[state_id]
                        last_rotation = rotation_info >= 0
                        kick_index = (
                            rotation_info & kick_index_mask
                            if last_rotation
                            else -1
                        )
                        rotation_from = (
                            rotation_info >> kick_index_bits
                            if last_rotation
                            else -1
                        )
                        best[key] = (
                            (0, 0, negative_depth),
                            state_id,
                            x,
                            landing_y,
                            rotation,
                            last_rotation,
                            kick_index,
                            rotation_from,
                            rotation if last_rotation else -1,
                        )

                profile.representative_nodes += 1
                profile.representative_seconds += max(
                    0.0,
                    time.perf_counter() - emit_started - landing_elapsed,
                )
    else:
        spin_cache: dict[tuple[int, int, int, int], str | None] = {}

        def emit(
            state_id: int,
            depth: int,
            rotation_info: int,
            order_rank: int,
        ) -> None:
            emit_started = time.perf_counter() if profiling else 0.0
            landing_elapsed = 0.0
            try:
                landing_started = time.perf_counter() if profiling else 0.0
                if profile is not None:
                    profile.landing_queries += 1

                final_state = landing_state[state_id]
                if final_state != no_landing:
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
                last_rotation = rotation_info >= 0
                kick_index = (
                    rotation_info & kick_index_mask if last_rotation else -1
                )
                rotation_from = (
                    rotation_info >> kick_index_bits if last_rotation else -1
                )

                spin_kind: str | None = None
                if last_rotation:
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
                    -depth,
                )
                previous = best.get(key)
                if (
                    previous is None
                    or preference > previous[0]
                    or (preference == previous[0] and order_rank < previous[1])
                ):
                    best[key] = (
                        preference,
                        order_rank,
                        x,
                        landing_y,
                        rotation,
                        last_rotation,
                        kick_index,
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
            emit(
                state_id,
                state_depths[state_id],
                state_kick_infos[state_id],
                state_id,
            )

        rotation_phase_base = packed_count
        for state_id in visited_rotation_ids:
            if rotation_is_geometry[state_id]:
                if profile is not None:
                    profile.representative_duplicate_skips += 1
                continue
            emit(
                state_id,
                rotation_depths[state_id],
                rotation_kick_infos[state_id],
                rotation_phase_base + state_id,
            )

    placement_started = time.perf_counter() if profiling else 0.0
    placements: list[Placement] = []
    append_placement = placements.append

    for (
        _preference,
        _order_rank,
        x,
        landing_y,
        rotation,
        last_rotation,
        kick_index,
        rotation_from,
        rotation_to,
    ) in best.values():
        append_placement(
            Placement(
                piece=piece,
                x=x,
                y=landing_y,
                rotation=rotation,
                cells=placement_cells(piece, x, landing_y, rotation),
                path=(),
                last_move_was_rotation=last_rotation,
                rotation_kick_index=kick_index if kick_index >= 0 else None,
                rotation_from=rotation_from if rotation_from >= 0 else None,
                rotation_to=rotation_to if rotation_to >= 0 else None,
            )
        )

    placements.sort(key=lambda item: (item.rotation, item.x, item.y, 0))

    if profile is not None:
        profile.collision_checks += collision_checks
        profile.collision_evaluations += collision_evaluations
        profile.collision_cache_hits += collision_cache_hits
        profile.kick_checks += kick_checks
        profile.placement_seconds += time.perf_counter() - placement_started
        profile.placements += len(placements)
        profile.total_seconds += time.perf_counter() - profile_started

    result = tuple(placements)
    return _reference._cache_reachability_result(cache_key, result)


def _dispatch_reachable_placements(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
    include_paths: bool = True,
) -> tuple[Placement, ...]:
    if include_paths:
        return _REFERENCE_REACHABLE_PLACEMENTS(
            game,
            allow_180=allow_180,
            max_nodes=max_nodes,
            include_paths=True,
        )
    return reachable_placements_pathless(
        game,
        allow_180=allow_180,
        max_nodes=max_nodes,
    )


def install_pathless_search_fast_path() -> None:
    """Install the pathless dispatcher before search.py binds reachable_placements."""

    _reference.reachable_placements = _dispatch_reachable_placements
