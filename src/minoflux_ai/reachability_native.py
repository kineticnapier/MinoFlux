from __future__ import annotations

from functools import cache
import os
import time
from typing import Sequence

from minoflux_engine import Game, Placement

from . import reachability as _reference
from . import reachability_pathless as _pathless
from .bitboard import board_row_masks, placement_cells

try:
    from . import _reachability_native as _native
except (ImportError, OSError):  # Optional extension: pure-Python installs remain valid.
    _native = None


_PYTHON_PATHLESS = _pathless.reachable_placements_pathless
_NATIVE_MASK_BYTES = 32
_NATIVE_DISABLE_ENV = "MINOFLUX_DISABLE_NATIVE_REACHABILITY"


def native_pathless_available() -> bool:
    """Return whether the optional native pathless exact-SRS core can be used."""

    if _native is None:
        return False
    disabled = os.environ.get(_NATIVE_DISABLE_ENV, "").strip().lower()
    return disabled not in {"1", "true", "yes", "on"}


def _native_board_supported(width: int, height: int) -> bool:
    return 0 < width <= 64 and 0 < height and width * height <= 256


def _pack_masks(values: Sequence[int]) -> tuple[bytes, bytes]:
    invalid = bytearray(len(values))
    packed = bytearray(len(values) * _NATIVE_MASK_BYTES)
    for index, value in enumerate(values):
        if value < 0:
            invalid[index] = 1
            continue
        start = index * _NATIVE_MASK_BYTES
        packed[start : start + _NATIVE_MASK_BYTES] = int(value).to_bytes(
            _NATIVE_MASK_BYTES,
            "little",
            signed=False,
        )
    return bytes(invalid), bytes(packed)


@cache
def _native_table_handle(
    piece: str,
    allow_180: bool,
    width: int,
    height: int,
) -> int:
    if _native is None:
        raise RuntimeError("native reachability extension is unavailable")
    tables = _reference._state_tables(piece, width, height)
    collision_invalid, collision_masks = _pack_masks(tables.collision_mask)
    geometry_invalid, geometry_masks = _pack_masks(tables.geometry_mask)
    rotation_transitions = _pathless._rotation_kick_groups(
        piece,
        bool(allow_180),
        width,
        height,
    )
    return int(
        _native.register_table(
            piece,
            width,
            height,
            tables.x_min,
            tables.x_max,
            tables.x_count,
            tables.y_min,
            tables.state_x,
            tables.state_y,
            tables.left_state,
            tables.right_state,
            tables.down_state,
            collision_invalid,
            collision_masks,
            geometry_invalid,
            geometry_masks,
            rotation_transitions,
        )
    )


def reachable_placements_pathless_python(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
) -> tuple[Placement, ...]:
    """Explicit correctness-oracle/fallback entry point for the Python search."""

    return _PYTHON_PATHLESS(
        game,
        allow_180=allow_180,
        max_nodes=max_nodes,
    )


def reachable_placements_pathless_native(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
) -> tuple[Placement, ...]:
    """Use one native call for an exact-SRS pathless reachability query."""

    if not native_pathless_available() or not _native_board_supported(game.width, game.height):
        return reachable_placements_pathless_python(
            game,
            allow_180=allow_180,
            max_nodes=max_nodes,
        )

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
    table_handle = _native_table_handle(
        game.current,
        normalized_allow_180,
        game.width,
        game.height,
    )
    python_setup_elapsed = time.perf_counter() - setup_started if profiling else 0.0

    native_result = _native.run(
        table_handle,
        rows,
        game.x,
        game.y,
        game.rotation & 3,
        normalized_max_nodes,
        profiling,
    )

    placement_started = time.perf_counter() if profiling else 0.0
    piece = game.current
    placements = tuple(
        Placement(
            piece=piece,
            x=int(record[0]),
            y=int(record[1]),
            rotation=int(record[2]),
            cells=placement_cells(piece, int(record[0]), int(record[1]), int(record[2])),
            path=(),
            last_move_was_rotation=bool(record[3]),
            rotation_kick_index=int(record[4]) if int(record[4]) >= 0 else None,
            rotation_from=int(record[5]) if int(record[5]) >= 0 else None,
            rotation_to=int(record[6]) if int(record[6]) >= 0 else None,
        )
        for record in native_result["placements"]
    )
    python_placement_elapsed = time.perf_counter() - placement_started if profiling else 0.0

    if profile is not None:
        counters = native_result["counters"]
        timings = native_result["timings"]
        profile.bfs_nodes += int(counters["bfsNodes"])
        profile.collision_checks += int(counters["collisionChecks"])
        profile.collision_evaluations += int(counters["collisionEvaluations"])
        profile.collision_cache_hits += int(counters["collisionCacheHits"])
        profile.kick_checks += int(counters["kickChecks"])
        profile.landing_queries += int(counters["landingQueries"])
        profile.landing_cache_hits += int(counters["landingCacheHits"])
        profile.representative_nodes += int(counters["representativeNodes"])
        profile.representative_duplicate_skips += int(
            counters["representativeDuplicateSkips"]
        )
        profile.setup_seconds += python_setup_elapsed + float(timings["setupSeconds"])
        profile.bfs_seconds += float(timings["bfsSeconds"])
        profile.rotation_seconds += float(timings["rotationSeconds"])
        profile.landing_seconds += float(timings["landingSeconds"])
        profile.representative_seconds += float(timings["representativeSeconds"])
        profile.placement_seconds += (
            float(timings["placementSeconds"]) + python_placement_elapsed
        )
        profile.placements += len(placements)
        profile.total_seconds += time.perf_counter() - profile_started

    return _reference._cache_reachability_result(cache_key, placements)


def install_native_pathless_search_fast_path() -> None:
    """Route pathless calls through native code while retaining Python fallback."""

    _pathless.reachable_placements_pathless = reachable_placements_pathless_native
