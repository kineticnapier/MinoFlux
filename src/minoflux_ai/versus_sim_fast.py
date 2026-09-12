from __future__ import annotations

from collections import deque
from copy import copy
import os
import time

from minoflux_engine import Game, VersusMatch, VersusResolution, VersusSide

from . import versus_search as _versus
from .search import SearchAction

_DISABLE_ENV = "MINOFLUX_DISABLE_VERSUS_SIM_FAST"
_ORIGINAL_SIMULATE_ACTION = _versus._simulate_action


def versus_sim_fast_enabled() -> bool:
    disabled = os.environ.get(_DISABLE_ENV, "").strip().lower()
    return disabled not in {"1", "true", "yes", "on"}


def _clone_game_for_action(
    game: Game,
    action: SearchAction,
    *,
    profile=None,
) -> Game:
    """Clone exactly enough Game state for one search action and one lock.

    A direct lock consumes at most one SevenBag item while an empty-Hold action
    can consume at most two. If the bag already contains those items, the RNG is
    provably not touched by the simulated action and can be shared read-only.
    """

    started = _versus.profile_timer_start(profile)
    cloned = copy(game)
    board_started = _versus.profile_timer_start(profile)
    cloned.board = [row.copy() for row in game.board]
    _versus.record_profile_elapsed(profile, "board_copy", board_started)
    cloned.queue = deque(game.queue)
    cloned._bag = copy(game._bag)
    cloned._bag._queue = deque(game._bag._queue)

    required_bag_items = 1 + int(action.use_hold and game.hold_piece is None)
    if len(game._bag._queue) < required_bag_items:
        rng_started = _versus.profile_timer_start(profile)
        cloned._bag._rng = _versus._clone_random(game._bag._rng)
        _versus.record_profile_elapsed(profile, "bag_rng_state_copy", rng_started)
    else:
        cloned._bag._rng = game._bag._rng

    _versus.record_profile_elapsed(profile, "clone_game", started)
    return cloned


def _copy_side_shell(side, *, game) -> VersusSide:
    return VersusSide(
        game=game,
        pending=side.pending,
        sent=side.sent,
        received=side.received,
        canceled=side.canceled,
        garbage_applied=side.garbage_applied,
    )


def _simulate_action_fast(
    match: VersusMatch,
    side_name: _versus.SideName,
    action: SearchAction,
    *,
    _stage: str = "root",
    _profile=None,
) -> tuple[VersusMatch, VersusResolution]:
    """Simulate one lock with copy-on-write queues/RNG and exact engine methods."""

    if not versus_sim_fast_enabled():
        return _ORIGINAL_SIMULATE_ACTION(
            match,
            side_name,
            action,
            _stage=_stage,
            _profile=_profile,
        )

    simulate_started = _versus.profile_timer_start(_profile)
    clone_started_ns = time.perf_counter_ns() if _profile is not None else 0

    simulated = copy(match)
    if side_name == "player":
        player_game = _clone_game_for_action(match.player.game, action, profile=_profile)
        simulated.player = _copy_side_shell(match.player, game=player_game)
        simulated.ai = _copy_side_shell(match.ai, game=match.ai.game)
    else:
        ai_game = _clone_game_for_action(match.ai.game, action, profile=_profile)
        simulated.player = _copy_side_shell(match.player, game=match.player.game)
        simulated.ai = _copy_side_shell(match.ai, game=ai_game)
    simulated._garbage_rng = match._garbage_rng

    initial_clone_ns = (
        time.perf_counter_ns() - clone_started_ns if _profile is not None else 0
    )

    apply_started = _versus.profile_timer_start(_profile)
    result = _versus.apply_search_action(
        _versus._side(simulated, side_name).game,
        action,
    )
    _versus.record_profile_elapsed(_profile, "apply_search_action", apply_started)

    deferred_clone_started_ns = time.perf_counter_ns() if _profile is not None else 0
    attack_possible = result.attack > 0
    acting_game = _versus._side(simulated, side_name).game
    pending_may_apply = result.lines == 0 and not acting_game.game_over

    if attack_possible or pending_may_apply:
        _versus._side(simulated, side_name).pending = _versus._clone_queue(
            _versus._side(match, side_name).pending,
            _profile=_profile,
        )
    if attack_possible:
        opponent_name = _versus._opponent_name(side_name)
        _versus._side(simulated, opponent_name).pending = _versus._clone_queue(
            _versus._side(match, opponent_name).pending,
            _profile=_profile,
        )
        rng_started = _versus.profile_timer_start(_profile)
        simulated._garbage_rng = _versus._clone_random(match._garbage_rng)
        _versus.record_profile_elapsed(
            _profile,
            "garbage_rng_state_copy",
            rng_started,
        )

    if _profile is not None:
        deferred_clone_ns = time.perf_counter_ns() - deferred_clone_started_ns
        _profile.record(
            "clone_versus_match",
            initial_clone_ns + deferred_clone_ns,
        )

    resolve_started = _versus.profile_timer_start(_profile)
    resolution = simulated.resolve_lock(side_name, result)
    _versus.record_profile_elapsed(_profile, "resolve_lock", resolve_started)
    _versus.record_profile_elapsed(
        _profile,
        "root_simulate_action" if _stage == "root" else "reply_simulate_action",
        simulate_started,
    )
    return simulated, resolution


def install_versus_sim_fast_path() -> None:
    _versus._simulate_action = _simulate_action_fast
