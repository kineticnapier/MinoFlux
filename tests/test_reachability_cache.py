from __future__ import annotations

from minoflux_ai import reachability
from minoflux_ai.reachability import (
    ReachabilityProfile,
    clear_reachability_cache,
    collect_reachability_profile,
    reachable_placements,
)
from minoflux_engine import Game


def _signature(placement):
    return (
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
        placement.cells,
        placement.path,
        placement.last_move_was_rotation,
        placement.rotation_kick_index,
        placement.rotation_from,
        placement.rotation_to,
    )


def _profiled(game: Game, **kwargs):
    profile = ReachabilityProfile()
    with collect_reachability_profile(profile):
        placements = reachable_placements(game, **kwargs)
    return placements, profile


def test_warm_cache_is_exact_and_reports_hit() -> None:
    clear_reachability_cache()
    game = Game(20260904)

    cold, cold_profile = _profiled(game, allow_180=True, include_paths=True)
    warm, warm_profile = _profiled(game, allow_180=True, include_paths=True)

    assert cold
    assert tuple(map(_signature, warm)) == tuple(map(_signature, cold))
    assert cold_profile.cache_misses == 1
    assert cold_profile.cache_hits == 0
    assert warm_profile.cache_hits == 1
    assert warm_profile.cache_misses == 0
    assert warm_profile.bfs_nodes == 0
    assert all(placement.path[-1] == "hard_drop" for placement in warm)


def test_cache_key_tracks_source_mutation_and_path_mode() -> None:
    clear_reachability_cache()
    game = Game(90210)

    with_paths, first_profile = _profiled(game, include_paths=True)
    pathless, pathless_profile = _profiled(game, include_paths=False)
    assert tuple(
        signature[:5] + signature[6:]
        for signature in map(_signature, with_paths)
    ) == tuple(
        signature[:5] + signature[6:]
        for signature in map(_signature, pathless)
    )
    assert all(not placement.path for placement in pathless)
    assert first_profile.cache_misses == 1
    assert pathless_profile.cache_misses == 1

    game.board[-1][0] = "G"
    changed, changed_profile = _profiled(game, include_paths=True)
    assert changed_profile.cache_misses == 1
    assert tuple(map(_signature, changed)) != tuple(map(_signature, with_paths))


def test_lru_is_bounded_and_evicts_least_recently_used(monkeypatch) -> None:
    clear_reachability_cache()
    monkeypatch.setattr(reachability, "_REACHABILITY_CACHE_MAXSIZE", 2)

    games = [Game(1), Game(2), Game(3)]
    for piece, game in zip(("I", "O", "T"), games):
        game.current = piece

    reachable_placements(games[0])
    reachable_placements(games[1])
    _, touched_profile = _profiled(games[0])
    assert touched_profile.cache_hits == 1

    reachable_placements(games[2])
    assert len(reachability._REACHABILITY_CACHE) == 2

    _, retained_profile = _profiled(games[0])
    _, evicted_profile = _profiled(games[1])
    assert retained_profile.cache_hits == 1
    assert evicted_profile.cache_misses == 1

    clear_reachability_cache()


def test_empty_invalid_spawn_result_is_cached() -> None:
    clear_reachability_cache()
    game = Game(77)
    game.x = -100

    cold, cold_profile = _profiled(game)
    warm, warm_profile = _profiled(game)

    assert cold == warm == ()
    assert cold_profile.cache_misses == 1
    assert warm_profile.cache_hits == 1
