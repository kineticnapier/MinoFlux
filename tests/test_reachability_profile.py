from __future__ import annotations

from minoflux_ai.reachability import (
    ReachabilityProfile,
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
        placement.last_move_was_rotation,
        placement.rotation_kick_index,
        placement.rotation_from,
        placement.rotation_to,
    )


def test_reachability_profile_preserves_placements() -> None:
    game = Game(20260902)
    baseline = reachable_placements(game, include_paths=False)

    profile = ReachabilityProfile()
    with collect_reachability_profile(profile):
        profiled = reachable_placements(game, include_paths=False)

    assert tuple(map(_signature, profiled)) == tuple(map(_signature, baseline))
    metrics = profile.to_dict()
    assert metrics["calls"] == 1
    assert metrics["placements"] == len(profiled)
    assert metrics["bfsNodes"] > 0
    assert metrics["collisionChecks"] > 0
    assert 0 < metrics["collisionEvaluations"] <= metrics["collisionChecks"]
    assert 0 <= metrics["collisionCacheHits"] <= metrics["collisionChecks"]
    assert metrics["landingQueries"] > 0
    assert metrics["representativeNodes"] >= len(profiled)
    assert metrics["totalSeconds"] >= 0.0
    assert metrics["bfsSeconds"] >= metrics["rotationSeconds"] >= 0.0
    assert 0.0 <= metrics["collisionCacheHitRate"] <= 1.0
    assert 0.0 <= metrics["landingCacheHitRate"] <= 1.0
