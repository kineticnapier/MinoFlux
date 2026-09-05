from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from minoflux_ai.features import BoardFeatures
from minoflux_ai.heuristic import (
    DEFAULT_WEIGHTS,
    HeuristicWeights,
    PlacementFeatures,
    _t_arrival_conversion_score,
    load_weights,
    rank_placements,
    save_weights,
)
from minoflux_engine import Game


def _features(*, spin_lines: int = 0, attack: int = 0, slot_delta: int = 0) -> PlacementFeatures:
    return PlacementFeatures(
        board=BoardFeatures(
            aggregate_height=0,
            max_height=0,
            holes=0,
            hole_depth=0,
            bumpiness=0,
            wells=0,
            t_spin_slots=max(0, slot_delta),
            occupied_cells=0,
        ),
        new_holes=0,
        lines=spin_lines,
        attack=attack,
        spin_lines=spin_lines,
        perfect_clear=False,
        game_over=False,
        spin="t_spin_double" if spin_lines else None,
        t_spin_slot_delta=slot_delta,
    )


class TArrivalConversionTests(unittest.TestCase):
    def test_ready_t_spin_conversion_formula(self) -> None:
        game = Game(7)
        game.current = "T"
        features = _features(spin_lines=2, attack=4, slot_delta=-1)
        self.assertAlmostEqual(_t_arrival_conversion_score(game, features), 3.2)

    def test_ready_t_penalizes_destroying_slot_without_spin(self) -> None:
        game = Game(11)
        game.current = "T"
        features = _features(slot_delta=-2)
        self.assertAlmostEqual(_t_arrival_conversion_score(game, features), -1.6)

    def test_non_t_piece_has_no_arrival_conversion_score(self) -> None:
        game = Game(13)
        game.current = "I"
        features = _features(spin_lines=2, attack=4, slot_delta=-1)
        self.assertEqual(_t_arrival_conversion_score(game, features), 0.0)

    def test_old_weight_mapping_uses_arrival_conversion_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_arrival_conversion")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.t_arrival_conversion,
            DEFAULT_WEIGHTS.t_arrival_conversion,
        )

    def test_arrival_conversion_weight_round_trip(self) -> None:
        custom = HeuristicWeights(t_arrival_conversion=0.75)
        with TemporaryDirectory() as directory:
            path = save_weights(f"{directory}/weights.json", custom)
            self.assertEqual(load_weights(path), custom)

    def test_arrival_conversion_ranking_does_not_mutate_game(self) -> None:
        game = Game(17)
        game.current = "T"
        before = game.snapshot()
        ranked = rank_placements(game, DEFAULT_WEIGHTS)
        self.assertTrue(ranked)
        self.assertEqual(game.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
