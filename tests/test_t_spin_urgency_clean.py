from __future__ import annotations

from collections import deque
from tempfile import TemporaryDirectory
import unittest

from minoflux_ai import DEFAULT_WEIGHTS, HeuristicWeights, load_weights, save_weights
from minoflux_ai.features import BoardFeatures
from minoflux_ai.heuristic import _t_spin_slot_urgency_clean_score
from minoflux_engine import Game


class TSpinUrgencyCleanTests(unittest.TestCase):
    def board(self, *, slots: int, holes: int) -> BoardFeatures:
        return BoardFeatures(
            aggregate_height=0,
            max_height=0,
            holes=holes,
            hole_depth=0,
            bumpiness=0,
            wells=0,
            t_spin_slots=slots,
            occupied_cells=0,
        )

    def test_score_rewards_clean_slots_more_when_t_is_near(self) -> None:
        game = Game(123)
        game.current = "I"
        game.queue = deque(["T", "O", "S", "Z", "J", "L", "I"])
        before = game.snapshot()
        clean = _t_spin_slot_urgency_clean_score(game, self.board(slots=2, holes=0))
        dirty = _t_spin_slot_urgency_clean_score(game, self.board(slots=2, holes=2))
        self.assertAlmostEqual(clean, 0.34 * (6 / 7) * 2)
        self.assertAlmostEqual(dirty, clean / 3)
        self.assertGreater(clean, dirty)
        self.assertEqual(game.snapshot(), before)

    def test_score_is_zero_without_a_near_t_or_slots(self) -> None:
        game = Game(456)
        game.current = "I"
        game.queue = deque(["I", "O", "S", "Z", "J", "L", "I"])
        self.assertEqual(_t_spin_slot_urgency_clean_score(game, self.board(slots=2, holes=0)), 0.0)
        game.current = "T"
        self.assertEqual(_t_spin_slot_urgency_clean_score(game, self.board(slots=0, holes=0)), 0.0)

    def test_old_weight_mapping_uses_urgency_clean_default(self) -> None:
        old_weights = DEFAULT_WEIGHTS.to_dict()
        old_weights.pop("t_spin_slot_urgency_clean")
        loaded = HeuristicWeights.from_mapping(old_weights)
        self.assertEqual(
            loaded.t_spin_slot_urgency_clean,
            DEFAULT_WEIGHTS.t_spin_slot_urgency_clean,
        )

    def test_weight_round_trip(self) -> None:
        custom = HeuristicWeights(t_spin_slot_urgency_clean=0.73)
        with TemporaryDirectory() as directory:
            path = save_weights(f"{directory}/weights.json", custom)
            self.assertEqual(load_weights(path), custom)


if __name__ == "__main__":
    unittest.main()
