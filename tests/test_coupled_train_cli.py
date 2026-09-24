from __future__ import annotations

import random
import unittest

from minoflux.coupled_train_cli import _build_coupled_sampler, _patched_coupled_training
import minoflux_ai.neural_weighted_train as weighted_train


def _record(seed: int, piece_index: int, weight: float = 1.0) -> dict[str, object]:
    return {
        "seed": seed,
        "pieceIndex": piece_index,
        "trainingWeight": weight,
    }


class CoupledTrainCliTests(unittest.TestCase):
    def test_coupled_sampler_preserves_base_rng_and_only_replaces_slots(self) -> None:
        base = [_record(1, index, 1.0) for index in range(8)]
        full = base + [_record(2, 100, 1.0), _record(2, 101, 1.0)]

        sampler, replacements, new_count, new_mass = _build_coupled_sampler(
            full,
            base,
            seed=12345,
        )

        baseline_rng = random.Random(77)
        coupled_rng = random.Random(77)
        baseline = weighted_train._weighted_epoch_indices(
            tuple(1.0 for _ in base),
            80,
            baseline_rng,
        )
        coupled = sampler(
            tuple(1.0 for _ in full),
            80,
            coupled_rng,
        )

        self.assertEqual(baseline_rng.getstate(), coupled_rng.getstate())
        self.assertEqual(new_count, 2)
        self.assertEqual(new_mass, 2.0)
        self.assertEqual(len(replacements), 1)
        self.assertGreater(replacements[0], 0)
        self.assertEqual(
            sum(left != right for left, right in zip(baseline, coupled, strict=True)),
            replacements[0],
        )
        self.assertTrue(all(index < len(full) for index in coupled))

    def test_disabled_validation_split_does_not_consume_rng_in_coupled_path(self) -> None:
        base = [_record(1, 0)]
        sampler, _replacements, _new_count, _new_mass = _build_coupled_sampler(
            base,
            base,
            seed=12345,
        )
        rng = random.Random(9)
        before = rng.getstate()

        with _patched_coupled_training(sampler):
            train, validation = weighted_train._split_by_game(base, 0.0, rng)

        self.assertEqual(train, base)
        self.assertEqual(validation, [])
        self.assertEqual(rng.getstate(), before)

    def test_shared_records_must_be_identical(self) -> None:
        base = [_record(1, 0, 1.0)]
        full = [_record(1, 0, 0.5)]
        with self.assertRaisesRegex(ValueError, "Shared record differs"):
            _build_coupled_sampler(full, base, seed=1)


if __name__ == "__main__":
    unittest.main()
