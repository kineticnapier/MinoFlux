from __future__ import annotations

import random
import unittest


class WeightedSamplingCliTests(unittest.TestCase):
    def test_parser_defaults_match_weighted_training_experiment(self) -> None:
        from minoflux.weighted_sampling_cli import build_parser

        args = build_parser().parse_args(["left.jsonl", "right.jsonl"])
        self.assertEqual(args.epochs, 8)
        self.assertEqual(args.samples_per_epoch, 2304)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.validation_fraction, 0.0)
        self.assertEqual(args.seed, 12345)

    def test_sampling_trace_is_reproducible(self) -> None:
        from minoflux.weighted_sampling_cli import _simulate_sampling

        keys = ((1, 0), (2, 0), (3, 0))
        weights = (1.0, 0.5, 0.25)
        left = _simulate_sampling(
            keys,
            weights,
            random.Random(12345),
            epochs=3,
            samples_per_epoch=64,
            batch_size=16,
        )
        right = _simulate_sampling(
            keys,
            weights,
            random.Random(12345),
            epochs=3,
            samples_per_epoch=64,
            batch_size=16,
        )
        self.assertEqual(left, right)
        self.assertEqual([len(epoch) for epoch in left], [64, 64, 64])

    def test_multiset_overlap_counts_duplicate_draws(self) -> None:
        from minoflux.weighted_sampling_cli import _multiset_overlap

        left = ((1, 0), (1, 0), (2, 0), (3, 0))
        right = ((1, 0), (2, 0), (2, 0), (4, 0))
        self.assertEqual(_multiset_overlap(left, right), 2)


if __name__ == "__main__":
    unittest.main()
