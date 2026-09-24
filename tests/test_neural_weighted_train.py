from __future__ import annotations

from collections import Counter
import importlib.util
import os
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class NeuralWeightedDeterminismTests(unittest.TestCase):
    def test_weighted_train_cli_deterministic_flag_is_opt_in(self) -> None:
        from minoflux.weighted_train_cli import build_parser

        parser = build_parser()
        self.assertFalse(parser.parse_args([]).deterministic)
        self.assertTrue(parser.parse_args(["--deterministic"]).deterministic)
        self.assertIsNone(parser.parse_args([]).samples_per_epoch)
        self.assertEqual(
            parser.parse_args(["--samples-per-epoch", "2304"]).samples_per_epoch,
            2304,
        )

    def test_deterministic_environment_sets_cublas_workspace(self) -> None:
        from minoflux_ai.neural_weighted_train import (
            _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            _prepare_deterministic_environment,
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            _prepare_deterministic_environment(True)
            self.assertEqual(
                os.environ["CUBLAS_WORKSPACE_CONFIG"],
                _DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            )

    def test_deterministic_torch_disables_nondeterministic_fast_paths(self) -> None:
        from minoflux_ai.neural_weighted_train import _configure_deterministic_torch

        calls: list[bool] = []
        cudnn = SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)
        matmul = SimpleNamespace(allow_tf32=True)
        fake_torch = SimpleNamespace(
            use_deterministic_algorithms=calls.append,
            backends=SimpleNamespace(
                cudnn=cudnn,
                cuda=SimpleNamespace(matmul=matmul),
            ),
        )

        _configure_deterministic_torch(fake_torch, True)

        self.assertEqual(calls, [True])
        self.assertFalse(cudnn.benchmark)
        self.assertTrue(cudnn.deterministic)
        self.assertFalse(cudnn.allow_tf32)
        self.assertFalse(matmul.allow_tf32)

    def test_zero_weight_records_are_removed_before_training(self) -> None:
        from minoflux_ai.neural_weighted_train import _positive_weight_records

        records = [
            {"seed": 1},
            {"seed": 2, "trainingWeight": 0.0},
            {"seed": 3, "trainingWeight": 0.25},
        ]

        filtered = _positive_weight_records(records)

        self.assertEqual([record["seed"] for record in filtered], [1, 3])

    def test_weighted_epoch_sampler_uses_weight_as_frequency(self) -> None:
        from minoflux_ai.neural_weighted_train import _weighted_epoch_indices

        sampled = _weighted_epoch_indices(
            (4.0, 2.0, 1.0),
            7,
            random.Random(12345),
        )

        self.assertEqual(len(sampled), 7)
        self.assertEqual(Counter(sampled), Counter({0: 4, 1: 2, 2: 1}))

    def test_weighted_epoch_sampler_is_reproducible(self) -> None:
        from minoflux_ai.neural_weighted_train import _weighted_epoch_indices

        left = _weighted_epoch_indices((1.0, 0.5, 0.25), 64, random.Random(7))
        right = _weighted_epoch_indices((1.0, 0.5, 0.25), 64, random.Random(7))

        self.assertEqual(left, right)

    def test_samples_per_epoch_defaults_to_effective_weight_mass(self) -> None:
        from minoflux_ai.neural_weighted_train import _resolve_samples_per_epoch

        self.assertEqual(_resolve_samples_per_epoch((1.0, 0.5, 0.25), None), 2)
        self.assertEqual(_resolve_samples_per_epoch((1.0, 0.5, 0.25), 2304), 2304)
        with self.assertRaises(ValueError):
            _resolve_samples_per_epoch((1.0,), 0)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch optional dependency not installed")
class NeuralWeightedTrainTests(unittest.TestCase):
    def test_weighted_loss_scales_each_sample_without_renormalizing(self) -> None:
        import torch
        from torch.nn import functional as F

        from minoflux_ai.neural_train import _PreparedGroup, _loss_only_vectorized
        from minoflux_ai.neural_weighted_train import _weighted_loss_only_vectorized

        groups = (
            _PreparedGroup(
                start=0,
                end=2,
                expert_indices=(0,),
                negative_indices=(1,),
                teacher_pairs=(),
                rollout_pairs=(),
            ),
            _PreparedGroup(
                start=2,
                end=4,
                expert_indices=(0,),
                negative_indices=(1,),
                teacher_pairs=(),
                rollout_pairs=(),
            ),
        )
        values = torch.tensor(
            [0.0, 0.0, -0.2, 0.0],
            dtype=torch.float32,
            requires_grad=True,
        )

        weighted = _weighted_loss_only_vectorized(
            values,
            groups,
            (1.0, 0.25),
            torch,
            F,
            margin=0.2,
            teacher_weight=0.0,
            rollout_weight=0.0,
        )
        self.assertAlmostEqual(float(weighted.detach()), 0.15, places=6)

        all_one = _weighted_loss_only_vectorized(
            values,
            groups,
            (1.0, 1.0),
            torch,
            F,
            margin=0.2,
            teacher_weight=0.0,
            rollout_weight=0.0,
        )
        reference = _loss_only_vectorized(
            values,
            groups,
            torch,
            F,
            margin=0.2,
            teacher_weight=0.0,
            rollout_weight=0.0,
        )
        self.assertTrue(torch.allclose(all_one, reference, rtol=0.0, atol=0.0))


if __name__ == "__main__":
    unittest.main()
