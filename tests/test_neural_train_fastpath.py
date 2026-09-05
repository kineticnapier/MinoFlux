from __future__ import annotations

import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "PyTorch optional dependency not installed")
class NeuralTrainFastPathTests(unittest.TestCase):
    def test_cached_features_and_metric_free_loss(self) -> None:
        import torch
        from torch.nn import functional as F

        from minoflux_ai.neural import NeuralValueConfig
        from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT
        from minoflux_ai.neural_train import (
            _build_dataset_cache,
            _loss_and_ranks,
            _loss_only_vectorized,
            _prepare_cached_batch,
        )

        cfg = NeuralValueConfig()
        zero_context = [0.0] * cfg.context_size
        one_context = [0.0] * cfg.context_size
        one_context[0] = 1.0
        record = {
            "format": NEURAL_DATASET_FORMAT,
            "seed": 1,
            "pieceIndex": 0,
            "expertIndex": 0,
            "expertIndices": [0],
            "candidates": [
                {
                    "rows": [1] + [0] * (cfg.board_height - 1),
                    "context": zero_context,
                    "teacherScore": 2.0,
                    "targetValue": 1.0,
                },
                {
                    "rows": [2] + [0] * (cfg.board_height - 1),
                    "context": one_context,
                    "teacherScore": 1.0,
                    "targetValue": 0.0,
                },
            ],
        }
        cache = _build_dataset_cache([record], cfg, torch, "cpu")
        boards, contexts, groups = _prepare_cached_batch(cache, [0], torch)

        self.assertEqual(tuple(boards.shape), (2, 1, cfg.board_height, cfg.board_width))
        self.assertEqual(boards.dtype, torch.float32)
        self.assertEqual(float(boards[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(boards[0, 0, 0, 1]), 0.0)
        self.assertEqual(float(boards[1, 0, 0, 0]), 0.0)
        self.assertEqual(float(boards[1, 0, 0, 1]), 1.0)
        self.assertEqual(float(contexts[1, 0]), 1.0)
        self.assertEqual(groups[0].teacher_pairs, ((0, 1),))
        self.assertEqual(groups[0].rollout_pairs, ((0, 1),))

        values = torch.tensor([0.3, 0.1], dtype=torch.float32, requires_grad=True)
        measured_loss, top1, top3, mean_rank = _loss_and_ranks(
            values,
            groups,
            torch,
            F,
            margin=0.2,
            teacher_weight=0.25,
            rollout_weight=0.5,
            collect_metrics=True,
        )
        reference_loss, fast_top1, fast_top3, fast_rank = _loss_and_ranks(
            values,
            groups,
            torch,
            F,
            margin=0.2,
            teacher_weight=0.25,
            rollout_weight=0.5,
            collect_metrics=False,
        )
        vectorized_loss = _loss_only_vectorized(
            values,
            groups,
            torch,
            F,
            margin=0.2,
            teacher_weight=0.25,
            rollout_weight=0.5,
        )
        self.assertTrue(torch.allclose(measured_loss, reference_loss, rtol=0.0, atol=0.0))
        self.assertTrue(torch.allclose(measured_loss, vectorized_loss, rtol=1e-6, atol=1e-7))
        self.assertEqual((top1, top3, mean_rank), (1, 1, 1.0))
        self.assertEqual((fast_top1, fast_top3, fast_rank), (0, 0, 0.0))

    def test_vectorized_loss_matches_reference_for_multiple_samples(self) -> None:
        import torch
        from torch.nn import functional as F

        from minoflux_ai.neural_train import _PreparedGroup, _loss_and_ranks, _loss_only_vectorized

        groups = (
            _PreparedGroup(
                start=0,
                end=3,
                expert_indices=(0,),
                negative_indices=(1, 2),
                teacher_pairs=((0, 1), (0, 2), (1, 2)),
                rollout_pairs=((1, 2),),
            ),
            _PreparedGroup(
                start=3,
                end=7,
                expert_indices=(0, 1),
                negative_indices=(2, 3),
                teacher_pairs=((1, 2),),
                rollout_pairs=((0, 3), (2, 3)),
            ),
        )
        values = torch.tensor(
            [0.5, 0.2, -0.1, 0.7, 0.6, 0.3, 0.0],
            dtype=torch.float32,
            requires_grad=True,
        )
        reference, _, _, _ = _loss_and_ranks(
            values,
            groups,
            torch,
            F,
            margin=0.2,
            teacher_weight=0.25,
            rollout_weight=0.5,
            collect_metrics=False,
        )
        vectorized = _loss_only_vectorized(
            values,
            groups,
            torch,
            F,
            margin=0.2,
            teacher_weight=0.25,
            rollout_weight=0.5,
        )
        self.assertTrue(torch.allclose(reference, vectorized, rtol=1e-6, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
