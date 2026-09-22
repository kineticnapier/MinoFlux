from __future__ import annotations

import importlib.util
import unittest


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
