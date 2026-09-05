from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai.neural_train import _load_jsonl
from minoflux_ai.placement_teacher import (
    PlacementTeacherConfig,
    PlacementTeacherWeights,
    PlacementV2DatasetConfig,
    placement_teacher_transition_score,
    rank_placement_teacher_actions,
    write_placement_v2_dataset,
)
from minoflux_ai.reachability import reachable_placements
from minoflux_ai.search import clone_game
from minoflux_engine import Game, LockResult


def _placement_key(placement) -> tuple[object, ...]:
    return (
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
        placement.last_move_was_rotation,
        placement.rotation_kick_index,
    )


class PlacementTeacherTests(unittest.TestCase):
    def test_teacher_does_not_mutate_source_game(self) -> None:
        game = Game(seed=701)
        before = game.snapshot()
        ranked = rank_placement_teacher_actions(
            game,
            config=PlacementTeacherConfig(depth=1, beam_width=4),
        )
        self.assertTrue(ranked)
        self.assertEqual(game.snapshot(), before)

    def test_teacher_is_deterministic(self) -> None:
        config = PlacementTeacherConfig(depth=2, beam_width=4)
        first = rank_placement_teacher_actions(Game(seed=702), config=config)
        second = rank_placement_teacher_actions(Game(seed=702), config=config)
        self.assertEqual(first, second)

    def test_depth_one_is_immediate_only(self) -> None:
        ranked = rank_placement_teacher_actions(
            Game(seed=703),
            config=PlacementTeacherConfig(depth=1, beam_width=3),
        )
        self.assertTrue(ranked)
        for score in ranked:
            self.assertEqual(score.future_value, 0.0)
            self.assertAlmostEqual(score.total, score.immediate)

    def test_hold_can_be_enabled_and_disabled(self) -> None:
        with_hold = rank_placement_teacher_actions(
            Game(seed=704),
            config=PlacementTeacherConfig(depth=1, allow_hold=True),
        )
        without_hold = rank_placement_teacher_actions(
            Game(seed=704),
            config=PlacementTeacherConfig(depth=1, allow_hold=False),
        )
        self.assertTrue(any(score.action.use_hold for score in with_hold))
        self.assertFalse(any(score.action.use_hold for score in without_hold))

    def test_direct_actions_are_exact_srs_reachable_actions(self) -> None:
        game = Game(seed=705)
        config = PlacementTeacherConfig(depth=1, allow_hold=False, allow_180=False)
        ranked = rank_placement_teacher_actions(game, config=config)
        expected = {
            _placement_key(placement)
            for placement in reachable_placements(
                game,
                allow_180=False,
                max_nodes=config.reachability_node_limit,
                include_paths=True,
            )
        }
        actual = {_placement_key(score.action.placement) for score in ranked}
        self.assertEqual(actual, expected)

    def test_engine_attack_is_directly_rewarded(self) -> None:
        before = Game(seed=706)
        after = clone_game(before)
        quiet = LockResult(0, 0, None, False, -1, False, False)
        attacking = LockResult(4, 4, None, False, 0, False, False)
        quiet_score = placement_teacher_transition_score(before, after, quiet)
        attack_score = placement_teacher_transition_score(before, after, attacking)
        self.assertGreater(attack_score.total, quiet_score.total)
        self.assertEqual(attack_score.attack, 4)

    def test_b2b_break_and_topout_are_penalized(self) -> None:
        before = Game(seed=707)
        before.back_to_back = True
        before.b2b_chain = 3
        safe_after = clone_game(before)
        safe_result = LockResult(0, 0, None, False, -1, True, False, b2b_chain=3)
        safe = placement_teacher_transition_score(before, safe_after, safe_result)

        broken_after = clone_game(before)
        broken_after.back_to_back = False
        broken_after.game_over = True
        broken_result = LockResult(1, 0, None, False, 0, False, True, b2b_chain=0)
        broken = placement_teacher_transition_score(before, broken_after, broken_result)
        self.assertTrue(broken.b2b_broken)
        self.assertGreater(broken.b2b_break_penalty, 0.0)
        self.assertGreater(broken.topout_penalty, 0.0)
        self.assertLess(broken.total, safe.total)

    def test_weights_reject_unknown_keys(self) -> None:
        with self.assertRaises(ValueError):
            PlacementTeacherWeights.from_mapping({"not_a_weight": 1.0})

    def test_dataset_is_compatible_with_existing_ranking_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "placement-v2.jsonl"
            result = write_placement_v2_dataset(
                path,
                PlacementV2DatasetConfig(
                    games=1,
                    max_pieces=1,
                    max_candidates=8,
                    teacher=PlacementTeacherConfig(depth=1, beam_width=2),
                ),
                workers=1,
            )
            self.assertEqual(result["samples"], 1)
            records = _load_jsonl(path)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["format"], "minoflux_neural_ranking_dataset_v1")
            self.assertEqual(record["teacher"], "placement-v2")
            self.assertTrue(record["expertIndices"])
            candidates = record["candidates"]
            self.assertTrue(candidates)
            self.assertIn("teacherScore", candidates[0])
            self.assertIn("teacherBreakdown", candidates[0])
            self.assertIn("teacherFutureValue", candidates[0])
            json.dumps(record)

    def test_config_normalization_bounds_search(self) -> None:
        config = PlacementTeacherConfig(depth=99, beam_width=999, discount=2.0).normalized()
        self.assertEqual(config.depth, 3)
        self.assertEqual(config.beam_width, 128)
        self.assertEqual(config.discount, 1.0)


if __name__ == "__main__":
    unittest.main()
