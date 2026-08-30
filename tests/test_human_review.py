from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux.neural_cli import build_parser
from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig
from minoflux_ai.human_review import (
    NEURAL_REVIEW_QUEUE_FORMAT,
    HumanReviewConfig,
    append_human_label,
    build_review_record,
    human_label_record,
)
from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT
from minoflux_engine import Game


def _queue_record() -> dict[str, object]:
    candidate = {
        "rows": [0] * 24,
        "context": [0.0] * 59,
        "move": {"hold": False, "piece": "T", "x": 3, "y": 20, "rotation": 0},
        "nnValue": 1.5,
    }
    other = {
        "rows": [1] * 24,
        "context": [0.0] * 59,
        "move": {"hold": True, "piece": "I", "x": 4, "y": 19, "rotation": 1},
        "nnValue": 1.4,
    }
    return {
        "format": NEURAL_REVIEW_QUEUE_FORMAT,
        "seed": 123,
        "pieceIndex": 17,
        "reasons": ["low_margin"],
        "candidates": [candidate, other],
    }


class HumanReviewTests(unittest.TestCase):
    def test_human_label_becomes_standard_ranking_sample(self) -> None:
        labeled = human_label_record(_queue_record(), 1)
        self.assertEqual(labeled["format"], NEURAL_DATASET_FORMAT)
        self.assertEqual(labeled["expertIndex"], 1)
        self.assertEqual(labeled["source"], "human_review")
        candidates = labeled["candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertNotIn("nnValue", candidates[0])

    def test_append_human_label_deduplicates_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.jsonl"
            self.assertTrue(append_human_label(path, _queue_record(), 0))
            self.assertFalse(append_human_label(path, _queue_record(), 1))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["expertIndex"], 0)

    def test_review_record_contains_source_and_candidate_states(self) -> None:
        class HeuristicLikeScorer:
            def score_many(self, game, evaluations):
                return [evaluation.score for evaluation in evaluations]

        record = build_review_record(
            Game(7),
            HeuristicLikeScorer(),
            weights=DEFAULT_WEIGHTS,
            config=HumanReviewConfig(
                max_candidates=4,
                search_config=SearchConfig(
                    allow_hold=True,
                    lookahead_pieces=0,
                    beam_width=2,
                    srs_reachable=True,
                ),
            ),
            reasons=("manual",),
        )
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["format"], NEURAL_REVIEW_QUEUE_FORMAT)
        self.assertEqual(len(record["source"]["rows"]), 24)
        self.assertGreaterEqual(len(record["candidates"]), 2)
        self.assertLessEqual(len(record["candidates"]), 4)

    def test_review_cli_is_available_without_ui_dependency(self) -> None:
        args = build_parser().parse_args(["review", "--collect-only"])
        self.assertEqual(args.command, "review")
        self.assertTrue(args.collect_only)
        self.assertEqual(args.max_candidates, 6)


if __name__ == "__main__":
    unittest.main()
