from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux.human_review_pygame import _find_candidate_index, _restore_game
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
        "displayRows": ["." * 10] * 20,
        "context": [0.0] * 59,
        "move": {"hold": False, "piece": "T", "x": 3, "y": 20, "rotation": 0},
        "nnValue": 1.5,
    }
    other = {
        "rows": [1] * 24,
        "displayRows": ["G" + "." * 9] * 20,
        "context": [0.0] * 59,
        "move": {"hold": True, "piece": "I", "x": 4, "y": 19, "rotation": 1},
        "nnValue": 1.4,
    }
    return {
        "format": NEURAL_REVIEW_QUEUE_FORMAT,
        "seed": 123,
        "pieceIndex": 17,
        "reasons": ["low_margin"],
        "source": {
            "rows": [0] * 24,
            "displayRows": ["." * 10] * 20,
            "current": "T",
            "hold": None,
            "next": ["I", "O", "S", "Z", "J"],
            "combo": -1,
            "b2b": False,
            "b2bChain": 0,
            "surgeCharge": 0,
        },
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
        self.assertNotIn("displayRows", candidates[0])

    def test_append_human_label_deduplicates_position(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.jsonl"
            self.assertTrue(append_human_label(path, _queue_record(), 0))
            self.assertFalse(append_human_label(path, _queue_record(), 1))
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["expertIndex"], 0)

    def test_review_record_contains_colored_display_states(self) -> None:
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
        source = record["source"]
        self.assertEqual(len(source["rows"]), 24)
        self.assertEqual(len(source["displayRows"]), 20)
        self.assertGreaterEqual(len(record["candidates"]), 2)
        self.assertLessEqual(len(record["candidates"]), 4)
        for candidate in record["candidates"]:
            self.assertEqual(len(candidate["displayRows"]), 20)

    def test_pygame_reviewer_restores_playable_position(self) -> None:
        game = _restore_game(_queue_record())
        self.assertEqual(game.current, "T")
        self.assertIsNone(game.hold_piece)
        self.assertEqual(list(game.queue)[:5], ["I", "O", "S", "Z", "J"])
        self.assertEqual(game.pieces_placed, 17)
        self.assertFalse(game.game_over)

    def test_manual_placement_maps_back_to_training_candidate(self) -> None:
        record = _queue_record()
        self.assertEqual(
            _find_candidate_index(
                record,
                use_hold=False,
                piece="T",
                x=3,
                y=20,
                rotation=0,
            ),
            0,
        )
        self.assertEqual(
            _find_candidate_index(
                record,
                use_hold=True,
                piece="I",
                x=4,
                y=19,
                rotation=1,
            ),
            1,
        )

    def test_review_cli_defaults_to_tqdm_and_all_legal_candidates(self) -> None:
        args = build_parser().parse_args(["review", "--collect-only"])
        self.assertEqual(args.command, "review")
        self.assertTrue(args.collect_only)
        self.assertEqual(args.max_candidates, 0)
        self.assertFalse(args.no_progress)
        self.assertEqual(HumanReviewConfig().normalized().max_candidates, 0)


if __name__ == "__main__":
    unittest.main()
