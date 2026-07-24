from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    CaptureAlignment,
    CaptureSample,
    ImitationConfig,
    prepare_imitation_examples,
    reachable_placements,
    train_imitation,
)
from minoflux_ai.search import clone_game
from minoflux_engine import Game


class ImitationTrainingTests(unittest.TestCase):
    def _fixture(self) -> tuple[CaptureSample, CaptureAlignment]:
        game = Game(777)
        placements = reachable_placements(game, allow_180=True)
        self.assertGreater(len(placements), 2)
        target = placements[-2]
        before = tuple(tuple(row) for row in game.board)
        simulated = clone_game(game)
        simulated.place(target)
        after = tuple(
            tuple(None if cell is None else str(cell).lower() for cell in row)
            for row in simulated.board
        )
        sample = CaptureSample(
            group_id="mochbot|r1|g1",
            split="train",
            username="mochbot",
            game_id=1,
            round=1,
            sequence=2,
            frame=30,
            frame_delta=30,
            piece_index=2,
            piece=game.current,
            x=float(target.x),
            y=float(target.y),
            rotation=target.rotation,
            hold_before=None,
            hold_after=None,
            used_hold=False,
            next_placed_piece=None,
            operations=target.path,
            board_before=before,
            board_after=after,
            estimated_lines=0,
            transition_confidence="clean-count-inference",
        )
        alignment = CaptureAlignment(
            group_id=sample.group_id,
            sequence=sample.sequence,
            piece=sample.piece,
            status="exact",
            candidate_count=1,
            x=target.x,
            y=target.y,
            rotation=target.rotation,
            path=target.path,
            last_move_was_rotation=target.last_move_was_rotation,
            rotation_kick_index=target.rotation_kick_index,
            rotation_from=target.rotation_from,
            rotation_to=target.rotation_to,
        )
        return sample, alignment

    def test_prepare_and_train_one_aligned_example(self) -> None:
        sample, alignment = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "dataset.jsonl"
            alignments = Path(directory) / "alignment.jsonl"
            dataset.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
            alignments.write_text(json.dumps(alignment.to_dict()) + "\n", encoding="utf-8")
            config = ImitationConfig(epochs=1, max_samples=10, allow_180=True)
            examples, skipped = prepare_imitation_examples(dataset, alignments, config)
            result = train_imitation(dataset, alignments, DEFAULT_WEIGHTS, config)

        self.assertEqual(skipped, {})
        self.assertEqual(len(examples), 1)
        self.assertEqual(result.prepared_examples, 1)
        self.assertEqual(result.train.samples, 1)
        self.assertEqual(len(result.epoch_losses), 1)
        self.assertEqual(result.learned_weights.game_over, DEFAULT_WEIGHTS.game_over)


if __name__ == "__main__":
    unittest.main()
