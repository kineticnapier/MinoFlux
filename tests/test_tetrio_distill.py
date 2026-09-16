from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import CaptureAlignment, CaptureSample, reachable_placements
from minoflux_ai.search import clone_game
from minoflux_ai.tetrio_distill import TetrioDistillConfig, write_tetrio_ranking_dataset
from minoflux_engine import Game


def _board(game: Game):
    return tuple(tuple(None if cell is None else str(cell).lower() for cell in row) for row in game.board)


def _alignment(sample: CaptureSample, target) -> CaptureAlignment:
    return CaptureAlignment(
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


class TetrioDistillTests(unittest.TestCase):
    def test_capture_and_alignment_write_neural_ranking_samples(self) -> None:
        game = Game(24680)
        samples: list[CaptureSample] = []
        alignments: list[CaptureAlignment] = []
        for sequence in range(1, 9):
            before = _board(game)
            piece = game.current
            placements = reachable_placements(game, allow_180=True)
            self.assertTrue(placements)
            target = placements[len(placements) // 2]
            simulated = clone_game(game)
            simulated.place(target)
            sample = CaptureSample(
                group_id="mochbot|r1|g1",
                split="train",
                username="mochbot",
                game_id=1,
                round=1,
                sequence=sequence,
                frame=sequence * 6,
                frame_delta=6,
                piece_index=sequence,
                piece=piece,
                x=float(target.x),
                y=float(target.y),
                rotation=target.rotation,
                hold_before=None,
                hold_after=None,
                used_hold=False,
                next_placed_piece=None,
                operations=(),
                board_before=before,
                board_after=_board(simulated),
                estimated_lines=None,
                transition_confidence="clean-count-inference",
            )
            samples.append(sample)
            alignments.append(_alignment(sample, target))
            game = simulated

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_path = root / "capture.jsonl"
            alignment_path = root / "alignment.jsonl"
            output_path = root / "ranking.jsonl"
            capture_path.write_text(
                "".join(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n" for sample in samples),
                encoding="utf-8",
            )
            alignment_path.write_text(
                "".join(json.dumps(item.to_dict(), separators=(",", ":")) + "\n" for item in alignments),
                encoding="utf-8",
            )
            result = write_tetrio_ranking_dataset(
                capture_path,
                alignment_path,
                output_path,
                TetrioDistillConfig(max_candidates=24, allow_180=True),
            )
            records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

        self.assertGreaterEqual(result["samples"], 2)
        self.assertEqual(result["samples"], len(records))
        self.assertGreater(result["candidates"], result["samples"])
        self.assertGreaterEqual(result["skipped"].get("insufficient-future-queue", 0), 6)
        for record in records:
            self.assertEqual(record["format"], "minoflux_neural_ranking_dataset_v1")
            self.assertEqual(record["teacher"], "tetrio-capture")
            self.assertTrue(record["expertIndices"])
            expert = record["candidates"][record["expertIndex"]]
            self.assertEqual(expert["samplingBucket"], "tetrio-expert")

    def test_used_hold_sample_is_skipped_in_first_distillation_pass(self) -> None:
        game = Game(13579)
        target = reachable_placements(game, allow_180=True)[0]
        simulated = clone_game(game)
        simulated.place(target)
        sample = CaptureSample(
            group_id="mochbot|r1|g2",
            split="train",
            username="mochbot",
            game_id=2,
            round=1,
            sequence=1,
            frame=1,
            frame_delta=None,
            piece_index=1,
            piece=game.current,
            x=float(target.x),
            y=float(target.y),
            rotation=target.rotation,
            hold_before=None,
            hold_after="T",
            used_hold=True,
            next_placed_piece=None,
            operations=(),
            board_before=_board(game),
            board_after=_board(simulated),
            estimated_lines=0,
            transition_confidence="clean-count-inference",
        )
        alignment = _alignment(sample, target)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture_path = root / "capture.jsonl"
            alignment_path = root / "alignment.jsonl"
            output_path = root / "ranking.jsonl"
            capture_path.write_text(json.dumps(sample.to_dict()) + "\n", encoding="utf-8")
            alignment_path.write_text(json.dumps(alignment.to_dict()) + "\n", encoding="utf-8")
            result = write_tetrio_ranking_dataset(capture_path, alignment_path, output_path)
        self.assertEqual(result["samples"], 0)
        self.assertEqual(result["skipped"].get("used-hold"), 1)


if __name__ == "__main__":
    unittest.main()
