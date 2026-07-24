from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import (
    CAPTURE_DATASET_FORMAT,
    CaptureSample,
    align_capture_sample,
    build_capture_samples,
    capture_summary,
    load_tetrio_capture,
    reachable_placements,
    save_capture_dataset,
)
from minoflux_ai.search import clone_game
from minoflux_engine import Game


def board_with(cells: list[tuple[int, int, str]]) -> list[list[str | None]]:
    board: list[list[str | None]] = [[None] * 10 for _ in range(40)]
    for x, y, piece in cells:
        board[y][x] = piece
    return board


class TetrioCaptureTests(unittest.TestCase):
    def test_capture_is_grouped_and_split_without_row_leakage(self) -> None:
        payload = {
            "placements": [
                {
                    "username": "moch",
                    "gameid": 11,
                    "pieceIndex": 2,
                    "piece": "j",
                    "x": 1,
                    "y": 38.3,
                    "rotation": 0,
                    "frame": 18,
                    "hold": None,
                    "round": 1,
                    "sequence": 1,
                    "board": board_with([(0, 38, "j"), (0, 39, "j"), (1, 39, "j"), (2, 39, "j")]),
                },
                {
                    "username": "moch",
                    "gameid": 11,
                    "pieceIndex": 3,
                    "piece": "l",
                    "x": 7,
                    "y": 38.6,
                    "rotation": 0,
                    "frame": 48,
                    "hold": "s",
                    "round": 1,
                    "sequence": 2,
                    "inputs": ["right", "cw", "hard_drop"],
                    "board": board_with([
                        (0, 38, "j"), (0, 39, "j"), (1, 39, "j"), (2, 39, "j"),
                        (7, 38, "l"), (6, 39, "l"), (7, 39, "l"), (8, 39, "l"),
                    ]),
                },
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            placements = load_tetrio_capture(source)
            samples = build_capture_samples(placements)

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].group_id, samples[1].group_id)
        self.assertEqual(samples[0].split, samples[1].split)
        self.assertIsNone(samples[0].board_before)
        self.assertEqual(samples[1].frame_delta, 30)
        self.assertTrue(samples[1].used_hold)
        self.assertEqual(samples[1].operations, ("right", "cw", "hard_drop"))
        self.assertEqual(samples[1].estimated_lines, 0)
        self.assertEqual(samples[1].transition_confidence, "clean-count-inference")

    def test_save_writes_jsonl_and_summary(self) -> None:
        payload = {
            "placements": [{
                "username": "player",
                "gameid": 1,
                "pieceIndex": 1,
                "piece": "i",
                "x": 3,
                "y": 39,
                "rotation": 0,
                "frame": 1,
                "hold": None,
                "round": 1,
                "sequence": 1,
                "board": board_with([(3, 39, "i"), (4, 39, "i"), (5, 39, "i"), (6, 39, "i")]),
            }]
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.json"
            output = Path(directory) / "dataset.jsonl"
            source.write_text(json.dumps(payload), encoding="utf-8")
            samples = build_capture_samples(load_tetrio_capture(source))
            dataset_path, summary_path = save_capture_dataset(output, samples)
            record = json.loads(dataset_path.read_text(encoding="utf-8").splitlines()[0])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(record["format"], CAPTURE_DATASET_FORMAT)
        self.assertEqual(summary, capture_summary(samples))
        self.assertEqual(summary["samples"], 1)

    def test_board_transition_aligns_to_an_exact_srs_route(self) -> None:
        game = Game(321)
        placements = reachable_placements(game, allow_180=True)
        self.assertTrue(placements)
        target = placements[len(placements) // 2]
        before = tuple(tuple(row) for row in game.board)
        simulated = clone_game(game)
        simulated.place(target)
        after = tuple(
            tuple(None if cell is None else str(cell).lower() for cell in row)
            for row in simulated.board
        )
        sample = CaptureSample(
            group_id="player|r1|g1",
            split="train",
            username="player",
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
            operations=(),
            board_before=before,
            board_after=after,
            estimated_lines=0,
            transition_confidence="clean-count-inference",
        )
        alignment = align_capture_sample(sample, allow_180=True)
        self.assertIn(alignment.status, ("exact", "ambiguous"))
        self.assertGreaterEqual(alignment.candidate_count, 1)
        self.assertTrue(alignment.path)
        self.assertEqual(alignment.path[-1], "hard_drop")


if __name__ == "__main__":
    unittest.main()
