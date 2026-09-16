from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import build_capture_samples, load_tetrio_capture


def empty_board() -> list[list[str | None]]:
    return [[None] * 10 for _ in range(40)]


class TetrioLineClearTests(unittest.TestCase):
    def test_completed_capture_rows_are_cleared_before_alignment(self) -> None:
        first = empty_board()
        first[38][0] = "t"
        first[39] = ["g"] * 6 + ["i"] * 4

        second = empty_board()
        second[39][4:8] = ["i"] * 4

        payload = {
            "placements": [
                {
                    "username": "player",
                    "gameid": 1,
                    "pieceIndex": 1,
                    "piece": "i",
                    "x": 6,
                    "y": 39,
                    "rotation": 0,
                    "frame": 10,
                    "hold": None,
                    "round": 1,
                    "sequence": 1,
                    "board": first,
                },
                {
                    "username": "player",
                    "gameid": 1,
                    "pieceIndex": 2,
                    "piece": "i",
                    "x": 4,
                    "y": 39,
                    "rotation": 0,
                    "frame": 20,
                    "hold": None,
                    "round": 1,
                    "sequence": 2,
                    "board": second,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            samples = build_capture_samples(load_tetrio_capture(source))

        # TETR.IO's lock hook sees the completed row before the game's line-clear
        # step.  The importer must reproduce that step before the board is used
        # as either the current post-placement board or the next pre-placement board.
        self.assertIsNone(samples[0].board_after[-2][0])
        self.assertEqual(samples[0].board_after[-1][0], "t")
        self.assertEqual(samples[1].board_before, samples[0].board_after)

    def test_capture_without_completed_rows_is_unchanged(self) -> None:
        board = empty_board()
        board[39][3:7] = ["i"] * 4
        payload = {
            "placements": [{
                "username": "player",
                "gameid": 2,
                "pieceIndex": 1,
                "piece": "i",
                "x": 3,
                "y": 39,
                "rotation": 0,
                "frame": 10,
                "hold": None,
                "round": 1,
                "sequence": 1,
                "board": board,
            }]
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            sample = build_capture_samples(load_tetrio_capture(source))[0]

        self.assertEqual(sample.board_after[-1][3:7], ("i", "i", "i", "i"))


if __name__ == "__main__":
    unittest.main()
