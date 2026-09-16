from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minoflux_ai import load_tetrio_capture


class TetrioCompactCaptureTests(unittest.TestCase):
    def test_loads_row_major_board_compact_capture(self) -> None:
        compact = "." * 396 + "IIII"
        payload = {
            "placements": [
                {
                    "username": "mochbot",
                    "gameid": 3321,
                    "pieceIndex": 2,
                    "piece": "i",
                    "x": 6,
                    "y": 39,
                    "rotation": 0,
                    "frame": 1,
                    "hold": None,
                    "round": 1,
                    "sequence": 1,
                    "boardCompact": compact,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            placements = load_tetrio_capture(source)

        self.assertEqual(len(placements), 1)
        self.assertEqual(len(placements[0].board40), 40)
        self.assertEqual(placements[0].board40[-1], (None,) * 6 + ("i",) * 4)

    def test_rejects_board_compact_with_wrong_length(self) -> None:
        payload = {
            "placements": [
                {
                    "username": "mochbot",
                    "piece": "t",
                    "boardCompact": "." * 399,
                }
            ]
        }

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "capture.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "400"):
                load_tetrio_capture(source)


if __name__ == "__main__":
    unittest.main()
