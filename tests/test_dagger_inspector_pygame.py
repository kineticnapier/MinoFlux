from __future__ import annotations

import unittest

from minoflux.dagger_inspector_pygame import (
    _best_index,
    _candidate_snapshot,
    _is_disagreement,
    _move_text,
    _visible_records,
)


def _candidate(*, mask: int, move: list[object], attack: int = 0) -> dict[str, object]:
    context = [0.0] * 57
    context[-4] = 0.25 if attack else 0.0
    context[-3] = attack / 20.0
    return {
        "rows": [0] * 23 + [mask],
        "context": context,
        "move": move,
    }


class DaggerInspectorTests(unittest.TestCase):
    def test_move_text_accepts_compact_dagger_move(self) -> None:
        candidate = _candidate(mask=1, move=[1, "T", 7, 18, 0])
        self.assertEqual(_move_text(candidate), "T@(7,18)r0 hold")

    def test_candidate_snapshot_reports_board_and_attack(self) -> None:
        candidate = _candidate(mask=0b11, move=[0, "I", 3, 20, 1], attack=4)
        snapshot = _candidate_snapshot(candidate)
        self.assertEqual(snapshot["max_height"], 1)
        self.assertEqual(snapshot["holes"], 0)
        self.assertEqual(snapshot["attack"], 4)
        self.assertEqual(snapshot["move"], "I@(3,20)r1")

    def test_visible_records_filters_and_sorts_confident_disagreements(self) -> None:
        records = [
            {"seed": 1, "pieceIndex": 10, "learnerMargin": 0.02, "learnerMatchedOracle": False},
            {"seed": 2, "pieceIndex": 20, "learnerMargin": 0.40, "learnerMatchedOracle": False},
            {"seed": 3, "pieceIndex": 30, "learnerMargin": 0.90, "learnerMatchedOracle": True},
        ]
        visible = _visible_records(
            records,
            disagreements_only=True,
            confident_only=True,
            sort_mode="margin",
        )
        self.assertEqual([(record["seed"], record["pieceIndex"]) for record in visible], [(2, 20)])
        self.assertTrue(_is_disagreement(records[0]))
        self.assertFalse(_is_disagreement(records[2]))

    def test_best_index_handles_disabled_and_scores(self) -> None:
        self.assertEqual(_best_index((), 3), -1)
        self.assertEqual(_best_index((0.1, 0.8, 0.3), 3), 1)


if __name__ == "__main__":
    unittest.main()
