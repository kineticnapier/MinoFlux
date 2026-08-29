from __future__ import annotations

from pathlib import Path

HEURISTIC = Path("src/minoflux_ai/heuristic.py")
TEST = Path("tests/test_garbage_slot_supply_floor.py")
WORKFLOW = Path(".github/workflows/self-improve-tournament.yml")
SELF = Path("tools/self_improve_tournament.py")

text = HEURISTIC.read_text(encoding="utf-8")

replacements = [
    (
        "    garbage_tspin_recovery: float = 1.000000\n    new_holes: float = -1.200000\n",
        "    garbage_tspin_recovery: float = 1.000000\n    garbage_slot_supply_floor: float = 1.000000\n    new_holes: float = -1.200000\n",
    ),
    (
        "    garbage_tspin_recovery: float = 0.0\n\n    def to_dict(self) -> dict[str, object]:\n",
        "    garbage_tspin_recovery: float = 0.0\n    garbage_t_spin_slot_floor: int = 0\n\n    def to_dict(self) -> dict[str, object]:\n",
    ),
    (
        '            "garbage_tspin_recovery": self.garbage_tspin_recovery,\n            "new_holes": self.new_holes,\n',
        '            "garbage_tspin_recovery": self.garbage_tspin_recovery,\n            "garbage_t_spin_slot_floor": self.garbage_t_spin_slot_floor,\n            "new_holes": self.new_holes,\n',
    ),
    (
        "    hold_t_supply_balance = _hold_t_supply_balance_score(\n        game,\n        features.board.t_spin_slots,\n    )\n    return (\n",
        "    hold_t_supply_balance = _hold_t_supply_balance_score(\n        game,\n        features.board.t_spin_slots,\n    )\n    garbage_slot_supply_floor = _garbage_slot_supply_floor_score(\n        game,\n        features.garbage_t_spin_slot_floor,\n    )\n    return (\n",
    ),
    (
        "        + hold_t_supply_balance * weights.hold_t_supply_balance\n    )\n\n\ndef _center_garbage_resilience_score",
        "        + hold_t_supply_balance * weights.hold_t_supply_balance\n        + garbage_slot_supply_floor * weights.garbage_slot_supply_floor\n    )\n\n\ndef _garbage_t_spin_slot_floor(board: list[list[str | None]], width: int) -> int:\n    \"\"\"Return the worst T-spin setup count after a four-line center garbage spike.\"\"\"\n\n    holes = tuple(sorted({min(width - 1, max(0, hole)) for hole in (3, 4, 5, 6)}))\n    floors: list[int] = []\n    for hole in holes:\n        stressed = [row.copy() for row in board]\n        for _ in range(4):\n            stressed.pop(0)\n            garbage: list[str | None] = [\"G\"] * width\n            garbage[hole] = None\n            stressed.append(garbage)\n        floors.append(extract_board_features(stressed).t_spin_slots)\n    return min(floors, default=0)\n\n\ndef _garbage_slot_supply_floor_score(game: Game, slot_floor: int) -> float:\n    supply = _t_supply_count(game) + int(game.hold_piece == \"T\")\n    matched = min(slot_floor, supply)\n    surplus = max(0, slot_floor - supply)\n    return 0.30 * matched - 0.22 * surplus\n\n\ndef _center_garbage_resilience_score",
    ),
    (
        "    garbage_tspin_recovery = _garbage_tspin_recovery_score(board, game.width)\n    return PlacementFeatures(\n",
        "    garbage_tspin_recovery = _garbage_tspin_recovery_score(board, game.width)\n    garbage_t_spin_slot_floor = _garbage_t_spin_slot_floor(board, game.width)\n    return PlacementFeatures(\n",
    ),
    (
        "        garbage_tspin_recovery=garbage_tspin_recovery,\n    )\n",
        "        garbage_tspin_recovery=garbage_tspin_recovery,\n        garbage_t_spin_slot_floor=garbage_t_spin_slot_floor,\n    )\n",
    ),
]

for old, new in replacements:
    if old not in text:
        raise RuntimeError(f"Expected heuristic anchor not found: {old[:80]!r}")
    text = text.replace(old, new, 1)

HEURISTIC.write_text(text, encoding="utf-8")

TEST.write_text(
    '''from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import (
    DEFAULT_WEIGHTS,
    MODEL_FORMAT,
    HeuristicWeights,
    evaluate_placement,
    load_weights,
    save_weights,
)
from minoflux_ai.search import clone_game
from minoflux_engine import Game


def _t_supply(game: Game) -> int:
    supply = int(game.current == "T") + int(game.hold_piece == "T")
    for index, piece in enumerate(game.queue):
        if index >= 6:
            break
        supply += int(piece == "T")
    return supply


def test_garbage_slot_floor_matches_four_center_probes() -> None:
    game = Game(891_001)
    placement = game.legal_placements()[0]
    evaluation = evaluate_placement(game, placement)

    slot_counts = []
    for hole in (3, 4, 5, 6):
        probe = clone_game(game)
        probe.place(placement)
        probe.add_garbage(4, hole)
        slot_counts.append(extract_board_features(probe.board).t_spin_slots)

    assert evaluation.features.garbage_t_spin_slot_floor == min(slot_counts)


def test_garbage_slot_supply_floor_score_matches_formula() -> None:
    game = Game(891_002)
    placement = game.legal_placements()[0]
    disabled = replace(DEFAULT_WEIGHTS, garbage_slot_supply_floor=0.0)

    without = evaluate_placement(game, placement, disabled)
    with_feature = evaluate_placement(game, placement, DEFAULT_WEIGHTS)
    floor = with_feature.features.garbage_t_spin_slot_floor
    supply = _t_supply(game)
    expected = 0.30 * min(floor, supply) - 0.22 * max(0, floor - supply)

    assert with_feature.score - without.score == pytest.approx(expected)


def test_garbage_slot_supply_floor_does_not_mutate_game() -> None:
    game = Game(891_003)
    before = game.snapshot(queue_size=7)
    placement = game.legal_placements()[0]

    evaluate_placement(game, placement)

    assert game.snapshot(queue_size=7) == before


def test_old_model_defaults_garbage_slot_supply_floor(tmp_path: Path) -> None:
    path = tmp_path / "old-model.json"
    path.write_text(
        json.dumps({"format": MODEL_FORMAT, "weights": {"attack": 0.91}}),
        encoding="utf-8",
    )

    weights = load_weights(path)

    assert weights.attack == 0.91
    assert weights.garbage_slot_supply_floor == DEFAULT_WEIGHTS.garbage_slot_supply_floor


def test_garbage_slot_supply_floor_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "model.json"
    weights = HeuristicWeights(garbage_slot_supply_floor=0.42)

    save_weights(path, weights)

    assert load_weights(path) == weights
''',
    encoding="utf-8",
)

# The tournament is complete; do not leave evaluation-only machinery in the branch.
if WORKFLOW.exists():
    WORKFLOW.unlink()
if SELF.exists():
    SELF.unlink()
