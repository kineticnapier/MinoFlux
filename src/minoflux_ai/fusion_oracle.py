from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from minoflux_engine import Game

from .search import SearchAction


_FUSION_PIECES = ("I", "O", "T", "L", "J", "S", "Z")
_FUSION_BASE_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "I": ((-1, 0), (1, 0), (2, 0)),
    "O": ((1, 0), (0, 1), (1, 1)),
    "T": ((-1, 0), (1, 0), (0, 1)),
    "L": ((-1, 0), (1, 0), (1, 1)),
    "J": ((-1, 0), (1, 0), (-1, 1)),
    "S": ((-1, 0), (0, 1), (1, 1)),
    "Z": ((-1, 1), (0, 1), (1, 0)),
}


@dataclass(frozen=True, slots=True)
class FusionOracleConfig:
    queue_length: int = 18
    max_candidates: int = 24
    workers: int = 1
    allow_180: bool = True
    reachability_node_limit: int = 8_000

    def normalized(self) -> "FusionOracleConfig":
        return FusionOracleConfig(
            queue_length=max(1, int(self.queue_length)),
            max_candidates=max(0, int(self.max_candidates)),
            workers=max(1, int(self.workers)),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
        )


@dataclass(frozen=True, slots=True)
class FusionOracleLabel:
    best_move_raw: int
    best_value: float
    best_hold_used: bool | None = None
    best_cells: frozenset[tuple[int, int]] | None = None


class FusionOracleMatchError(ValueError):
    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(self.reason)


def game_to_fusion_request(
    game: Game,
    *,
    request_id: str,
    queue_length: int,
) -> dict[str, object]:
    count = max(1, int(queue_length))
    if len(game.queue) < count:
        raise ValueError("insufficient-queue")

    rows: list[int] = []
    for row in reversed(game.board):
        mask = 0
        for x, cell in enumerate(row):
            if cell is not None:
                mask |= 1 << x
        rows.append(mask)
    while rows and rows[-1] == 0:
        rows.pop()

    return {
        "schema_version": "phase1-v1",
        "replay_id": str(request_id),
        "round_id": 0,
        "player_id": 0,
        "frame_id": int(game.pieces_placed),
        "group_id": str(request_id),
        "player_board_rows": rows,
        "opponent_board_rows": [],
        "current_piece": game.current.lower(),
        "hold_piece": None if game.hold_piece is None else game.hold_piece.lower(),
        "queue": [piece.lower() for piece in list(game.queue)[:count]],
        "combo": max(0, int(game.combo) + 1),
        "b2b": 0 if not game.back_to_back else int(game.b2b_chain) + 1,
        "lines": int(game.lines),
        "pending_garbage": 0,
        "bag_number": 0,
    }


def parse_fusion_label(value: Mapping[str, object]) -> FusionOracleLabel:
    if bool(value.get("skipped", False)):
        raise ValueError("oracle-output-skipped")
    if "best_move_raw" not in value or "best_value" not in value:
        raise ValueError("oracle-output-invalid")

    raw = int(value["best_move_raw"])
    if raw < 0 or raw > 0xFFFF:
        raise ValueError("oracle-output-invalid")

    hold_raw = value.get("bestHoldUsed")
    cells_raw = value.get("bestCells")
    if (hold_raw is None) != (cells_raw is None):
        raise ValueError("oracle-output-invalid")

    best_hold_used: bool | None = None
    best_cells: frozenset[tuple[int, int]] | None = None
    if hold_raw is not None and cells_raw is not None:
        if not isinstance(hold_raw, bool):
            raise ValueError("oracle-output-invalid")
        if not isinstance(cells_raw, Sequence) or isinstance(cells_raw, (str, bytes)):
            raise ValueError("oracle-output-invalid")
        parsed: list[tuple[int, int]] = []
        for item in cells_raw:
            if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) != 2:
                raise ValueError("oracle-output-invalid")
            parsed.append((int(item[0]), int(item[1])))
        if len(parsed) != 4 or len(set(parsed)) != 4:
            raise ValueError("oracle-output-invalid")
        best_hold_used = hold_raw
        best_cells = frozenset(parsed)

    return FusionOracleLabel(
        best_move_raw=raw,
        best_value=float(value["best_value"]),
        best_hold_used=best_hold_used,
        best_cells=best_cells,
    )


def _fusion_piece_from_raw(raw: int) -> str:
    piece_raw = (int(raw) >> 10) & 0x7
    if piece_raw == 7:
        return "T"
    return _FUSION_PIECES[piece_raw]


def _rotate_offset(rotation: int, dx: int, dy: int) -> tuple[int, int]:
    rotation %= 4
    if rotation == 0:
        return dx, dy
    if rotation == 1:
        return dy, -dx
    if rotation == 2:
        return -dx, -dy
    return -dy, dx


def _fusion_cells_from_raw(raw: int) -> frozenset[tuple[int, int]]:
    raw = int(raw)
    piece = _fusion_piece_from_raw(raw)
    rotation = (raw >> 13) & 0x3
    x = (raw >> 6) & 0xF
    y = raw & 0x3F
    cells = {(x, y)}
    for dx, dy in _FUSION_BASE_OFFSETS[piece]:
        ox, oy = _rotate_offset(rotation, dx, dy)
        cells.add((x + ox, y + oy))
    return frozenset(cells)


def _action_cells_bottom_up(game: Game, action: SearchAction) -> frozenset[tuple[int, int]]:
    return frozenset(
        (int(x), (int(game.height) - 1) - int(y))
        for x, y in action.placement.cells
    )


def match_fusion_action(
    game: Game,
    actions: Sequence[SearchAction],
    label: FusionOracleLabel,
) -> SearchAction:
    piece = _fusion_piece_from_raw(label.best_move_raw)
    cells = label.best_cells or _fusion_cells_from_raw(label.best_move_raw)

    matches = [
        action
        for action in actions
        if action.placement.piece == piece
        and _action_cells_bottom_up(game, action) == cells
        and (
            label.best_hold_used is None
            or bool(action.use_hold) == label.best_hold_used
        )
    ]
    if not matches:
        raise FusionOracleMatchError("oracle-action-unmatched")
    if len(matches) != 1:
        raise FusionOracleMatchError("oracle-action-ambiguous")
    return matches[0]
