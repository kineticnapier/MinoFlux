from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Mapping, Sequence

from minoflux_engine import Game, Placement
from minoflux_engine.pieces import LINE_ATTACK, SHAPES

from .features import BoardFeatures, extract_board_features

MODEL_FORMAT = "minoflux_heuristic_v1"


@dataclass(frozen=True, slots=True)
class HeuristicWeights:
    aggregate_height: float = -0.510066
    max_height: float = -0.080000
    holes: float = -0.800000
    hole_depth: float = -0.120000
    bumpiness: float = -0.184483
    wells: float = -0.060000
    new_holes: float = -1.200000
    lines: float = 0.760666
    attack: float = 0.850000
    spin_lines: float = 1.250000
    perfect_clear: float = 8.000000
    game_over: float = -1_000_000.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "HeuristicWeights":
        names = {item.name for item in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"Unknown heuristic weights: {sorted(unknown)}")
        defaults = cls().to_dict()
        defaults.update({key: float(value) for key, value in values.items()})
        return cls(**defaults)


DEFAULT_WEIGHTS = HeuristicWeights()


@dataclass(frozen=True, slots=True)
class PlacementFeatures:
    board: BoardFeatures
    new_holes: int
    lines: int
    attack: int
    spin_lines: int
    perfect_clear: bool
    game_over: bool

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = self.board.to_dict()
        value.update({
            "new_holes": self.new_holes,
            "lines": self.lines,
            "attack": self.attack,
            "spin_lines": self.spin_lines,
            "perfect_clear": self.perfect_clear,
            "game_over": self.game_over,
        })
        return value


@dataclass(frozen=True, slots=True)
class PlacementEvaluation:
    placement: Placement
    score: float
    features: PlacementFeatures


def score_features(features: PlacementFeatures, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    board = features.board
    return (
        board.aggregate_height * weights.aggregate_height
        + board.max_height * weights.max_height
        + board.holes * weights.holes
        + board.hole_depth * weights.hole_depth
        + board.bumpiness * weights.bumpiness
        + board.wells * weights.wells
        + features.new_holes * weights.new_holes
        + features.lines * weights.lines
        + features.attack * weights.attack
        + features.spin_lines * weights.spin_lines
        + int(features.perfect_clear) * weights.perfect_clear
        + int(features.game_over) * weights.game_over
    )


def _collides(
    board: Sequence[Sequence[object | None]],
    piece: str,
    x: int,
    y: int,
    rotation: int,
) -> bool:
    height = len(board)
    width = len(board[0]) if board else 0
    for dx, dy in SHAPES[piece][rotation % 4]:
        cell_x, cell_y = x + dx, y + dy
        if cell_x < 0 or cell_x >= width or cell_y >= height:
            return True
        if cell_y >= 0 and board[cell_y][cell_x] is not None:
            return True
    return False


def _detect_direct_placement_spin(game: Game, placement: Placement) -> str | None:
    if placement.rotation == game.rotation or placement.piece == "O":
        return None
    board = game.board
    if placement.piece == "T":
        pivot_x, pivot_y = placement.x + 1, placement.y + 1
        occupied = 0
        for cell_x, cell_y in (
            (pivot_x - 1, pivot_y - 1),
            (pivot_x + 1, pivot_y - 1),
            (pivot_x - 1, pivot_y + 1),
            (pivot_x + 1, pivot_y + 1),
        ):
            if (
                cell_x < 0
                or cell_x >= game.width
                or cell_y < 0
                or cell_y >= game.height
                or board[cell_y][cell_x] is not None
            ):
                occupied += 1
        if occupied >= 3:
            return "T"
    blocked = (
        _collides(board, placement.piece, placement.x - 1, placement.y, placement.rotation)
        and _collides(board, placement.piece, placement.x + 1, placement.y, placement.rotation)
        and _collides(board, placement.piece, placement.x, placement.y + 1, placement.rotation)
    )
    return placement.piece if blocked else None


def _placement_features_fast(
    game: Game,
    placement: Placement,
    before: BoardFeatures,
) -> PlacementFeatures:
    # Only the 24x10 board is copied. Bag, queue, RNG and the rest of Game are not.
    board = [row.copy() for row in game.board]
    spin = _detect_direct_placement_spin(game, placement)
    topped_out = False
    for cell_x, cell_y in placement.cells:
        if cell_y < 0:
            topped_out = True
        else:
            board[cell_y][cell_x] = placement.piece

    full_rows = [index for index, row in enumerate(board) if all(cell is not None for cell in row)]
    lines = len(full_rows)
    if full_rows:
        full_set = set(full_rows)
        board = [[None] * game.width for _ in full_rows] + [
            row for index, row in enumerate(board) if index not in full_set
        ]

    perfect_clear = all(cell is None for row in board for cell in row)
    difficult = lines == 4 or (spin is not None and lines > 0)
    attack = LINE_ATTACK.get(lines, 0)
    if spin is not None and lines:
        attack += max(1, lines)
    if difficult and game.back_to_back:
        attack += 1
    combo = game.combo + 1 if lines else -1
    if lines and combo > 0:
        attack += min(4, combo // 2 + 1)
    if perfect_clear and lines:
        attack += 10

    hidden_occupied = any(
        cell is not None
        for row in board[: game.hidden_rows]
        for cell in row
    )
    after = extract_board_features(board)
    return PlacementFeatures(
        board=after,
        new_holes=max(0, after.holes - before.holes),
        lines=lines,
        attack=attack,
        spin_lines=lines if spin is not None else 0,
        perfect_clear=perfect_clear,
        game_over=topped_out or hidden_occupied,
    )


def evaluate_placement(
    game: Game,
    placement: Placement,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> PlacementEvaluation:
    before = extract_board_features(game.board)
    placement_features = _placement_features_fast(game, placement, before)
    return PlacementEvaluation(
        placement=placement,
        score=score_features(placement_features, weights),
        features=placement_features,
    )


def rank_placements(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> tuple[PlacementEvaluation, ...]:
    before = extract_board_features(game.board)
    evaluated = [
        PlacementEvaluation(
            placement=placement,
            score=score_features(features, weights),
            features=features,
        )
        for placement in game.legal_placements()
        for features in (_placement_features_fast(game, placement, before),)
    ]
    evaluated.sort(
        key=lambda item: (
            item.score,
            item.features.attack,
            item.features.lines,
            -item.features.board.holes,
            -item.features.board.max_height,
            -item.placement.rotation,
            -item.placement.x,
        ),
        reverse=True,
    )
    return tuple(evaluated)


def choose_placement(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> PlacementEvaluation | None:
    ranked = rank_placements(game, weights)
    return ranked[0] if ranked else None


def save_weights(path: str | Path, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"format": MODEL_FORMAT, "weights": weights.to_dict()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target


def load_weights(path: str | Path) -> HeuristicWeights:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"Unsupported heuristic model format: {payload.get('format')!r}")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Model weights must be an object")
    return HeuristicWeights.from_mapping(weights)
