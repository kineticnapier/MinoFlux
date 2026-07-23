from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Mapping

from minoflux_engine import Game, Placement

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


def evaluate_placement(
    game: Game,
    placement: Placement,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> PlacementEvaluation:
    before = extract_board_features(game.board)
    simulation = deepcopy(game)
    result = simulation.place(placement)
    after = extract_board_features(simulation.board)
    placement_features = PlacementFeatures(
        board=after,
        new_holes=max(0, after.holes - before.holes),
        lines=result.lines,
        attack=result.attack,
        spin_lines=result.lines if result.spin is not None else 0,
        perfect_clear=result.perfect_clear,
        game_over=result.game_over,
    )
    return PlacementEvaluation(
        placement=placement,
        score=score_features(placement_features, weights),
        features=placement_features,
    )


def rank_placements(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> tuple[PlacementEvaluation, ...]:
    evaluated = [evaluate_placement(game, placement, weights) for placement in game.legal_placements()]
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
