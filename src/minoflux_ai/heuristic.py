from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from heapq import nlargest
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from minoflux_engine import Game, Placement
from minoflux_engine.b2b import resolve_b2b_charging
from minoflux_engine.spin import base_attack, classify_t_spin, is_difficult_clear, t_spin_event

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
    t_spin_slots: float = 1.200000
    t_spin_slot_density: float = 0.280000
    t_spin_slot_delta: float = 0.250000
    t_spin_slot_height_quality: float = 0.700000
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
    spin: str | None = None
    t_spin_slot_delta: int = 0

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = self.board.to_dict()
        value.update({
            "t_spin_slot_delta": self.t_spin_slot_delta,
            "new_holes": self.new_holes,
            "lines": self.lines,
            "attack": self.attack,
            "spin_lines": self.spin_lines,
            "perfect_clear": self.perfect_clear,
            "game_over": self.game_over,
            "spin": self.spin,
        })
        return value


@dataclass(frozen=True, slots=True)
class PlacementEvaluation:
    placement: Placement
    score: float
    features: PlacementFeatures


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    if game.hold_piece == "T":
        return 1
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
    return 8


def _tournament_bonus(game: Game, features: PlacementFeatures) -> float:
    candidate = os.environ.get("MINOFLUX_TOURNAMENT_CANDIDATE", "baseline")
    if candidate == "baseline":
        return 0.0
    b = features.board
    tdist = _next_t_distance(game)
    difficult = features.spin_lines > 0 or features.lines == 4
    breaks_b2b = game.back_to_back and features.lines > 0 and not difficult
    keeps_or_starts_b2b = difficult and features.lines > 0
    near_t = max(0.0, 4.0 - float(tdist)) / 4.0
    far_t = min(1.0, float(tdist) / 5.0)
    slot_gain = max(0, features.t_spin_slot_delta)
    slot_loss = max(0, -features.t_spin_slot_delta)
    danger = max(0.0, b.max_height - 8.0)
    garbage_present = any(cell == "G" for row in game.board for cell in row)
    terms = {
        "t_arrival_slot_preserve": 0.75 * near_t * (b.t_spin_slots - 1.5 * slot_loss),
        "t_arrival_slot_create": 0.70 * near_t * slot_gain,
        "t_far_clean_preserve": 0.55 * far_t * b.t_spin_slots / (1.0 + b.holes + features.new_holes),
        "t_hold_slot_reserve": 0.75 * int(game.hold_piece == "T") * b.t_spin_slots / (1.0 + b.max_height / 10.0),
        "t_current_conversion": 0.70 * int(game.current == "T") * (features.spin_lines + 0.4 * features.attack),
        "b2b_break_cost": -1.15 * int(breaks_b2b) * (1.0 + 0.15 * game.b2b_chain),
        "b2b_start_value": 0.75 * int((not game.back_to_back) and keeps_or_starts_b2b),
        "b2b_chain_value": 0.45 * int(game.back_to_back and keeps_or_starts_b2b) * (1.0 + min(5, game.b2b_chain) / 5.0),
        "surge_preservation": 0.28 * int(keeps_or_starts_b2b) * game.surge_charge - 0.32 * int(breaks_b2b) * game.surge_charge,
        "danger_spin_escape": 0.11 * danger * (features.attack + features.spin_lines),
        "danger_b2b_escape": 0.09 * danger * int(keeps_or_starts_b2b) * (1.0 + features.attack),
        "garbage_downstack_attack": 0.35 * int(garbage_present) * features.lines * (1.0 + 0.35 * features.attack) / (1.0 + b.holes),
        "garbage_hole_pressure": -0.20 * int(garbage_present) * (b.hole_depth + 2.0 * features.new_holes),
        "combo_b2b_tradeoff": 0.28 * max(0, game.combo + 1) * features.lines + 0.60 * int(keeps_or_starts_b2b) - 0.75 * int(breaks_b2b),
    }
    return terms.get(candidate, 0.0)


def score_features(features: PlacementFeatures, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    board = features.board
    t_spin_slot_height_quality = board.t_spin_slots / (1.0 + board.max_height / 6.0)
    return (
        board.aggregate_height * weights.aggregate_height
        + board.max_height * weights.max_height
        + board.holes * weights.holes
        + board.hole_depth * weights.hole_depth
        + board.bumpiness * weights.bumpiness
        + board.wells * weights.wells
        + board.t_spin_slots * weights.t_spin_slots
        + board.t_spin_slot_density * weights.t_spin_slot_density
        + features.t_spin_slot_delta * weights.t_spin_slot_delta
        + t_spin_slot_height_quality * weights.t_spin_slot_height_quality
        + features.new_holes * weights.new_holes
        + features.lines * weights.lines
        + features.attack * weights.attack
        + features.spin_lines * weights.spin_lines
        + int(features.perfect_clear) * weights.perfect_clear
        + int(features.game_over) * weights.game_over
    )


def _placement_features_fast(game: Game, placement: Placement, before: BoardFeatures) -> PlacementFeatures:
    spin_kind = classify_t_spin(
        game.board,
        piece=placement.piece,
        x=placement.x,
        y=placement.y,
        rotation=placement.rotation,
        last_move_was_rotation=placement.last_move_was_rotation,
        rotation_kick_index=placement.rotation_kick_index,
    )
    board = list(game.board)
    copied_rows: set[int] = set()
    topped_out = False
    for cell_x, cell_y in placement.cells:
        if cell_y < 0:
            topped_out = True
        else:
            if cell_y not in copied_rows:
                board[cell_y] = game.board[cell_y].copy()
                copied_rows.add(cell_y)
            board[cell_y][cell_x] = placement.piece
    full_rows = [index for index, row in enumerate(board) if all(cell is not None for cell in row)]
    lines = len(full_rows)
    if full_rows:
        full_set = set(full_rows)
        board = [[None] * game.width for _ in full_rows] + [row for index, row in enumerate(board) if index not in full_set]
    spin = t_spin_event(spin_kind, lines)
    perfect_clear = all(cell is None for row in board for cell in row)
    difficult = is_difficult_clear(lines, spin)
    b2b = resolve_b2b_charging(
        active=game.back_to_back,
        chain=game.b2b_chain,
        difficult=difficult,
        lines=lines,
        perfect_clear=perfect_clear and lines > 0,
    )
    attack = base_attack(lines, spin) + b2b.attack_bonus + b2b.released
    combo = game.combo + 1 if lines else -1
    if lines and combo > 0:
        attack += min(4, combo // 2 + 1)
    if perfect_clear and lines:
        attack += 10
    hidden_occupied = any(cell is not None for row in board[: game.hidden_rows] for cell in row)
    after = extract_board_features(board)
    return PlacementFeatures(
        board=after,
        new_holes=max(0, after.holes - before.holes),
        lines=lines,
        attack=attack,
        spin_lines=lines if spin is not None else 0,
        perfect_clear=perfect_clear,
        game_over=topped_out or hidden_occupied,
        spin=spin,
        t_spin_slot_delta=after.t_spin_slots - before.t_spin_slots,
    )


def evaluate_placement(game: Game, placement: Placement, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> PlacementEvaluation:
    before = extract_board_features(game.board)
    features = _placement_features_fast(game, placement, before)
    return PlacementEvaluation(placement=placement, score=score_features(features, weights) + _tournament_bonus(game, features), features=features)


def _placement_key(item: PlacementEvaluation) -> tuple[float, int, int, int, int, int, int, int]:
    return (item.score, item.features.attack, item.features.spin_lines, item.features.lines, -item.features.board.holes, -item.features.board.max_height, -item.placement.rotation, -item.placement.x)


def rank_placements(game: Game, weights: HeuristicWeights = DEFAULT_WEIGHTS, *, placements: Iterable[Placement] | None = None, limit: int | None = None) -> tuple[PlacementEvaluation, ...]:
    before = extract_board_features(game.board)
    source = game.legal_placements() if placements is None else placements
    evaluated = [
        PlacementEvaluation(placement=p, score=score_features(f, weights) + _tournament_bonus(game, f), features=f)
        for p in source
        for f in (_placement_features_fast(game, p, before),)
    ]
    if limit is not None:
        count = max(0, int(limit))
        if count == 0:
            return ()
        if count < len(evaluated):
            return tuple(nlargest(count, evaluated, key=_placement_key))
    evaluated.sort(key=_placement_key, reverse=True)
    return tuple(evaluated)


def choose_placement(game: Game, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> PlacementEvaluation | None:
    ranked = rank_placements(game, weights, limit=1)
    return ranked[0] if ranked else None


def save_weights(path: str | Path, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"format": MODEL_FORMAT, "weights": weights.to_dict()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def load_weights(path: str | Path) -> HeuristicWeights:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != MODEL_FORMAT:
        raise ValueError(f"Unsupported heuristic model format: {payload.get('format')!r}")
    weights = payload.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("Model weights must be an object")
    return HeuristicWeights.from_mapping(weights)
