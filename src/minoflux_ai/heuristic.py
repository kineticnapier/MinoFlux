from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from heapq import nlargest
import json
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
    t_spin_slot_low_clean: float = 0.900000
    t_spin_slot_supply_match: float = 0.550000
    t_spin_slot_queue_match: float = 1.000000
    t_spin_slot_urgency_clean: float = 1.000000
    t_arrival_conversion: float = 1.000000
    hold_t_supply_balance: float = 1.000000
    center_garbage_resilience: float = 1.000000
    garbage_tspin_recovery: float = 1.000000
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
    center_garbage_resilience: float = 0.0
    garbage_tspin_recovery: float = 0.0

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = self.board.to_dict()
        value.update({
            "t_spin_slot_delta": self.t_spin_slot_delta,
            "center_garbage_resilience": self.center_garbage_resilience,
            "garbage_tspin_recovery": self.garbage_tspin_recovery,
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


def score_features(features: PlacementFeatures, weights: HeuristicWeights = DEFAULT_WEIGHTS) -> float:
    board = features.board
    t_spin_slot_height_quality = board.t_spin_slots / (1.0 + board.max_height / 6.0)
    t_spin_slot_low_clean = board.t_spin_slots / (
        1.0 + board.holes + board.max_height / 6.0
    )
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
        + t_spin_slot_low_clean * weights.t_spin_slot_low_clean
        + features.center_garbage_resilience * weights.center_garbage_resilience
        + features.garbage_tspin_recovery * weights.garbage_tspin_recovery
        + features.new_holes * weights.new_holes
        + features.lines * weights.lines
        + features.attack * weights.attack
        + features.spin_lines * weights.spin_lines
        + int(features.perfect_clear) * weights.perfect_clear
        + int(features.game_over) * weights.game_over
    )


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 5:
            break
    return 7


def _t_supply_count(game: Game) -> int:
    count = 1 if game.current == "T" else 0
    for index, piece in enumerate(game.queue):
        if index >= 6:
            break
        if piece == "T":
            count += 1
    return count


def _t_spin_slot_supply_match(t_spin_slots: int, next_t_distance: int) -> float:
    availability = max(0.0, (6.0 - next_t_distance) / 6.0)
    excess_slots = max(0, t_spin_slots - 1)
    return availability * min(1, t_spin_slots) - 0.35 * excess_slots * (1.0 - availability)


def _t_spin_slot_queue_match_score(game: Game, t_spin_slots: int) -> float:
    held_t = 1 if game.hold_piece == "T" else 0
    supply = _t_supply_count(game) + held_t
    matched = min(t_spin_slots, supply)
    return 0.24 * matched - 0.22 * abs(t_spin_slots - supply)


def _t_spin_slot_urgency_clean_score(game: Game, board: BoardFeatures) -> float:
    next_t_distance = _next_t_distance(game)
    urgency = max(0.0, (7.0 - next_t_distance) / 7.0)
    return 0.34 * urgency * min(2, board.t_spin_slots) / (1.0 + board.holes)


def _t_arrival_conversion_score(game: Game, features: PlacementFeatures) -> float:
    if game.current != "T":
        return 0.0
    if features.spin_lines > 0:
        return 1.20 * features.spin_lines + 0.20 * features.attack
    slot_loss = max(0, -features.t_spin_slot_delta)
    return -0.80 * slot_loss


def _hold_t_supply_balance_score(game: Game, t_spin_slots: int) -> float:
    if not game.hold_used:
        return 0.0
    held_t = 1 if game.hold_piece == "T" else 0
    unmet_slots = max(0, t_spin_slots - (_t_supply_count(game) + held_t))
    return -0.28 * unmet_slots + 0.12 * held_t * min(1, t_spin_slots)


def _context_score(game: Game, features: PlacementFeatures, weights: HeuristicWeights) -> float:
    supply_match = _t_spin_slot_supply_match(
        features.board.t_spin_slots,
        _next_t_distance(game),
    )
    queue_match = _t_spin_slot_queue_match_score(
        game,
        features.board.t_spin_slots,
    )
    urgency_clean = _t_spin_slot_urgency_clean_score(game, features.board)
    arrival_conversion = _t_arrival_conversion_score(game, features)
    hold_t_supply_balance = _hold_t_supply_balance_score(
        game,
        features.board.t_spin_slots,
    )
    return (
        supply_match * weights.t_spin_slot_supply_match
        + queue_match * weights.t_spin_slot_queue_match
        + urgency_clean * weights.t_spin_slot_urgency_clean
        + arrival_conversion * weights.t_arrival_conversion
        + hold_t_supply_balance * weights.hold_t_supply_balance
    )


def _center_garbage_resilience_score(board: list[list[str | None]], width: int) -> float:
    """Score the post-placement board after a representative four-line center garbage spike."""

    stressed = [row.copy() for row in board]
    hole = min(width - 1, width // 2 - 1)
    for _ in range(4):
        stressed.pop(0)
        garbage: list[str | None] = ["G"] * width
        garbage[hole] = None
        stressed.append(garbage)
    features = extract_board_features(stressed)
    return -(
        0.075 * features.holes
        + 0.018 * features.hole_depth
        + 0.025 * features.max_height
    )


def _garbage_tspin_recovery_score(board: list[list[str | None]], width: int) -> float:
    """Reward boards that keep clean T-spin recovery options across likely garbage-hole positions."""

    holes = (0, min(width - 1, 3), min(width - 1, 6), width - 1)
    slot_quality_total = 0.0
    board_quality_total = 0.0
    for hole in holes:
        stressed = [row.copy() for row in board]
        for _ in range(4):
            stressed.pop(0)
            garbage: list[str | None] = ["G"] * width
            garbage[hole] = None
            stressed.append(garbage)
        features = extract_board_features(stressed)
        slot_quality_total += features.t_spin_slots / (
            1.0 + features.holes + features.max_height / 6.0
        )
        board_quality_total -= (
            0.080 * features.holes
            + 0.020 * features.hole_depth
            + 0.030 * features.max_height
        )
    count = float(len(holes))
    return 1.10 * slot_quality_total / count + 0.20 * board_quality_total / count


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
        board = [[None] * game.width for _ in full_rows] + [
            row for index, row in enumerate(board) if index not in full_set
        ]

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
    center_garbage_resilience = _center_garbage_resilience_score(board, game.width)
    garbage_tspin_recovery = _garbage_tspin_recovery_score(board, game.width)
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
        center_garbage_resilience=center_garbage_resilience,
        garbage_tspin_recovery=garbage_tspin_recovery,
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
        score=score_features(placement_features, weights) + _context_score(game, placement_features, weights),
        features=placement_features,
    )


def _placement_key(item: PlacementEvaluation) -> tuple[float, int, int, int, int, int, int, int]:
    return (
        item.score,
        item.features.attack,
        item.features.spin_lines,
        item.features.lines,
        -item.features.board.holes,
        -item.features.board.max_height,
        -item.placement.rotation,
        -item.placement.x,
    )


def rank_placements(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    *,
    placements: Iterable[Placement] | None = None,
    limit: int | None = None,
) -> tuple[PlacementEvaluation, ...]:
    before = extract_board_features(game.board)
    source = game.legal_placements() if placements is None else placements
    evaluated = [
        PlacementEvaluation(
            placement=placement,
            score=score_features(features, weights) + _context_score(game, features, weights),
            features=features,
        )
        for placement in source
        for features in (_placement_features_fast(game, placement, before),)
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
