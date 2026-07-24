from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import exp
from pathlib import Path
import random
from typing import Mapping, Sequence

from minoflux_engine import Game

from .heuristic import HeuristicWeights, PlacementFeatures, evaluate_placement
from .reachability import reachable_placements

IMITATION_FORMAT = "minoflux_imitation_result_v1"
FEATURE_NAMES = (
    "aggregate_height",
    "max_height",
    "holes",
    "hole_depth",
    "bumpiness",
    "wells",
    "new_holes",
    "lines",
    "attack",
    "spin_lines",
    "perfect_clear",
)
FEATURE_SCALES = {
    "aggregate_height": 100.0,
    "max_height": 24.0,
    "holes": 20.0,
    "hole_depth": 100.0,
    "bumpiness": 60.0,
    "wells": 100.0,
    "new_holes": 8.0,
    "lines": 4.0,
    "attack": 20.0,
    "spin_lines": 3.0,
    "perfect_clear": 1.0,
}


@dataclass(frozen=True, slots=True)
class ImitationConfig:
    epochs: int = 4
    learning_rate: float = 0.08
    l2: float = 0.0001
    random_seed: int = 12345
    max_samples: int = 5_000
    allow_180: bool = True
    reachability_node_limit: int = 8_000
    include_ambiguous: bool = True

    def normalized(self) -> "ImitationConfig":
        return ImitationConfig(
            epochs=max(1, int(self.epochs)),
            learning_rate=max(1e-6, float(self.learning_rate)),
            l2=max(0.0, float(self.l2)),
            random_seed=int(self.random_seed),
            max_samples=max(1, int(self.max_samples)),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
            include_ambiguous=bool(self.include_ambiguous),
        )


@dataclass(frozen=True, slots=True)
class CandidateVector:
    values: tuple[float, ...]
    game_over: bool


@dataclass(frozen=True, slots=True)
class ImitationExample:
    key: str
    split: str
    candidates: tuple[CandidateVector, ...]
    expert_index: int


@dataclass(frozen=True, slots=True)
class RankingMetrics:
    samples: int
    top1: int
    top3: int
    mean_rank: float

    @property
    def top1_accuracy(self) -> float:
        return self.top1 / self.samples if self.samples else 0.0

    @property
    def top3_accuracy(self) -> float:
        return self.top3 / self.samples if self.samples else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "top1": self.top1,
            "top3": self.top3,
            "top1Accuracy": self.top1_accuracy,
            "top3Accuracy": self.top3_accuracy,
            "meanRank": self.mean_rank,
        }


@dataclass(frozen=True, slots=True)
class ImitationResult:
    config: ImitationConfig
    initial_weights: HeuristicWeights
    learned_weights: HeuristicWeights
    prepared_examples: int
    skipped: dict[str, int]
    epoch_losses: tuple[float, ...]
    train: RankingMetrics
    validation: RankingMetrics
    test: RankingMetrics

    def to_dict(self) -> dict[str, object]:
        return {
            "format": IMITATION_FORMAT,
            "config": asdict(self.config),
            "initialWeights": self.initial_weights.to_dict(),
            "learnedWeights": self.learned_weights.to_dict(),
            "preparedExamples": self.prepared_examples,
            "skipped": dict(sorted(self.skipped.items())),
            "epochLosses": list(self.epoch_losses),
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "test": self.test.to_dict(),
        }


def _load_jsonl(path: str | Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record {line_number} must be an object")
            records.append(value)
    return tuple(records)


def _key(value: Mapping[str, object]) -> str:
    return f"{value.get('group_id', value.get('groupId', ''))}|{int(value.get('sequence', 0))}"


def _board(value: object) -> list[list[str | None]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("board_before must be an array")
    rows: list[list[str | None]] = []
    for raw_row in value:
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
            raise ValueError("board row must be an array")
        row = [None if cell is None else str(cell).lower() for cell in raw_row]
        if len(row) != 10:
            raise ValueError("board row must contain 10 cells")
        rows.append(row)
    if len(rows) != 24:
        raise ValueError("imitation board must contain 24 rows")
    return rows


def _feature_vector(features: PlacementFeatures) -> CandidateVector:
    board = features.board
    values = (
        float(board.aggregate_height),
        float(board.max_height),
        float(board.holes),
        float(board.hole_depth),
        float(board.bumpiness),
        float(board.wells),
        float(features.new_holes),
        float(features.lines),
        float(features.attack),
        float(features.spin_lines),
        float(features.perfect_clear),
    )
    return CandidateVector(values, features.game_over)


def _game_from_record(record: Mapping[str, object]) -> Game:
    piece = str(record.get("piece", "")).upper()
    if piece not in {"I", "O", "T", "S", "Z", "J", "L"}:
        raise ValueError("sample has no supported piece")
    game = Game(0)
    game.board = _board(record.get("board_before"))
    game.current = piece
    game.x, game.y, game.rotation = 3, 1, 0
    hold = record.get("hold_before")
    game.hold_piece = None if hold is None else str(hold).upper()
    game.hold_used = False
    game.combo = -1
    game.back_to_back = False
    game.b2b_chain = 0
    game.surge_charge = 0
    game.game_over = game._collides(game.current, game.x, game.y, game.rotation)
    game.paused = False
    return game


def prepare_imitation_examples(
    dataset_path: str | Path,
    alignment_path: str | Path,
    config: ImitationConfig = ImitationConfig(),
) -> tuple[tuple[ImitationExample, ...], dict[str, int]]:
    cfg = config.normalized()
    dataset = {_key(item): item for item in _load_jsonl(dataset_path)}
    alignments = {_key(item): item for item in _load_jsonl(alignment_path)}
    skipped: dict[str, int] = {}
    examples: list[ImitationExample] = []

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for key in sorted(dataset):
        if len(examples) >= cfg.max_samples:
            break
        record = dataset[key]
        alignment = alignments.get(key)
        if alignment is None:
            skip("missing-alignment")
            continue
        status = str(alignment.get("status", ""))
        if status not in ({"exact", "ambiguous"} if cfg.include_ambiguous else {"exact"}):
            skip(f"alignment-{status or 'unknown'}")
            continue
        if record.get("board_before") is None:
            skip("no-before-board")
            continue
        if record.get("used_hold") is True:
            skip("hold-transition")
            continue
        try:
            game = _game_from_record(record)
        except ValueError:
            skip("invalid-state")
            continue
        if game.game_over:
            skip("spawn-collision")
            continue
        placements = reachable_placements(
            game,
            allow_180=cfg.allow_180,
            max_nodes=cfg.reachability_node_limit,
        )
        if not placements:
            skip("no-candidates")
            continue
        target_path = tuple(str(item) for item in alignment.get("path", ()))
        target_x = alignment.get("x")
        target_y = alignment.get("y")
        target_rotation = alignment.get("rotation")
        candidate_vectors: list[CandidateVector] = []
        expert_index: int | None = None
        geometry_matches: list[int] = []
        for placement in placements:
            evaluation = evaluate_placement(game, placement)
            candidate_vectors.append(_feature_vector(evaluation.features))
            if (
                placement.x == target_x
                and placement.y == target_y
                and placement.rotation == target_rotation
            ):
                geometry_matches.append(len(candidate_vectors) - 1)
                if target_path and placement.path == target_path:
                    expert_index = len(candidate_vectors) - 1
        if expert_index is None and geometry_matches:
            expert_index = geometry_matches[0]
        if expert_index is None:
            skip("expert-not-in-candidates")
            continue
        examples.append(
            ImitationExample(
                key=key,
                split=str(record.get("split", "train")),
                candidates=tuple(candidate_vectors),
                expert_index=expert_index,
            )
        )
    return tuple(examples), skipped


def _theta_from_weights(weights: HeuristicWeights) -> list[float]:
    return [getattr(weights, name) * FEATURE_SCALES[name] for name in FEATURE_NAMES]


def _weights_from_theta(theta: Sequence[float], base: HeuristicWeights) -> HeuristicWeights:
    values = base.to_dict()
    for index, name in enumerate(FEATURE_NAMES):
        values[name] = float(theta[index]) / FEATURE_SCALES[name]
    return HeuristicWeights.from_mapping(values)


def _normalized(candidate: CandidateVector) -> tuple[float, ...]:
    return tuple(
        candidate.values[index] / FEATURE_SCALES[name]
        for index, name in enumerate(FEATURE_NAMES)
    )


def _score(theta: Sequence[float], candidate: CandidateVector, game_over_weight: float) -> float:
    value = sum(weight * feature for weight, feature in zip(theta, _normalized(candidate)))
    if candidate.game_over:
        value += game_over_weight
    return value


def _metrics(
    examples: Sequence[ImitationExample],
    theta: Sequence[float],
    game_over_weight: float,
) -> RankingMetrics:
    top1 = top3 = 0
    rank_total = 0.0
    for example in examples:
        ranked = sorted(
            range(len(example.candidates)),
            key=lambda index: (_score(theta, example.candidates[index], game_over_weight), -index),
            reverse=True,
        )
        rank = ranked.index(example.expert_index) + 1
        rank_total += rank
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
    return RankingMetrics(
        samples=len(examples),
        top1=top1,
        top3=top3,
        mean_rank=rank_total / len(examples) if examples else 0.0,
    )


def train_imitation(
    dataset_path: str | Path,
    alignment_path: str | Path,
    initial_weights: HeuristicWeights,
    config: ImitationConfig = ImitationConfig(),
) -> ImitationResult:
    cfg = config.normalized()
    examples, skipped = prepare_imitation_examples(dataset_path, alignment_path, cfg)
    train_examples = [item for item in examples if item.split == "train"]
    validation_examples = [item for item in examples if item.split == "validation"]
    test_examples = [item for item in examples if item.split == "test"]
    rng = random.Random(cfg.random_seed)
    theta = _theta_from_weights(initial_weights)
    losses: list[float] = []

    for _ in range(cfg.epochs):
        rng.shuffle(train_examples)
        loss_total = 0.0
        updates = 0
        for example in train_examples:
            expert = example.candidates[example.expert_index]
            rivals = [
                index for index in range(len(example.candidates))
                if index != example.expert_index
            ]
            if not rivals:
                continue
            rival_index = max(
                rivals,
                key=lambda index: _score(theta, example.candidates[index], initial_weights.game_over),
            )
            rival = example.candidates[rival_index]
            expert_features = _normalized(expert)
            rival_features = _normalized(rival)
            difference = tuple(a - b for a, b in zip(expert_features, rival_features))
            margin = sum(weight * feature for weight, feature in zip(theta, difference))
            margin += (int(expert.game_over) - int(rival.game_over)) * initial_weights.game_over
            clipped = min(40.0, max(-40.0, margin))
            factor = 1.0 / (1.0 + exp(clipped))
            loss_total += max(0.0, -margin) + (0.0 if margin >= 0 else 1.0)
            for index, gradient in enumerate(difference):
                theta[index] += cfg.learning_rate * (
                    factor * gradient - cfg.l2 * theta[index]
                )
            updates += 1
        losses.append(loss_total / updates if updates else 0.0)

    learned = _weights_from_theta(theta, initial_weights)
    return ImitationResult(
        config=cfg,
        initial_weights=initial_weights,
        learned_weights=learned,
        prepared_examples=len(examples),
        skipped=skipped,
        epoch_losses=tuple(losses),
        train=_metrics(train_examples, theta, initial_weights.game_over),
        validation=_metrics(validation_examples, theta, initial_weights.game_over),
        test=_metrics(test_examples, theta, initial_weights.game_over),
    )
