from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

from .bitboard import ROW_OCCUPANCY_BYTES
from .neural import (
    NEURAL_VALUE_FORMAT,
    NeuralValueConfig,
    build_neural_value_model,
    save_neural_value_checkpoint,
)
from .neural_dataset import NEURAL_DATASET_FORMAT


@dataclass(frozen=True, slots=True)
class NeuralTrainConfig:
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.10
    margin: float = 0.20
    human_weight: float = 5.0
    teacher_weight: float = 0.25
    rollout_weight: float = 0.50
    seed: int = 12345
    device: str = "auto"

    def normalized(self) -> "NeuralTrainConfig":
        return NeuralTrainConfig(
            epochs=max(1, int(self.epochs)),
            batch_size=max(1, int(self.batch_size)),
            learning_rate=max(1e-7, float(self.learning_rate)),
            weight_decay=max(0.0, float(self.weight_decay)),
            validation_fraction=min(0.5, max(0.0, float(self.validation_fraction))),
            margin=max(0.0, float(self.margin)),
            human_weight=max(0.0, float(self.human_weight)),
            teacher_weight=max(0.0, float(self.teacher_weight)),
            rollout_weight=max(0.0, float(self.rollout_weight)),
            seed=int(self.seed),
            device=str(self.device or "auto"),
        )


@dataclass(frozen=True, slots=True)
class NeuralTrainMetrics:
    samples: int
    top1_accuracy: float
    top3_accuracy: float
    mean_rank: float
    loss: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeuralTrainResult:
    checkpoint_path: str
    device: str
    train: NeuralTrainMetrics
    validation: NeuralTrainMetrics
    human: NeuralTrainMetrics | None
    epoch_losses: tuple[float, ...]
    config: NeuralTrainConfig
    human_dataset_path: str | None = None
    cache_device: str = "cpu"
    elapsed_seconds: float = 0.0
    cache_seconds: float = 0.0
    epoch_seconds: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "format": NEURAL_VALUE_FORMAT,
            "checkpointPath": self.checkpoint_path,
            "device": self.device,
            "cacheDevice": self.cache_device,
            "elapsedSeconds": self.elapsed_seconds,
            "cacheSeconds": self.cache_seconds,
            "epochSeconds": list(self.epoch_seconds),
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "human": None if self.human is None else self.human.to_dict(),
            "humanDatasetPath": self.human_dataset_path,
            "epochLosses": list(self.epoch_losses),
            "config": asdict(self.config),
        }


@dataclass(frozen=True, slots=True)
class _PreparedGroup:
    start: int
    end: int
    expert_indices: tuple[int, ...]
    negative_indices: tuple[int, ...]
    teacher_pairs: tuple[tuple[int, int], ...]
    rollout_pairs: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _DatasetCache:
    boards: Any
    contexts: Any
    groups: tuple[_PreparedGroup, ...]
    device: str


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch.nn import functional as F
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for neural training. Install it with `uv sync --extra ml`."
        ) from error
    return torch, F


def _resolve_device(torch: Any, requested: str) -> str:
    value = requested.strip().lower()
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return value


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict) or value.get("format") != NEURAL_DATASET_FORMAT:
                raise ValueError(f"Invalid neural dataset record at line {line_number}")
            records.append(value)
    if not records:
        raise ValueError("Neural dataset is empty")
    return records


def _infer_config(record: Mapping[str, object]) -> NeuralValueConfig:
    raw_candidates = record.get("candidates")
    if not isinstance(raw_candidates, Sequence) or not raw_candidates:
        raise ValueError("Neural dataset record has no candidates")
    first = raw_candidates[0]
    if not isinstance(first, Mapping):
        raise ValueError("Neural candidate must be an object")
    rows = first.get("rows")
    context = first.get("context")
    if not isinstance(rows, Sequence) or not isinstance(context, Sequence):
        raise ValueError("Neural candidate is missing rows/context arrays")
    cfg = NeuralValueConfig()
    if len(rows) != cfg.board_height or len(context) != cfg.context_size:
        raise ValueError(
            f"Dataset feature shape does not match neural config: rows={len(rows)}, context={len(context)}"
        )
    return cfg


def _split_by_game(
    records: Sequence[dict[str, object]],
    validation_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    seeds = sorted({int(record.get("seed", 0)) for record in records})
    rng.shuffle(seeds)
    if validation_fraction <= 0.0 or len(seeds) < 2:
        return list(records), []
    validation_games = max(1, int(round(len(seeds) * validation_fraction)))
    validation_games = min(len(seeds) - 1, validation_games)
    validation_seeds = set(seeds[:validation_games])
    train = [record for record in records if int(record.get("seed", 0)) not in validation_seeds]
    validation = [record for record in records if int(record.get("seed", 0)) in validation_seeds]
    return train, validation


def _expert_indices(record: Mapping[str, object], count: int) -> tuple[int, ...]:
    raw = record.get("expertIndices")
    indices: list[int] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        indices.extend(int(value) for value in raw)
    if not indices:
        indices.append(int(record.get("expertIndex", -1)))
    unique = tuple(sorted(set(indices)))
    if not unique or any(index < 0 or index >= count for index in unique):
        raise ValueError("expertIndex/expertIndices is out of range")
    return unique


def _optional_number(candidate: Mapping[str, object], key: str) -> float | None:
    value = candidate.get(key)
    if value is None:
        return None
    return float(value)


def _ordered_pairs(labels: Sequence[float | None]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for left in range(len(labels)):
        left_label = labels[left]
        if left_label is None:
            continue
        for right in range(left + 1, len(labels)):
            right_label = labels[right]
            if right_label is None:
                continue
            difference = float(left_label) - float(right_label)
            if abs(difference) <= 1e-12:
                continue
            pairs.append((left, right) if difference > 0.0 else (right, left))
    return tuple(pairs)


def _build_dataset_cache(
    records: Sequence[dict[str, object]],
    cfg: NeuralValueConfig,
    torch: Any,
    device: str,
) -> _DatasetCache:
    if not records:
        raise ValueError("Cannot cache an empty neural dataset")

    board_bytes = bytearray()
    context_values = array("f")
    groups: list[_PreparedGroup] = []
    candidate_cursor = 0

    for record in records:
        raw_candidates = record.get("candidates")
        if not isinstance(raw_candidates, Sequence) or not raw_candidates:
            raise ValueError("Sample has no candidates")
        count = len(raw_candidates)
        positives = _expert_indices(record, count)
        positive_set = set(positives)
        teacher_scores: list[float | None] = []
        target_values: list[float | None] = []
        start = candidate_cursor

        for candidate in raw_candidates:
            if not isinstance(candidate, Mapping):
                raise ValueError("Candidate must be an object")
            rows = candidate.get("rows")
            context = candidate.get("context")
            if not isinstance(rows, Sequence) or not isinstance(context, Sequence):
                raise ValueError("Candidate is missing rows/context")
            if len(rows) != cfg.board_height or len(context) != cfg.context_size:
                raise ValueError("Candidate feature shape does not match neural config")
            for value in rows:
                mask = int(value)
                if mask < 0 or mask >= len(ROW_OCCUPANCY_BYTES):
                    raise ValueError(f"Packed board row mask is out of range: {mask}")
                board_bytes.extend(ROW_OCCUPANCY_BYTES[mask])
            context_values.extend(float(value) for value in context)
            teacher_scores.append(_optional_number(candidate, "teacherScore"))
            target_values.append(_optional_number(candidate, "targetValue"))
            candidate_cursor += 1

        groups.append(
            _PreparedGroup(
                start=start,
                end=candidate_cursor,
                expert_indices=positives,
                negative_indices=tuple(index for index in range(count) if index not in positive_set),
                teacher_pairs=_ordered_pairs(teacher_scores),
                rollout_pairs=_ordered_pairs(target_values),
            )
        )

    board_cpu = torch.frombuffer(board_bytes, dtype=torch.uint8).reshape(
        candidate_cursor, 1, cfg.board_height, cfg.board_width
    )
    context_cpu = torch.frombuffer(context_values, dtype=torch.float32).reshape(
        candidate_cursor, cfg.context_size
    )

    # The full 22.9k-sample training set is small enough that float32 boards use
    # only tens of MB. Convert once here instead of converting every batch.
    boards = board_cpu.to(device=device, dtype=torch.float32)
    contexts = context_cpu.clone() if device == "cpu" else context_cpu.to(device=device)
    return _DatasetCache(
        boards=boards,
        contexts=contexts,
        groups=tuple(groups),
        device=device,
    )


def _prepare_cached_batch(
    cache: _DatasetCache,
    sample_indices: Sequence[int],
    torch: Any,
) -> tuple[Any, Any, tuple[_PreparedGroup, ...]]:
    flat_indices: list[int] = []
    local_groups: list[_PreparedGroup] = []
    cursor = 0
    for sample_index in sample_indices:
        prepared = cache.groups[int(sample_index)]
        count = prepared.end - prepared.start
        flat_indices.extend(range(prepared.start, prepared.end))
        local_groups.append(
            _PreparedGroup(
                start=cursor,
                end=cursor + count,
                expert_indices=prepared.expert_indices,
                negative_indices=prepared.negative_indices,
                teacher_pairs=prepared.teacher_pairs,
                rollout_pairs=prepared.rollout_pairs,
            )
        )
        cursor += count

    index = torch.tensor(flat_indices, dtype=torch.long, device=cache.device)
    boards = cache.boards.index_select(0, index)
    contexts = cache.contexts.index_select(0, index)
    return boards, contexts, tuple(local_groups)


def _pairwise_order_loss(
    group: Any,
    pairs: Sequence[tuple[int, int]],
    torch: Any,
    F: Any,
    margin: float,
) -> Any:
    if not pairs:
        return group.sum() * 0.0
    winners = torch.tensor(
        [winner for winner, _loser in pairs],
        dtype=torch.long,
        device=group.device,
    )
    losers = torch.tensor(
        [loser for _winner, loser in pairs],
        dtype=torch.long,
        device=group.device,
    )
    differences = group.index_select(0, winners) - group.index_select(0, losers)
    return F.relu(margin - differences).mean()


def _loss_and_ranks(
    values: Any,
    groups: Sequence[_PreparedGroup],
    torch: Any,
    F: Any,
    margin: float,
    teacher_weight: float,
    rollout_weight: float,
    *,
    collect_metrics: bool = True,
) -> tuple[Any, int, int, float]:
    """Reference loss path used for final metrics and regression checking."""

    losses = []
    top1 = 0
    top3 = 0
    rank_total = 0.0
    metric_values = values.detach().cpu().tolist() if collect_metrics else None

    for prepared in groups:
        group = values[prepared.start : prepared.end]
        positive_indices = prepared.expert_indices
        positives = group[
            torch.tensor(positive_indices, dtype=torch.long, device=group.device)
        ]
        if prepared.negative_indices:
            negatives = group[
                torch.tensor(prepared.negative_indices, dtype=torch.long, device=group.device)
            ]
            ranking_loss = F.relu(
                margin - (positives[:, None] - negatives[None, :])
            ).mean()
        else:
            ranking_loss = group.sum() * 0.0

        teacher_loss = _pairwise_order_loss(
            group, prepared.teacher_pairs, torch, F, margin
        )
        rollout_loss = _pairwise_order_loss(
            group, prepared.rollout_pairs, torch, F, margin
        )
        losses.append(
            ranking_loss
            + float(teacher_weight) * teacher_loss
            + float(rollout_weight) * rollout_loss
        )

        if metric_values is not None:
            group_values = metric_values[prepared.start : prepared.end]
            best_positive = max(float(group_values[index]) for index in positive_indices)
            rank = 1 + sum(float(value) > best_positive for value in group_values)
            top1 += int(rank == 1)
            top3 += int(rank <= 3)
            rank_total += rank

    loss = values.sum() * 0.0 if not losses else torch.stack(losses).mean()
    return loss, top1, top3, rank_total


def _batched_pair_means(
    values: Any,
    groups: Sequence[_PreparedGroup],
    torch: Any,
    F: Any,
    margin: float,
    kind: str,
) -> Any:
    """Return one mean margin loss per sample for one pair family.

    Pair indices are assembled on the CPU once per batch, then all margin losses
    and per-sample reductions happen in a handful of GPU tensor operations.
    """

    winners: list[int] = []
    losers: list[int] = []
    sample_ids: list[int] = []

    for sample_id, prepared in enumerate(groups):
        base = prepared.start
        if kind == "ranking":
            for winner in prepared.expert_indices:
                for loser in prepared.negative_indices:
                    winners.append(base + winner)
                    losers.append(base + loser)
                    sample_ids.append(sample_id)
        elif kind == "teacher":
            for winner, loser in prepared.teacher_pairs:
                winners.append(base + winner)
                losers.append(base + loser)
                sample_ids.append(sample_id)
        elif kind == "rollout":
            for winner, loser in prepared.rollout_pairs:
                winners.append(base + winner)
                losers.append(base + loser)
                sample_ids.append(sample_id)
        else:
            raise ValueError(f"Unknown neural pair kind: {kind}")

    sample_count = len(groups)
    if not winners:
        return values.new_zeros(sample_count)

    winner_index = torch.tensor(winners, dtype=torch.long, device=values.device)
    loser_index = torch.tensor(losers, dtype=torch.long, device=values.device)
    sample_index = torch.tensor(sample_ids, dtype=torch.long, device=values.device)
    pair_losses = F.relu(
        margin
        - (
            values.index_select(0, winner_index)
            - values.index_select(0, loser_index)
        )
    )

    sums = values.new_zeros(sample_count)
    counts = values.new_zeros(sample_count)
    sums.index_add_(0, sample_index, pair_losses)
    counts.index_add_(0, sample_index, torch.ones_like(pair_losses))
    return sums / counts.clamp_min(1.0)


def _loss_only_vectorized(
    values: Any,
    groups: Sequence[_PreparedGroup],
    torch: Any,
    F: Any,
    margin: float,
    teacher_weight: float,
    rollout_weight: float,
) -> Any:
    """Training-only loss path with no per-sample GPU kernels or synchronizations."""

    if not groups:
        return values.sum() * 0.0

    sample_losses = _batched_pair_means(
        values,
        groups,
        torch,
        F,
        margin,
        "ranking",
    )
    if teacher_weight > 0.0:
        sample_losses = sample_losses + float(teacher_weight) * _batched_pair_means(
            values,
            groups,
            torch,
            F,
            margin,
            "teacher",
        )
    if rollout_weight > 0.0:
        sample_losses = sample_losses + float(rollout_weight) * _batched_pair_means(
            values,
            groups,
            torch,
            F,
            margin,
            "rollout",
        )
    return sample_losses.mean()


def _iter_index_batches(
    count: int,
    batch_size: int,
    rng: random.Random | None = None,
) -> list[list[int]]:
    ordered = list(range(count))
    if rng is not None:
        rng.shuffle(ordered)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def _evaluate_cached(
    model: Any,
    cache: _DatasetCache | None,
    torch: Any,
    F: Any,
    margin: float,
    batch_size: int,
    teacher_weight: float,
    rollout_weight: float,
) -> NeuralTrainMetrics:
    if cache is None or not cache.groups:
        return NeuralTrainMetrics(0, 0.0, 0.0, 0.0, 0.0)
    loss_total = 0.0
    top1 = 0
    top3 = 0
    rank_total = 0.0
    samples = 0
    model.eval()
    with torch.inference_mode():
        for indices in _iter_index_batches(len(cache.groups), batch_size):
            boards, contexts, groups = _prepare_cached_batch(cache, indices, torch)
            values = model(boards, contexts)
            loss, batch_top1, batch_top3, batch_rank_total = _loss_and_ranks(
                values,
                groups,
                torch,
                F,
                margin,
                teacher_weight,
                rollout_weight,
                collect_metrics=True,
            )
            count = len(groups)
            loss_total += float(loss.detach().cpu().item()) * count
            top1 += batch_top1
            top3 += batch_top3
            rank_total += batch_rank_total
            samples += count
    return NeuralTrainMetrics(
        samples=samples,
        top1_accuracy=top1 / samples,
        top3_accuracy=top3 / samples,
        mean_rank=rank_total / samples,
        loss=loss_total / samples,
    )


def _load_resume(model: Any, path: str | Path, cfg: NeuralValueConfig, torch: Any, device: str) -> None:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != NEURAL_VALUE_FORMAT:
        raise ValueError("Resume checkpoint is not a MinoFlux neural value model")
    raw_config = payload.get("config")
    if not isinstance(raw_config, Mapping) or NeuralValueConfig.from_mapping(raw_config) != cfg:
        raise ValueError("Resume checkpoint neural config does not match the dataset")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("Resume checkpoint has no state_dict")
    model.load_state_dict(state_dict)


def train_neural_value_model(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    config: NeuralTrainConfig = NeuralTrainConfig(),
    *,
    human_dataset_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> NeuralTrainResult:
    started = time.perf_counter()
    cfg = config.normalized()
    torch, F = _require_torch()
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    rng = random.Random(cfg.seed)
    records = _load_jsonl(dataset_path)
    neural_config = _infer_config(records[0])
    train_records, validation_records = _split_by_game(
        records, cfg.validation_fraction, rng
    )
    if not train_records:
        raise ValueError("Training split is empty")

    human_records: list[dict[str, object]] = []
    if human_dataset_path is not None:
        human_records = _load_jsonl(human_dataset_path)
        for record in human_records:
            if _infer_config(record) != neural_config:
                raise ValueError("Human review dataset neural config does not match the base dataset")

    device = _resolve_device(torch, cfg.device)
    model = build_neural_value_model(neural_config).to(device)
    if resume_from is not None:
        _load_resume(model, resume_from, neural_config, torch, device)

    cache_started = time.perf_counter()
    train_cache = _build_dataset_cache(train_records, neural_config, torch, device)
    validation_cache = (
        _build_dataset_cache(validation_records, neural_config, torch, device)
        if validation_records
        else None
    )
    human_cache = (
        _build_dataset_cache(human_records, neural_config, torch, device)
        if human_records
        else None
    )
    cache_seconds = time.perf_counter() - cache_started

    del records, train_records, validation_records, human_records

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    epoch_losses: list[float] = []
    epoch_seconds: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_parts: list[Any] = []
        weighted_samples = 0.0
        batch_jobs: list[tuple[_DatasetCache, list[int], float]] = [
            (train_cache, batch, 1.0)
            for batch in _iter_index_batches(len(train_cache.groups), cfg.batch_size, rng)
        ]
        if human_cache is not None and cfg.human_weight > 0.0:
            batch_jobs.extend(
                (human_cache, batch, cfg.human_weight)
                for batch in _iter_index_batches(len(human_cache.groups), cfg.batch_size, rng)
            )
        rng.shuffle(batch_jobs)

        for cache, indices, source_weight in batch_jobs:
            boards, contexts, groups = _prepare_cached_batch(cache, indices, torch)
            optimizer.zero_grad(set_to_none=True)
            values = model(boards, contexts)
            raw_loss = _loss_only_vectorized(
                values,
                groups,
                torch,
                F,
                cfg.margin,
                cfg.teacher_weight,
                cfg.rollout_weight,
            )
            loss = raw_loss * source_weight
            loss.backward()
            optimizer.step()
            contribution = len(groups) * source_weight
            loss_parts.append(raw_loss.detach() * contribution)
            weighted_samples += contribution

        if loss_parts:
            loss_total = float(torch.stack(loss_parts).sum().cpu().item())
        else:
            loss_total = 0.0
        epoch_loss = loss_total / max(1.0, weighted_samples)
        epoch_losses.append(epoch_loss)
        epoch_seconds.append(time.perf_counter() - epoch_started)
        if progress is not None:
            progress(epoch, epoch_loss)

    train_metrics = _evaluate_cached(
        model,
        train_cache,
        torch,
        F,
        cfg.margin,
        cfg.batch_size,
        cfg.teacher_weight,
        cfg.rollout_weight,
    )
    validation_metrics = _evaluate_cached(
        model,
        validation_cache,
        torch,
        F,
        cfg.margin,
        cfg.batch_size,
        cfg.teacher_weight,
        cfg.rollout_weight,
    )
    human_metrics = (
        _evaluate_cached(
            model,
            human_cache,
            torch,
            F,
            cfg.margin,
            cfg.batch_size,
            cfg.teacher_weight,
            cfg.rollout_weight,
        )
        if human_cache is not None
        else None
    )
    elapsed_seconds = time.perf_counter() - started
    saved = save_neural_value_checkpoint(
        checkpoint_path,
        model,
        neural_config,
        metadata={
            "dataset": str(dataset_path),
            "humanDataset": None if human_dataset_path is None else str(human_dataset_path),
            "humanWeight": cfg.human_weight,
            "teacherWeight": cfg.teacher_weight,
            "rolloutWeight": cfg.rollout_weight,
            "resumeFrom": None if resume_from is None else str(resume_from),
            "trainConfig": asdict(cfg),
            "trainMetrics": train_metrics.to_dict(),
            "validationMetrics": validation_metrics.to_dict(),
            "humanMetrics": None if human_metrics is None else human_metrics.to_dict(),
            "epochLosses": epoch_losses,
            "cacheDevice": device,
            "cacheSeconds": cache_seconds,
            "epochSeconds": epoch_seconds,
            "elapsedSeconds": elapsed_seconds,
        },
    )
    return NeuralTrainResult(
        checkpoint_path=str(saved),
        device=device,
        train=train_metrics,
        validation=validation_metrics,
        human=human_metrics,
        epoch_losses=tuple(epoch_losses),
        config=cfg,
        human_dataset_path=None if human_dataset_path is None else str(human_dataset_path),
        cache_device=device,
        elapsed_seconds=elapsed_seconds,
        cache_seconds=cache_seconds,
        epoch_seconds=tuple(epoch_seconds),
    )
