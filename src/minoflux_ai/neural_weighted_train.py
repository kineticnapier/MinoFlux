from __future__ import annotations

from dataclasses import asdict
import math
from pathlib import Path
import random
import time
from typing import Any, Callable

from .neural import build_neural_value_model, save_neural_value_checkpoint
from .neural_train import (
    NeuralTrainConfig,
    NeuralTrainResult,
    _batched_pair_means,
    _build_dataset_cache,
    _evaluate_cached,
    _infer_config,
    _iter_index_batches,
    _load_jsonl,
    _load_resume,
    _prepare_cached_batch,
    _require_torch,
    _resolve_device,
    _split_by_game,
)


def _record_training_weight(record: dict[str, object]) -> float:
    weight = float(record.get("trainingWeight", 1.0))
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("trainingWeight must be finite and non-negative")
    return weight


def _record_training_weights(records: list[dict[str, object]]) -> tuple[float, ...]:
    return tuple(_record_training_weight(record) for record in records)


def _weighted_loss_only_vectorized(
    values: Any,
    groups: Any,
    sample_weights: tuple[float, ...] | list[float],
    torch: Any,
    F: Any,
    margin: float,
    teacher_weight: float,
    rollout_weight: float,
) -> Any:
    """Training loss with one multiplicative weight per ranking sample.

    The denominator remains the physical batch size rather than the sum of weights,
    so a sample weighted 0.25 contributes one quarter of the gradient it would have
    at weight 1.0 instead of being renormalized back to full strength.
    """

    if not groups:
        return values.sum() * 0.0
    if len(sample_weights) != len(groups):
        raise ValueError("sample_weights must match the number of ranking groups")

    weights = values.new_tensor(sample_weights)
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
    return (sample_losses * weights).sum() / len(groups)


def train_weighted_neural_value_model(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    config: NeuralTrainConfig = NeuralTrainConfig(),
    *,
    human_dataset_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> NeuralTrainResult:
    """Train a neural value model while honoring per-record ``trainingWeight``."""

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
        records,
        cfg.validation_fraction,
        rng,
    )
    if not train_records:
        raise ValueError("Training split is empty")
    train_weights = _record_training_weights(train_records)
    validation_weights = _record_training_weights(validation_records)

    human_records: list[dict[str, object]] = []
    if human_dataset_path is not None:
        human_records = _load_jsonl(human_dataset_path)
        for record in human_records:
            if _infer_config(record) != neural_config:
                raise ValueError("Human review dataset neural config does not match the base dataset")
    human_weights = _record_training_weights(human_records)

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
        batch_jobs: list[tuple[Any, list[int], tuple[float, ...], float]] = [
            (
                train_cache,
                batch,
                tuple(train_weights[index] for index in batch),
                1.0,
            )
            for batch in _iter_index_batches(len(train_cache.groups), cfg.batch_size, rng)
        ]
        if human_cache is not None and cfg.human_weight > 0.0:
            batch_jobs.extend(
                (
                    human_cache,
                    batch,
                    tuple(human_weights[index] for index in batch),
                    cfg.human_weight,
                )
                for batch in _iter_index_batches(len(human_cache.groups), cfg.batch_size, rng)
            )
        rng.shuffle(batch_jobs)

        for cache, indices, base_weights, source_weight in batch_jobs:
            effective_weights = tuple(weight * source_weight for weight in base_weights)
            effective_weight_sum = sum(effective_weights)
            if effective_weight_sum <= 0.0:
                continue

            boards, contexts, groups = _prepare_cached_batch(cache, indices, torch)
            optimizer.zero_grad(set_to_none=True)
            values = model(boards, contexts)
            loss = _weighted_loss_only_vectorized(
                values,
                groups,
                effective_weights,
                torch,
                F,
                cfg.margin,
                cfg.teacher_weight,
                cfg.rollout_weight,
            )
            loss.backward()
            optimizer.step()

            loss_parts.append(loss.detach() * len(groups))
            weighted_samples += effective_weight_sum

        if loss_parts:
            weighted_loss_total = float(torch.stack(loss_parts).sum().cpu().item())
        else:
            weighted_loss_total = 0.0
        epoch_loss = weighted_loss_total / max(1.0, weighted_samples)
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
            "weightedTraining": True,
            "trainWeightSum": sum(train_weights),
            "validationWeightSum": sum(validation_weights),
            "humanRecordWeightSum": sum(human_weights),
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
