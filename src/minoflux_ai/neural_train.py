from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any, Callable, Mapping, Sequence

from .neural import (
    NEURAL_VALUE_FORMAT,
    NeuralValueConfig,
    build_neural_value_model,
    save_neural_value_checkpoint,
)
from .neural_dataset import NEURAL_DATASET_FORMAT, unpack_board_rows


@dataclass(frozen=True, slots=True)
class NeuralTrainConfig:
    epochs: int = 8
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    validation_fraction: float = 0.10
    margin: float = 0.20
    human_weight: float = 5.0
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

    def to_dict(self) -> dict[str, object]:
        return {
            "format": NEURAL_VALUE_FORMAT,
            "checkpointPath": self.checkpoint_path,
            "device": self.device,
            "train": self.train.to_dict(),
            "validation": self.validation.to_dict(),
            "human": None if self.human is None else self.human.to_dict(),
            "humanDatasetPath": self.human_dataset_path,
            "epochLosses": list(self.epoch_losses),
            "config": asdict(self.config),
        }


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


def _candidate_arrays(
    record: Mapping[str, object],
    cfg: NeuralValueConfig,
) -> tuple[list[tuple[float, ...]], list[list[float]], int]:
    raw = record.get("candidates")
    if not isinstance(raw, Sequence) or not raw:
        raise ValueError("Sample has no candidates")
    boards: list[tuple[float, ...]] = []
    contexts: list[list[float]] = []
    for candidate in raw:
        if not isinstance(candidate, Mapping):
            raise ValueError("Candidate must be an object")
        rows = candidate.get("rows")
        context = candidate.get("context")
        if not isinstance(rows, Sequence) or not isinstance(context, Sequence):
            raise ValueError("Candidate is missing rows/context")
        boards.append(unpack_board_rows([int(value) for value in rows], cfg))
        contexts.append([float(value) for value in context])
    expert_index = int(record.get("expertIndex", -1))
    if expert_index < 0 or expert_index >= len(boards):
        raise ValueError("expertIndex is out of range")
    return boards, contexts, expert_index


def _prepare_batch(
    records: Sequence[dict[str, object]],
    cfg: NeuralValueConfig,
    torch: Any,
    device: str,
) -> tuple[Any, Any, tuple[tuple[int, int, int], ...]]:
    flat_boards: list[tuple[float, ...]] = []
    flat_contexts: list[list[float]] = []
    groups: list[tuple[int, int, int]] = []
    for record in records:
        boards, contexts, expert_index = _candidate_arrays(record, cfg)
        start = len(flat_boards)
        flat_boards.extend(boards)
        flat_contexts.extend(contexts)
        groups.append((start, len(flat_boards), expert_index))
    boards_tensor = torch.tensor(flat_boards, dtype=torch.float32, device=device).reshape(
        len(flat_boards), 1, cfg.board_height, cfg.board_width
    )
    contexts_tensor = torch.tensor(flat_contexts, dtype=torch.float32, device=device)
    return boards_tensor, contexts_tensor, tuple(groups)


def _loss_and_ranks(
    values: Any,
    groups: Sequence[tuple[int, int, int]],
    torch: Any,
    F: Any,
    margin: float,
) -> tuple[Any, int, int, float]:
    losses = []
    top1 = 0
    top3 = 0
    rank_total = 0.0
    for start, end, expert_index in groups:
        group = values[start:end]
        expert_value = group[expert_index]
        rivals = torch.cat((group[:expert_index], group[expert_index + 1 :]))
        if rivals.numel() == 0:
            losses.append(group.sum() * 0.0)
        else:
            losses.append(F.relu(margin - (expert_value - rivals)).mean())
        rank = 1 + int((group > expert_value).sum().detach().cpu().item())
        top1 += int(rank == 1)
        top3 += int(rank <= 3)
        rank_total += rank
    loss = values.sum() * 0.0 if not losses else torch.stack(losses).mean()
    return loss, top1, top3, rank_total


def _iter_batches(
    records: Sequence[dict[str, object]],
    batch_size: int,
    rng: random.Random | None = None,
) -> list[list[dict[str, object]]]:
    ordered = list(records)
    if rng is not None:
        rng.shuffle(ordered)
    return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def _evaluate(
    model: Any,
    records: Sequence[dict[str, object]],
    cfg: NeuralValueConfig,
    torch: Any,
    F: Any,
    device: str,
    margin: float,
    batch_size: int,
) -> NeuralTrainMetrics:
    if not records:
        return NeuralTrainMetrics(0, 0.0, 0.0, 0.0, 0.0)
    loss_total = 0.0
    top1 = 0
    top3 = 0
    rank_total = 0.0
    samples = 0
    model.eval()
    with torch.inference_mode():
        for batch in _iter_batches(records, batch_size):
            boards, contexts, groups = _prepare_batch(batch, cfg, torch, device)
            values = model(boards, contexts)
            loss, batch_top1, batch_top3, batch_rank_total = _loss_and_ranks(
                values, groups, torch, F, margin
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    epoch_losses: list[float] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss_total = 0.0
        weighted_samples = 0.0
        batch_jobs: list[tuple[list[dict[str, object]], float]] = [
            (batch, 1.0)
            for batch in _iter_batches(train_records, cfg.batch_size, rng)
        ]
        if human_records and cfg.human_weight > 0.0:
            batch_jobs.extend(
                (batch, cfg.human_weight)
                for batch in _iter_batches(human_records, cfg.batch_size, rng)
            )
        rng.shuffle(batch_jobs)
        for batch, source_weight in batch_jobs:
            boards, contexts, groups = _prepare_batch(batch, neural_config, torch, device)
            optimizer.zero_grad(set_to_none=True)
            values = model(boards, contexts)
            raw_loss, _, _, _ = _loss_and_ranks(values, groups, torch, F, cfg.margin)
            loss = raw_loss * source_weight
            loss.backward()
            optimizer.step()
            contribution = len(groups) * source_weight
            loss_total += float(raw_loss.detach().cpu().item()) * contribution
            weighted_samples += contribution
        epoch_loss = loss_total / max(1.0, weighted_samples)
        epoch_losses.append(epoch_loss)
        if progress is not None:
            progress(epoch, epoch_loss)

    train_metrics = _evaluate(
        model,
        train_records,
        neural_config,
        torch,
        F,
        device,
        cfg.margin,
        cfg.batch_size,
    )
    validation_metrics = _evaluate(
        model,
        validation_records,
        neural_config,
        torch,
        F,
        device,
        cfg.margin,
        cfg.batch_size,
    )
    human_metrics = (
        _evaluate(
            model,
            human_records,
            neural_config,
            torch,
            F,
            device,
            cfg.margin,
            cfg.batch_size,
        )
        if human_records
        else None
    )
    saved = save_neural_value_checkpoint(
        checkpoint_path,
        model,
        neural_config,
        metadata={
            "dataset": str(dataset_path),
            "humanDataset": None if human_dataset_path is None else str(human_dataset_path),
            "humanWeight": cfg.human_weight,
            "resumeFrom": None if resume_from is None else str(resume_from),
            "trainConfig": asdict(cfg),
            "trainMetrics": train_metrics.to_dict(),
            "validationMetrics": validation_metrics.to_dict(),
            "humanMetrics": None if human_metrics is None else human_metrics.to_dict(),
            "epochLosses": epoch_losses,
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
    )
