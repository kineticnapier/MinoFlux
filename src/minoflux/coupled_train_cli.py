from __future__ import annotations

import json
import math
import random
import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence

import minoflux_ai.neural_weighted_train as weighted_train
from minoflux_ai.neural_train import NeuralTrainConfig, _load_jsonl

from . import weighted_train_cli

RecordKey = tuple[int, int]


def _record_key(record: dict[str, object]) -> RecordKey:
    return int(record.get("seed", 0)), int(record.get("pieceIndex", 0))


def _index_unique_records(records: Sequence[dict[str, object]]) -> dict[RecordKey, int]:
    result: dict[RecordKey, int] = {}
    for index, record in enumerate(records):
        key = _record_key(record)
        if key in result:
            raise ValueError(f"Duplicate record key in coupled dataset: {key}")
        result[key] = index
    return result


def _build_coupled_sampler(
    full_records: Sequence[dict[str, object]],
    base_records: Sequence[dict[str, object]],
    *,
    seed: int,
) -> tuple[
    Callable[[Sequence[float], int, random.Random], list[int]],
    list[int],
    int,
    float,
]:
    """Build a sampler that preserves the base schedule and only replaces slots.

    The caller's RNG is used only for the base weighted schedule, exactly as in the
    base-only run. A separate RNG chooses replacement positions and new records, so
    adding new data cannot perturb old-record order or batch-job shuffling.
    """
    if not base_records:
        raise ValueError("Coupled base dataset is empty")

    full_index = _index_unique_records(full_records)
    base_index = _index_unique_records(base_records)
    base_keys = set(base_index)

    base_to_full: list[int] = []
    base_weights: list[float] = []
    for base_record in base_records:
        key = _record_key(base_record)
        if key not in full_index:
            raise ValueError(f"Base record is missing from full dataset: {key}")
        full_record = full_records[full_index[key]]
        if full_record != base_record:
            raise ValueError(f"Shared record differs between base and full dataset: {key}")
        base_to_full.append(full_index[key])
        base_weights.append(weighted_train._record_training_weight(base_record))

    new_indices = [
        index
        for index, record in enumerate(full_records)
        if _record_key(record) not in base_keys
    ]
    new_weights = [
        weighted_train._record_training_weight(full_records[index])
        for index in new_indices
    ]
    base_mass = sum(base_weights)
    new_mass = sum(new_weights)
    if base_mass <= 0.0:
        raise ValueError("Coupled base dataset has no positive training weight")

    original_sampler = weighted_train._weighted_epoch_indices
    replacement_rng = random.Random(int(seed) ^ 0xC0A1D5EED)
    replacements_per_epoch: list[int] = []

    def sample(
        _full_weights: Sequence[float],
        sample_count: int,
        rng: random.Random,
    ) -> list[int]:
        base_selected = original_sampler(base_weights, sample_count, rng)
        selected = [base_to_full[index] for index in base_selected]

        replacement_count = 0
        if new_indices and new_mass > 0.0:
            expected = sample_count * new_mass / (base_mass + new_mass)
            replacement_count = min(
                sample_count,
                int(math.floor(expected + replacement_rng.random())),
            )
            if replacement_count > 0:
                positions = replacement_rng.sample(range(sample_count), replacement_count)
                new_selected = original_sampler(
                    new_weights,
                    replacement_count,
                    replacement_rng,
                )
                for position, new_local_index in zip(positions, new_selected, strict=True):
                    selected[position] = new_indices[new_local_index]

        replacements_per_epoch.append(replacement_count)
        return selected

    return sample, replacements_per_epoch, len(new_indices), new_mass


@contextmanager
def _patched_coupled_training(
    sampler: Callable[[Sequence[float], int, random.Random], list[int]],
) -> Iterator[None]:
    original_sampler = weighted_train._weighted_epoch_indices
    original_split = weighted_train._split_by_game

    def split_without_disabled_rng(
        records: Sequence[dict[str, object]],
        validation_fraction: float,
        rng: random.Random,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if validation_fraction <= 0.0:
            return list(records), []
        return original_split(records, validation_fraction, rng)

    weighted_train._weighted_epoch_indices = sampler
    weighted_train._split_by_game = split_without_disabled_rng
    try:
        yield
    finally:
        weighted_train._weighted_epoch_indices = original_sampler
        weighted_train._split_by_game = original_split


def build_parser():
    parser = weighted_train_cli.build_parser()
    parser.description = (
        "Train with a coupled weighted schedule: preserve the base sample order and "
        "replace only the slots allocated to records new in the full dataset"
    )
    parser.add_argument(
        "--base-dataset",
        required=True,
        help="Base weighted dataset whose exact sampling schedule is preserved",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.samples_per_epoch is None:
        raise ValueError("--samples-per-epoch is required for coupled A/B training")
    if args.samples_per_epoch < 1:
        raise ValueError("--samples-per-epoch must be at least 1")
    if abs(float(args.validation_fraction)) > 1e-12:
        raise ValueError("Coupled A/B training currently requires --validation-fraction 0")

    full_records = weighted_train._positive_weight_records(_load_jsonl(args.dataset))
    base_records = weighted_train._positive_weight_records(_load_jsonl(args.base_dataset))
    sampler, replacements, new_record_count, new_weight_sum = _build_coupled_sampler(
        full_records,
        base_records,
        seed=args.seed,
    )

    def progress(epoch: int, loss: float) -> None:
        replacement_count = replacements[-1] if replacements else 0
        print(
            f"epoch {epoch}: ranking_loss={loss:.6f} coupled_replacements={replacement_count}",
            file=sys.stderr,
            flush=True,
        )

    with _patched_coupled_training(sampler):
        result = weighted_train.train_weighted_neural_value_model(
            args.dataset,
            args.output,
            NeuralTrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                validation_fraction=args.validation_fraction,
                margin=args.margin,
                human_weight=args.human_weight,
                teacher_weight=args.teacher_weight,
                rollout_weight=args.rollout_weight,
                seed=args.seed,
                device=args.device,
            ),
            human_dataset_path=args.human_dataset,
            resume_from=args.resume,
            progress=progress,
            deterministic=args.deterministic,
            samples_per_epoch=args.samples_per_epoch,
        )

    print(
        "coupled schedule: "
        f"base={len(base_records)} full={len(full_records)} new={new_record_count} "
        f"newWeight={new_weight_sum:.6f} replacements={','.join(map(str, replacements))} "
        f"total={sum(replacements)}",
        file=sys.stderr,
        flush=True,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
