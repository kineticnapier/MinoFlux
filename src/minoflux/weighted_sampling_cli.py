from __future__ import annotations

import argparse
from collections import Counter
import random
from typing import Sequence

from minoflux_ai.neural_train import _load_jsonl, _split_by_game
from minoflux_ai.neural_weighted_train import (
    _positive_weight_records,
    _record_training_weights,
    _weighted_epoch_indices,
)

RecordKey = tuple[int, int]


def _record_key(record: dict[str, object]) -> RecordKey:
    return int(record.get("seed", 0)), int(record.get("pieceIndex", 0))


def _load_side(
    path: str,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[tuple[RecordKey, ...], tuple[float, ...], random.Random]:
    rng = random.Random(seed)
    records = _positive_weight_records(_load_jsonl(path))
    train_records, _validation_records = _split_by_game(
        records,
        validation_fraction,
        rng,
    )
    keys = tuple(_record_key(record) for record in train_records)
    weights = _record_training_weights(train_records)
    return keys, weights, rng


def _simulate_sampling(
    keys: Sequence[RecordKey],
    weights: Sequence[float],
    rng: random.Random,
    *,
    epochs: int,
    samples_per_epoch: int,
    batch_size: int,
) -> list[tuple[RecordKey, ...]]:
    """Reproduce weighted-trainer sampling and RNG advancement without training."""
    traces: list[tuple[RecordKey, ...]] = []
    batch_count = (samples_per_epoch + batch_size - 1) // batch_size

    for _epoch in range(epochs):
        sampled_indices = _weighted_epoch_indices(weights, samples_per_epoch, rng)
        traces.append(tuple(keys[index] for index in sampled_indices))

        # train_weighted_neural_value_model shuffles its batch-job list after
        # sampling. Consume exactly the same RNG operation so the next epoch's
        # systematic-resampling offset matches the real training run.
        batch_jobs = list(range(batch_count))
        rng.shuffle(batch_jobs)

    return traces


def _multiset_overlap(left: Sequence[RecordKey], right: Sequence[RecordKey]) -> int:
    return sum((Counter(left) & Counter(right)).values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the exact weighted-sampling schedules of two neural datasets"
    )
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=2304)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=12345)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if args.samples_per_epoch < 1:
        raise ValueError("samples-per-epoch must be at least 1")
    if args.batch_size < 1:
        raise ValueError("batch-size must be at least 1")

    left_keys, left_weights, left_rng = _load_side(
        args.left,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    right_keys, right_weights, right_rng = _load_side(
        args.right,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    left_trace = _simulate_sampling(
        left_keys,
        left_weights,
        left_rng,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
    )
    right_trace = _simulate_sampling(
        right_keys,
        right_weights,
        right_rng,
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
    )

    left_available = set(left_keys)
    right_available = set(right_keys)
    shared_available = left_available & right_available
    right_only_available = right_available - left_available
    print(
        f"dataset keys: left={len(left_available)} right={len(right_available)} "
        f"shared={len(shared_available)} right-only={len(right_only_available)}"
    )

    total_overlap = 0
    total_right_old = 0
    total_right_new = 0
    total_same_position = 0
    total_draws = 0

    for epoch, (left_epoch, right_epoch) in enumerate(
        zip(left_trace, right_trace, strict=True),
        start=1,
    ):
        overlap = _multiset_overlap(left_epoch, right_epoch)
        right_new = sum(key not in left_available for key in right_epoch)
        right_old = len(right_epoch) - right_new
        same_position = sum(left == right for left, right in zip(left_epoch, right_epoch, strict=True))
        old_overlap_pct = 100.0 * overlap / right_old if right_old else 100.0
        same_position_pct = 100.0 * same_position / len(right_epoch)
        print(
            f"epoch {epoch}: old-overlap {overlap}/{right_old} ({old_overlap_pct:.2f}%) "
            f"| right-new {right_new} | same-pos {same_position}/{len(right_epoch)} "
            f"({same_position_pct:.2f}%)"
        )
        total_overlap += overlap
        total_right_old += right_old
        total_right_new += right_new
        total_same_position += same_position
        total_draws += len(right_epoch)

    old_overlap_pct = 100.0 * total_overlap / total_right_old if total_right_old else 100.0
    same_position_pct = 100.0 * total_same_position / total_draws if total_draws else 100.0
    print(
        f"total: old-overlap {total_overlap}/{total_right_old} ({old_overlap_pct:.2f}%) "
        f"| right-new {total_right_new} | same-pos {total_same_position}/{total_draws} "
        f"({same_position_pct:.2f}%)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
