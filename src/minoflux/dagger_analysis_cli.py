from __future__ import annotations

import argparse
from collections import Counter
import math
from pathlib import Path
from statistics import mean, median
from typing import Mapping, Sequence

from minoflux_ai.features import extract_board_features_from_masks
from minoflux_ai.neural_train import _load_jsonl

_SELECTION_REASONS = ("low_margin", "high_stack", "holes", "random_control")
_CONFIDENCE_THRESHOLDS = (0.08, 0.15, 0.30)


def _percent(count: int, total: int) -> float:
    return 0.0 if total <= 0 else 100.0 * count / total


def _finite_number(value: object) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(round((len(ordered) - 1) * min(1.0, max(0.0, fraction))))
    return ordered[index]


def _reasons(record: Mapping[str, object]) -> tuple[str, ...]:
    raw = record.get("daggerReasons")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(sorted({str(value) for value in raw}))


def _is_disagreement(record: Mapping[str, object]) -> bool:
    reasons = _reasons(record)
    matched = record.get("learnerMatchedOracle")
    return matched is False or "nn_oracle_disagree" in reasons


def _expert_candidate(record: Mapping[str, object]) -> Mapping[str, object] | None:
    candidates = record.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        return None
    index = int(record.get("expertIndex", -1))
    if index < 0 or index >= len(candidates):
        return None
    candidate = candidates[index]
    return candidate if isinstance(candidate, Mapping) else None


def _expert_snapshot(record: Mapping[str, object]) -> dict[str, object]:
    candidate = _expert_candidate(record)
    if candidate is None:
        return {}

    rows = candidate.get("rows")
    features = None
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and rows:
        row_masks = tuple(int(value) for value in rows)
        features = extract_board_features_from_masks(row_masks, width=10)

    context = candidate.get("context")
    lines = 0
    attack = 0
    spin = False
    perfect_clear = False
    game_over = False
    if isinstance(context, Sequence) and not isinstance(context, (str, bytes)) and len(context) >= 9:
        values = [float(value) for value in context[-9:]]
        game_over = values[4] >= 0.5
        lines = int(round(values[5] * 4.0))
        attack = int(round(values[6] * 20.0))
        spin = values[7] >= 0.5
        perfect_clear = values[8] >= 0.5

    move = candidate.get("move")
    move_text = "?"
    if isinstance(move, Sequence) and not isinstance(move, (str, bytes)) and len(move) >= 5:
        hold, piece, x, y, rotation = move[:5]
        move_text = f"{piece}@({int(x)},{int(y)})r{int(rotation)}" + (" hold" if int(hold) else "")

    result: dict[str, object] = {
        "lines": lines,
        "attack": attack,
        "spin": spin,
        "perfect_clear": perfect_clear,
        "game_over": game_over,
        "move": move_text,
    }
    if features is not None:
        result.update(
            {
                "max_height": features.max_height,
                "holes": features.holes,
                "aggregate_height": features.aggregate_height,
                "bumpiness": features.bumpiness,
                "t_spin_slots": features.t_spin_slots,
            }
        )
    return result


def _bucket_name(piece_index: int) -> str:
    start = max(0, int(piece_index) // 100 * 100)
    return f"{start:03d}-{start + 99:03d}"


def _mean_snapshot(snapshots: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(snapshot[key]) for snapshot in snapshots if key in snapshot]
    return mean(values) if values else 0.0


def _summarize_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    samples = len(records)
    disagreements = [record for record in records if _is_disagreement(record)]
    margins = [
        margin
        for record in records
        if (margin := _finite_number(record.get("learnerMargin"))) is not None
    ]
    disagree_margins = [
        margin
        for record in disagreements
        if (margin := _finite_number(record.get("learnerMargin"))) is not None
    ]

    reason_counts = Counter()
    reason_disagreements = Counter()
    reason_combinations = Counter()
    piece_buckets: dict[str, list[int]] = {}
    snapshots: list[dict[str, object]] = []
    disagreement_snapshots: list[dict[str, object]] = []

    for record in records:
        reasons = _reasons(record)
        selected_reasons = tuple(reason for reason in _SELECTION_REASONS if reason in reasons)
        reason_combinations[selected_reasons or ("none",)] += 1
        disagreed = _is_disagreement(record)
        for reason in selected_reasons:
            reason_counts[reason] += 1
            if disagreed:
                reason_disagreements[reason] += 1

        bucket = _bucket_name(int(record.get("pieceIndex", 0)))
        counts = piece_buckets.setdefault(bucket, [0, 0])
        counts[0] += 1
        counts[1] += int(disagreed)

        snapshot = _expert_snapshot(record)
        if snapshot:
            snapshots.append(snapshot)
            if disagreed:
                disagreement_snapshots.append(snapshot)

    confident = {
        threshold: sum(
            1
            for margin in disagree_margins
            if margin > threshold
        )
        for threshold in _CONFIDENCE_THRESHOLDS
    }
    attack_positive = sum(int(snapshot.get("attack", 0)) > 0 for snapshot in snapshots)
    clear_positive = sum(int(snapshot.get("lines", 0)) > 0 for snapshot in snapshots)
    spin_positive = sum(bool(snapshot.get("spin", False)) for snapshot in snapshots)
    game_over = sum(bool(snapshot.get("game_over", False)) for snapshot in snapshots)

    return {
        "samples": samples,
        "disagreements": len(disagreements),
        "disagreement_rate": _percent(len(disagreements), samples),
        "margin_mean": mean(margins) if margins else 0.0,
        "margin_median": median(margins) if margins else 0.0,
        "margin_p90": _quantile(margins, 0.90),
        "disagree_margin_mean": mean(disagree_margins) if disagree_margins else 0.0,
        "confident_disagreements": confident,
        "reason_counts": reason_counts,
        "reason_disagreements": reason_disagreements,
        "reason_combinations": reason_combinations,
        "piece_buckets": piece_buckets,
        "expert": {
            "max_height": _mean_snapshot(snapshots, "max_height"),
            "holes": _mean_snapshot(snapshots, "holes"),
            "aggregate_height": _mean_snapshot(snapshots, "aggregate_height"),
            "bumpiness": _mean_snapshot(snapshots, "bumpiness"),
            "t_spin_slots": _mean_snapshot(snapshots, "t_spin_slots"),
            "attack": _mean_snapshot(snapshots, "attack"),
            "attack_rate": _percent(attack_positive, len(snapshots)),
            "clear_rate": _percent(clear_positive, len(snapshots)),
            "spin_rate": _percent(spin_positive, len(snapshots)),
            "game_over_rate": _percent(game_over, len(snapshots)),
        },
        "expert_disagree": {
            "max_height": _mean_snapshot(disagreement_snapshots, "max_height"),
            "holes": _mean_snapshot(disagreement_snapshots, "holes"),
            "attack": _mean_snapshot(disagreement_snapshots, "attack"),
        },
    }


def _print_summary(label: str, summary: Mapping[str, object]) -> None:
    samples = int(summary["samples"])
    disagreements = int(summary["disagreements"])
    print(
        f"{label}: samples={samples} disagree={disagreements} "
        f"({float(summary['disagreement_rate']):.2f}%) | "
        f"margin mean={float(summary['margin_mean']):.4f} "
        f"p50={float(summary['margin_median']):.4f} "
        f"p90={float(summary['margin_p90']):.4f}"
    )

    confident = summary["confident_disagreements"]
    assert isinstance(confident, Mapping)
    print(
        "  confident disagreements: "
        + " | ".join(
            f">{threshold:.2f} {int(confident[threshold])}"
            for threshold in _CONFIDENCE_THRESHOLDS
        )
    )


def _print_reason_table(
    target: Mapping[str, object],
    baseline: Mapping[str, object] | None,
) -> None:
    print("selection reasons:")
    target_samples = int(target["samples"])
    baseline_samples = int(baseline["samples"]) if baseline is not None else 0
    target_counts = target["reason_counts"]
    target_disagreements = target["reason_disagreements"]
    baseline_counts = baseline["reason_counts"] if baseline is not None else {}
    assert isinstance(target_counts, Mapping)
    assert isinstance(target_disagreements, Mapping)
    assert isinstance(baseline_counts, Mapping)

    for reason in _SELECTION_REASONS:
        target_count = int(target_counts.get(reason, 0))
        target_disagree = int(target_disagreements.get(reason, 0))
        text = (
            f"  {reason:14s} target {target_count:3d}/{target_samples} "
            f"({_percent(target_count, target_samples):5.1f}%) "
            f"disagree={_percent(target_disagree, target_count):5.1f}%"
        )
        if baseline is not None:
            baseline_count = int(baseline_counts.get(reason, 0))
            text += (
                f" | baseline {baseline_count:3d}/{baseline_samples} "
                f"({_percent(baseline_count, baseline_samples):5.1f}%)"
            )
        print(text)


def _print_expert_features(
    target: Mapping[str, object],
    baseline: Mapping[str, object] | None,
) -> None:
    target_expert = target["expert"]
    assert isinstance(target_expert, Mapping)
    print("oracle expert post-state:")
    for key, label in (
        ("max_height", "maxHeight"),
        ("holes", "holes"),
        ("attack", "immediateAttack"),
        ("attack_rate", "attack>0%"),
        ("clear_rate", "lineClear%"),
        ("spin_rate", "spin%"),
        ("game_over_rate", "gameOver%"),
    ):
        target_value = float(target_expert[key])
        text = f"  {label:16s} target={target_value:.3f}"
        if baseline is not None:
            baseline_expert = baseline["expert"]
            assert isinstance(baseline_expert, Mapping)
            baseline_value = float(baseline_expert[key])
            text += f" | baseline={baseline_value:.3f} | delta={target_value - baseline_value:+.3f}"
        print(text)


def _print_piece_buckets(summary: Mapping[str, object]) -> None:
    buckets = summary["piece_buckets"]
    assert isinstance(buckets, Mapping)
    print("target piece buckets:")
    for name in sorted(buckets):
        values = buckets[name]
        assert isinstance(values, Sequence)
        count, disagree = int(values[0]), int(values[1])
        print(f"  {name}: {count:3d} samples | disagree={_percent(disagree, count):5.1f}%")


def _print_top_disagreements(
    records: Sequence[dict[str, object]],
    *,
    limit: int,
) -> None:
    ranked: list[tuple[float, dict[str, object], dict[str, object]]] = []
    for record in records:
        if not _is_disagreement(record):
            continue
        margin = _finite_number(record.get("learnerMargin"))
        if margin is None:
            continue
        ranked.append((margin, record, _expert_snapshot(record)))
    ranked.sort(key=lambda item: item[0], reverse=True)

    print(f"top confident target disagreements ({min(limit, len(ranked))}):")
    for margin, record, snapshot in ranked[:limit]:
        reasons = ",".join(reason for reason in _reasons(record) if reason != "nn_oracle_disagree")
        height = snapshot.get("max_height", "?")
        holes = snapshot.get("holes", "?")
        attack = snapshot.get("attack", "?")
        lines = snapshot.get("lines", "?")
        spin = int(bool(snapshot.get("spin", False)))
        move = snapshot.get("move", "?")
        print(
            f"  seed={int(record.get('seed', 0))} piece={int(record.get('pieceIndex', 0)):3d} "
            f"margin={margin:.4f} reasons={reasons or 'none'} | "
            f"expert h={height} holes={holes} lines={lines} atk={attack} spin={spin} | {move}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Oracle DAgger samples by confidence, selection reason, and expert outcome"
    )
    parser.add_argument(
        "--dataset",
        default="data/neural/native-oracle-dagger-r4.jsonl",
        help="Target DAgger dataset to analyze",
    )
    parser.add_argument(
        "--baseline",
        default="data/neural/native-oracle-dagger-r3.jsonl",
        help="Optional earlier DAgger dataset for side-by-side comparison",
    )
    parser.add_argument("--top", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target_path = Path(args.dataset)
    if not target_path.exists():
        raise FileNotFoundError(target_path)

    target_records = _load_jsonl(target_path)
    target = _summarize_records(target_records)

    baseline = None
    baseline_path = Path(args.baseline) if args.baseline else None
    if baseline_path is not None and baseline_path.exists():
        baseline = _summarize_records(_load_jsonl(baseline_path))

    _print_summary("target", target)
    if baseline is not None:
        _print_summary("baseline", baseline)
        print(
            f"delta: disagreement {float(target['disagreement_rate']) - float(baseline['disagreement_rate']):+.2f}pp "
            f"| margin p50 {float(target['margin_median']) - float(baseline['margin_median']):+.4f}"
        )
    elif baseline_path is not None:
        print(f"baseline: {baseline_path} not found (skipped)")

    _print_reason_table(target, baseline)
    _print_expert_features(target, baseline)
    _print_piece_buckets(target)
    _print_top_disagreements(target_records, limit=max(0, int(args.top)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
