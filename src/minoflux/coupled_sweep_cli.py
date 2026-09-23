from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys


@dataclass(frozen=True, slots=True)
class DuelSummary:
    left_wins: int
    right_wins: int
    draws: int
    left_win_rate: float
    right_win_rate: float
    left_app: float
    right_app: float
    left_sent: float
    right_sent: float


_DUEL_SCORE_RE = re.compile(r"^.+? (\d+) - (\d+) .+? \(draw (\d+)\)$")
_DUEL_METRICS_RE = re.compile(
    r"^win ([0-9.]+)% - ([0-9.]+)% \| "
    r"APP ([0-9.]+) - ([0-9.]+) \| "
    r"sent/piece ([0-9.]+) - ([0-9.]+)$"
)


def _parse_seeds(text: str) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    for raw in str(text).split(","):
        item = raw.strip()
        if not item:
            continue
        seed = int(item)
        if seed in seen:
            raise ValueError(f"Duplicate training seed: {seed}")
        seen.add(seed)
        values.append(seed)
    if not values:
        raise ValueError("At least one training seed is required")
    return tuple(values)


def _parse_duel_summary(text: str) -> DuelSummary:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Duel output did not contain the expected two summary lines")
    score = _DUEL_SCORE_RE.match(lines[-2])
    metrics = _DUEL_METRICS_RE.match(lines[-1])
    if score is None or metrics is None:
        raise ValueError("Could not parse compact duel output")
    return DuelSummary(
        left_wins=int(score.group(1)),
        right_wins=int(score.group(2)),
        draws=int(score.group(3)),
        left_win_rate=float(metrics.group(1)),
        right_win_rate=float(metrics.group(2)),
        left_app=float(metrics.group(3)),
        right_app=float(metrics.group(4)),
        left_sent=float(metrics.group(5)),
        right_sent=float(metrics.group(6)),
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def _train_command(
    *,
    dataset: str,
    base_dataset: str,
    output: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    samples_per_epoch: int,
    device: str,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "minoflux.neural_dispatch_cli",
        "train-coupled",
        "--dataset",
        dataset,
        "--base-dataset",
        base_dataset,
        "--output",
        str(output),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--samples-per-epoch",
        str(samples_per_epoch),
        "--validation-fraction",
        "0",
        "--seed",
        str(seed),
        "--device",
        device,
        "--deterministic",
    ]


def _duel_command(
    *,
    left: Path,
    right: Path,
    games: int,
    max_turns: int,
    duel_seed_base: int,
    game_batch: int,
    beam: int,
    device: str,
    allow_180: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "minoflux.neural_dispatch_cli",
        "duel",
        str(left),
        str(right),
        "-n",
        str(games),
        "-t",
        str(max_turns),
        "--seed-base",
        str(duel_seed_base),
        "--game-batch",
        str(game_batch),
        "--beam",
        str(beam),
        "--device",
        device,
    ]
    if not allow_180:
        command.append("--no-180")
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-seed coupled R3/R4 training A/B and compact mirrored duels. "
            "Each pair uses the same training seed and the same evaluation seeds."
        )
    )
    parser.add_argument(
        "--base-dataset",
        default="data/neural/native-oracle-round3-weighted.jsonl",
    )
    parser.add_argument(
        "--full-dataset",
        default="data/neural/native-oracle-round4-weighted.jsonl",
    )
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--seeds", default="12345,23456,34567,45678,56789")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples-per-epoch", type=int, default=2304)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--duel-seed-base", type=int, default=10_000_001)
    parser.add_argument("--game-batch", type=int, default=16)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-180", action="store_true")
    return parser


def _print_failure(label: str, completed: subprocess.CompletedProcess[str]) -> None:
    print(f"{label} failed with exit code {completed.returncode}", file=sys.stderr)
    if completed.stderr.strip():
        print(completed.stderr.rstrip(), file=sys.stderr)
    if completed.stdout.strip():
        print(completed.stdout.rstrip(), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    if args.epochs < 1 or args.batch_size < 1 or args.samples_per_epoch < 1:
        raise ValueError("epochs, batch-size, and samples-per-epoch must be at least 1")
    if args.games < 1 or args.max_turns < 1 or args.game_batch < 1 or args.beam < 1:
        raise ValueError("games, max-turns, game-batch, and beam must be at least 1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[DuelSummary] = []
    for seed in seeds:
        left = output_dir / f"native-oracle-value-r3w-coupled-s{seed}.pt"
        right = output_dir / f"native-oracle-value-r4w-coupled-s{seed}.pt"

        print(f"seed {seed}: train R3", flush=True)
        completed = _run(
            _train_command(
                dataset=args.base_dataset,
                base_dataset=args.base_dataset,
                output=left,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                samples_per_epoch=args.samples_per_epoch,
                device=args.device,
            )
        )
        if completed.returncode != 0:
            _print_failure(f"seed {seed} R3 training", completed)
            return int(completed.returncode)

        print(f"seed {seed}: train R4", flush=True)
        completed = _run(
            _train_command(
                dataset=args.full_dataset,
                base_dataset=args.base_dataset,
                output=right,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                samples_per_epoch=args.samples_per_epoch,
                device=args.device,
            )
        )
        if completed.returncode != 0:
            _print_failure(f"seed {seed} R4 training", completed)
            return int(completed.returncode)

        completed = _run(
            _duel_command(
                left=left,
                right=right,
                games=args.games,
                max_turns=args.max_turns,
                duel_seed_base=args.duel_seed_base,
                game_batch=args.game_batch,
                beam=args.beam,
                device=args.device,
                allow_180=not args.no_180,
            )
        )
        if completed.returncode != 0:
            _print_failure(f"seed {seed} duel", completed)
            return int(completed.returncode)

        try:
            summary = _parse_duel_summary(completed.stdout)
        except ValueError as error:
            print(f"seed {seed}: {error}", file=sys.stderr)
            print(completed.stdout.rstrip(), file=sys.stderr)
            return 2
        summaries.append(summary)
        print(
            f"seed {seed}: R3 {summary.left_wins} - {summary.right_wins} R4 "
            f"(draw {summary.draws}) | win {summary.left_win_rate:.1f}% - "
            f"{summary.right_win_rate:.1f}% | APP {summary.left_app:.4f} - "
            f"{summary.right_app:.4f} | sent/piece {summary.left_sent:.4f} - "
            f"{summary.right_sent:.4f}",
            flush=True,
        )

    left_wins = sum(summary.left_wins for summary in summaries)
    right_wins = sum(summary.right_wins for summary in summaries)
    draws = sum(summary.draws for summary in summaries)
    decided = left_wins + right_wins
    left_rate = 100.0 * left_wins / decided if decided else 0.0
    right_rate = 100.0 * right_wins / decided if decided else 0.0
    left_seed_wins = sum(summary.left_wins > summary.right_wins for summary in summaries)
    right_seed_wins = sum(summary.right_wins > summary.left_wins for summary in summaries)
    seed_ties = len(summaries) - left_seed_wins - right_seed_wins
    print(
        f"aggregate: R3 {left_wins} - {right_wins} R4 (draw {draws}) | "
        f"win {left_rate:.1f}% - {right_rate:.1f}% | "
        f"seed-pairs {left_seed_wins}-{right_seed_wins}-{seed_ties}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
