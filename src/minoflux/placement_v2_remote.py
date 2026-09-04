from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

DATASET = Path("data/neural/placement-v2-ranking.jsonl")
MODEL = Path("data/models/placement-v2.pt")


def _run(label: str, *args: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join((str(sys.executable), *args)), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def _generate(*, games: int, max_pieces: int, teacher_depth: int, teacher_beam: int, workers: int) -> None:
    _run(
        "placement-v2 teacher dataset",
        "-m",
        "minoflux.placement_v2_cli",
        "generate",
        "--output",
        str(DATASET),
        "--games",
        str(games),
        "--max-pieces",
        str(max_pieces),
        "--seed-base",
        "8000001",
        "--seed-step",
        "31",
        "--max-candidates",
        "24",
        "--teacher-depth",
        str(teacher_depth),
        "--teacher-beam",
        str(teacher_beam),
        "--workers",
        str(workers),
    )


def _train(*, epochs: int) -> None:
    if not DATASET.is_file():
        raise SystemExit(f"Dataset not found: {DATASET}")
    _run(
        "placement-v2 neural distillation",
        "-m",
        "minoflux.neural_cli",
        "train",
        "--dataset",
        str(DATASET),
        "--output",
        str(MODEL),
        "--epochs",
        str(epochs),
        "--batch-size",
        "64",
        "--learning-rate",
        "3e-4",
        "--teacher-weight",
        "0.5",
        "--rollout-weight",
        "0",
        "--device",
        "auto",
    )


def _evaluate(model: Path, label: str) -> None:
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    _run(
        label,
        "-m",
        "minoflux.neural_cli",
        "evaluate",
        "--model",
        str(model),
        "--games",
        "8",
        "--max-pieces",
        "300",
        "--seed-base",
        "8100001",
        "--seed-step",
        "97",
        "--game-batch-size",
        "8",
        "--device",
        "auto",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed Placement-v2 jobs used by the MinoFlux remote agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Generate, train, then compare Placement-v2")
    full.add_argument("--games", type=int, default=50)
    full.add_argument("--max-pieces", type=int, default=300)
    full.add_argument("--teacher-depth", type=int, default=2)
    full.add_argument("--teacher-beam", type=int, default=24)
    full.add_argument("--workers", type=int, default=6)
    full.add_argument("--epochs", type=int, default=8)

    generate = sub.add_parser("generate", help="Generate the fixed Placement-v2 dataset")
    generate.add_argument("--games", type=int, default=50)
    generate.add_argument("--max-pieces", type=int, default=300)
    generate.add_argument("--teacher-depth", type=int, default=2)
    generate.add_argument("--teacher-beam", type=int, default=24)
    generate.add_argument("--workers", type=int, default=6)

    train = sub.add_parser("train", help="Train Placement-v2 from the fixed dataset")
    train.add_argument("--epochs", type=int, default=8)

    sub.add_parser("evaluate", help="Evaluate Placement-v2 and the current human model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"full", "generate"}:
        _generate(
            games=args.games,
            max_pieces=args.max_pieces,
            teacher_depth=args.teacher_depth,
            teacher_beam=args.teacher_beam,
            workers=args.workers,
        )
        if args.command == "generate":
            return 0

    if args.command in {"full", "train"}:
        _train(epochs=args.epochs)
        if args.command == "train":
            return 0

    if args.command in {"full", "evaluate"}:
        _evaluate(MODEL, "placement-v2 evaluation")
        _evaluate(Path("data/models/neural-value-human.pt"), "current human-model baseline")
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
