from __future__ import annotations

import argparse
import json
import sys

from minoflux_ai.neural_train import NeuralTrainConfig
from minoflux_ai.neural_weighted_train import train_weighted_neural_value_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a MinoFlux neural value model with per-record training weights"
    )
    parser.add_argument("--dataset", default="data/neural/native-oracle-round2-weighted.jsonl")
    parser.add_argument("--human-dataset", default=None)
    parser.add_argument("--human-weight", type=float, default=5.0)
    parser.add_argument("--teacher-weight", type=float, default=0.25)
    parser.add_argument("--rollout-weight", type=float, default=0.50)
    parser.add_argument("--output", default="data/models/native-oracle-value-r2w.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Force deterministic PyTorch/CUDA training. "
            "This may be slower and fails if an operation has no deterministic implementation."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def progress(epoch: int, loss: float) -> None:
        print(f"epoch {epoch}: ranking_loss={loss:.6f}", file=sys.stderr, flush=True)

    result = train_weighted_neural_value_model(
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
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
