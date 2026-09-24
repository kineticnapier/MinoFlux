from __future__ import annotations

import argparse
import json

from minoflux_ai.neural_mix import merge_neural_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge MinoFlux neural ranking datasets")
    parser.add_argument("inputs", nargs="+", help="Input JSONL datasets in precedence order")
    parser.add_argument("--output", default="data/neural/mixed-ranking.jsonl")
    parser.add_argument("--keep-duplicates", action="store_true")
    parser.add_argument(
        "--weights",
        default=None,
        help="Comma-separated training weights aligned with input datasets",
    )
    return parser


def _parse_weights(raw: str | None, input_count: int) -> list[float] | None:
    if raw is None:
        return None
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as error:
        raise SystemExit("--weights must be a comma-separated list of numbers") from error
    if len(values) != input_count:
        raise SystemExit(
            f"--weights expected {input_count} values for {input_count} inputs, got {len(values)}"
        )
    return values


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = merge_neural_datasets(
        args.output,
        args.inputs,
        deduplicate=not args.keep_duplicates,
        input_weights=_parse_weights(args.weights, len(args.inputs)),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
