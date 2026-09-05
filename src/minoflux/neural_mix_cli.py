from __future__ import annotations

import argparse
import json

from minoflux_ai.neural_mix import merge_neural_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge MinoFlux neural ranking datasets")
    parser.add_argument("inputs", nargs="+", help="Input JSONL datasets in precedence order")
    parser.add_argument("--output", default="data/neural/mixed-ranking.jsonl")
    parser.add_argument("--keep-duplicates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = merge_neural_datasets(
        args.output,
        args.inputs,
        deduplicate=not args.keep_duplicates,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
