from __future__ import annotations

from argparse import ArgumentParser
import json

from minoflux_ai import DEFAULT_WEIGHTS, load_weights, run_heuristic_benchmark
from minoflux_engine import BOARD_HEIGHT, BOARD_WIDTH, HIDDEN_ROWS, PIECE_NAMES, VISIBLE_HEIGHT
from .run_store import RunStore
from .simulation import run_smoke


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="minoflux", description="Pure-Python tetromino game and AI laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Print engine capabilities")

    smoke = sub.add_parser("smoke", help="Run random legal-placement games")
    smoke.add_argument("--games", type=int, default=4)
    smoke.add_argument("--max-pieces", type=int, default=200)
    smoke.add_argument("--seed-base", type=int, default=1)
    smoke.add_argument("--save", action="store_true")

    benchmark = sub.add_parser("benchmark", help="Run the deterministic heuristic placement bot")
    benchmark.add_argument("--games", type=int, default=8)
    benchmark.add_argument("--max-pieces", type=int, default=500)
    benchmark.add_argument("--seed-base", type=int, default=1)
    benchmark.add_argument("--seed-step", type=int, default=31)
    benchmark.add_argument("--model", help="Path to a minoflux_heuristic_v1 JSON model")
    benchmark.add_argument("--save", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "info":
        _print({
            "name": "MinoFlux",
            "engine": "minoflux_engine",
            "board": {"width": BOARD_WIDTH, "visibleHeight": VISIBLE_HEIGHT, "hiddenRows": HIDDEN_ROWS, "height": BOARD_HEIGHT},
            "pieces": PIECE_NAMES,
            "runtime": "pure-python",
            "ai": {"features": "minoflux_ai", "baseline": "heuristic", "modelFormat": "minoflux_heuristic_v1"},
        })
        return 0
    if args.command == "smoke":
        result = run_smoke(max(1, args.games), max(1, args.max_pieces), args.seed_base)
        if args.save:
            run = RunStore().create("engine-smoke", vars(args))
            run.save_result(result)
            run.append_metric({"type": "complete", "meanPieces": result["meanPieces"], "topouts": result["topouts"]})
            result["runPath"] = str(run.path)
        _print(result)
        return 0
    if args.command == "benchmark":
        weights = load_weights(args.model) if args.model else DEFAULT_WEIGHTS
        benchmark = run_heuristic_benchmark(
            max(1, args.games),
            max(1, args.max_pieces),
            args.seed_base,
            args.seed_step,
            weights,
        )
        result = benchmark.to_dict()
        result["weights"] = weights.to_dict()
        if args.save:
            config = {**vars(args), "weights": weights.to_dict()}
            run = RunStore().create("heuristic-benchmark", config)
            run.save_result(result)
            run.append_metric({
                "type": "complete",
                "meanPieces": result["meanPieces"],
                "meanLines": result["meanLines"],
                "topouts": result["topouts"],
            })
            result["runPath"] = str(run.path)
        _print(result)
        return 0
    return 1
