from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path

from minoflux_ai import (
    CEMConfig,
    DEFAULT_WEIGHTS,
    REPLAY_FORMAT,
    SearchConfig,
    load_weights,
    run_heuristic_benchmark,
    save_replay,
    save_weights,
    train_cem,
)
from minoflux_engine import BOARD_HEIGHT, BOARD_WIDTH, HIDDEN_ROWS, PIECE_NAMES, VISIBLE_HEIGHT

from .run_store import RunStore
from .simulation import run_smoke


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _add_search_arguments(parser: ArgumentParser, *, lookahead_default: int) -> None:
    parser.add_argument(
        "--hold",
        action=BooleanOptionalAction,
        default=True,
        help="Include Hold placements (use --no-hold to disable)",
    )
    parser.add_argument(
        "--lookahead-pieces",
        type=int,
        default=lookahead_default,
        help="Future pieces considered beyond the current placement",
    )
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--lookahead-discount", type=float, default=0.90)


def _search_config_from_args(args) -> SearchConfig:
    return SearchConfig(
        allow_hold=args.hold,
        lookahead_pieces=args.lookahead_pieces,
        beam_width=args.beam_width,
        discount=args.lookahead_discount,
    ).normalized()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="minoflux", description="Pure-Python tetromino game and AI laboratory")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="Print engine capabilities")

    smoke = sub.add_parser("smoke", help="Run random legal-placement games")
    smoke.add_argument("--games", type=int, default=4)
    smoke.add_argument("--max-pieces", type=int, default=200)
    smoke.add_argument("--seed-base", type=int, default=1)
    smoke.add_argument("--save", action="store_true")

    benchmark = sub.add_parser("benchmark", help="Run the Hold-aware heuristic beam-search bot")
    benchmark.add_argument("--games", type=int, default=8)
    benchmark.add_argument("--max-pieces", type=int, default=500)
    benchmark.add_argument("--seed-base", type=int, default=1)
    benchmark.add_argument("--seed-step", type=int, default=31)
    benchmark.add_argument("--model", help="Path to a minoflux_heuristic_v1 JSON model")
    benchmark.add_argument("--replay-out", help=f"Write the best game as {REPLAY_FORMAT} JSON")
    _add_search_arguments(benchmark, lookahead_default=1)
    benchmark.add_argument("--save", action="store_true")

    cem = sub.add_parser("train-cem", help="Tune heuristic weights with the Cross-Entropy Method")
    cem.add_argument("--generations", type=int, default=8)
    cem.add_argument("--population", type=int, default=16)
    cem.add_argument("--elite-fraction", type=float, default=0.25)
    cem.add_argument("--games", type=int, default=3, help="Full training games per retained candidate")
    cem.add_argument("--max-pieces", type=int, default=200)
    cem.add_argument("--seed-base", type=int, default=1)
    cem.add_argument("--seed-step", type=int, default=31)
    cem.add_argument("--validation-games", type=int, default=4)
    cem.add_argument("--initial-sigma", type=float, default=0.35)
    cem.add_argument("--learning-rate", type=float, default=0.7)
    cem.add_argument("--random-seed", type=int, default=12345)
    cem.add_argument("--workers", type=int, default=0, help="Worker processes; 0 uses available CPUs")
    cem.add_argument("--screen-games", type=int, default=1, help="Short games used to reject weak candidates; 0 disables screening")
    cem.add_argument("--screen-max-pieces", type=int, default=60)
    cem.add_argument("--screen-fraction", type=float, default=0.5, help="Fraction retained for full evaluation")
    cem.add_argument("--initial-model", help="Optional starting minoflux_heuristic_v1 model")
    cem.add_argument("--model-out", help="Write the trained heuristic model")
    cem.add_argument("--replay-out", help=f"Write the best validation game as {REPLAY_FORMAT} JSON")
    _add_search_arguments(cem, lookahead_default=0)
    cem.add_argument("--save", action="store_true")

    replay = sub.add_parser("replay", help="Launch the Pygame replay viewer")
    replay.add_argument("path")
    replay.add_argument("--interval-ms", type=int, default=250)
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
            "ai": {
                "features": "minoflux_ai",
                "baseline": "heuristic",
                "search": ["hold candidates", "lookahead", "beam search"],
                "trainer": "cem",
                "acceleration": ["board-only placement simulation", "process workers", "candidate screening"],
                "modelFormat": "minoflux_heuristic_v1",
                "replayFormat": REPLAY_FORMAT,
            },
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
        search_config = _search_config_from_args(args)
        benchmark = run_heuristic_benchmark(
            max(1, args.games),
            max(1, args.max_pieces),
            args.seed_base,
            args.seed_step,
            weights,
            search_config,
            record_best_replay=True,
        )
        result = benchmark.to_dict()
        result["weights"] = weights.to_dict()
        replay_path: Path | None = None
        run = None
        if args.save:
            config = {**vars(args), "weights": weights.to_dict(), "searchConfig": search_config.to_dict()}
            run = RunStore().create("heuristic-benchmark", config)
        if benchmark.best_replay is not None:
            if args.replay_out:
                replay_path = save_replay(args.replay_out, benchmark.best_replay)
            elif run is not None:
                replay_path = save_replay(run.path / "best_replay.json", benchmark.best_replay)
        if run is not None:
            result["bestReplayPath"] = str(replay_path) if replay_path else None
            run.save_result(result)
            run.append_metric({
                "type": "complete",
                "meanPieces": result["meanPieces"],
                "meanLines": result["meanLines"],
                "topouts": result["topouts"],
                "searchConfig": search_config.to_dict(),
            })
            result["runPath"] = str(run.path)
        elif replay_path is not None:
            result["bestReplayPath"] = str(replay_path)
        _print(result)
        return 0
    if args.command == "train-cem":
        initial = load_weights(args.initial_model) if args.initial_model else DEFAULT_WEIGHTS
        search_config = _search_config_from_args(args)
        config = CEMConfig(
            generations=args.generations,
            population=args.population,
            elite_fraction=args.elite_fraction,
            games_per_candidate=args.games,
            max_pieces=args.max_pieces,
            seed_base=args.seed_base,
            seed_step=args.seed_step,
            validation_games=args.validation_games,
            initial_sigma=args.initial_sigma,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed,
            workers=args.workers,
            screen_games=args.screen_games,
            screen_max_pieces=args.screen_max_pieces,
            screen_fraction=args.screen_fraction,
            allow_hold=search_config.allow_hold,
            lookahead_pieces=search_config.lookahead_pieces,
            beam_width=search_config.beam_width,
            lookahead_discount=search_config.discount,
        )
        run = RunStore().create(
            "cem-training",
            {**vars(args), "initialWeights": initial.to_dict(), "searchConfig": search_config.to_dict()},
        ) if args.save else None

        def on_generation(generation) -> None:
            if run is not None:
                run.append_metric({"type": "generation", **generation.to_dict()})

        trained = train_cem(config, initial, on_generation)
        model_path: Path | None = None
        replay_path: Path | None = None
        if args.model_out:
            model_path = save_weights(args.model_out, trained.best_weights)
        elif run is not None:
            model_path = save_weights(run.path / "model.json", trained.best_weights)
        if trained.validation.best_replay is not None:
            if args.replay_out:
                replay_path = save_replay(args.replay_out, trained.validation.best_replay)
            elif run is not None:
                replay_path = save_replay(run.path / "best_validation_replay.json", trained.validation.best_replay)
        result = trained.to_dict()
        result["modelPath"] = str(model_path) if model_path else None
        result["bestReplayPath"] = str(replay_path) if replay_path else None
        if run is not None:
            run.save_result(result)
            run.append_metric({
                "type": "complete",
                "bestTrainingFitness": trained.best_training_fitness,
                "validationFitness": trained.validation_fitness,
                "workers": trained.workers,
                "elapsedSeconds": trained.elapsed_seconds,
                "searchConfig": search_config.to_dict(),
            })
            result["runPath"] = str(run.path)
        _print(result)
        return 0
    if args.command == "replay":
        from .replay import play_replay
        return play_replay(args.path, args.interval_ms)
    return 1
