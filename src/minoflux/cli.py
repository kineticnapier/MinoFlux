from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path

from minoflux_ai import (
    CEMConfig,
    DEFAULT_WEIGHTS,
    FITNESS_PROFILE_ATTACK_SPIN,
    FITNESS_PROFILE_NAMES,
    PromotionConfig,
    REPLAY_FORMAT,
    SearchConfig,
    benchmark_fitness,
    bootstrap_champion,
    evaluate_and_promote_model,
    load_weights,
    resolve_fitness_profile,
    run_heuristic_benchmark,
    save_replay,
    save_weights,
    train_cem,
)
from minoflux_engine import BOARD_HEIGHT, BOARD_WIDTH, HIDDEN_ROWS, PIECE_NAMES, VISIBLE_HEIGHT

from .run_store import RunStore
from .simulation import run_smoke

CHAMPION_MODEL = Path("data/models/champion-cem.json")
CANDIDATE_MODEL = Path("data/models/candidate-cem.json")
LEGACY_LATEST_MODEL = Path("data/models/latest-cem.json")
MODEL_HISTORY_DIR = Path("data/models/history")
RECOVERED_ATTACK_MODEL = Path("presets/recovered-attack-20260723.json")


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
    benchmark.add_argument("--workers", type=int, default=0, help="Parallel game workers; 0 uses available CPUs")
    benchmark.add_argument("--model", help="Path to a minoflux_heuristic_v1 JSON model")
    benchmark.add_argument("--fitness-profile", choices=FITNESS_PROFILE_NAMES, default=FITNESS_PROFILE_ATTACK_SPIN)
    benchmark.add_argument("--replay-out", help=f"Write the best game as {REPLAY_FORMAT} JSON")
    _add_search_arguments(benchmark, lookahead_default=1)
    benchmark.add_argument("--save", action="store_true")

    cem = sub.add_parser("train-cem", help="Tune heuristic weights with the Cross-Entropy Method")
    cem.add_argument("--generations", type=int, default=10)
    cem.add_argument("--population", type=int, default=24)
    cem.add_argument("--elite-fraction", type=float, default=0.25)
    cem.add_argument("--games", type=int, default=3, help="Full training games per retained candidate")
    cem.add_argument("--max-pieces", type=int, default=300)
    cem.add_argument("--seed-base", type=int, default=1)
    cem.add_argument("--seed-step", type=int, default=31)
    cem.add_argument("--validation-games", type=int, default=6)
    cem.add_argument("--initial-sigma", type=float, default=0.25)
    cem.add_argument("--learning-rate", type=float, default=0.7)
    cem.add_argument("--random-seed", type=int, default=12345)
    cem.add_argument("--workers", type=int, default=0, help="Worker processes; 0 uses available CPUs")
    cem.add_argument("--screen-games", type=int, default=1, help="Short games used to reject weak candidates; 0 disables screening")
    cem.add_argument("--screen-max-pieces", type=int, default=60)
    cem.add_argument("--screen-fraction", type=float, default=0.5, help="Fraction retained for full evaluation")
    cem.add_argument("--fitness-profile", choices=FITNESS_PROFILE_NAMES, default=FITNESS_PROFILE_ATTACK_SPIN)
    cem.add_argument("--initial-model", help="Optional starting minoflux_heuristic_v1 model")
    cem.add_argument("--model-out", help="Write the trained candidate model")
    cem.add_argument("--replay-out", help=f"Write the best validation game as {REPLAY_FORMAT} JSON")
    cem.add_argument("--promote", action=BooleanOptionalAction, default=True)
    cem.add_argument("--champion-model", default=str(CHAMPION_MODEL))
    cem.add_argument("--promotion-games", type=int, default=10)
    cem.add_argument("--promotion-max-pieces", type=int, default=1000)
    cem.add_argument("--promotion-min-gain", type=float, default=0.0)
    cem.add_argument("--promotion-max-completion-loss", type=int, default=1)
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
                "fitnessProfiles": list(FITNESS_PROFILE_NAMES),
                "modelPromotion": "candidate versus champion on unseen seeds",
                "acceleration": [
                    "board-only placement simulation",
                    "bounded top-k beam ranking",
                    "parallel benchmark games",
                    "process workers",
                    "candidate screening",
                ],
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
        bootstrap_champion(CHAMPION_MODEL, recovery_path=RECOVERED_ATTACK_MODEL, legacy_path=LEGACY_LATEST_MODEL)
        weights = load_weights(args.model) if args.model else (
            load_weights(CHAMPION_MODEL) if CHAMPION_MODEL.is_file() else DEFAULT_WEIGHTS
        )
        search_config = _search_config_from_args(args)
        profile = resolve_fitness_profile(args.fitness_profile)
        benchmark = run_heuristic_benchmark(
            max(1, args.games),
            max(1, args.max_pieces),
            args.seed_base,
            args.seed_step,
            weights,
            search_config,
            workers=args.workers,
            record_best_replay=True,
        )
        result = benchmark.to_dict()
        result["weights"] = weights.to_dict()
        result["fitnessProfile"] = profile.to_dict()
        result["fitness"] = benchmark_fitness(benchmark, profile)
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
                "meanAttack": result["meanAttack"],
                "meanSpins": result["meanSpins"],
                "meanSpinLines": result["meanSpinLines"],
                "fitness": result["fitness"],
                "topouts": result["topouts"],
                "workers": benchmark.workers,
                "searchConfig": search_config.to_dict(),
            })
            result["runPath"] = str(run.path)
        elif replay_path is not None:
            result["bestReplayPath"] = str(replay_path)
        _print(result)
        return 0
    if args.command == "train-cem":
        champion_path = Path(args.champion_model)
        bootstrap_champion(champion_path, recovery_path=RECOVERED_ATTACK_MODEL, legacy_path=LEGACY_LATEST_MODEL)
        initial = load_weights(args.initial_model) if args.initial_model else (
            load_weights(champion_path) if champion_path.is_file() else DEFAULT_WEIGHTS
        )
        search_config = _search_config_from_args(args)
        profile = resolve_fitness_profile(args.fitness_profile)
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
            fitness_profile=profile.name,
        )
        run = RunStore().create(
            "cem-training",
            {**vars(args), "initialWeights": initial.to_dict(), "searchConfig": search_config.to_dict()},
        ) if args.save else None

        def on_generation(generation) -> None:
            if run is not None:
                run.append_metric({"type": "generation", **generation.to_dict()})

        trained = train_cem(config, initial, on_generation)
        candidate_path = save_weights(args.model_out or CANDIDATE_MODEL, trained.best_weights)
        replay_path: Path | None = None
        if trained.validation.best_replay is not None:
            if args.replay_out:
                replay_path = save_replay(args.replay_out, trained.validation.best_replay)
            elif run is not None:
                replay_path = save_replay(run.path / "best_validation_replay.json", trained.validation.best_replay)

        promotion = None
        if args.promote:
            promotion = evaluate_and_promote_model(
                trained.best_weights,
                search_config,
                champion_path=champion_path,
                candidate_path=candidate_path,
                history_dir=MODEL_HISTORY_DIR,
                compatibility_latest_path=LEGACY_LATEST_MODEL,
                fitness_profile=profile,
                config=PromotionConfig(
                    games=args.promotion_games,
                    max_pieces=args.promotion_max_pieces,
                    seed_base=args.seed_base + 2_000_033,
                    seed_step=97,
                    minimum_fitness_gain=args.promotion_min_gain,
                    max_completion_loss=args.promotion_max_completion_loss,
                    workers=0,
                ),
            )

        result = trained.to_dict()
        result["candidateModelPath"] = str(candidate_path)
        result["championModelPath"] = str(champion_path)
        result["bestReplayPath"] = str(replay_path) if replay_path else None
        result["promotion"] = promotion.to_dict() if promotion is not None else None
        if run is not None:
            run.save_result(result)
            run.append_metric({
                "type": "complete",
                "bestTrainingFitness": trained.best_training_fitness,
                "validationFitness": trained.validation_fitness,
                "workers": trained.workers,
                "elapsedSeconds": trained.elapsed_seconds,
                "promoted": promotion.promoted if promotion is not None else None,
                "searchConfig": search_config.to_dict(),
            })
            result["runPath"] = str(run.path)
        _print(result)
        return 0
    if args.command == "replay":
        from .replay import play_replay
        return play_replay(args.path, args.interval_ms)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
