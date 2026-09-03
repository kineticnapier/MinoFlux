from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
import json
from pathlib import Path

from minoflux_ai.heuristic import DEFAULT_WEIGHTS, load_weights
from minoflux_ai.neural import NeuralValueEvaluator
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_neural import (
    VersusSelfPlayConfig,
    VersusTrainConfig,
    VersusValueEvaluator,
    generate_versus_selfplay_dataset,
    train_versus_value_model,
)
from minoflux_ai.versus_search import VersusSearchConfig

DEFAULT_SOLO_MODEL = "data/models/neural-value-human.pt"
DEFAULT_VERSUS_MODEL = "data/models/versus-value.pt"
DEFAULT_SELFPLAY = "data/neural/versus-selfplay.jsonl"


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _add_search_args(parser: ArgumentParser) -> None:
    parser.add_argument("--candidate-width", type=int, default=16)
    parser.add_argument("--reply-width", type=int, default=4)
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--hold", action=BooleanOptionalAction, default=True)
    parser.add_argument("--allow-180", action=BooleanOptionalAction, default=False)
    parser.add_argument("--reachability-nodes", type=int, default=8000)


def _search_config(args) -> VersusSearchConfig:
    placement = SearchConfig(
        allow_hold=args.hold,
        lookahead_pieces=args.lookahead,
        beam_width=args.beam,
        discount=args.discount,
        srs_reachable=True,
        allow_180=args.allow_180,
        reachability_node_limit=args.reachability_nodes,
    ).normalized()
    return VersusSearchConfig(
        placement_search=placement,
        candidate_width=args.candidate_width,
        opponent_reply_width=args.reply_width,
    ).normalized()


def _load_solo(path: str | None, args, cache: dict[str, NeuralValueEvaluator]):
    if not path:
        return None
    key = str(Path(path))
    cached = cache.get(key)
    if cached is not None:
        return cached
    evaluator = NeuralValueEvaluator.from_checkpoint(
        path,
        device=args.device,
        precision=args.precision,
        compile_model=args.torch_compile,
    )
    cache[key] = evaluator
    return evaluator


def _load_versus_value(path: str | None, args, cache: dict[str, VersusValueEvaluator]):
    if not path:
        return None
    key = str(Path(path))
    cached = cache.get(key)
    if cached is not None:
        return cached
    evaluator = VersusValueEvaluator.from_checkpoint(
        path,
        device=args.device,
        compile_model=args.torch_compile,
    )
    cache[key] = evaluator
    return evaluator


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="minoflux-versus-neural",
        description="Neural versus search, mirrored benchmarks, self-play, and match-value training",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    benchmark = sub.add_parser("benchmark", help="Run mirrored neural-versus-baseline matches")
    benchmark.add_argument("--games", type=int, default=20)
    benchmark.add_argument("--max-turns", type=int, default=500)
    benchmark.add_argument("--seed-base", type=int, default=7_000_001)
    benchmark.add_argument("--seed-step", type=int, default=31)
    benchmark.add_argument("--garbage-cap", type=int, default=8)
    benchmark.add_argument("--player-neural-model", default=DEFAULT_SOLO_MODEL)
    benchmark.add_argument("--ai-neural-model", default=None, help="Omit to use heuristic candidate ranking")
    benchmark.add_argument("--player-versus-value-model", default=None)
    benchmark.add_argument("--ai-versus-value-model", default=None)
    benchmark.add_argument("--player-heuristic-model", default=None)
    benchmark.add_argument("--ai-heuristic-model", default=None)
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--precision", choices=("float32", "float16", "bfloat16", "auto"), default="float32")
    benchmark.add_argument("--torch-compile", action="store_true")
    _add_search_args(benchmark)

    selfplay = sub.add_parser("selfplay", help="Generate win/loss-labelled versus self-play states")
    selfplay.add_argument("--output", default=DEFAULT_SELFPLAY)
    selfplay.add_argument("--games", type=int, default=50)
    selfplay.add_argument("--max-turns", type=int, default=500)
    selfplay.add_argument("--seed-base", type=int, default=6_000_001)
    selfplay.add_argument("--seed-step", type=int, default=31)
    selfplay.add_argument("--garbage-cap", type=int, default=8)
    selfplay.add_argument("--solo-model", default=DEFAULT_SOLO_MODEL)
    selfplay.add_argument("--versus-value-model", default=None, help="Optional previous match-value checkpoint for iterative self-play")
    selfplay.add_argument("--heuristic-model", default=None)
    selfplay.add_argument("--device", default="auto")
    selfplay.add_argument("--precision", choices=("float32", "float16", "bfloat16", "auto"), default="float32")
    selfplay.add_argument("--torch-compile", action="store_true")
    _add_search_args(selfplay)

    train = sub.add_parser("train", help="Train a two-board match-value network from self-play")
    train.add_argument("dataset", nargs="?", default=DEFAULT_SELFPLAY)
    train.add_argument("--output", default=DEFAULT_VERSUS_MODEL)
    train.add_argument("--resume", default=None, help="Optional previous versus-value checkpoint")
    train.add_argument("--epochs", type=int, default=6)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--validation-fraction", type=float, default=0.10)
    train.add_argument("--teacher-weight", type=float, default=0.25)
    train.add_argument("--seed", type=int, default=20260903)
    train.add_argument("--device", default="auto")

    return parser


def _benchmark(args) -> int:
    solo_cache: dict[str, NeuralValueEvaluator] = {}
    value_cache: dict[str, VersusValueEvaluator] = {}
    player_scorer = _load_solo(args.player_neural_model, args, solo_cache)
    ai_scorer = _load_solo(args.ai_neural_model, args, solo_cache)
    player_value = _load_versus_value(args.player_versus_value_model, args, value_cache)
    ai_value = _load_versus_value(args.ai_versus_value_model, args, value_cache)
    player_weights = load_weights(args.player_heuristic_model) if args.player_heuristic_model else DEFAULT_WEIGHTS
    ai_weights = load_weights(args.ai_heuristic_model) if args.ai_heuristic_model else DEFAULT_WEIGHTS
    search_config = _search_config(args)
    result = run_versus_benchmark(
        args.games,
        max_turns=args.max_turns,
        seed_base=args.seed_base,
        seed_step=args.seed_step,
        player_weights=player_weights,
        ai_weights=ai_weights,
        player_config=search_config,
        ai_config=search_config,
        garbage_cap=args.garbage_cap,
        player_scorer=player_scorer,
        ai_scorer=ai_scorer,
        player_state_scorer=player_value,
        ai_state_scorer=ai_value,
    ).to_dict()
    result["playerPolicy"] = {
        "soloNeural": args.player_neural_model,
        "versusValue": args.player_versus_value_model,
    }
    result["aiPolicy"] = {
        "soloNeural": args.ai_neural_model,
        "versusValue": args.ai_versus_value_model,
    }
    result["versusSearchConfig"] = search_config.to_dict()
    _print(result)
    return 0


def _selfplay(args) -> int:
    solo_cache: dict[str, NeuralValueEvaluator] = {}
    value_cache: dict[str, VersusValueEvaluator] = {}
    solo = _load_solo(args.solo_model, args, solo_cache)
    if solo is None:
        raise SystemExit("--solo-model is required for neural self-play")
    value = _load_versus_value(args.versus_value_model, args, value_cache)
    weights = load_weights(args.heuristic_model) if args.heuristic_model else DEFAULT_WEIGHTS
    result = generate_versus_selfplay_dataset(
        args.output,
        solo,
        VersusSelfPlayConfig(
            games=args.games,
            max_turns=args.max_turns,
            seed_base=args.seed_base,
            seed_step=args.seed_step,
            garbage_cap=args.garbage_cap,
            search_config=_search_config(args),
        ),
        heuristic_weights=weights,
        value_scorer=value,
    )
    result["soloModel"] = args.solo_model
    result["versusValueModel"] = args.versus_value_model
    _print(result)
    return 0


def _train(args) -> int:
    result = train_versus_value_model(
        args.dataset,
        args.output,
        VersusTrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            validation_fraction=args.validation_fraction,
            teacher_weight=args.teacher_weight,
            seed=args.seed,
            device=args.device,
        ),
        resume_from=args.resume,
    )
    _print(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        return _benchmark(args)
    if args.command == "selfplay":
        return _selfplay(args)
    if args.command == "train":
        return _train(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
