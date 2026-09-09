from __future__ import annotations

from argparse import ArgumentParser, BooleanOptionalAction
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from minoflux_ai.heuristic import DEFAULT_WEIGHTS, load_weights
from minoflux_ai.neural import NeuralValueEvaluator
from minoflux_ai.neural_promotion import (
    NeuralModelSpec,
    build_neural_promotion_report,
    reverse_versus_benchmark_result,
    run_neural_solo_benchmark,
    same_neural_model,
    summarize_paired_versus,
)
from minoflux_ai.progress import progress_bar
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_benchmark import run_versus_benchmark
from minoflux_ai.versus_neural import (
    VersusSelfPlayConfig,
    VersusTrainConfig,
    VersusValueEvaluator,
)
from minoflux_ai.versus_progress import (
    generate_versus_selfplay_dataset_progress,
    train_versus_value_model_progress,
)
from minoflux_ai.versus_search import VersusSearchConfig
from minoflux_ai.versus_profile import collect_versus_profile

DEFAULT_SOLO_MODEL = "data/models/neural-value-human.pt"
DEFAULT_VERSUS_MODEL = "data/models/versus-value.pt"
DEFAULT_SELFPLAY = "data/neural/versus-selfplay.jsonl"


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _execution_note(payload: dict[str, object]) -> str:
    if payload.get("skipped"):
        return f"skipped: {payload.get('reason', 'skipped')}"
    if payload.get("reused"):
        return f"reused from {payload.get('reusedFrom', '?')}"
    return "executed"


def _promotion_summary(report: dict[str, object]) -> str:
    lines = ["Neural promotion benchmark", "Solo:"]
    solo = report["solo"]
    assert isinstance(solo, dict)
    for role in ("candidate", "champion", "reference"):
        payload = solo[role]
        assert isinstance(payload, dict)
        model_name = str(payload.get("modelName", role))
        app = float(payload.get("attackPerPiece", 0.0))
        survived = float(payload.get("meanPiecesSurvived", 0.0))
        completion = 100.0 * float(payload.get("completionRate", 0.0))
        topouts = int(payload.get("topouts", 0))
        games = int(payload.get("games", 0))
        lines.append(
            f"  {role} ({model_name}): APP {app:.4f}, survival {survived:.1f}, "
            f"completion {completion:.1f}%, topouts {topouts}/{games} "
            f"[{_execution_note(payload)}]"
        )

    lines.append("Versus:")
    versus = report["versus"]
    assert isinstance(versus, dict)
    for label, raw_payload in versus.items():
        assert isinstance(raw_payload, dict)
        payload = raw_payload
        if payload.get("skipped"):
            lines.append(f"  {label}: skipped ({payload.get('reason', 'skipped')})")
            continue
        model_a = str(payload.get("modelA", "A"))
        model_b = str(payload.get("modelB", "B"))
        a_rate = 100.0 * float(payload.get("aWinRate", 0.0))
        a_wins = int(payload.get("aWins", 0))
        b_wins = int(payload.get("bWins", 0))
        draws = int(payload.get("draws", 0))
        pairs = int(payload.get("pairs", 0))
        lines.append(
            f"  {label}: {model_a} vs {model_b}, {model_a} {a_rate:.1f}% "
            f"({a_wins}-{b_wins}-{draws}), pairs {pairs} "
            f"[{_execution_note(payload)}]"
        )

    lines.append(f"Report: {report.get('outputPath', '-')}")
    return "\n".join(lines)


def _print_promotion_report(
    report: dict[str, object],
    *,
    print_json: bool = False,
    pretty_json: bool = False,
) -> None:
    if print_json:
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
        return
    if pretty_json:
        _print(report)
        return
    print(_promotion_summary(report))


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
    bar = progress_bar(total=1, desc=f"Load solo model: {Path(path).name}", unit="model")
    try:
        evaluator = NeuralValueEvaluator.from_checkpoint(
            path,
            device=args.device,
            precision=args.precision,
            compile_model=args.torch_compile,
        )
        cache[key] = evaluator
        bar.update(1)
        return evaluator
    finally:
        bar.close()


def _load_versus_value(path: str | None, args, cache: dict[str, VersusValueEvaluator]):
    if not path:
        return None
    key = str(Path(path))
    cached = cache.get(key)
    if cached is not None:
        return cached
    bar = progress_bar(total=1, desc=f"Load versus model: {Path(path).name}", unit="model")
    try:
        evaluator = VersusValueEvaluator.from_checkpoint(
            path,
            device=args.device,
            compile_model=args.torch_compile,
        )
        cache[key] = evaluator
        bar.update(1)
        return evaluator
    finally:
        bar.close()


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
    benchmark.add_argument(
        "--game-batch",
        type=int,
        default=1,
        help="Concurrent games sharing each neural forward pass (1 keeps the serial reference path)",
    )
    benchmark.add_argument(
        "--profile",
        action="store_true",
        help="Collect detailed versus-search CPU timings",
    )
    _add_search_args(benchmark)

    promotion = sub.add_parser(
        "promotion",
        help="Compare a neural candidate against stable and aggressive baselines without auto-promoting",
    )
    promotion.add_argument("--candidate", required=True)
    promotion.add_argument(
        "--champion",
        default="data/models/placement-v2-human3-home.pt",
        help="Stable current champion baseline",
    )
    promotion.add_argument(
        "--reference",
        default="data/models/placement-v2-human4-e5.pt",
        help="Aggressive reference/challenger baseline",
    )
    promotion.add_argument("--candidate-name", default=None)
    promotion.add_argument("--champion-name", default="human3")
    promotion.add_argument("--reference-name", default="e5")
    promotion.add_argument("--output", default=None)
    output_mode = promotion.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--print-json",
        action="store_true",
        help="Print the complete report as compact one-line JSON",
    )
    output_mode.add_argument(
        "--pretty-json",
        action="store_true",
        help="Print the complete report as indented JSON (legacy stdout format)",
    )
    promotion.add_argument(
        "--solo-games",
        type=int,
        default=100,
        help="Solo games per unique checkpoint; 500 is recommended for a standard promotion baseline",
    )
    promotion.add_argument("--solo-max-pieces", type=int, default=300)
    promotion.add_argument("--solo-seed-base", type=int, default=8_100_001)
    promotion.add_argument("--solo-seed-step", type=int, default=31)
    promotion.add_argument("--solo-game-batch", type=int, default=20)
    promotion.add_argument(
        "--versus-pairs",
        type=int,
        default=50,
        help=(
            "Same-seed mirrored pairs; each pair runs A-left/B-right and B-left/A-right. "
            "300 pairs (600 games) is recommended for a standard promotion baseline"
        ),
    )
    promotion.add_argument("--versus-max-turns", type=int, default=500)
    promotion.add_argument("--versus-seed-base", type=int, default=8_200_001)
    promotion.add_argument("--versus-seed-step", type=int, default=31)
    promotion.add_argument("--garbage-cap", type=int, default=8)
    promotion.add_argument("--heuristic-model", default=None)
    promotion.add_argument("--device", default="auto")
    promotion.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16", "auto"),
        default="float32",
    )
    promotion.add_argument("--torch-compile", action="store_true")
    promotion.add_argument(
        "--game-batch",
        type=int,
        default=1,
        help="Concurrent versus games sharing neural forwards; 1 is the serial reference",
    )
    promotion.add_argument(
        "--progress",
        action=BooleanOptionalAction,
        default=True,
    )
    promotion.add_argument(
        "--baseline-match",
        action=BooleanOptionalAction,
        default=True,
        help="Also measure human3 vs e5 so candidate results have a baseline distribution",
    )
    promotion.add_argument(
        "--same-model-match",
        action=BooleanOptionalAction,
        default=False,
        help="Run same-checkpoint versus matchups instead of skipping them (default: skip)",
    )
    _add_search_args(promotion)

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
    selfplay.add_argument(
        "--game-batch",
        type=int,
        default=1,
        help="Concurrent rolling self-play games sharing each neural forward pass",
    )
    selfplay.add_argument(
        "--profile",
        action="store_true",
        help="Collect detailed versus-search CPU timings",
    )
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
    profile_context = collect_versus_profile() if args.profile else nullcontext(None)
    with profile_context as profile:
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
            progress=True,
            game_batch=args.game_batch,
        ).to_dict()
    result["gameBatch"] = max(1, int(args.game_batch))
    result["playerPolicy"] = {
        "soloNeural": args.player_neural_model,
        "versusValue": args.player_versus_value_model,
    }
    result["aiPolicy"] = {
        "soloNeural": args.ai_neural_model,
        "versusValue": args.ai_versus_value_model,
    }
    result["versusSearchConfig"] = search_config.to_dict()
    if profile is not None:
        result["versusProfile"] = profile.to_dict()
        print(profile.format_table(), file=sys.stderr)
    _print(result)
    return 0


def _promotion(args) -> int:
    solo_loader_cache: dict[str, NeuralValueEvaluator] = {}
    candidate = NeuralModelSpec.from_path(args.candidate, name=args.candidate_name)
    champion = NeuralModelSpec.from_path(args.champion, name=args.champion_name)
    reference = NeuralModelSpec.from_path(args.reference, name=args.reference_name)
    specs = {
        "candidate": candidate,
        "champion": champion,
        "reference": reference,
    }
    role_order = ("candidate", "champion", "reference")

    representatives: dict[str, str] = {}
    scorers: dict[str, NeuralValueEvaluator] = {}
    for key in role_order:
        reused_from = next(
            (
                previous
                for previous in role_order
                if previous in representatives and same_neural_model(specs[key], specs[previous])
            ),
            None,
        )
        if reused_from is not None:
            representative = representatives[reused_from]
            representatives[key] = representative
            scorers[key] = scorers[reused_from]
            continue
        scorer = _load_solo(specs[key].path, args, solo_loader_cache)
        if scorer is None:
            raise SystemExit("promotion benchmark requires all three neural model paths")
        representatives[key] = key
        scorers[key] = scorer

    weights = load_weights(args.heuristic_model) if args.heuristic_model else DEFAULT_WEIGHTS
    versus_config = _search_config(args)
    solo_config = versus_config.placement_search

    solo_results = {}
    solo_execution: dict[str, dict[str, object]] = {}
    solo_by_representative = {}
    for key in role_order:
        representative = representatives[key]
        existing = solo_by_representative.get(representative)
        if existing is not None:
            solo_results[key] = existing
            solo_execution[key] = {
                "executed": False,
                "reused": True,
                "skipped": False,
                "reusedFrom": representative,
            }
            continue
        result = run_neural_solo_benchmark(
            scorers[key],
            model=specs[key].path,
            games=args.solo_games,
            max_pieces=args.solo_max_pieces,
            seed_base=args.solo_seed_base,
            seed_step=args.solo_seed_step,
            game_batch_size=args.solo_game_batch,
            weights=weights,
            search_config=solo_config,
            progress=args.progress,
        )
        solo_by_representative[representative] = result
        solo_results[key] = result
        solo_execution[key] = {
            "executed": True,
            "reused": False,
            "skipped": False,
        }

    versus_games = max(1, int(args.versus_pairs)) * 2
    versus_conditions_key = (
        versus_games,
        max(1, int(args.versus_max_turns)),
        int(args.versus_seed_base),
        int(args.versus_seed_step),
        max(1, int(args.garbage_cap)),
        max(1, int(args.game_batch)),
        json.dumps(versus_config.to_dict(), sort_keys=True),
        str(args.heuristic_model or ""),
    )
    versus_cache: dict[
        tuple[tuple[str, str], tuple[object, ...]],
        tuple[str, str, str, object],
    ] = {}

    def paired(label: str, a_key: str, b_key: str) -> tuple[str, dict[str, object]]:
        rep_a = representatives[a_key]
        rep_b = representatives[b_key]
        if rep_a == rep_b and not args.same_model_match:
            return label, {
                "modelA": specs[a_key].name,
                "modelB": specs[b_key].name,
                "games": 0,
                "pairs": 0,
                "executed": False,
                "reused": False,
                "skipped": True,
                "reason": "same-model matchup",
            }

        canonical_pair = tuple(sorted((rep_a, rep_b)))
        cache_key = (canonical_pair, versus_conditions_key)
        cached = versus_cache.get(cache_key)
        if cached is not None:
            source_label, source_a, source_b, raw = cached
            reversed_from_source = (source_a, source_b) != (rep_a, rep_b)
            effective = (
                reverse_versus_benchmark_result(raw)  # type: ignore[arg-type]
                if reversed_from_source
                else raw
            )
            summary = summarize_paired_versus(
                effective,  # type: ignore[arg-type]
                model_a=specs[a_key].name,
                model_b=specs[b_key].name,
            )
            summary.update(
                {
                    "executed": False,
                    "reused": True,
                    "skipped": False,
                    "reusedFrom": source_label,
                    "reversedFromSource": reversed_from_source,
                }
            )
            return label, summary

        raw = run_versus_benchmark(
            versus_games,
            max_turns=args.versus_max_turns,
            seed_base=args.versus_seed_base,
            seed_step=args.versus_seed_step,
            player_weights=weights,
            ai_weights=weights,
            player_config=versus_config,
            ai_config=versus_config,
            garbage_cap=args.garbage_cap,
            player_scorer=scorers[a_key],
            ai_scorer=scorers[b_key],
            progress=args.progress,
            game_batch=args.game_batch,
        )
        versus_cache[cache_key] = (label, rep_a, rep_b, raw)
        summary = summarize_paired_versus(
            raw,
            model_a=specs[a_key].name,
            model_b=specs[b_key].name,
        )
        summary.update(
            {
                "executed": True,
                "reused": False,
                "skipped": False,
            }
        )
        return label, summary

    versus_results = dict([
        paired("candidateVsChampion", "candidate", "champion"),
        paired("candidateVsReference", "candidate", "reference"),
    ])
    if args.baseline_match:
        label, value = paired("championVsReference", "champion", "reference")
        versus_results[label] = value

    conditions = {
        "solo": {
            "games": max(1, int(args.solo_games)),
            "maxPieces": max(1, int(args.solo_max_pieces)),
            "seedBase": int(args.solo_seed_base),
            "seedStep": int(args.solo_seed_step),
            "gameBatch": max(1, int(args.solo_game_batch)),
        },
        "versus": {
            "pairs": max(1, int(args.versus_pairs)),
            "games": versus_games,
            "seedBase": int(args.versus_seed_base),
            "seedStep": int(args.versus_seed_step),
            "maxTurns": max(1, int(args.versus_max_turns)),
            "garbageCap": max(1, int(args.garbage_cap)),
            "gameBatch": max(1, int(args.game_batch)),
            "sideSwap": True,
            "sameSeedPerPair": True,
            "baselineMatch": bool(args.baseline_match),
            "sameModelMatch": bool(args.same_model_match),
        },
        "recommendedBaseline": {
            "soloGamesPerUniqueModel": 500,
            "versusPairsPerUniqueMatchup": 300,
        },
        "searchConfig": versus_config.to_dict(),
        "inference": {
            "device": args.device,
            "precision": args.precision,
            "torchCompile": bool(args.torch_compile),
        },
    }
    report = build_neural_promotion_report(
        candidate=candidate,
        champion=champion,
        reference=reference,
        solo=solo_results,
        versus=versus_results,
        conditions=conditions,
    )
    for key in role_order:
        payload = report["solo"][key]
        payload["model"] = specs[key].path
        payload["modelName"] = specs[key].name
        payload.update(solo_execution[key])

    output = args.output
    if not output:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        output = f"data/benchmarks/neural-promotion-{stamp}.json"
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    report["outputPath"] = str(target)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _print_promotion_report(
        report,
        print_json=bool(args.print_json),
        pretty_json=bool(args.pretty_json),
    )
    return 0


def _selfplay(args) -> int:
    solo_cache: dict[str, NeuralValueEvaluator] = {}
    value_cache: dict[str, VersusValueEvaluator] = {}
    solo = _load_solo(args.solo_model, args, solo_cache)
    if solo is None:
        raise SystemExit("--solo-model is required for neural self-play")
    value = _load_versus_value(args.versus_value_model, args, value_cache)
    weights = load_weights(args.heuristic_model) if args.heuristic_model else DEFAULT_WEIGHTS
    profile_context = collect_versus_profile() if args.profile else nullcontext(None)
    with profile_context as profile:
        result = generate_versus_selfplay_dataset_progress(
            args.output,
            solo,
            VersusSelfPlayConfig(
                games=args.games,
                max_turns=args.max_turns,
                seed_base=args.seed_base,
                seed_step=args.seed_step,
                garbage_cap=args.garbage_cap,
                search_config=_search_config(args),
                game_batch=args.game_batch,
            ),
            heuristic_weights=weights,
            value_scorer=value,
        )
    result["soloModel"] = args.solo_model
    result["versusValueModel"] = args.versus_value_model
    if profile is not None:
        result["versusProfile"] = profile.to_dict()
        print(profile.format_table(), file=sys.stderr)
    _print(result)
    return 0


def _train(args) -> int:
    result = train_versus_value_model_progress(
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
    if args.command == "promotion":
        return _promotion(args)
    if args.command == "selfplay":
        return _selfplay(args)
    if args.command == "train":
        return _train(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
