from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, apply_search_action, load_weights
from minoflux_ai.human_review import HumanReviewConfig, collect_neural_review_queue
from minoflux_ai.neural import NeuralValueEvaluator
from minoflux_ai.search import choose_search_actions_batch
from minoflux.human_review_pygame import launch_human_review_app
from minoflux_ai.neural_dataset import NeuralDatasetConfig, write_neural_ranking_dataset
from minoflux_ai.neural_train import NeuralTrainConfig, train_neural_value_model
from minoflux_engine import Game


def _search_config(args: argparse.Namespace) -> SearchConfig:
    return SearchConfig(
        allow_hold=not args.no_hold,
        lookahead_pieces=args.lookahead,
        beam_width=args.beam,
        discount=args.discount,
        srs_reachable=not args.no_srs,
        allow_180=args.allow_180,
        reachability_node_limit=args.reachability_nodes,
    )


def _weights(path: str | None):
    return DEFAULT_WEIGHTS if path is None else load_weights(path)


def _champion_weights(path: str | None):
    return load_weights(path or "data/models/champion-cem.json")


def _generate(args: argparse.Namespace) -> int:
    model_path = args.model or "data/models/champion-cem.json"
    weights = load_weights(model_path)

    def progress(samples: int, candidates: int) -> None:
        print(
            f"generated {samples:,} samples / {candidates:,} candidate states",
            file=sys.stderr,
            flush=True,
        )

    result = write_neural_ranking_dataset(
        args.output,
        NeuralDatasetConfig(
            games=args.games,
            max_pieces=args.max_pieces,
            seed_base=args.seed_base,
            seed_step=args.seed_step,
            max_candidates=args.max_candidates,
            sampling_mode=args.sampling_mode,
            hard_candidates=args.hard_candidates,
            medium_candidates=args.medium_candidates,
            random_candidates=args.random_candidates,
            bad_candidates=args.bad_candidates,
            random_seed=args.sampling_seed,
            teacher_lookahead=args.teacher_lookahead,
            teacher_beam_width=args.teacher_beam,
            teacher_acceptable_margin=args.teacher_acceptable_margin,
            teacher_score_candidates=args.teacher_score_candidates,
            rollout_horizon=args.rollout_horizon,
            rollout_candidates=args.rollout_candidates,
            rollout_lookahead=args.rollout_lookahead,
            rollout_beam_width=args.rollout_beam,
            search_config=_search_config(args),
        ),
        weights,
        progress=progress,
        progress_every=args.progress_every,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    def progress(epoch: int, loss: float) -> None:
        print(f"epoch {epoch}: ranking_loss={loss:.6f}", file=sys.stderr, flush=True)

    result = train_neural_value_model(
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
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    evaluator = NeuralValueEvaluator.from_checkpoint(
        args.model,
        device=args.device,
        precision=args.precision,
        compile_model=args.torch_compile,
    )
    weights = _weights(args.heuristic_model)
    config = _search_config(args)
    games = [
        Game(args.seed_base + game_index * args.seed_step)
        for game_index in range(args.games)
    ]
    done: set[int] = set()
    total_pieces = 0
    total_attack = 0
    topouts = 0
    completed = 0
    progress_bar = None
    if not args.no_progress:
        try:
            from tqdm import tqdm
        except ImportError as error:
            raise SystemExit(
                "tqdm is required for neural evaluation progress. Install it with `uv pip install tqdm` "
                "or pass --no-progress."
            ) from error
        progress_bar = tqdm(
            total=args.games * args.max_pieces,
            desc="Evaluate pieces",
            unit="piece",
        )

    def finish(index: int) -> None:
        nonlocal topouts, completed
        if index in done:
            return
        game = games[index]
        done.add(index)
        topouts += int(game.game_over)
        completed += int(not game.game_over and game.pieces_placed >= args.max_pieces)
        if progress_bar is not None and game.pieces_placed < args.max_pieces:
            progress_bar.total -= args.max_pieces - game.pieces_placed
            progress_bar.refresh()

    batch_size = max(1, int(args.game_batch_size))
    started = time.perf_counter()
    try:
        while len(done) < len(games):
            active = [
                index
                for index, game in enumerate(games)
                if index not in done
                and not game.game_over
                and game.pieces_placed < args.max_pieces
            ]
            active_set = set(active)
            for index in range(len(games)):
                if index not in done and index not in active_set:
                    finish(index)
            if not active:
                break

            for start in range(0, len(active), batch_size):
                indices = active[start : start + batch_size]
                batch_games = tuple(games[index] for index in indices)
                choices = choose_search_actions_batch(
                    batch_games,
                    weights,
                    config,
                    scorer=evaluator,
                )
                batch_gained = 0
                for game_index, choice in zip(indices, choices):
                    game = games[game_index]
                    if choice is None:
                        finish(game_index)
                        continue
                    before_attack = game.attack
                    before_pieces = game.pieces_placed
                    apply_search_action(game, choice.action)
                    gained_pieces = max(0, game.pieces_placed - before_pieces)
                    batch_gained += gained_pieces
                    total_pieces += gained_pieces
                    total_attack += game.attack - before_attack
                    if game.game_over or game.pieces_placed >= args.max_pieces:
                        finish(game_index)
                if progress_bar is not None and batch_gained:
                    progress_bar.update(batch_gained)

            if progress_bar is not None:
                elapsed = max(time.perf_counter() - started, 1e-9)
                progress_bar.set_postfix(
                    pps=f"{total_pieces / elapsed:.1f}",
                    app=f"{total_attack / max(1, total_pieces):.3f}",
                    active=len(games) - len(done),
                    topouts=topouts,
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    total_pieces = sum(game.pieces_placed for game in games)
    total_attack = sum(game.attack for game in games)
    topouts = sum(int(game.game_over) for game in games)
    completed = sum(
        int(not game.game_over and game.pieces_placed >= args.max_pieces)
        for game in games
    )
    elapsed = time.perf_counter() - started
    result = {
        "model": args.model,
        "device": evaluator.device,
        "precision": evaluator.precision,
        "torchCompiled": evaluator.compiled,
        "games": args.games,
        "maxPieces": args.max_pieces,
        "gameBatchSize": batch_size,
        "pieces": total_pieces,
        "attack": total_attack,
        "attackPerPiece": total_attack / max(1, total_pieces),
        "topouts": topouts,
        "completed": completed,
        "elapsedSeconds": elapsed,
        "piecesPerSecond": total_pieces / max(elapsed, 1e-9),
        "searchConfig": config.to_dict(),
    }
    print(json.dumps(result, indent=2))
    return 0


def _review(args: argparse.Namespace) -> int:
    queue_path = Path(args.queue)
    if args.regenerate or not queue_path.is_file():
        evaluator = NeuralValueEvaluator.from_checkpoint(
            args.model,
            device=args.device,
            precision=args.precision,
            compile_model=args.torch_compile,
        )
        result = collect_neural_review_queue(
            queue_path,
            evaluator,
            _champion_weights(args.heuristic_model),
            HumanReviewConfig(
                games=args.games,
                max_pieces=args.max_pieces,
                seed_base=args.seed_base,
                seed_step=args.seed_step,
                max_samples=args.max_samples,
                max_candidates=args.max_candidates,
                uncertainty_margin=args.uncertainty_margin,
                danger_height=args.danger_height,
                danger_holes=args.danger_holes,
                topout_tail=args.topout_tail,
                random_sample_rate=args.random_sample_rate,
                random_seed=args.review_random_seed,
                teacher_lookahead=args.teacher_lookahead,
                teacher_beam_width=args.teacher_beam,
                search_config=_search_config(args),
                neural_config=evaluator.config,
            ),
            show_progress=not args.no_progress,
        )
        print(json.dumps(result, indent=2))
    if args.collect_only:
        return 0
    return launch_human_review_app(queue_path, args.output)


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--no-srs", action="store_true")
    parser.add_argument("--allow-180", action="store_true")
    parser.add_argument("--reachability-nodes", type=int, default=8000)


def _add_inference_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16", "auto"),
        default="float32",
        help="Inference precision. float32 preserves existing ranking most strictly.",
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Use torch.compile(reduce-overhead); optional because support varies by platform.",
    )


def _add_generate_args(parser: argparse.ArgumentParser, *, strong: bool) -> None:
    parser.add_argument(
        "--output",
        default=(
            "data/neural/strong-ranking.jsonl"
            if strong
            else "data/neural/champion-ranking.jsonl"
        ),
    )
    parser.add_argument("--model", default=None, help="Heuristic model JSON; defaults to Champion")
    parser.add_argument("--games", type=int, default=8 if strong else 40)
    parser.add_argument("--max-pieces", type=int, default=300 if strong else 500)
    parser.add_argument("--seed-base", type=int, default=3_500_001 if strong else 3_000_001)
    parser.add_argument("--seed-step", type=int, default=31)
    parser.add_argument("--max-candidates", type=int, default=24, help="0 keeps every legal root candidate")
    parser.add_argument("--sampling-mode", choices=("diverse", "hard"), default="diverse")
    parser.add_argument("--hard-candidates", type=int, default=8)
    parser.add_argument("--medium-candidates", type=int, default=5)
    parser.add_argument("--random-candidates", type=int, default=5)
    parser.add_argument("--bad-candidates", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=24680)
    parser.add_argument("--teacher-lookahead", type=int, default=1 if strong else 0)
    parser.add_argument("--teacher-beam", type=int, default=4)
    parser.add_argument("--teacher-acceptable-margin", type=float, default=0.0)
    parser.add_argument(
        "--teacher-score-candidates",
        type=int,
        default=8 if strong else 0,
        help="Candidates receiving expensive deep teacherScore; 0 scores all",
    )
    parser.add_argument("--rollout-horizon", type=int, default=12 if strong else 0)
    parser.add_argument("--rollout-candidates", type=int, default=4 if strong else 0)
    parser.add_argument(
        "--rollout-lookahead",
        type=int,
        default=0 if strong else 1,
        help="Lookahead used by the policy inside a long rollout; 0 avoids nested beam search",
    )
    parser.add_argument("--rollout-beam", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100 if strong else 500)
    parser.add_argument(
        "--workers",
        type=int,
        default=0 if strong else 1,
        help="Parallel game workers; 0 uses available CPU cores",
    )
    _add_search_args(parser)
    parser.set_defaults(func=_generate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinoFlux neural value learning tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate imitation ranking data with diverse negatives")
    _add_generate_args(generate, strong=False)

    generate_strong = subparsers.add_parser(
        "generate-strong",
        help="Generate diverse lookahead-distilled data with clean-attack rollouts",
    )
    _add_generate_args(generate_strong, strong=True)

    train = subparsers.add_parser("train", help="Train a neural value network from ranking data")
    train.add_argument("--dataset", default="data/neural/champion-ranking.jsonl")
    train.add_argument("--human-dataset", default=None, help="Optional human-reviewed ranking JSONL")
    train.add_argument("--human-weight", type=float, default=5.0, help="Loss multiplier for human labels")
    train.add_argument("--teacher-weight", type=float, default=0.25, help="Auxiliary pairwise weight for teacherScore")
    train.add_argument("--rollout-weight", type=float, default=0.50, help="Auxiliary pairwise weight for targetValue")
    train.add_argument("--output", default="data/models/neural-value.pt")
    train.add_argument("--resume", default=None, help="Optional neural checkpoint to continue from")
    train.add_argument("--epochs", type=int, default=8)
    train.add_argument("--batch-size", type=int, default=64, help="Ranking samples per GPU batch")
    train.add_argument("--learning-rate", type=float, default=0.001)
    train.add_argument("--weight-decay", type=float, default=0.0001)
    train.add_argument("--validation-fraction", type=float, default=0.10)
    train.add_argument("--margin", type=float, default=0.20)
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    train.set_defaults(func=_train)

    evaluate = subparsers.add_parser("evaluate", help="Run solo games with a trained neural scorer")
    evaluate.add_argument("--model", default="data/models/neural-value.pt")
    evaluate.add_argument("--heuristic-model", default=None, help="Fallback heuristic model")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--games", type=int, default=8)
    evaluate.add_argument("--max-pieces", type=int, default=500)
    evaluate.add_argument("--seed-base", type=int, default=4000001)
    evaluate.add_argument("--seed-step", type=int, default=97)
    evaluate.add_argument(
        "--game-batch-size",
        type=int,
        default=16,
        help="Games advanced together per neural GPU batch",
    )
    evaluate.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    _add_inference_args(evaluate)
    _add_search_args(evaluate)
    evaluate.set_defaults(func=_evaluate)

    review = subparsers.add_parser(
        "review",
        help="Collect DAgger positions with tqdm and label them in the Pygame reviewer",
    )
    review.add_argument("--model", default="data/models/neural-value.pt")
    review.add_argument("--heuristic-model", default=None, help="Heuristic teacher model")
    review.add_argument("--queue", default="data/neural/review-queue.jsonl")
    review.add_argument("--output", default="data/neural/human-ranking.jsonl")
    review.add_argument("--device", default="auto")
    review.add_argument("--games", type=int, default=8)
    review.add_argument("--max-pieces", type=int, default=500)
    review.add_argument("--seed-base", type=int, default=5000001)
    review.add_argument("--seed-step", type=int, default=97)
    review.add_argument("--max-samples", type=int, default=320)
    review.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Maximum placements shown per position; 0 shows every legal placement",
    )
    review.add_argument("--uncertainty-margin", type=float, default=0.08)
    review.add_argument("--danger-height", type=int, default=12)
    review.add_argument("--danger-holes", type=int, default=4)
    review.add_argument("--topout-tail", type=int, default=30)
    review.add_argument("--random-sample-rate", type=float, default=0.03)
    review.add_argument("--review-random-seed", type=int, default=13579)
    review.add_argument("--teacher-lookahead", type=int, default=1)
    review.add_argument("--teacher-beam", type=int, default=4)
    review.add_argument("--regenerate", action="store_true", help="Replace an existing review queue")
    review.add_argument("--collect-only", action="store_true", help="Build the queue without launching Pygame")
    review.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    _add_inference_args(review)
    _add_search_args(review)
    review.set_defaults(func=_review)
    return parser


def _run_in_utf8_mode_if_needed(raw_argv: list[str]) -> int | None:
    """Relaunch torch.compile commands in UTF-8 mode on non-UTF-8 Windows."""

    if (
        sys.platform != "win32"
        or sys.flags.utf8_mode
        or "--torch-compile" not in raw_argv
    ):
        return None

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "minoflux.neural_cli",
        *raw_argv,
    ]
    print(
        "torch.compile requires UTF-8 mode on this Windows locale; relaunching automatically...",
        file=sys.stderr,
        flush=True,
    )
    return int(subprocess.run(command, check=False).returncode)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    relaunched = _run_in_utf8_mode_if_needed(raw_argv)
    if relaunched is not None:
        return relaunched
    args = build_parser().parse_args(raw_argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
