from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig, apply_search_action, choose_search_action, load_weights
from minoflux_ai.human_review import HumanReviewConfig, collect_neural_review_queue
from minoflux_ai.neural import NeuralValueEvaluator
from minoflux.human_review_web import launch_human_review_app
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
            search_config=_search_config(args),
        ),
        weights,
        progress=progress,
        progress_every=args.progress_every,
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
    evaluator = NeuralValueEvaluator.from_checkpoint(args.model, device=args.device)
    weights = _weights(args.heuristic_model)
    config = _search_config(args)
    total_pieces = 0
    total_attack = 0
    topouts = 0
    completed = 0
    for game_index in range(args.games):
        seed = args.seed_base + game_index * args.seed_step
        game = Game(seed)
        while not game.game_over and game.pieces_placed < args.max_pieces:
            choice = choose_search_action(
                game,
                weights,
                config,
                scorer=evaluator,
            )
            if choice is None:
                break
            apply_search_action(game, choice.action)
        total_pieces += game.pieces_placed
        total_attack += game.attack
        topouts += int(game.game_over)
        completed += int(not game.game_over and game.pieces_placed >= args.max_pieces)
    result = {
        "model": args.model,
        "device": evaluator.device,
        "games": args.games,
        "maxPieces": args.max_pieces,
        "pieces": total_pieces,
        "attack": total_attack,
        "attackPerPiece": total_attack / max(1, total_pieces),
        "topouts": topouts,
        "completed": completed,
        "searchConfig": config.to_dict(),
    }
    print(json.dumps(result, indent=2))
    return 0


def _review(args: argparse.Namespace) -> int:
    queue_path = Path(args.queue)
    if args.regenerate or not queue_path.is_file():
        evaluator = NeuralValueEvaluator.from_checkpoint(args.model, device=args.device)
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
                search_config=_search_config(args),
                neural_config=evaluator.config,
            ),
        )
        print(json.dumps(result, indent=2))
    if args.collect_only:
        return 0
    launch_human_review_app(queue_path, args.output, port=args.port)
    return 0


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--no-srs", action="store_true")
    parser.add_argument("--allow-180", action="store_true")
    parser.add_argument("--reachability-nodes", type=int, default=8000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinoFlux neural value learning tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate Champion imitation ranking data")
    generate.add_argument("--output", default="data/neural/champion-ranking.jsonl")
    generate.add_argument("--model", default=None, help="Heuristic model JSON; defaults to Champion")
    generate.add_argument("--games", type=int, default=40)
    generate.add_argument("--max-pieces", type=int, default=500)
    generate.add_argument("--seed-base", type=int, default=3000001)
    generate.add_argument("--seed-step", type=int, default=31)
    generate.add_argument("--max-candidates", type=int, default=24, help="0 keeps every legal root candidate")
    generate.add_argument("--progress-every", type=int, default=500)
    _add_search_args(generate)
    generate.set_defaults(func=_generate)

    train = subparsers.add_parser("train", help="Train a neural value network from ranking data")
    train.add_argument("--dataset", default="data/neural/champion-ranking.jsonl")
    train.add_argument("--human-dataset", default=None, help="Optional human-reviewed ranking JSONL")
    train.add_argument("--human-weight", type=float, default=5.0, help="Loss multiplier for human labels")
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
    evaluate.add_argument("--heuristic-model", default=None, help="Only used for feature extraction/ties")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--games", type=int, default=8)
    evaluate.add_argument("--max-pieces", type=int, default=500)
    evaluate.add_argument("--seed-base", type=int, default=4000001)
    evaluate.add_argument("--seed-step", type=int, default=97)
    _add_search_args(evaluate)
    evaluate.set_defaults(func=_evaluate)

    review = subparsers.add_parser(
        "review",
        help="Collect uncertain NN positions and label them in a local browser UI",
    )
    review.add_argument("--model", default="data/models/neural-value.pt")
    review.add_argument("--heuristic-model", default=None, help="Champion model used only for disagreement sampling")
    review.add_argument("--queue", default="data/neural/review-queue.jsonl")
    review.add_argument("--output", default="data/neural/human-ranking.jsonl")
    review.add_argument("--device", default="auto")
    review.add_argument("--games", type=int, default=8)
    review.add_argument("--max-pieces", type=int, default=500)
    review.add_argument("--seed-base", type=int, default=5000001)
    review.add_argument("--seed-step", type=int, default=97)
    review.add_argument("--max-samples", type=int, default=160)
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
    review.add_argument("--regenerate", action="store_true", help="Replace an existing review queue")
    review.add_argument("--collect-only", action="store_true", help="Build the queue without launching the browser UI")
    review.add_argument("--port", type=int, default=7861)
    _add_search_args(review)
    review.set_defaults(func=_review)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
