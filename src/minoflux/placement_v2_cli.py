from __future__ import annotations

from argparse import ArgumentParser
import json

from minoflux_ai.placement_teacher import (
    DEFAULT_PLACEMENT_TEACHER_WEIGHTS,
    PlacementTeacherConfig,
    PlacementV2DatasetConfig,
    load_placement_teacher_weights,
    write_placement_v2_dataset,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="minoflux-placement-v2",
        description="Generate Placement Teacher v2 ranking data for neural distillation",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser(
        "generate",
        help="Generate ranking JSONL with the non-heuristic multi-ply placement teacher",
    )
    generate.add_argument("--output", default="data/neural/placement-v2-ranking.jsonl")
    generate.add_argument("--games", type=int, default=50)
    generate.add_argument("--max-pieces", type=int, default=300)
    generate.add_argument("--seed-base", type=int, default=8_000_001)
    generate.add_argument("--seed-step", type=int, default=31)
    generate.add_argument("--max-candidates", type=int, default=24)
    generate.add_argument("--hard-candidates", type=int, default=8)
    generate.add_argument("--medium-candidates", type=int, default=5)
    generate.add_argument("--random-candidates", type=int, default=5)
    generate.add_argument("--bad-candidates", type=int, default=5)
    generate.add_argument("--acceptable-margin", type=float, default=0.50)
    generate.add_argument("--sampling-seed", type=int, default=26_090_904)
    generate.add_argument("--teacher-depth", type=int, default=2)
    generate.add_argument("--teacher-beam", type=int, default=24)
    generate.add_argument("--teacher-discount", type=float, default=0.92)
    generate.add_argument("--high-stack-height", type=int, default=12)
    generate.add_argument("--no-hold", action="store_true")
    generate.add_argument("--allow-180", action="store_true")
    generate.add_argument("--reachability-nodes", type=int, default=8_000)
    generate.add_argument(
        "--teacher-weights",
        default=None,
        help="Optional JSON object overriding PlacementTeacherWeights",
    )
    generate.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel game workers; 0 uses available CPU cores",
    )
    return parser


def _generate(args) -> int:
    weights = (
        DEFAULT_PLACEMENT_TEACHER_WEIGHTS
        if args.teacher_weights is None
        else load_placement_teacher_weights(args.teacher_weights)
    )
    teacher = PlacementTeacherConfig(
        depth=args.teacher_depth,
        beam_width=args.teacher_beam,
        discount=args.teacher_discount,
        allow_hold=not args.no_hold,
        allow_180=args.allow_180,
        reachability_node_limit=args.reachability_nodes,
        high_stack_height=args.high_stack_height,
    ).normalized()
    config = PlacementV2DatasetConfig(
        games=args.games,
        max_pieces=args.max_pieces,
        seed_base=args.seed_base,
        seed_step=args.seed_step,
        max_candidates=args.max_candidates,
        hard_candidates=args.hard_candidates,
        medium_candidates=args.medium_candidates,
        random_candidates=args.random_candidates,
        bad_candidates=args.bad_candidates,
        acceptable_margin=args.acceptable_margin,
        random_seed=args.sampling_seed,
        teacher=teacher,
    ).normalized()

    try:
        from tqdm import tqdm
    except ImportError:
        progress_bar = None
    else:
        progress_bar = tqdm(total=config.games, desc="Placement-v2 teacher", unit="game")

    completed_games = 0

    def progress(samples: int, candidates: int) -> None:
        nonlocal completed_games
        completed_games += 1
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(samples=samples, candidates=candidates)

    try:
        result = write_placement_v2_dataset(
            args.output,
            config,
            weights,
            workers=args.workers,
            progress=progress,
        )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return _generate(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
