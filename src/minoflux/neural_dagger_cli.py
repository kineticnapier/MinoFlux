from __future__ import annotations

import argparse
import json
import sys

from minoflux_ai import SearchConfig, load_weights
from minoflux_ai.neural import NeuralValueEvaluator
from minoflux_ai.neural_dagger import write_neural_dagger_dataset
from minoflux_ai.neural_dataset import NeuralDatasetConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate teacher-labelled DAgger data on neural self-play states"
    )
    parser.add_argument("--model", default="data/models/neural-value-human.pt")
    parser.add_argument("--heuristic-model", default="data/models/champion-cem.json")
    parser.add_argument("--output", default="data/neural/dagger-ranking.jsonl")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16", "auto"), default="float32")
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=6_000_001)
    parser.add_argument("--seed-step", type=int, default=97)
    parser.add_argument("--max-samples", type=int, default=2000, help="0 means unlimited")
    parser.add_argument("--sample-rate", type=float, default=0.25)
    parser.add_argument("--uncertainty-margin", type=float, default=0.08)
    parser.add_argument("--danger-height", type=int, default=12)
    parser.add_argument("--danger-holes", type=int, default=4)

    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--sampling-mode", choices=("diverse", "hard"), default="diverse")
    parser.add_argument("--hard-candidates", type=int, default=8)
    parser.add_argument("--medium-candidates", type=int, default=5)
    parser.add_argument("--random-candidates", type=int, default=5)
    parser.add_argument("--bad-candidates", type=int, default=5)
    parser.add_argument("--sampling-seed", type=int, default=24680)

    parser.add_argument("--teacher-lookahead", type=int, default=1)
    parser.add_argument("--teacher-beam", type=int, default=4)
    parser.add_argument("--teacher-acceptable-margin", type=float, default=0.0)
    parser.add_argument("--rollout-horizon", type=int, default=12)
    parser.add_argument("--rollout-candidates", type=int, default=4)
    parser.add_argument("--rollout-lookahead", type=int, default=1)
    parser.add_argument("--rollout-beam", type=int, default=4)

    parser.add_argument("--no-hold", action="store_true")
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--no-srs", action="store_true")
    parser.add_argument("--allow-180", action="store_true")
    parser.add_argument("--reachability-nodes", type=int, default=8000)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluator = NeuralValueEvaluator.from_checkpoint(
        args.model,
        device=args.device,
        precision=args.precision,
    )
    weights = load_weights(args.heuristic_model)
    search = SearchConfig(
        allow_hold=not args.no_hold,
        lookahead_pieces=args.lookahead,
        beam_width=args.beam,
        discount=args.discount,
        srs_reachable=not args.no_srs,
        allow_180=args.allow_180,
        reachability_node_limit=args.reachability_nodes,
    )

    def progress(samples: int, candidates: int) -> None:
        print(
            f"dagger {samples:,} samples / {candidates:,} candidate states",
            file=sys.stderr,
            flush=True,
        )

    result = write_neural_dagger_dataset(
        args.output,
        evaluator,
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
            rollout_horizon=args.rollout_horizon,
            rollout_candidates=args.rollout_candidates,
            rollout_lookahead=args.rollout_lookahead,
            rollout_beam_width=args.rollout_beam,
            search_config=search,
            neural_config=evaluator.config,
        ),
        weights,
        sample_rate=args.sample_rate,
        uncertainty_margin=args.uncertainty_margin,
        danger_height=args.danger_height,
        danger_holes=args.danger_holes,
        max_samples=args.max_samples,
        progress=progress,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
