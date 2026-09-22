from __future__ import annotations

import argparse
import json
import sys

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # Optional convenience only.
    _tqdm = None

from minoflux_ai.neural import NeuralValueEvaluator
from minoflux_ai.oracle import OracleConfig, oracle_native_available, profile_oracle
from minoflux_ai.oracle_dagger import write_oracle_dagger_dataset
from minoflux_ai.oracle_dataset import OracleDatasetConfig, run_oracle_smoke, write_oracle_ranking_dataset
from minoflux_ai.search import SearchConfig
from minoflux_engine import Game


def _require_native() -> None:
    if not oracle_native_available():
        raise SystemExit(
            "MinoFlux native oracle extension is unavailable. Rebuild it with "
            "`uv run --with pybind11 --with setuptools --no-sync python setup.py build_ext --inplace --force`."
        )


def _oracle_config(args: argparse.Namespace) -> OracleConfig:
    return OracleConfig(
        beam_width=args.beam,
        depth=args.depth,
        allow_180=not args.no_180,
        reachability_node_limit=args.reachability_nodes,
    ).normalized()


def _add_oracle_search_args(
    parser: argparse.ArgumentParser,
    *,
    beam: int,
    depth: int,
) -> None:
    parser.add_argument("--beam", type=int, default=beam)
    parser.add_argument("--depth", type=int, default=depth)
    parser.add_argument("--no-180", action="store_true", help="Disable 180-degree SRS rotations")
    parser.add_argument("--reachability-nodes", type=int, default=8_000)


def _smoke(args: argparse.Namespace) -> int:
    _require_native()
    result = run_oracle_smoke(
        games=args.games,
        max_pieces=args.max_pieces,
        seed_base=args.seed_base,
        seed_step=args.seed_step,
        oracle=_oracle_config(args),
    )
    print(json.dumps(result, indent=2))
    return 0


def _annotate_detailed_profile(result: dict[str, object]) -> None:
    reachability = result.get("reachability")
    if not isinstance(reachability, dict):
        return

    bfs_seconds = float(reachability.get("bfsSeconds", 0.0))
    rotation_seconds = float(reachability.get("rotationSeconds", 0.0))
    landing_seconds = float(reachability.get("landingSeconds", 0.0))
    representative_seconds = float(reachability.get("representativeSeconds", 0.0))

    reachability["movementSeconds"] = max(0.0, bfs_seconds - rotation_seconds)
    reachability["representativeCoreSeconds"] = representative_seconds
    reachability["representativeTotalSeconds"] = representative_seconds + landing_seconds
    reachability["coarseRepresentativeSeconds"] = representative_seconds + landing_seconds
    result["profileMode"] = "sampled-ratio-v2"
    result["timingSampleStride"] = 257
    result["timingNote"] = (
        "Rotation/BFS and landing/representative ratios are measured on one detailed "
        "reachability call per 257 calls, then projected onto the low-overhead coarse "
        "BFS and representative totals. Use oracle-smoke A/B timings for final "
        "performance decisions."
    )


def _profile(args: argparse.Namespace) -> int:
    _require_native()
    config = _oracle_config(args)
    result = profile_oracle(Game(args.seed), config)
    _annotate_detailed_profile(result)
    result = {
        "teacher": "minoflux-native-oracle",
        "seed": args.seed,
        **result,
        "oracle": {
            "beamWidth": config.beam_width,
            "depth": config.depth,
            "allow180": config.allow_180,
            "reachabilityNodeLimit": config.reachability_node_limit,
        },
    }
    print(json.dumps(result, indent=2))
    return 0


def _dataset(args: argparse.Namespace) -> int:
    _require_native()
    total_samples = max(1, int(args.games)) * max(1, int(args.max_pieces))
    progress_bar = None
    last_samples = 0

    if _tqdm is not None:
        progress_bar = _tqdm(
            total=total_samples,
            desc="oracle dataset",
            unit="sample",
            dynamic_ncols=True,
            disable=None,
            file=sys.stderr,
        )

    def progress(samples: int, candidates: int) -> None:
        nonlocal last_samples
        if progress_bar is None:
            print(
                f"generated {samples:,} oracle samples / {candidates:,} candidate states",
                file=sys.stderr,
                flush=True,
            )
            last_samples = samples
            return
        if samples > last_samples:
            progress_bar.update(samples - last_samples)
            last_samples = samples
        progress_bar.set_postfix_str(f"candidates={candidates:,}", refresh=True)

    try:
        result = write_oracle_ranking_dataset(
            args.output,
            OracleDatasetConfig(
                games=args.games,
                max_pieces=args.max_pieces,
                seed_base=args.seed_base,
                seed_step=args.seed_step,
                max_candidates=args.max_candidates,
                oracle=_oracle_config(args),
            ),
            progress=progress,
            progress_every=args.progress_every,
            workers=args.workers,
        )
        final_samples = int(result.get("samples", last_samples))
        final_candidates = int(result.get("candidates", 0))
        if progress_bar is not None:
            if final_samples > last_samples:
                progress_bar.update(final_samples - last_samples)
                last_samples = final_samples
            progress_bar.set_postfix_str(f"candidates={final_candidates:,}", refresh=True)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(json.dumps(result, indent=2))
    return 0


def _dagger(args: argparse.Namespace) -> int:
    _require_native()
    oracle = _oracle_config(args)
    evaluator = NeuralValueEvaluator.from_checkpoint(
        args.model,
        device=args.device,
        precision=args.precision,
    )
    learner_search = SearchConfig(
        allow_hold=True,
        lookahead_pieces=args.learner_lookahead,
        beam_width=args.learner_beam,
        discount=0.9,
        srs_reachable=True,
        allow_180=oracle.allow_180,
        reachability_node_limit=oracle.reachability_node_limit,
    )

    total = (
        max(1, int(args.max_samples))
        if int(args.max_samples) > 0
        else max(1, int(args.games)) * max(1, int(args.max_pieces))
    )
    progress_bar = None
    last_samples = 0
    if _tqdm is not None:
        progress_bar = _tqdm(
            total=total,
            desc="oracle dagger",
            unit="label",
            dynamic_ncols=True,
            disable=None,
            file=sys.stderr,
        )

    def progress(samples: int, candidates: int, visited: int, queries: int) -> None:
        nonlocal last_samples
        if progress_bar is None:
            print(
                f"oracle dagger {samples:,} labels / {visited:,} visited / {queries:,} oracle queries",
                file=sys.stderr,
                flush=True,
            )
            last_samples = samples
            return
        if samples > last_samples:
            progress_bar.update(samples - last_samples)
            last_samples = samples
        progress_bar.set_postfix_str(
            f"visited={visited:,}, queries={queries:,}, candidates={candidates:,}",
            refresh=True,
        )

    try:
        result = write_oracle_dagger_dataset(
            args.output,
            evaluator,
            OracleDatasetConfig(
                games=args.games,
                max_pieces=args.max_pieces,
                seed_base=args.seed_base,
                seed_step=args.seed_step,
                max_candidates=args.max_candidates,
                oracle=oracle,
                neural_config=evaluator.config,
            ),
            learner_search=learner_search,
            sample_rate=args.sample_rate,
            uncertainty_margin=args.uncertainty_margin,
            danger_height=args.danger_height,
            danger_holes=args.danger_holes,
            max_samples=args.max_samples,
            progress=progress,
            progress_every=args.progress_every,
        )
        final_samples = int(result.get("samples", last_samples))
        if progress_bar is not None:
            if final_samples > last_samples:
                progress_bar.update(final_samples - last_samples)
                last_samples = final_samples
            progress_bar.set_postfix_str(
                "visited={:,}, queries={:,}, candidates={:,}".format(
                    int(result.get("visitedStates", 0)),
                    int(result.get("oracleQueries", 0)),
                    int(result.get("candidates", 0)),
                ),
                refresh=True,
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinoFlux native offline oracle tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "oracle-smoke",
        help="Run a short self-play smoke test using only the native oracle",
    )
    smoke.add_argument("--games", type=int, default=1)
    smoke.add_argument("--max-pieces", type=int, default=20)
    smoke.add_argument("--seed-base", type=int, default=8_100_001)
    smoke.add_argument("--seed-step", type=int, default=97)
    _add_oracle_search_args(smoke, beam=64, depth=2)
    smoke.set_defaults(func=_smoke)

    profile = subparsers.add_parser(
        "oracle-profile",
        help="Profile one native oracle search and report reachability hot spots",
    )
    profile.add_argument("--seed", type=int, default=8_100_001)
    _add_oracle_search_args(profile, beam=2_000, depth=18)
    profile.set_defaults(func=_profile)

    dataset = subparsers.add_parser(
        "oracle-dataset",
        help="Generate neural ranking JSONL from the MinoFlux native oracle",
    )
    dataset.add_argument("--output", default="data/neural/native-oracle.jsonl")
    dataset.add_argument("--games", type=int, default=40)
    dataset.add_argument("--max-pieces", type=int, default=500)
    dataset.add_argument("--seed-base", type=int, default=6_000_001)
    dataset.add_argument("--seed-step", type=int, default=97)
    dataset.add_argument("--max-candidates", type=int, default=24, help="0 keeps every legal root candidate")
    dataset.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel game workers; 0 uses available CPU cores minus one",
    )
    dataset.add_argument("--progress-every", type=int, default=100)
    _add_oracle_search_args(dataset, beam=2_000, depth=18)
    dataset.set_defaults(func=_dataset)

    dagger = subparsers.add_parser(
        "oracle-dagger",
        help="Label selected neural self-play states with the native oracle",
    )
    dagger.add_argument("--model", default="data/models/native-oracle-value.pt")
    dagger.add_argument("--output", default="data/neural/native-oracle-dagger.jsonl")
    dagger.add_argument("--device", default="auto")
    dagger.add_argument(
        "--precision",
        choices=("float32", "float16", "bfloat16", "auto"),
        default="float32",
    )
    dagger.add_argument("--games", type=int, default=16)
    dagger.add_argument("--max-pieces", type=int, default=500)
    dagger.add_argument("--seed-base", type=int, default=7_000_001)
    dagger.add_argument("--seed-step", type=int, default=97)
    dagger.add_argument("--max-samples", type=int, default=500, help="0 means unlimited")
    dagger.add_argument("--sample-rate", type=float, default=0.05)
    dagger.add_argument("--uncertainty-margin", type=float, default=0.08)
    dagger.add_argument("--danger-height", type=int, default=12)
    dagger.add_argument("--danger-holes", type=int, default=4)
    dagger.add_argument("--max-candidates", type=int, default=24)
    dagger.add_argument("--learner-lookahead", type=int, default=0)
    dagger.add_argument("--learner-beam", type=int, default=4)
    dagger.add_argument("--progress-every", type=int, default=10)
    _add_oracle_search_args(dagger, beam=2_000, depth=18)
    dagger.set_defaults(func=_dagger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
