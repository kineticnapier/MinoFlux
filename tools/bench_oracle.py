from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    name: str
    seconds: float


@dataclass(frozen=True)
class PairResult:
    control_seconds: float
    candidate_seconds: float

    @property
    def improvement_percent(self) -> float:
        return (1.0 - self.candidate_seconds / self.control_seconds) * 100.0


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL = ROOT.parent / "MinoFlux-group-control"
DEFAULT_CONTROL_REF = "cadbb1b"


def run(cmd: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build_native(repo: Path) -> None:
    print(f"[build] {repo}")
    run(
        [
            "uv",
            "run",
            "--active",
            "--with",
            "pybind11",
            "--with",
            "setuptools",
            "--no-sync",
            "python",
            "setup.py",
            "build_ext",
            "--inplace",
        ],
        repo,
    )


def native_module_path(repo: Path) -> Path | None:
    package_dir = repo / "src" / "minoflux_ai"
    matches = sorted(package_dir.glob("_oracle_native*.pyd"))
    return matches[0] if matches else None


def ensure_control_worktree(control: Path, control_ref: str) -> bool:
    if control.is_dir():
        return False

    print(f"[worktree] {control_ref} -> {control}")
    run(
        ["git", "worktree", "add", "--detach", str(control), control_ref],
        ROOT,
    )
    return True


def oracle_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "minoflux.neural_dispatch_cli",
        "oracle-smoke",
        "--games",
        str(args.games),
        "--max-pieces",
        str(args.max_pieces),
        "--beam",
        str(args.beam),
        "--depth",
        str(args.depth),
    ]


def oracle_env(repo: Path) -> dict[str, str]:
    env = os.environ.copy()
    source_dir = str(repo / "src")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_dir if not existing else source_dir + os.pathsep + existing
    return env


def run_oracle(repo: Path, args: argparse.Namespace) -> float:
    cmd = oracle_command(args)
    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=repo,
        env=oracle_env(repo),
        stdout=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)
    return elapsed


def benchmark_once(repo: Path, name: str, args: argparse.Namespace) -> Sample:
    elapsed = run_oracle(repo, args)
    positions = args.games * args.max_pieces
    rate = positions / elapsed if elapsed > 0 else float("inf")
    print(f"{name:16} {elapsed:8.3f}s  {rate:8.4f} pos/s")
    return Sample(name=name, seconds=elapsed)


def warm_up(control: Path, current: Path, args: argparse.Namespace) -> None:
    print("[warmup] control")
    run_oracle(control, args)
    print("[warmup] current")
    run_oracle(current, args)


def summarize(
    samples: list[Sample],
    pairs: list[PairResult],
    baseline_name: str,
    candidate_name: str,
) -> None:
    groups: dict[str, list[float]] = {}
    for sample in samples:
        groups.setdefault(sample.name, []).append(sample.seconds)

    print()
    print(f"{'Version':16} {'Runs':>4} {'Avg':>9} {'Median':>9} {'Min':>9} {'Max':>9}")
    print("-" * 62)
    summaries: dict[str, tuple[float, float]] = {}
    for name in (baseline_name, candidate_name):
        values = groups[name]
        avg = statistics.fmean(values)
        median = statistics.median(values)
        summaries[name] = (avg, median)
        print(
            f"{name:16} {len(values):4d} {avg:8.3f}s {median:8.3f}s "
            f"{min(values):8.3f}s {max(values):8.3f}s"
        )

    baseline_avg, baseline_median = summaries[baseline_name]
    candidate_avg, candidate_median = summaries[candidate_name]
    avg_improvement = (1.0 - candidate_avg / baseline_avg) * 100.0
    median_improvement = (1.0 - candidate_median / baseline_median) * 100.0

    pair_improvements = [pair.improvement_percent for pair in pairs]
    pair_mean = statistics.fmean(pair_improvements)
    pair_median = statistics.median(pair_improvements)

    print()
    print(f"Average improvement: {avg_improvement:+.2f}%")
    print(f"Median improvement:  {median_improvement:+.2f}%")
    print(f"Paired improvement:  mean {pair_mean:+.2f}% / median {pair_median:+.2f}%")
    print("Pair deltas:         " + "  ".join(f"{value:+.2f}%" for value in pair_improvements))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interleaved A/B benchmark for the native offline oracle.",
    )
    parser.add_argument(
        "--control",
        type=Path,
        default=DEFAULT_CONTROL,
        help=f"control worktree (default: {DEFAULT_CONTROL})",
    )
    parser.add_argument(
        "--control-ref",
        default=DEFAULT_CONTROL_REF,
        help=f"commit/ref used when creating the control worktree (default: {DEFAULT_CONTROL_REF})",
    )
    parser.add_argument("--control-name", default="control")
    parser.add_argument("--candidate-name", default="current")
    parser.add_argument(
        "--pairs",
        type=int,
        default=5,
        help="number of ABBA pairs; 5 gives 10 measured runs per version",
    )
    parser.add_argument("--build-current", action="store_true")
    parser.add_argument("--build-control", action="store_true")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--max-pieces", type=int, default=5)
    parser.add_argument("--beam", type=int, default=2000)
    parser.add_argument("--depth", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current = ROOT
    control = args.control.resolve()

    if args.pairs <= 0:
        raise SystemExit("--pairs must be positive")

    created_control = ensure_control_worktree(control, args.control_ref)

    if args.build_current:
        build_native(current)
    if args.build_control or created_control:
        build_native(control)

    for label, repo in ((args.candidate_name, current), (args.control_name, control)):
        module = native_module_path(repo)
        if module is None:
            raise SystemExit(
                f"native module not found for {label}: {repo}\n"
                f"build it once with --build-current/--build-control as appropriate"
            )
        print(f"[{label}] {module}")

    if not args.no_warmup:
        print()
        warm_up(control, current, args)

    print()
    samples: list[Sample] = []
    pairs: list[PairResult] = []
    for _ in range(args.pairs):
        control_a = benchmark_once(control, args.control_name, args)
        candidate_a = benchmark_once(current, args.candidate_name, args)
        candidate_b = benchmark_once(current, args.candidate_name, args)
        control_b = benchmark_once(control, args.control_name, args)

        samples.extend((control_a, candidate_a, candidate_b, control_b))
        pairs.append(
            PairResult(
                control_seconds=(control_a.seconds + control_b.seconds) / 2.0,
                candidate_seconds=(candidate_a.seconds + candidate_b.seconds) / 2.0,
            )
        )

    summarize(samples, pairs, args.control_name, args.candidate_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
