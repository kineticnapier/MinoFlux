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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTROL = ROOT.parent / "MinoFlux-group-control"


def run(cmd: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def build_native(repo: Path) -> None:
    print(f"[build] {repo}")
    run(
        [
            "uv",
            "run",
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


def benchmark_once(repo: Path, name: str, args: argparse.Namespace) -> Sample:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")

    cmd = [
        "uv",
        "run",
        "--no-sync",
        "python",
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

    started = time.perf_counter()
    completed = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, cmd)

    positions = args.games * args.max_pieces
    rate = positions / elapsed if elapsed > 0 else float("inf")
    print(f"{name:16} {elapsed:8.3f}s  {rate:8.4f} pos/s")
    return Sample(name=name, seconds=elapsed)


def summarize(samples: list[Sample], baseline_name: str, candidate_name: str) -> None:
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

    print()
    print(f"Average improvement: {avg_improvement:+.2f}%")
    print(f"Median improvement:  {median_improvement:+.2f}%")


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
    parser.add_argument("--control-name", default="control")
    parser.add_argument("--candidate-name", default="current")
    parser.add_argument(
        "--pairs",
        type=int,
        default=5,
        help="number of ABBA pairs; 5 gives 10 runs per version",
    )
    parser.add_argument("--build-current", action="store_true")
    parser.add_argument("--build-control", action="store_true")
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
    if not control.is_dir():
        raise SystemExit(f"control worktree does not exist: {control}")

    if args.build_current:
        build_native(current)
    if args.build_control:
        build_native(control)

    for label, repo in ((args.candidate_name, current), (args.control_name, control)):
        module = native_module_path(repo)
        if module is None:
            raise SystemExit(
                f"native module not found for {label}: {repo}\n"
                f"build it once with --build-current/--build-control as appropriate"
            )
        print(f"[{label}] {module}")

    print()
    samples: list[Sample] = []
    for _ in range(args.pairs):
        samples.append(benchmark_once(control, args.control_name, args))
        samples.append(benchmark_once(current, args.candidate_name, args))
        samples.append(benchmark_once(current, args.candidate_name, args))
        samples.append(benchmark_once(control, args.control_name, args))

    summarize(samples, args.control_name, args.candidate_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
