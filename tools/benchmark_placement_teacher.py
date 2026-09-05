"""Reproducible Placement Teacher timings and byte-for-byte dataset comparison.

Run from the repository root with ``uv run python tools/benchmark_placement_teacher.py``.
Reference mode loads only placement_teacher.py from the requested commit; engine,
reachability and other shared dependencies use the current checkout. Workers
load the reference module at startup, including with Windows' spawn method.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import cProfile
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pstats
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
REFERENCE_ENV = "MINOFLUX_TEACHER_BENCH_REFERENCE"


def _load_teacher():
    path = os.environ.get(REFERENCE_ENV)
    if not path:
        from minoflux_ai import placement_teacher
        return placement_teacher
    name = "minoflux_ai._teacher_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load teacher reference: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# Spawned workers need this module before pickle resolves reference dataclasses.
if os.environ.get(REFERENCE_ENV):
    _load_teacher()


def _profile_rows(profiler: cProfile.Profile):
    stats = pstats.Stats(profiler)
    denominator = stats.total_tt
    rows = []
    for (filename, line, name), (_primitive, calls, exclusive, inclusive, _callers) in stats.stats.items():
        rows.append({"function": name, "file": filename, "line": line,
                     "calls": calls, "exclusiveMs": exclusive * 1000,
                     "inclusiveMs": inclusive * 1000,
                     "meanInclusiveUs": inclusive * 1_000_000 / calls if calls else 0,
                     "meanExclusiveUs": exclusive * 1_000_000 / calls if calls else 0,
                     "exclusivePercent": 100 * exclusive / denominator if denominator else 0,
                     "inclusivePercent": 100 * inclusive / denominator if denominator else 0})
    return sorted(rows, key=lambda row: row["inclusiveMs"], reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=12)
    parser.add_argument("--pieces", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam", type=int, default=24)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--seed-base", type=int, default=8_000_001)
    parser.add_argument("--seed-step", type=int, default=31)
    parser.add_argument("--reference-commit")
    parser.add_argument("--reference-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--compare", type=Path, help="Fail unless output bytes equal this JSONL")
    parser.add_argument("--profile", action="store_true", help="Separate instrumented single-worker run")
    parser.add_argument("--spawn", action="store_true", help="Use Windows-style process startup on any OS")
    args = parser.parse_args()
    if args.profile and args.workers != 1:
        parser.error("--profile requires --workers 1; instrumented timings are separate from throughput")
    if args.reference_commit and args.reference_file:
        parser.error("Choose only one reference source")
    if args.spawn:
        import multiprocessing
        multiprocessing.set_start_method("spawn", force=True)

    with tempfile.TemporaryDirectory(prefix="minoflux-teacher-reference-") as temporary:
        if args.reference_commit:
            source = subprocess.check_output(["git", "show", f"{args.reference_commit}:src/minoflux_ai/placement_teacher.py"], cwd=ROOT)
            reference = Path(temporary) / "placement_teacher.py"
            reference.write_bytes(source)
            os.environ[REFERENCE_ENV] = str(reference)
        elif args.reference_file:
            os.environ[REFERENCE_ENV] = str(args.reference_file.resolve())
        teacher = _load_teacher()
        from minoflux_ai.reachability import clear_reachability_cache, collect_reachability_profile
        clear_reachability_cache()
        config = teacher.PlacementV2DatasetConfig(
            games=args.games, max_pieces=args.pieces, seed_base=args.seed_base,
            seed_step=args.seed_step, max_candidates=args.max_candidates,
            teacher=teacher.PlacementTeacherConfig(depth=args.depth, beam_width=args.beam))
        profiler = cProfile.Profile() if args.profile else None
        with collect_reachability_profile() if args.profile else nullcontext() as reachability:
            started = time.perf_counter()
            if profiler is not None:
                profiler.enable()
            result = teacher.write_placement_v2_dataset(args.output, config, workers=args.workers)
            if profiler is not None:
                profiler.disable()
            elapsed = time.perf_counter() - started
        payload = args.output.read_bytes()
        report = {"reference": args.reference_commit or (str(args.reference_file) if args.reference_file else None),
                  "seconds": elapsed, "secondsPerGame": elapsed / args.games,
                  "samplesPerSecond": result["samples"] / elapsed,
                  "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
                  "instrumented": args.profile, "result": result}
        if args.compare:
            report["bytesMatchReference"] = payload == args.compare.read_bytes()
        if profiler is not None:
            report["profile"] = _profile_rows(profiler)
            report["reachabilityProfile"] = reachability.to_dict()
        serialized = json.dumps(report, indent=2)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        if args.compare and not report["bytesMatchReference"]:
            raise SystemExit("Dataset bytes differ from reference")


if __name__ == "__main__":
    main()
