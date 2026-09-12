from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterator, Sequence

from minoflux_ai import (
    NeuralValueConfig,
    NeuralValueEvaluator,
    SearchConfig,
    VersusSearchConfig,
    VersusSelfPlayConfig,
    VersusValueConfig,
    VersusValueEvaluator,
    build_neural_value_model,
    build_versus_value_model,
    generate_versus_selfplay_dataset,
    run_versus_benchmark,
)
from minoflux_ai.versus_profile import collect_versus_profile


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(value) for value in values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


class _ForwardStats:
    def __init__(self) -> None:
        self.calls = 0
        self.total_seconds = 0.0
        self.batch_sizes: list[int] = []

    def record(self, batch_size: int, elapsed: float) -> None:
        self.calls += 1
        self.total_seconds += max(0.0, float(elapsed))
        self.batch_sizes.append(max(0, int(batch_size)))

    def to_dict(self) -> dict[str, object]:
        sizes = self.batch_sizes
        return {
            "calls": self.calls,
            "totalSeconds": self.total_seconds,
            "meanBatchSize": statistics.fmean(sizes) if sizes else 0.0,
            "p50BatchSize": _percentile(sizes, 0.50),
            "p95BatchSize": _percentile(sizes, 0.95),
            "maxBatchSize": max(sizes, default=0),
        }


class _ProfiledModule:
    def __init__(self, torch: Any, module: Any, device: str, stats: _ForwardStats) -> None:
        self._torch = torch
        self._module = module
        self._device = str(device)
        self._stats = stats

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        batch_size = int(args[0].shape[0]) if args else 0
        if self._device.startswith("cuda"):
            self._torch.cuda.synchronize()
        started = time.perf_counter()
        output = self._module(*args, **kwargs)
        if self._device.startswith("cuda"):
            self._torch.cuda.synchronize()
        self._stats.record(batch_size, time.perf_counter() - started)
        return output


class _GpuSampler:
    def __init__(self, interval: float = 0.20) -> None:
        self.interval = max(0.05, float(interval))
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.available = shutil.which("nvidia-smi") is not None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                first = completed.stdout.strip().splitlines()[0]
                util_raw, memory_raw = first.split(",", 1)
                self.samples.append((float(util_raw.strip()), float(memory_raw.strip())))
            except (IndexError, OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "_GpuSampler":
        if self.available:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def to_dict(self) -> dict[str, object]:
        utils = [sample[0] for sample in self.samples]
        memory = [sample[1] for sample in self.samples]
        return {
            "available": self.available,
            "samples": len(self.samples),
            "meanUtilizationPercent": statistics.fmean(utils) if utils else None,
            "p50UtilizationPercent": _percentile([round(value) for value in utils], 0.50) if utils else None,
            "p95UtilizationPercent": _percentile([round(value) for value in utils], 0.95) if utils else None,
            "peakMemoryMiB": max(memory, default=None),
        }


def _zero_parameters(model: Any) -> None:
    for parameter in model.parameters():
        parameter.data.zero_()


def _build_profiled_evaluators(device: str):
    import torch

    torch.manual_seed(20260913)
    solo_config = NeuralValueConfig()
    solo_model = build_neural_value_model(solo_config)
    _zero_parameters(solo_model)
    solo = NeuralValueEvaluator(
        solo_model,
        solo_config,
        device=device,
        precision="float32",
        compile_model=False,
    )
    solo_stats = _ForwardStats()
    solo.model = _ProfiledModule(torch, solo.model, solo.device, solo_stats)

    versus_config = VersusValueConfig()
    versus_model = build_versus_value_model(versus_config)
    _zero_parameters(versus_model)
    versus = VersusValueEvaluator(
        versus_model,
        versus_config,
        device=device,
        compile_model=False,
    )
    versus_stats = _ForwardStats()
    versus.model = _ProfiledModule(torch, versus.model, versus.device, versus_stats)
    return solo, versus, solo_stats, versus_stats


def _phase_seconds(profile, name: str) -> float:
    for row in profile.phase_rows():
        if row["name"] == name:
            return float(row["totalMs"]) / 1000.0
    return 0.0


def _summarize_run(
    *,
    kind: str,
    elapsed: float,
    games: int,
    turns: int,
    profile,
    solo_stats: _ForwardStats,
    versus_stats: _ForwardStats,
    gpu: _GpuSampler,
) -> dict[str, object]:
    placement_scoring = _phase_seconds(profile, "neural_placement_scoring")
    versus_inference = _phase_seconds(profile, "versus_value_inference")
    forward_seconds = solo_stats.total_seconds + versus_stats.total_seconds
    total_search = _phase_seconds(profile, "total_search")
    root_expansion = _phase_seconds(profile, "root_placement_generation")
    reply_expansion = _phase_seconds(profile, "opponent_placement_generation")
    board_update = _phase_seconds(profile, "apply_search_action")
    garbage_update = _phase_seconds(profile, "resolve_lock")
    state_encoding = (
        _phase_seconds(profile, "root_state_encoding")
        + _phase_seconds(profile, "reply_state_encoding")
    )
    placement_encode_estimate = max(0.0, placement_scoring - solo_stats.total_seconds)
    orchestration = max(0.0, elapsed - total_search)
    return {
        "kind": kind,
        "elapsedSeconds": elapsed,
        "games": games,
        "turns": turns,
        "gamesPerSecond": games / elapsed if elapsed else 0.0,
        "turnsPerSecond": turns / elapsed if elapsed else 0.0,
        "modelForwards": {
            "placement": solo_stats.to_dict(),
            "versusValue": versus_stats.to_dict(),
            "totalCalls": solo_stats.calls + versus_stats.calls,
        },
        "neuralInferenceSeconds": forward_seconds,
        "neuralInferenceDutyCycle": forward_seconds / elapsed if elapsed else 0.0,
        "reachabilitySeconds": float(profile.reachability.total_seconds),
        "candidateExpansionSeconds": root_expansion,
        "replyExpansionSeconds": reply_expansion,
        "boardUpdateSeconds": board_update,
        "garbageUpdateSeconds": garbage_update,
        "featureEncodingSeconds": state_encoding + placement_encode_estimate,
        "pythonOrchestrationSeconds": orchestration,
        "profile": profile.to_dict(),
        "gpu": gpu.to_dict(),
        "gpuIdleInference": (
            "Measured by nvidia-smi samples"
            if gpu.samples
            else "No GPU telemetry available; neuralInferenceDutyCycle is the idle proxy"
        ),
        "notes": {
            "placementEncodingEstimate": "neural_placement_scoring minus measured placement model forward wall time",
            "phaseTimings": "inclusive; phase totals are not additive",
            "model": "default-size zero-weight float32 networks; computation shape matches production defaults while ranking is deterministic",
        },
        "rawPhaseSeconds": {
            "neuralPlacementScoring": placement_scoring,
            "versusValueInference": versus_inference,
            "totalSearch": total_search,
        },
    }


def _search_config(args) -> VersusSearchConfig:
    return VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=4,
            discount=0.9,
            srs_reachable=True,
            allow_180=False,
            reachability_node_limit=8000,
        ),
        candidate_width=args.candidate_width,
        opponent_reply_width=args.reply_width,
    ).normalized()


def _run_benchmark(args) -> dict[str, object]:
    solo, versus, solo_stats, versus_stats = _build_profiled_evaluators(args.device)
    search_config = _search_config(args)
    with _GpuSampler() as gpu, collect_versus_profile() as profile:
        started = time.perf_counter()
        result = run_versus_benchmark(
            args.games,
            max_turns=args.max_turns,
            seed_base=args.seed_base,
            seed_step=args.seed_step,
            player_config=search_config,
            ai_config=search_config,
            player_scorer=solo,
            ai_scorer=solo,
            player_state_scorer=versus,
            ai_state_scorer=versus,
            progress=False,
            game_batch=args.game_batch,
        )
        elapsed = time.perf_counter() - started
    turns = sum(game.turns for game in result.per_game)
    return _summarize_run(
        kind="benchmark",
        elapsed=elapsed,
        games=args.games,
        turns=turns,
        profile=profile,
        solo_stats=solo_stats,
        versus_stats=versus_stats,
        gpu=gpu,
    )


def _run_selfplay(args) -> dict[str, object]:
    solo, versus, solo_stats, versus_stats = _build_profiled_evaluators(args.device)
    search_config = _search_config(args)
    with tempfile.TemporaryDirectory(prefix="minoflux-versus-profile-") as directory:
        target = Path(directory) / "selfplay.jsonl"
        with _GpuSampler() as gpu, collect_versus_profile() as profile:
            started = time.perf_counter()
            result = generate_versus_selfplay_dataset(
                target,
                solo,
                VersusSelfPlayConfig(
                    games=args.games,
                    max_turns=args.max_turns,
                    seed_base=args.seed_base,
                    seed_step=args.seed_step,
                    garbage_cap=8,
                    search_config=search_config,
                    game_batch=args.game_batch,
                    max_records_per_game=0,
                ),
                value_scorer=versus,
                ai_scorer=solo,
                ai_value_scorer=versus,
            )
            elapsed = time.perf_counter() - started
    turns = round(float(result["meanTurns"]) * int(result["games"]))
    summary = _summarize_run(
        kind="selfplay",
        elapsed=elapsed,
        games=args.games,
        turns=turns,
        profile=profile,
        solo_stats=solo_stats,
        versus_stats=versus_stats,
        gpu=gpu,
    )
    summary["records"] = int(result["records"])
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile exact neural versus throughput")
    parser.add_argument("--mode", choices=("benchmark", "selfplay", "both"), default="both")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--seed-base", type=int, default=9_100_001)
    parser.add_argument("--seed-step", type=int, default=31)
    parser.add_argument("--candidate-width", type=int, default=16)
    parser.add_argument("--reply-width", type=int, default=4)
    parser.add_argument("--game-batch", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = []
    if args.mode in {"benchmark", "both"}:
        runs.append(_run_benchmark(args))
    if args.mode in {"selfplay", "both"}:
        runs.append(_run_selfplay(args))
    report = {
        "format": "minoflux_versus_neural_profile_v1",
        "conditions": {
            "games": args.games,
            "maxTurns": args.max_turns,
            "seedBase": args.seed_base,
            "seedStep": args.seed_step,
            "candidateWidth": args.candidate_width,
            "opponentReplyWidth": args.reply_width,
            "gameBatch": args.game_batch,
            "device": args.device,
            "precision": "float32",
            "hold": True,
            "exactSrs": True,
        },
        "runs": runs,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
