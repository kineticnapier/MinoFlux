from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import current_process
import os

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .replay import Replay, ReplayStep, ReplaySummary
from .search import (
    DEFAULT_SEARCH_CONFIG,
    SearchConfig,
    apply_search_action,
    choose_search_action,
)


@dataclass(frozen=True, slots=True)
class BenchmarkGame:
    seed: int
    pieces: int
    lines: int
    attack: int
    spins: int
    spin_lines: int
    perfect_clears: int
    score: int
    topout: bool
    completed: bool


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    games: int
    max_pieces: int
    seed_base: int
    seed_step: int
    search_config: SearchConfig
    workers: int = field(compare=False)
    pieces: int
    mean_pieces: float
    lines: int
    mean_lines: float
    attack: int
    mean_attack: float
    spins: int
    mean_spins: float
    spin_lines: int
    mean_spin_lines: float
    perfect_clears: int
    mean_perfect_clears: float
    topouts: int
    completed: int
    per_game: tuple[BenchmarkGame, ...]
    best_game: BenchmarkGame
    best_replay: Replay | None = None

    def to_dict(self, *, include_replay: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "games": self.games,
            "maxPieces": self.max_pieces,
            "seedBase": self.seed_base,
            "seedStep": self.seed_step,
            "searchConfig": self.search_config.to_dict(),
            "workers": self.workers,
            "pieces": self.pieces,
            "meanPieces": self.mean_pieces,
            "lines": self.lines,
            "meanLines": self.mean_lines,
            "attack": self.attack,
            "meanAttack": self.mean_attack,
            "spins": self.spins,
            "meanSpins": self.mean_spins,
            "spinLines": self.spin_lines,
            "meanSpinLines": self.mean_spin_lines,
            "perfectClears": self.perfect_clears,
            "meanPerfectClears": self.mean_perfect_clears,
            "topouts": self.topouts,
            "completed": self.completed,
            "bestGame": asdict(self.best_game),
            "perGame": [asdict(item) for item in self.per_game],
        }
        if self.best_replay is not None:
            result["bestReplay"] = (
                self.best_replay.to_dict()
                if include_replay
                else {
                    "format": self.best_replay.format,
                    "seed": self.best_replay.seed,
                    "steps": len(self.best_replay.steps),
                }
            )
        return result


def _play_heuristic_game(
    seed: int,
    max_pieces: int,
    weights: HeuristicWeights,
    search_config: SearchConfig,
    *,
    record_replay: bool,
) -> tuple[BenchmarkGame, Replay | None]:
    limit = max(1, int(max_pieces))
    cfg = search_config.normalized()
    game = Game(int(seed))
    steps: list[ReplayStep] = []
    spins = 0
    spin_lines = 0
    perfect_clears = 0
    while not game.game_over and game.pieces_placed < limit:
        choice = choose_search_action(game, weights, cfg)
        if choice is None:
            break
        action = choice.action
        result = apply_search_action(game, action)
        placement = action.placement
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
        if result.perfect_clear:
            perfect_clears += 1
        if record_replay:
            steps.append(
                ReplayStep(
                    piece=placement.piece,
                    x=placement.x,
                    y=placement.y,
                    rotation=placement.rotation,
                    lines=result.lines,
                    attack=result.attack,
                    score=game.score,
                    total_lines=game.lines,
                    total_attack=game.attack,
                    hold=action.use_hold,
                    spin=result.spin,
                    perfect_clear=result.perfect_clear,
                )
            )
    summary = BenchmarkGame(
        seed=int(seed),
        pieces=game.pieces_placed,
        lines=game.lines,
        attack=game.attack,
        spins=spins,
        spin_lines=spin_lines,
        perfect_clears=perfect_clears,
        score=game.score,
        topout=game.game_over,
        completed=not game.game_over and game.pieces_placed >= limit,
    )
    replay = None
    if record_replay:
        replay = Replay(
            seed=int(seed),
            max_pieces=limit,
            weights=weights.to_dict(),
            search_config=cfg.to_dict(),
            steps=tuple(steps),
            final=ReplaySummary(
                pieces=summary.pieces,
                lines=summary.lines,
                attack=summary.attack,
                score=summary.score,
                topout=summary.topout,
            ),
        )
    return summary, replay


def run_heuristic_game(
    seed: int,
    max_pieces: int = 500,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> BenchmarkGame:
    return _play_heuristic_game(
        seed,
        max_pieces,
        weights,
        search_config,
        record_replay=False,
    )[0]


def record_heuristic_game(
    seed: int,
    max_pieces: int = 500,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[BenchmarkGame, Replay]:
    summary, replay = _play_heuristic_game(
        seed,
        max_pieces,
        weights,
        search_config,
        record_replay=True,
    )
    assert replay is not None
    return summary, replay


def _best_game_key(game: BenchmarkGame) -> tuple[int, int, int, int, int, int, int]:
    return (
        game.pieces,
        game.attack,
        game.spin_lines,
        game.spins,
        game.lines,
        game.score,
        -game.seed,
    )


def _resolve_benchmark_workers(requested: int, games: int) -> int:
    count = max(1, int(games))
    value = int(requested)
    if value > 0:
        return min(count, value)
    # Candidate benchmarks already run inside CEM worker processes. Never create
    # a nested pool there, even when an API caller explicitly requests auto mode.
    if current_process().name != "MainProcess":
        return 1
    available = max(1, (os.cpu_count() or 1) - 1)
    return min(count, available)


def _run_game_task(
    task: tuple[int, int, HeuristicWeights, SearchConfig],
) -> BenchmarkGame:
    seed, max_pieces, weights, search_config = task
    return run_heuristic_game(seed, max_pieces, weights, search_config)


def run_heuristic_benchmark(
    games: int = 8,
    max_pieces: int = 500,
    seed_base: int = 1,
    seed_step: int = 31,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    *,
    workers: int | None = None,
    record_best_replay: bool = False,
) -> BenchmarkResult:
    count = max(1, int(games))
    limit = max(1, int(max_pieces))
    step = int(seed_step)
    cfg = search_config.normalized()
    # Interactive benchmark/replay calls default to auto parallelism. Internal
    # fitness calls omit replay recording and stay serial unless workers is set.
    requested_workers = (0 if record_best_replay else 1) if workers is None else int(workers)
    resolved_workers = _resolve_benchmark_workers(requested_workers, count)
    tasks = [
        (int(seed_base) + index * step, limit, weights, cfg)
        for index in range(count)
    ]
    if resolved_workers > 1:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            results = tuple(executor.map(_run_game_task, tasks, chunksize=1))
    else:
        results = tuple(map(_run_game_task, tasks))

    pieces = sum(item.pieces for item in results)
    lines = sum(item.lines for item in results)
    attack = sum(item.attack for item in results)
    spins = sum(item.spins for item in results)
    spin_lines = sum(item.spin_lines for item in results)
    perfect_clears = sum(item.perfect_clears for item in results)
    best_game = max(results, key=_best_game_key)
    best_replay = None
    if record_best_replay:
        replay_summary, best_replay = record_heuristic_game(
            best_game.seed,
            limit,
            weights,
            cfg,
        )
        if replay_summary != best_game:
            raise RuntimeError("Recorded replay did not reproduce the benchmark result")
    return BenchmarkResult(
        games=count,
        max_pieces=limit,
        seed_base=int(seed_base),
        seed_step=step,
        search_config=cfg,
        workers=resolved_workers,
        pieces=pieces,
        mean_pieces=pieces / count,
        lines=lines,
        mean_lines=lines / count,
        attack=attack,
        mean_attack=attack / count,
        spins=spins,
        mean_spins=spins / count,
        spin_lines=spin_lines,
        mean_spin_lines=spin_lines / count,
        perfect_clears=perfect_clears,
        mean_perfect_clears=perfect_clears / count,
        topouts=sum(item.topout for item in results),
        completed=sum(item.completed for item in results),
        per_game=results,
        best_game=best_game,
        best_replay=best_replay,
    )
