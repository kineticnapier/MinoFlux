from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from multiprocessing import current_process
import os

from minoflux_engine import (
    Game,
    T_SPIN_DOUBLE,
    T_SPIN_MINI,
    T_SPIN_MINI_SINGLE,
    T_SPIN_SINGLE,
    T_SPIN_TRIPLE,
)

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .replay import Replay, ReplayStep, ReplaySummary
from .search import DEFAULT_SEARCH_CONFIG, SearchConfig, apply_search_action, choose_search_action


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
    t_spin_minis: int = 0
    t_spin_mini_singles: int = 0
    t_spin_singles: int = 0
    t_spin_doubles: int = 0
    t_spin_triples: int = 0


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    games: int
    max_pieces: int
    seed_base: int
    seed_step: int
    search_config: SearchConfig
    workers: int = field(compare=False)
    pieces: int = 0
    mean_pieces: float = 0.0
    lines: int = 0
    mean_lines: float = 0.0
    attack: int = 0
    mean_attack: float = 0.0
    spins: int = 0
    mean_spins: float = 0.0
    spin_lines: int = 0
    mean_spin_lines: float = 0.0
    t_spin_minis: int = 0
    mean_t_spin_minis: float = 0.0
    t_spin_mini_singles: int = 0
    mean_t_spin_mini_singles: float = 0.0
    t_spin_singles: int = 0
    mean_t_spin_singles: float = 0.0
    t_spin_doubles: int = 0
    mean_t_spin_doubles: float = 0.0
    t_spin_triples: int = 0
    mean_t_spin_triples: float = 0.0
    perfect_clears: int = 0
    mean_perfect_clears: float = 0.0
    topouts: int = 0
    completed: int = 0
    per_game: tuple[BenchmarkGame, ...] = ()
    best_game: BenchmarkGame | None = None
    best_replay: Replay | None = None

    def to_dict(self, *, include_replay: bool = False) -> dict[str, object]:
        if self.best_game is None:
            raise ValueError("Benchmark result has no best game")
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
            "tSpinMinis": self.t_spin_minis,
            "meanTSpinMinis": self.mean_t_spin_minis,
            "tSpinMiniSingles": self.t_spin_mini_singles,
            "meanTSpinMiniSingles": self.mean_t_spin_mini_singles,
            "tSpinSingles": self.t_spin_singles,
            "meanTSpinSingles": self.mean_t_spin_singles,
            "tSpinDoubles": self.t_spin_doubles,
            "meanTSpinDoubles": self.mean_t_spin_doubles,
            "tSpinTriples": self.t_spin_triples,
            "meanTSpinTriples": self.mean_t_spin_triples,
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
    counts = {
        "spins": 0,
        "spin_lines": 0,
        "t_spin_minis": 0,
        "t_spin_mini_singles": 0,
        "t_spin_singles": 0,
        "t_spin_doubles": 0,
        "t_spin_triples": 0,
        "perfect_clears": 0,
    }
    while not game.game_over and game.pieces_placed < limit:
        choice = choose_search_action(game, weights, cfg)
        if choice is None:
            break
        action = choice.action
        result = apply_search_action(game, action)
        placement = action.placement
        if result.spin is not None:
            counts["spins"] += 1
            counts["spin_lines"] += result.lines
        if result.spin == T_SPIN_MINI:
            counts["t_spin_minis"] += 1
        elif result.spin == T_SPIN_MINI_SINGLE:
            counts["t_spin_mini_singles"] += 1
        elif result.spin == T_SPIN_SINGLE:
            counts["t_spin_singles"] += 1
        elif result.spin == T_SPIN_DOUBLE:
            counts["t_spin_doubles"] += 1
        elif result.spin == T_SPIN_TRIPLE:
            counts["t_spin_triples"] += 1
        if result.perfect_clear:
            counts["perfect_clears"] += 1
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
                    path=placement.path,
                    last_move_was_rotation=placement.last_move_was_rotation,
                    rotation_kick_index=placement.rotation_kick_index,
                    rotation_from=placement.rotation_from,
                    rotation_to=placement.rotation_to,
                )
            )
    summary = BenchmarkGame(
        seed=int(seed),
        pieces=game.pieces_placed,
        lines=game.lines,
        attack=game.attack,
        score=game.score,
        topout=game.game_over,
        completed=not game.game_over and game.pieces_placed >= limit,
        **counts,
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
    return _play_heuristic_game(seed, max_pieces, weights, search_config, record_replay=False)[0]


def record_heuristic_game(
    seed: int,
    max_pieces: int = 500,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[BenchmarkGame, Replay]:
    summary, replay = _play_heuristic_game(seed, max_pieces, weights, search_config, record_replay=True)
    assert replay is not None
    return summary, replay


def _best_game_key(game: BenchmarkGame) -> tuple[int, int, int, int, int, int]:
    return game.pieces, game.attack, game.spin_lines, game.lines, game.score, -game.seed


def _resolve_benchmark_workers(requested: int, games: int) -> int:
    count = max(1, int(games))
    value = int(requested)
    if value > 0:
        return min(count, value)
    if current_process().name != "MainProcess":
        return 1
    available = max(1, (os.cpu_count() or 1) - 1)
    return min(count, available)


def _run_game_task(task: tuple[int, int, HeuristicWeights, SearchConfig]) -> BenchmarkGame:
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
    requested_workers = (0 if record_best_replay else 1) if workers is None else int(workers)
    resolved_workers = _resolve_benchmark_workers(requested_workers, count)
    tasks = [(int(seed_base) + index * step, limit, weights, cfg) for index in range(count)]
    if resolved_workers > 1:
        with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
            results = tuple(executor.map(_run_game_task, tasks, chunksize=1))
    else:
        results = tuple(map(_run_game_task, tasks))

    totals = {
        name: sum(getattr(item, name) for item in results)
        for name in (
            "pieces", "lines", "attack", "spins", "spin_lines", "t_spin_minis",
            "t_spin_mini_singles", "t_spin_singles", "t_spin_doubles", "t_spin_triples",
            "perfect_clears",
        )
    }
    best_game = max(results, key=_best_game_key)
    best_replay = None
    if record_best_replay:
        replay_summary, best_replay = record_heuristic_game(best_game.seed, limit, weights, cfg)
        if replay_summary != best_game:
            raise RuntimeError("Recorded replay did not reproduce the benchmark result")
    return BenchmarkResult(
        games=count,
        max_pieces=limit,
        seed_base=int(seed_base),
        seed_step=step,
        search_config=cfg,
        workers=resolved_workers,
        pieces=totals["pieces"],
        mean_pieces=totals["pieces"] / count,
        lines=totals["lines"],
        mean_lines=totals["lines"] / count,
        attack=totals["attack"],
        mean_attack=totals["attack"] / count,
        spins=totals["spins"],
        mean_spins=totals["spins"] / count,
        spin_lines=totals["spin_lines"],
        mean_spin_lines=totals["spin_lines"] / count,
        t_spin_minis=totals["t_spin_minis"],
        mean_t_spin_minis=totals["t_spin_minis"] / count,
        t_spin_mini_singles=totals["t_spin_mini_singles"],
        mean_t_spin_mini_singles=totals["t_spin_mini_singles"] / count,
        t_spin_singles=totals["t_spin_singles"],
        mean_t_spin_singles=totals["t_spin_singles"] / count,
        t_spin_doubles=totals["t_spin_doubles"],
        mean_t_spin_doubles=totals["t_spin_doubles"] / count,
        t_spin_triples=totals["t_spin_triples"],
        mean_t_spin_triples=totals["t_spin_triples"] / count,
        perfect_clears=totals["perfect_clears"],
        mean_perfect_clears=totals["perfect_clears"] / count,
        topouts=sum(item.topout for item in results),
        completed=sum(item.completed for item in results),
        per_game=results,
        best_game=best_game,
        best_replay=best_replay,
    )
