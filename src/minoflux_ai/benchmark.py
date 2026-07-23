from __future__ import annotations

from dataclasses import asdict, dataclass

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
    pieces: int
    mean_pieces: float
    lines: int
    mean_lines: float
    attack: int
    mean_attack: float
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
            "pieces": self.pieces,
            "meanPieces": self.mean_pieces,
            "lines": self.lines,
            "meanLines": self.mean_lines,
            "attack": self.attack,
            "meanAttack": self.mean_attack,
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
    while not game.game_over and game.pieces_placed < limit:
        choice = choose_search_action(game, weights, cfg)
        if choice is None:
            break
        action = choice.action
        result = apply_search_action(game, action)
        placement = action.placement
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


def _best_game_key(game: BenchmarkGame) -> tuple[int, int, int, int, int]:
    return game.pieces, game.lines, game.attack, game.score, -game.seed


def run_heuristic_benchmark(
    games: int = 8,
    max_pieces: int = 500,
    seed_base: int = 1,
    seed_step: int = 31,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    *,
    record_best_replay: bool = False,
) -> BenchmarkResult:
    count = max(1, int(games))
    limit = max(1, int(max_pieces))
    step = int(seed_step)
    cfg = search_config.normalized()
    results = tuple(
        run_heuristic_game(
            int(seed_base) + index * step,
            limit,
            weights,
            cfg,
        )
        for index in range(count)
    )
    pieces = sum(item.pieces for item in results)
    lines = sum(item.lines for item in results)
    attack = sum(item.attack for item in results)
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
        pieces=pieces,
        mean_pieces=pieces / count,
        lines=lines,
        mean_lines=lines / count,
        attack=attack,
        mean_attack=attack / count,
        topouts=sum(item.topout for item in results),
        completed=sum(item.completed for item in results),
        per_game=results,
        best_game=best_game,
        best_replay=best_replay,
    )
