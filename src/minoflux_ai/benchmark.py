from __future__ import annotations

from dataclasses import asdict, dataclass

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights, choose_placement


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
    pieces: int
    mean_pieces: float
    lines: int
    mean_lines: float
    attack: int
    mean_attack: float
    topouts: int
    completed: int
    per_game: tuple[BenchmarkGame, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "maxPieces": self.max_pieces,
            "seedBase": self.seed_base,
            "seedStep": self.seed_step,
            "pieces": self.pieces,
            "meanPieces": self.mean_pieces,
            "lines": self.lines,
            "meanLines": self.mean_lines,
            "attack": self.attack,
            "meanAttack": self.mean_attack,
            "topouts": self.topouts,
            "completed": self.completed,
            "perGame": [asdict(item) for item in self.per_game],
        }


def run_heuristic_game(
    seed: int,
    max_pieces: int = 500,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> BenchmarkGame:
    limit = max(1, int(max_pieces))
    game = Game(int(seed))
    while not game.game_over and game.pieces_placed < limit:
        choice = choose_placement(game, weights)
        if choice is None:
            break
        game.place(choice.placement)
    return BenchmarkGame(
        seed=int(seed),
        pieces=game.pieces_placed,
        lines=game.lines,
        attack=game.attack,
        score=game.score,
        topout=game.game_over,
        completed=not game.game_over and game.pieces_placed >= limit,
    )


def run_heuristic_benchmark(
    games: int = 8,
    max_pieces: int = 500,
    seed_base: int = 1,
    seed_step: int = 31,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> BenchmarkResult:
    count = max(1, int(games))
    limit = max(1, int(max_pieces))
    step = int(seed_step)
    results = tuple(run_heuristic_game(int(seed_base) + index * step, limit, weights) for index in range(count))
    pieces = sum(item.pieces for item in results)
    lines = sum(item.lines for item in results)
    attack = sum(item.attack for item in results)
    return BenchmarkResult(
        games=count,
        max_pieces=limit,
        seed_base=int(seed_base),
        seed_step=step,
        pieces=pieces,
        mean_pieces=pieces / count,
        lines=lines,
        mean_lines=lines / count,
        attack=attack,
        mean_attack=attack / count,
        topouts=sum(item.topout for item in results),
        completed=sum(item.completed for item in results),
        per_game=results,
    )
