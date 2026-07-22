from __future__ import annotations

from dataclasses import asdict, dataclass
import random

from minoflux_engine import Game


@dataclass(frozen=True, slots=True)
class SimulationResult:
    seed: int
    pieces: int
    lines: int
    attack: int
    score: int
    topout: bool


def choose_random_placement(game: Game, rng: random.Random):
    placements = game.legal_placements()
    return rng.choice(placements) if placements else None


def simulate(seed: int, max_pieces: int = 200) -> SimulationResult:
    game = Game(seed)
    rng = random.Random(seed ^ 0x5F3759DF)
    while not game.game_over and game.pieces_placed < max_pieces:
        placement = choose_random_placement(game, rng)
        if placement is None:
            break
        game.place(placement)
    return SimulationResult(
        seed=seed,
        pieces=game.pieces_placed,
        lines=game.lines,
        attack=game.attack,
        score=game.score,
        topout=game.game_over,
    )


def run_smoke(games: int = 4, max_pieces: int = 200, seed_base: int = 1) -> dict:
    results = [simulate(seed_base + index * 31, max_pieces) for index in range(max(1, games))]
    total_pieces = sum(item.pieces for item in results)
    return {
        "games": len(results),
        "maxPieces": max_pieces,
        "pieces": total_pieces,
        "meanPieces": total_pieces / len(results),
        "lines": sum(item.lines for item in results),
        "attack": sum(item.attack for item in results),
        "topouts": sum(item.topout for item in results),
        "perGame": [asdict(item) for item in results],
    }
