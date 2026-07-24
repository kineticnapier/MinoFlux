from __future__ import annotations

from dataclasses import asdict, dataclass

from minoflux_engine import VersusMatch

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .search import apply_search_action
from .versus_search import (
    DEFAULT_VERSUS_SEARCH_CONFIG,
    VersusSearchConfig,
    choose_versus_action,
)


@dataclass(frozen=True, slots=True)
class VersusGameResult:
    seed: int
    winner: str
    turns: int
    player_pieces: int
    ai_pieces: int
    player_attack: int
    ai_attack: int
    player_sent: int
    ai_sent: int
    player_canceled: int
    ai_canceled: int
    player_received: int
    ai_received: int
    player_max_b2b: int
    ai_max_b2b: int
    player_max_surge: int
    ai_max_surge: int


@dataclass(frozen=True, slots=True)
class VersusBenchmarkResult:
    games: int
    max_turns: int
    seed_base: int
    seed_step: int
    player_wins: int
    ai_wins: int
    draws: int
    mean_turns: float
    player_mean_attack: float
    ai_mean_attack: float
    player_mean_sent: float
    ai_mean_sent: float
    per_game: tuple[VersusGameResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "maxTurns": self.max_turns,
            "seedBase": self.seed_base,
            "seedStep": self.seed_step,
            "playerWins": self.player_wins,
            "aiWins": self.ai_wins,
            "draws": self.draws,
            "playerWinRate": self.player_wins / self.games,
            "aiWinRate": self.ai_wins / self.games,
            "meanTurns": self.mean_turns,
            "playerMeanAttack": self.player_mean_attack,
            "aiMeanAttack": self.ai_mean_attack,
            "playerMeanSent": self.player_mean_sent,
            "aiMeanSent": self.ai_mean_sent,
            "perGame": [asdict(item) for item in self.per_game],
        }


def run_versus_game(
    seed: int,
    *,
    max_turns: int = 1000,
    player_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    ai_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    player_config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG,
    ai_config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG,
    garbage_cap: int = 8,
    player_starts: bool = True,
) -> VersusGameResult:
    match = VersusMatch(seed, garbage_cap=garbage_cap)
    limit = max(1, int(max_turns))
    turn_side = "player" if player_starts else "ai"
    turns = 0
    player_max_b2b = ai_max_b2b = 0
    player_max_surge = ai_max_surge = 0

    while match.winner is None and turns < limit:
        weights = player_weights if turn_side == "player" else ai_weights
        config = player_config if turn_side == "player" else ai_config
        choice = choose_versus_action(match, turn_side, weights, config)
        if choice is None:
            match.side(turn_side).game.game_over = True
            match._update_winner()
            break
        side = match.side(turn_side)
        result = apply_search_action(side.game, choice.action)
        match.resolve_lock(turn_side, result)
        turns += 1
        player_max_b2b = max(player_max_b2b, match.player.game.b2b_chain)
        ai_max_b2b = max(ai_max_b2b, match.ai.game.b2b_chain)
        player_max_surge = max(player_max_surge, match.player.game.surge_charge)
        ai_max_surge = max(ai_max_surge, match.ai.game.surge_charge)
        turn_side = "ai" if turn_side == "player" else "player"

    winner = match.winner or "draw"
    return VersusGameResult(
        seed=int(seed),
        winner=winner,
        turns=turns,
        player_pieces=match.player.game.pieces_placed,
        ai_pieces=match.ai.game.pieces_placed,
        player_attack=match.player.game.attack,
        ai_attack=match.ai.game.attack,
        player_sent=match.player.sent,
        ai_sent=match.ai.sent,
        player_canceled=match.player.canceled,
        ai_canceled=match.ai.canceled,
        player_received=match.player.received,
        ai_received=match.ai.received,
        player_max_b2b=player_max_b2b,
        ai_max_b2b=ai_max_b2b,
        player_max_surge=player_max_surge,
        ai_max_surge=ai_max_surge,
    )


def run_versus_benchmark(
    games: int = 10,
    *,
    max_turns: int = 1000,
    seed_base: int = 1,
    seed_step: int = 31,
    player_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    ai_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    player_config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG,
    ai_config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG,
    garbage_cap: int = 8,
) -> VersusBenchmarkResult:
    count = max(1, int(games))
    results = tuple(
        run_versus_game(
            int(seed_base) + index * int(seed_step),
            max_turns=max_turns,
            player_weights=player_weights,
            ai_weights=ai_weights,
            player_config=player_config,
            ai_config=ai_config,
            garbage_cap=garbage_cap,
            player_starts=index % 2 == 0,
        )
        for index in range(count)
    )
    return VersusBenchmarkResult(
        games=count,
        max_turns=max(1, int(max_turns)),
        seed_base=int(seed_base),
        seed_step=int(seed_step),
        player_wins=sum(item.winner == "player" for item in results),
        ai_wins=sum(item.winner == "ai" for item in results),
        draws=sum(item.winner == "draw" for item in results),
        mean_turns=sum(item.turns for item in results) / count,
        player_mean_attack=sum(item.player_attack for item in results) / count,
        ai_mean_attack=sum(item.ai_attack for item in results) / count,
        player_mean_sent=sum(item.player_sent for item in results) / count,
        ai_mean_sent=sum(item.ai_sent for item in results) / count,
        per_game=results,
    )
