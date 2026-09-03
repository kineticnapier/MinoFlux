from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from minoflux_engine import VersusMatch

from .features import extract_board_features
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .search import SearchScorer, apply_search_action
from .versus_search import (
    DEFAULT_VERSUS_SEARCH_CONFIG,
    VersusSearchConfig,
    VersusStateScorer,
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
    player_garbage_applied: int
    ai_garbage_applied: int
    player_pending: int
    ai_pending: int
    player_final_height: int
    ai_final_height: int
    player_final_holes: int
    ai_final_holes: int
    player_max_b2b: int
    ai_max_b2b: int
    player_max_surge: int
    ai_max_surge: int
    models_swapped: bool = False


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
    player_mean_canceled: float
    ai_mean_canceled: float
    player_mean_received: float
    ai_mean_received: float
    player_mean_pieces: float
    ai_mean_pieces: float
    per_game: tuple[VersusGameResult, ...]

    def to_dict(self) -> dict[str, object]:
        player_pieces = max(1.0, sum(item.player_pieces for item in self.per_game))
        ai_pieces = max(1.0, sum(item.ai_pieces for item in self.per_game))
        return {
            "games": self.games,
            "maxTurns": self.max_turns,
            "seedBase": self.seed_base,
            "seedStep": self.seed_step,
            "mirroredSides": True,
            "playerWins": self.player_wins,
            "aiWins": self.ai_wins,
            "draws": self.draws,
            "playerWinRate": self.player_wins / self.games,
            "aiWinRate": self.ai_wins / self.games,
            "meanTurns": self.mean_turns,
            "playerMeanPieces": self.player_mean_pieces,
            "aiMeanPieces": self.ai_mean_pieces,
            "playerMeanAttack": self.player_mean_attack,
            "aiMeanAttack": self.ai_mean_attack,
            "playerMeanSent": self.player_mean_sent,
            "aiMeanSent": self.ai_mean_sent,
            "playerMeanCanceled": self.player_mean_canceled,
            "aiMeanCanceled": self.ai_mean_canceled,
            "playerMeanReceived": self.player_mean_received,
            "aiMeanReceived": self.ai_mean_received,
            "playerSentPerPiece": sum(item.player_sent for item in self.per_game) / player_pieces,
            "aiSentPerPiece": sum(item.ai_sent for item in self.per_game) / ai_pieces,
            "playerAttackPerPiece": sum(item.player_attack for item in self.per_game) / player_pieces,
            "aiAttackPerPiece": sum(item.ai_attack for item in self.per_game) / ai_pieces,
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
    player_scorer: SearchScorer | None = None,
    ai_scorer: SearchScorer | None = None,
    player_state_scorer: VersusStateScorer | None = None,
    ai_state_scorer: VersusStateScorer | None = None,
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
        scorer = player_scorer if turn_side == "player" else ai_scorer
        opponent_scorer = ai_scorer if turn_side == "player" else player_scorer
        state_scorer = player_state_scorer if turn_side == "player" else ai_state_scorer
        choice = choose_versus_action(
            match,
            turn_side,
            weights,
            config,
            scorer=scorer,
            opponent_scorer=opponent_scorer,
            state_scorer=state_scorer,
        )
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
    player_board = extract_board_features(match.player.game.board)
    ai_board = extract_board_features(match.ai.game.board)
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
        player_garbage_applied=match.player.garbage_applied,
        ai_garbage_applied=match.ai.garbage_applied,
        player_pending=match.player.pending.pending_lines,
        ai_pending=match.ai.pending.pending_lines,
        player_final_height=player_board.max_height,
        ai_final_height=ai_board.max_height,
        player_final_holes=player_board.holes,
        ai_final_holes=ai_board.holes,
        player_max_b2b=player_max_b2b,
        ai_max_b2b=ai_max_b2b,
        player_max_surge=player_max_surge,
        ai_max_surge=ai_max_surge,
    )


def _remap_swapped_result(result: VersusGameResult) -> VersusGameResult:
    winner = {"player": "ai", "ai": "player", "draw": "draw"}[result.winner]
    return replace(
        result,
        winner=winner,
        player_pieces=result.ai_pieces,
        ai_pieces=result.player_pieces,
        player_attack=result.ai_attack,
        ai_attack=result.player_attack,
        player_sent=result.ai_sent,
        ai_sent=result.player_sent,
        player_canceled=result.ai_canceled,
        ai_canceled=result.player_canceled,
        player_received=result.ai_received,
        ai_received=result.player_received,
        player_garbage_applied=result.ai_garbage_applied,
        ai_garbage_applied=result.player_garbage_applied,
        player_pending=result.ai_pending,
        ai_pending=result.player_pending,
        player_final_height=result.ai_final_height,
        ai_final_height=result.player_final_height,
        player_final_holes=result.ai_final_holes,
        ai_final_holes=result.player_final_holes,
        player_max_b2b=result.ai_max_b2b,
        ai_max_b2b=result.player_max_b2b,
        player_max_surge=result.ai_max_surge,
        ai_max_surge=result.player_max_surge,
        models_swapped=True,
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
    player_scorer: SearchScorer | None = None,
    ai_scorer: SearchScorer | None = None,
    player_state_scorer: VersusStateScorer | None = None,
    ai_state_scorer: VersusStateScorer | None = None,
) -> VersusBenchmarkResult:
    count = max(1, int(games))
    results: list[VersusGameResult] = []
    for index in range(count):
        swapped = index % 2 == 1
        seed = int(seed_base) + (index // 2) * int(seed_step)
        physical = run_versus_game(
            seed,
            max_turns=max_turns,
            player_weights=ai_weights if swapped else player_weights,
            ai_weights=player_weights if swapped else ai_weights,
            player_config=ai_config if swapped else player_config,
            ai_config=player_config if swapped else ai_config,
            garbage_cap=garbage_cap,
            player_starts=True,
            player_scorer=ai_scorer if swapped else player_scorer,
            ai_scorer=player_scorer if swapped else ai_scorer,
            player_state_scorer=ai_state_scorer if swapped else player_state_scorer,
            ai_state_scorer=player_state_scorer if swapped else ai_state_scorer,
        )
        results.append(_remap_swapped_result(physical) if swapped else physical)

    result_tuple = tuple(results)
    return VersusBenchmarkResult(
        games=count,
        max_turns=max(1, int(max_turns)),
        seed_base=int(seed_base),
        seed_step=int(seed_step),
        player_wins=sum(item.winner == "player" for item in result_tuple),
        ai_wins=sum(item.winner == "ai" for item in result_tuple),
        draws=sum(item.winner == "draw" for item in result_tuple),
        mean_turns=sum(item.turns for item in result_tuple) / count,
        player_mean_attack=sum(item.player_attack for item in result_tuple) / count,
        ai_mean_attack=sum(item.ai_attack for item in result_tuple) / count,
        player_mean_sent=sum(item.player_sent for item in result_tuple) / count,
        ai_mean_sent=sum(item.ai_sent for item in result_tuple) / count,
        player_mean_canceled=sum(item.player_canceled for item in result_tuple) / count,
        ai_mean_canceled=sum(item.ai_canceled for item in result_tuple) / count,
        player_mean_received=sum(item.player_received for item in result_tuple) / count,
        ai_mean_received=sum(item.ai_received for item in result_tuple) / count,
        player_mean_pieces=sum(item.player_pieces for item in result_tuple) / count,
        ai_mean_pieces=sum(item.ai_pieces for item in result_tuple) / count,
        per_game=result_tuple,
    )
