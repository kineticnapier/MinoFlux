from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from minoflux_engine import VersusMatch

from .features import extract_board_features
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .progress import progress_bar
from .search import SearchScorer, apply_search_action
from .versus_search import (
    DEFAULT_VERSUS_SEARCH_CONFIG,
    SideName,
    VersusSearchConfig,
    VersusSearchRequest,
    VersusStateScorer,
    choose_versus_actions_batch,
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
    player_mean_canceled: float = 0.0
    ai_mean_canceled: float = 0.0
    player_mean_received: float = 0.0
    ai_mean_received: float = 0.0
    player_mean_pieces: float = 0.0
    ai_mean_pieces: float = 0.0
    per_game: tuple[VersusGameResult, ...] = ()

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
    progress: bool = False,
    progress_desc: str | None = None,
) -> VersusGameResult:
    match = VersusMatch(seed, garbage_cap=garbage_cap)
    limit = max(1, int(max_turns))
    turn_side = "player" if player_starts else "ai"
    turns = 0
    player_max_b2b = ai_max_b2b = 0
    player_max_surge = ai_max_surge = 0
    turn_bar = progress_bar(
        total=limit,
        desc=progress_desc or f"seed {seed}",
        unit="turn",
        leave=False,
        disable=not progress,
    )

    try:
        while match.winner is None and turns < limit:
            weights = player_weights if turn_side == "player" else ai_weights
            opponent_weights = ai_weights if turn_side == "player" else player_weights
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
                opponent_heuristic_weights=opponent_weights,
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
            turn_bar.update(1)
            player_max_b2b = max(player_max_b2b, match.player.game.b2b_chain)
            ai_max_b2b = max(ai_max_b2b, match.ai.game.b2b_chain)
            player_max_surge = max(player_max_surge, match.player.game.surge_charge)
            ai_max_surge = max(ai_max_surge, match.ai.game.surge_charge)
            turn_side = "ai" if turn_side == "player" else "player"
    finally:
        turn_bar.close()

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


@dataclass(slots=True)
class _BatchedVersusGame:
    index: int
    seed: int
    swapped: bool
    match: VersusMatch
    turn_side: SideName = "player"
    turns: int = 0
    player_max_b2b: int = 0
    ai_max_b2b: int = 0
    player_max_surge: int = 0
    ai_max_surge: int = 0


def _batched_game_result(game: _BatchedVersusGame) -> VersusGameResult:
    match = game.match
    player_board = extract_board_features(match.player.game.board)
    ai_board = extract_board_features(match.ai.game.board)
    physical = VersusGameResult(
        seed=game.seed,
        winner=match.winner or "draw",
        turns=game.turns,
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
        player_max_b2b=game.player_max_b2b,
        ai_max_b2b=game.ai_max_b2b,
        player_max_surge=game.player_max_surge,
        ai_max_surge=game.ai_max_surge,
    )
    return _remap_swapped_result(physical) if game.swapped else physical


def _summarize_benchmark(
    results: tuple[VersusGameResult, ...],
    *,
    max_turns: int,
    seed_base: int,
    seed_step: int,
) -> VersusBenchmarkResult:
    count = len(results)
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
        player_mean_canceled=sum(item.player_canceled for item in results) / count,
        ai_mean_canceled=sum(item.ai_canceled for item in results) / count,
        player_mean_received=sum(item.player_received for item in results) / count,
        ai_mean_received=sum(item.ai_received for item in results) / count,
        player_mean_pieces=sum(item.player_pieces for item in results) / count,
        ai_mean_pieces=sum(item.ai_pieces for item in results) / count,
        per_game=results,
    )


def _run_versus_benchmark_batched(
    count: int,
    *,
    game_batch: int,
    max_turns: int,
    seed_base: int,
    seed_step: int,
    player_weights: HeuristicWeights,
    ai_weights: HeuristicWeights,
    player_config: VersusSearchConfig,
    ai_config: VersusSearchConfig,
    garbage_cap: int,
    player_scorer: SearchScorer | None,
    ai_scorer: SearchScorer | None,
    player_state_scorer: VersusStateScorer | None,
    ai_state_scorer: VersusStateScorer | None,
    progress: bool,
) -> VersusBenchmarkResult:
    limit = max(1, int(max_turns))
    batch_size = max(1, int(game_batch))
    results: list[VersusGameResult | None] = [None] * count
    active: list[_BatchedVersusGame] = []
    next_index = 0
    completed = 0
    game_bar = progress_bar(
        total=count,
        desc="Benchmark",
        unit="game",
        disable=not progress,
    )

    def refill() -> None:
        nonlocal next_index
        while len(active) < batch_size and next_index < count:
            index = next_index
            seed = int(seed_base) + (index // 2) * int(seed_step)
            active.append(
                _BatchedVersusGame(
                    index=index,
                    seed=seed,
                    swapped=index % 2 == 1,
                    match=VersusMatch(seed, garbage_cap=garbage_cap),
                )
            )
            next_index += 1

    try:
        refill()
        while active:
            requests: list[VersusSearchRequest] = []
            for game in active:
                swapped = game.swapped
                physical_player_weights = ai_weights if swapped else player_weights
                physical_ai_weights = player_weights if swapped else ai_weights
                physical_player_config = ai_config if swapped else player_config
                physical_ai_config = player_config if swapped else ai_config
                physical_player_scorer = ai_scorer if swapped else player_scorer
                physical_ai_scorer = player_scorer if swapped else ai_scorer
                physical_player_state = ai_state_scorer if swapped else player_state_scorer
                physical_ai_state = player_state_scorer if swapped else ai_state_scorer
                player_turn = game.turn_side == "player"
                requests.append(
                    VersusSearchRequest(
                        match=game.match,
                        side_name=game.turn_side,
                        heuristic_weights=(
                            physical_player_weights if player_turn else physical_ai_weights
                        ),
                        config=physical_player_config if player_turn else physical_ai_config,
                        scorer=physical_player_scorer if player_turn else physical_ai_scorer,
                        opponent_scorer=(
                            physical_ai_scorer if player_turn else physical_player_scorer
                        ),
                        opponent_heuristic_weights=(
                            physical_ai_weights if player_turn else physical_player_weights
                        ),
                        state_scorer=physical_player_state if player_turn else physical_ai_state,
                    )
                )

            choices = choose_versus_actions_batch(tuple(requests))
            finished: list[_BatchedVersusGame] = []
            for game, choice in zip(active, choices):
                if choice is None:
                    game.match.side(game.turn_side).game.game_over = True
                    game.match._update_winner()
                else:
                    side = game.match.side(game.turn_side)
                    result = apply_search_action(side.game, choice.action)
                    game.match.resolve_lock(game.turn_side, result)
                    game.turns += 1
                    game.player_max_b2b = max(
                        game.player_max_b2b, game.match.player.game.b2b_chain
                    )
                    game.ai_max_b2b = max(game.ai_max_b2b, game.match.ai.game.b2b_chain)
                    game.player_max_surge = max(
                        game.player_max_surge, game.match.player.game.surge_charge
                    )
                    game.ai_max_surge = max(
                        game.ai_max_surge, game.match.ai.game.surge_charge
                    )
                    game.turn_side = "ai" if game.turn_side == "player" else "player"
                if game.match.winner is not None or game.turns >= limit:
                    finished.append(game)

            if finished:
                finished_ids = {id(game) for game in finished}
                active[:] = [game for game in active if id(game) not in finished_ids]
                for game in finished:
                    logical = _batched_game_result(game)
                    results[game.index] = logical
                    completed += 1
                    game_bar.update(1)
                    completed_results = [item for item in results if item is not None]
                    player_wins = sum(item.winner == "player" for item in completed_results)
                    ai_wins = sum(item.winner == "ai" for item in completed_results)
                    draws = len(completed_results) - player_wins - ai_wins
                    game_bar.set_postfix(
                        P=player_wins,
                        A=ai_wins,
                        D=draws,
                        turns=logical.turns,
                        active=len(active),
                    )
                refill()
    finally:
        game_bar.close()

    if completed != count or any(item is None for item in results):
        raise AssertionError("Batched versus benchmark did not finish every game")
    return _summarize_benchmark(
        tuple(item for item in results if item is not None),
        max_turns=limit,
        seed_base=seed_base,
        seed_step=seed_step,
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
    progress: bool = False,
    game_batch: int = 1,
) -> VersusBenchmarkResult:
    count = max(1, int(games))
    if int(game_batch) > 1:
        return _run_versus_benchmark_batched(
            count,
            game_batch=game_batch,
            max_turns=max_turns,
            seed_base=seed_base,
            seed_step=seed_step,
            player_weights=player_weights,
            ai_weights=ai_weights,
            player_config=player_config,
            ai_config=ai_config,
            garbage_cap=garbage_cap,
            player_scorer=player_scorer,
            ai_scorer=ai_scorer,
            player_state_scorer=player_state_scorer,
            ai_state_scorer=ai_state_scorer,
            progress=progress,
        )
    results: list[VersusGameResult] = []
    game_bar = progress_bar(
        range(count),
        total=count,
        desc="Benchmark",
        unit="game",
        disable=not progress,
    )
    for index in game_bar:
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
            progress=progress,
            progress_desc=f"Game {index + 1}/{count}",
        )
        logical = _remap_swapped_result(physical) if swapped else physical
        results.append(logical)
        if progress:
            player_wins = sum(item.winner == "player" for item in results)
            ai_wins = sum(item.winner == "ai" for item in results)
            draws = len(results) - player_wins - ai_wins
            game_bar.set_postfix(P=player_wins, A=ai_wins, D=draws, turns=logical.turns)
    game_bar.close()

    return _summarize_benchmark(
        tuple(results),
        max_turns=max_turns,
        seed_base=seed_base,
        seed_step=seed_step,
    )
