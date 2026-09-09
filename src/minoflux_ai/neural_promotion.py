from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

from minoflux_engine import Game, LockResult
from minoflux_engine.spin import is_difficult_clear

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .progress import progress_bar
from .search import (
    DEFAULT_SEARCH_CONFIG,
    SearchConfig,
    SearchScorer,
    apply_search_action,
    choose_search_actions_batch,
)
from .versus_benchmark import VersusBenchmarkResult, VersusGameResult

NEURAL_PROMOTION_FORMAT = "minoflux_neural_promotion_benchmark_v1"


@dataclass(frozen=True, slots=True)
class NeuralModelSpec:
    name: str
    path: str
    sha256: str | None = None

    @classmethod
    def from_path(cls, path: str | Path, *, name: str | None = None) -> "NeuralModelSpec":
        target = Path(path)
        digest: str | None = None
        try:
            if target.is_file():
                hasher = hashlib.sha256()
                with target.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
                digest = hasher.hexdigest()
        except OSError:
            digest = None
        text = str(path)
        return cls(name=(name or target.stem), path=text, sha256=digest)

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def normalized_model_path(path: str | Path) -> str:
    """Return a stable local path identity when a checkpoint hash is unavailable."""

    return os.path.normcase(str(Path(path).expanduser().resolve(strict=False)))


def same_neural_model(a: NeuralModelSpec, b: NeuralModelSpec) -> bool:
    """Compare checkpoint identity, preferring content hashes over path aliases."""

    if a.sha256 is not None and b.sha256 is not None:
        return a.sha256.casefold() == b.sha256.casefold()
    return normalized_model_path(a.path) == normalized_model_path(b.path)


@dataclass(frozen=True, slots=True)
class NeuralSoloGameResult:
    seed: int
    pieces: int
    lines: int
    attack: int
    topout: bool
    completed: bool
    t_spin_locks: int
    t_spin_lines: int
    t_spin_attack: int
    difficult_clears: int
    b2b_active_clears: int
    max_b2b_chain: int
    surge_released: int
    surge_release_events: int
    combo_clears: int
    max_combo: int
    perfect_clears: int

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["attackPerPiece"] = self.attack / max(1, self.pieces)
        result["tSpinAttackPerPiece"] = self.t_spin_attack / max(1, self.pieces)
        return result


@dataclass(frozen=True, slots=True)
class NeuralSoloBenchmarkResult:
    model: str
    games: int
    max_pieces: int
    seed_base: int
    seed_step: int
    game_batch_size: int
    search_config: SearchConfig
    elapsed_seconds: float
    per_game: tuple[NeuralSoloGameResult, ...]

    def to_dict(self) -> dict[str, object]:
        games = max(1, self.games)
        pieces = sum(item.pieces for item in self.per_game)
        lines = sum(item.lines for item in self.per_game)
        attack = sum(item.attack for item in self.per_game)
        completed = sum(item.completed for item in self.per_game)
        topouts = sum(item.topout for item in self.per_game)
        t_spin_locks = sum(item.t_spin_locks for item in self.per_game)
        t_spin_lines = sum(item.t_spin_lines for item in self.per_game)
        t_spin_attack = sum(item.t_spin_attack for item in self.per_game)
        difficult_clears = sum(item.difficult_clears for item in self.per_game)
        b2b_active_clears = sum(item.b2b_active_clears for item in self.per_game)
        surge_released = sum(item.surge_released for item in self.per_game)
        surge_release_events = sum(item.surge_release_events for item in self.per_game)
        combo_clears = sum(item.combo_clears for item in self.per_game)
        perfect_clears = sum(item.perfect_clears for item in self.per_game)
        return {
            "model": self.model,
            "games": self.games,
            "maxPieces": self.max_pieces,
            "seedBase": self.seed_base,
            "seedStep": self.seed_step,
            "gameBatchSize": self.game_batch_size,
            "searchConfig": self.search_config.to_dict(),
            "pieces": pieces,
            "meanPieces": pieces / games,
            "meanPiecesSurvived": pieces / games,
            "lines": lines,
            "meanLines": lines / games,
            "attack": attack,
            "meanAttack": attack / games,
            "attackPerPiece": attack / max(1, pieces),
            "topouts": topouts,
            "completed": completed,
            "completionRate": completed / games,
            "tSpinLocks": t_spin_locks,
            "tSpinLines": t_spin_lines,
            "tSpinAttack": t_spin_attack,
            "tSpinAttackPerPiece": t_spin_attack / max(1, pieces),
            "difficultClears": difficult_clears,
            "b2bActiveClears": b2b_active_clears,
            "maxB2bChain": max((item.max_b2b_chain for item in self.per_game), default=0),
            "meanMaxB2bChain": sum(item.max_b2b_chain for item in self.per_game) / games,
            "surgeReleased": surge_released,
            "surgeReleaseEvents": surge_release_events,
            "comboClears": combo_clears,
            "maxCombo": max((item.max_combo for item in self.per_game), default=0),
            "meanMaxCombo": sum(item.max_combo for item in self.per_game) / games,
            "perfectClears": perfect_clears,
            "elapsedSeconds": self.elapsed_seconds,
            "piecesPerSecond": pieces / max(self.elapsed_seconds, 1e-9),
            "perGame": [item.to_dict() for item in self.per_game],
        }


@dataclass(slots=True)
class _SoloCounters:
    t_spin_locks: int = 0
    t_spin_lines: int = 0
    t_spin_attack: int = 0
    difficult_clears: int = 0
    b2b_active_clears: int = 0
    max_b2b_chain: int = 0
    surge_released: int = 0
    surge_release_events: int = 0
    combo_clears: int = 0
    max_combo: int = 0
    perfect_clears: int = 0

    def observe(self, result: LockResult) -> None:
        if result.spin is not None:
            self.t_spin_locks += 1
            self.t_spin_lines += result.lines
            self.t_spin_attack += result.attack
        if is_difficult_clear(result.lines, result.spin):
            self.difficult_clears += 1
        if result.lines > 0 and result.back_to_back:
            self.b2b_active_clears += 1
        self.max_b2b_chain = max(self.max_b2b_chain, result.b2b_chain)
        if result.surge_released > 0:
            self.surge_release_events += 1
            self.surge_released += result.surge_released
        if result.lines > 0 and result.combo > 0:
            self.combo_clears += 1
        self.max_combo = max(self.max_combo, result.combo)
        if result.perfect_clear:
            self.perfect_clears += 1


def run_neural_solo_benchmark(
    scorer: SearchScorer,
    *,
    model: str,
    games: int = 20,
    max_pieces: int = 300,
    seed_base: int = 8_100_001,
    seed_step: int = 31,
    game_batch_size: int = 20,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    search_config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    progress: bool = True,
) -> NeuralSoloBenchmarkResult:
    """Run deterministic solo strength evaluation using the normal neural search path."""

    count = max(1, int(games))
    limit = max(1, int(max_pieces))
    step = int(seed_step)
    batch_size = max(1, int(game_batch_size))
    cfg = search_config.normalized()
    game_states = [Game(int(seed_base) + index * step) for index in range(count)]
    counters = [_SoloCounters() for _ in range(count)]
    done: set[int] = set()
    bar = progress_bar(
        total=count * limit,
        desc=f"Solo: {Path(model).name}",
        unit="piece",
        disable=not progress,
    )
    started = time.perf_counter()

    def finish(index: int) -> None:
        if index in done:
            return
        done.add(index)
        game = game_states[index]
        if progress and game.pieces_placed < limit:
            bar.total -= limit - game.pieces_placed
            bar.refresh()

    try:
        while len(done) < count:
            active = [
                index
                for index, game in enumerate(game_states)
                if index not in done and not game.game_over and game.pieces_placed < limit
            ]
            active_set = set(active)
            for index in range(count):
                if index not in done and index not in active_set:
                    finish(index)
            if not active:
                break

            for start in range(0, len(active), batch_size):
                indices = active[start : start + batch_size]
                choices = choose_search_actions_batch(
                    tuple(game_states[index] for index in indices),
                    weights,
                    cfg,
                    scorer=scorer,
                )
                gained = 0
                for index, choice in zip(indices, choices):
                    game = game_states[index]
                    if choice is None:
                        finish(index)
                        continue
                    before_pieces = game.pieces_placed
                    result = apply_search_action(game, choice.action)
                    counters[index].observe(result)
                    gained += max(0, game.pieces_placed - before_pieces)
                    if game.game_over or game.pieces_placed >= limit:
                        finish(index)
                if gained:
                    bar.update(gained)
    finally:
        bar.close()

    elapsed = time.perf_counter() - started
    per_game = tuple(
        NeuralSoloGameResult(
            seed=int(seed_base) + index * step,
            pieces=game.pieces_placed,
            lines=game.lines,
            attack=game.attack,
            topout=bool(game.game_over),
            completed=bool(not game.game_over and game.pieces_placed >= limit),
            t_spin_locks=counter.t_spin_locks,
            t_spin_lines=counter.t_spin_lines,
            t_spin_attack=counter.t_spin_attack,
            difficult_clears=counter.difficult_clears,
            b2b_active_clears=counter.b2b_active_clears,
            max_b2b_chain=counter.max_b2b_chain,
            surge_released=counter.surge_released,
            surge_release_events=counter.surge_release_events,
            combo_clears=counter.combo_clears,
            max_combo=counter.max_combo,
            perfect_clears=counter.perfect_clears,
        )
        for index, (game, counter) in enumerate(zip(game_states, counters))
    )
    return NeuralSoloBenchmarkResult(
        model=str(model),
        games=count,
        max_pieces=limit,
        seed_base=int(seed_base),
        seed_step=step,
        game_batch_size=batch_size,
        search_config=cfg,
        elapsed_seconds=elapsed,
        per_game=per_game,
    )


def reverse_versus_benchmark_result(result: VersusBenchmarkResult) -> VersusBenchmarkResult:
    """Return the same mirrored matches with logical model A/B exchanged."""

    def reverse_game(game: VersusGameResult) -> VersusGameResult:
        return replace(
            game,
            winner={"player": "ai", "ai": "player", "draw": "draw"}[game.winner],
            player_pieces=game.ai_pieces,
            ai_pieces=game.player_pieces,
            player_attack=game.ai_attack,
            ai_attack=game.player_attack,
            player_sent=game.ai_sent,
            ai_sent=game.player_sent,
            player_canceled=game.ai_canceled,
            ai_canceled=game.player_canceled,
            player_received=game.ai_received,
            ai_received=game.player_received,
            player_garbage_applied=game.ai_garbage_applied,
            ai_garbage_applied=game.player_garbage_applied,
            player_pending=game.ai_pending,
            ai_pending=game.player_pending,
            player_final_height=game.ai_final_height,
            ai_final_height=game.player_final_height,
            player_final_holes=game.ai_final_holes,
            ai_final_holes=game.player_final_holes,
            player_max_b2b=game.ai_max_b2b,
            ai_max_b2b=game.player_max_b2b,
            player_max_surge=game.ai_max_surge,
            ai_max_surge=game.player_max_surge,
            models_swapped=not game.models_swapped,
        )

    return replace(
        result,
        player_wins=result.ai_wins,
        ai_wins=result.player_wins,
        player_mean_attack=result.ai_mean_attack,
        ai_mean_attack=result.player_mean_attack,
        player_mean_sent=result.ai_mean_sent,
        ai_mean_sent=result.player_mean_sent,
        player_mean_canceled=result.ai_mean_canceled,
        ai_mean_canceled=result.player_mean_canceled,
        player_mean_received=result.ai_mean_received,
        ai_mean_received=result.player_mean_received,
        player_mean_pieces=result.ai_mean_pieces,
        ai_mean_pieces=result.player_mean_pieces,
        per_game=tuple(reverse_game(game) for game in result.per_game),
    )


def summarize_paired_versus(
    result: VersusBenchmarkResult,
    *,
    model_a: str,
    model_b: str,
) -> dict[str, object]:
    """Summarize an existing mirrored benchmark as complete same-seed pairs."""

    grouped: dict[int, list[VersusGameResult]] = {}
    for game in result.per_game:
        grouped.setdefault(game.seed, []).append(game)

    pair_a_wins = pair_b_wins = pair_ties = 0
    paired: list[dict[str, object]] = []
    for seed in sorted(grouped):
        games = grouped[seed]
        if len(games) != 2 or {bool(item.models_swapped) for item in games} != {False, True}:
            raise ValueError(f"Versus seed {seed} is not a complete mirrored pair")
        a_wins = sum(item.winner == "player" for item in games)
        b_wins = sum(item.winner == "ai" for item in games)
        draws = sum(item.winner == "draw" for item in games)
        if a_wins > b_wins:
            pair_winner = "a"
            pair_a_wins += 1
        elif b_wins > a_wins:
            pair_winner = "b"
            pair_b_wins += 1
        else:
            pair_winner = "tie"
            pair_ties += 1
        a_left = next(item for item in games if not item.models_swapped)
        a_right = next(item for item in games if item.models_swapped)
        logical_winner = {"player": "a", "ai": "b", "draw": "draw"}
        paired.append(
            {
                "seed": seed,
                "aWins": a_wins,
                "bWins": b_wins,
                "draws": draws,
                "winner": pair_winner,
                "aLeftResult": logical_winner[a_left.winner],
                "aRightResult": logical_winner[a_right.winner],
                "aLeftTurns": a_left.turns,
                "aRightTurns": a_right.turns,
            }
        )

    games = max(1, result.games)
    physical_a_left = [item for item in result.per_game if not item.models_swapped]
    physical_a_right = [item for item in result.per_game if item.models_swapped]
    return {
        "modelA": model_a,
        "modelB": model_b,
        "games": result.games,
        "pairs": len(grouped),
        "aWins": result.player_wins,
        "bWins": result.ai_wins,
        "draws": result.draws,
        "aWinRate": result.player_wins / games,
        "bWinRate": result.ai_wins / games,
        "pairWinsA": pair_a_wins,
        "pairWinsB": pair_b_wins,
        "pairTies": pair_ties,
        "aWinsAsLeft": sum(item.winner == "player" for item in physical_a_left),
        "aWinsAsRight": sum(item.winner == "player" for item in physical_a_right),
        "bWinsAsLeft": sum(item.winner == "ai" for item in physical_a_right),
        "bWinsAsRight": sum(item.winner == "ai" for item in physical_a_left),
        "meanPiecesA": result.player_mean_pieces,
        "meanPiecesB": result.ai_mean_pieces,
        "meanAttackA": result.player_mean_attack,
        "meanAttackB": result.ai_mean_attack,
        "meanSentA": result.player_mean_sent,
        "meanSentB": result.ai_mean_sent,
        "meanCanceledA": result.player_mean_canceled,
        "meanCanceledB": result.ai_mean_canceled,
        "meanReceivedA": result.player_mean_received,
        "meanReceivedB": result.ai_mean_received,
        "meanGarbageAppliedA": sum(item.player_garbage_applied for item in result.per_game) / games,
        "meanGarbageAppliedB": sum(item.ai_garbage_applied for item in result.per_game) / games,
        "meanPendingA": sum(item.player_pending for item in result.per_game) / games,
        "meanPendingB": sum(item.ai_pending for item in result.per_game) / games,
        "raw": result.to_dict(),
        "pairedSeeds": paired,
    }


def solo_deltas(candidate: Mapping[str, object], baseline: Mapping[str, object]) -> dict[str, float]:
    keys = (
        "attackPerPiece",
        "completionRate",
        "meanPiecesSurvived",
        "meanLines",
        "tSpinAttackPerPiece",
        "meanMaxB2bChain",
        "meanMaxCombo",
    )
    return {
        key: float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0))
        for key in keys
    }


def build_neural_promotion_report(
    *,
    candidate: NeuralModelSpec,
    champion: NeuralModelSpec,
    reference: NeuralModelSpec,
    solo: Mapping[str, NeuralSoloBenchmarkResult],
    versus: Mapping[str, dict[str, object]],
    conditions: Mapping[str, object],
    created_at: str | None = None,
) -> dict[str, object]:
    solo_dict = {name: value.to_dict() for name, value in solo.items()}
    candidate_solo = solo_dict["candidate"]
    return {
        "format": NEURAL_PROMOTION_FORMAT,
        "createdAt": created_at
        or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "decision": None,
        "decisionPolicy": "distribution-only; no promotion threshold is fixed by this benchmark",
        "models": {
            "candidate": candidate.to_dict(),
            "champion": champion.to_dict(),
            "reference": reference.to_dict(),
        },
        "conditions": dict(conditions),
        "solo": solo_dict,
        "versus": dict(versus),
        "comparisons": {
            "candidateMinusChampionSolo": solo_deltas(candidate_solo, solo_dict["champion"]),
            "candidateMinusReferenceSolo": solo_deltas(candidate_solo, solo_dict["reference"]),
        },
    }
