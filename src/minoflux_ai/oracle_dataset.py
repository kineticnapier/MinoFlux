from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Sequence

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS
from .neural import NeuralValueConfig, encode_game_state
from .neural_dataset import (
    NEURAL_DATASET_FORMAT,
    NeuralRankingCandidate,
    NeuralRankingSample,
    pack_board_rows,
)
from .oracle import OracleConfig, search_oracle
from .search import SearchAction, SearchConfig, apply_search_action, clone_game, rank_search_actions

ORACLE_TEACHER_NAME = "minoflux-native-oracle"


@dataclass(frozen=True, slots=True)
class OracleDatasetConfig:
    games: int = 40
    max_pieces: int = 500
    seed_base: int = 6_000_001
    seed_step: int = 97
    max_candidates: int = 24
    oracle: OracleConfig = OracleConfig()
    neural_config: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "OracleDatasetConfig":
        return OracleDatasetConfig(
            games=max(1, int(self.games)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=max(1, int(self.seed_step)),
            max_candidates=max(0, int(self.max_candidates)),
            oracle=self.oracle.normalized(),
            neural_config=self.neural_config.normalized(),
        )


@dataclass(frozen=True, slots=True)
class _GameSamples:
    samples: tuple[NeuralRankingSample, ...]
    attempted: int
    skipped: tuple[tuple[str, int], ...]
    pieces: int
    topout: bool


def _action_key(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        int(action.use_hold),
        placement.piece,
        int(placement.x),
        int(placement.y),
        int(placement.rotation),
    )


def _candidate_state(
    game: Game,
    action: SearchAction,
    neural_config: NeuralValueConfig,
    *,
    sampling_bucket: str,
) -> NeuralRankingCandidate:
    child = clone_game(game)
    apply_search_action(child, action)
    state = encode_game_state(child, neural_config)
    placement = action.placement
    return NeuralRankingCandidate(
        board_rows=pack_board_rows(state.board, neural_config),
        context=state.context,
        move=(
            int(action.use_hold),
            placement.piece,
            int(placement.x),
            int(placement.y),
            int(placement.rotation),
        ),
        sampling_bucket=sampling_bucket,
    )


def _negative_search_config(config: OracleConfig) -> SearchConfig:
    return SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=1,
        srs_reachable=True,
        allow_180=config.allow_180,
        reachability_node_limit=config.reachability_node_limit,
    )


def _selected_actions(
    game: Game,
    teacher_action: SearchAction,
    config: OracleDatasetConfig,
) -> tuple[tuple[SearchAction, ...], int] | None:
    ranked = rank_search_actions(
        game,
        DEFAULT_WEIGHTS,
        _negative_search_config(config.oracle),
        limit=None,
    )
    if not ranked:
        return None

    teacher_key = _action_key(teacher_action)
    teacher_rank = next(
        (
            index
            for index, (action, _evaluation) in enumerate(ranked)
            if _action_key(action) == teacher_key
        ),
        None,
    )
    if teacher_rank is None:
        return None

    cap = len(ranked) if config.max_candidates <= 0 else min(config.max_candidates, len(ranked))
    if cap <= 0:
        return None
    selected_indices = list(range(cap))
    if teacher_rank not in selected_indices:
        selected_indices[-1] = teacher_rank
        selected_indices.sort()

    actions: list[SearchAction] = []
    expert_index = -1
    for rank in selected_indices:
        action = teacher_action if rank == teacher_rank else ranked[rank][0]
        if rank == teacher_rank:
            expert_index = len(actions)
        actions.append(action)
    if expert_index < 0:
        return None
    return tuple(actions), expert_index


def _generate_game_samples(config: OracleDatasetConfig, game_index: int) -> _GameSamples:
    cfg = config.normalized()
    seed = cfg.seed_base + game_index * cfg.seed_step
    game = Game(seed)
    samples: list[NeuralRankingSample] = []
    skipped: Counter[str] = Counter()
    attempted = 0

    while not game.game_over and game.pieces_placed < cfg.max_pieces:
        attempted += 1
        choice = search_oracle(game, cfg.oracle)
        if choice is None:
            skipped["oracle-no-choice"] += 1
            break

        selected = _selected_actions(game, choice.action, cfg)
        if selected is None:
            skipped["oracle-action-unmatched"] += 1
            apply_search_action(game, choice.action)
            continue

        actions, expert_index = selected
        candidates = tuple(
            _candidate_state(
                game,
                action,
                cfg.neural_config,
                sampling_bucket="expert" if index == expert_index else "negative",
            )
            for index, action in enumerate(actions)
        )
        samples.append(
            NeuralRankingSample(
                seed=seed,
                piece_index=game.pieces_placed,
                expert_index=expert_index,
                expert_indices=(expert_index,),
                candidates=candidates,
            )
        )
        apply_search_action(game, choice.action)

    return _GameSamples(
        samples=tuple(samples),
        attempted=attempted,
        skipped=tuple(sorted(skipped.items())),
        pieces=game.pieces_placed,
        topout=bool(game.game_over),
    )


def _generate_game_task(task: tuple[OracleDatasetConfig, int]) -> _GameSamples:
    return _generate_game_samples(*task)


def _resolved_workers(requested: int, games: int) -> int:
    value = int(requested)
    if value > 0:
        return min(max(1, games), value)
    available = max(1, (os.cpu_count() or 1) - 1)
    return min(max(1, games), available)


def write_oracle_ranking_dataset(
    path: str | Path,
    config: OracleDatasetConfig = OracleDatasetConfig(),
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 100,
    workers: int = 1,
) -> dict[str, object]:
    cfg = config.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    resolved_workers = _resolved_workers(workers, cfg.games)

    samples = 0
    candidates = 0
    attempted = 0
    pieces = 0
    topouts = 0
    skipped: Counter[str] = Counter()

    def consume(stream, result: _GameSamples) -> None:
        nonlocal samples, candidates, attempted, pieces, topouts
        attempted += result.attempted
        pieces += result.pieces
        topouts += int(result.topout)
        skipped.update(dict(result.skipped))
        for sample in result.samples:
            stream.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
            samples += 1
            candidates += len(sample.candidates)
            if progress is not None and samples % max(1, int(progress_every)) == 0:
                progress(samples, candidates)

    with temporary.open("w", encoding="utf-8") as stream:
        if resolved_workers <= 1:
            for game_index in range(cfg.games):
                consume(stream, _generate_game_samples(cfg, game_index))
        else:
            tasks = tuple((cfg, game_index) for game_index in range(cfg.games))
            with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                for result in executor.map(_generate_game_task, tasks, chunksize=1):
                    consume(stream, result)

    temporary.replace(target)
    result: dict[str, object] = {
        "format": NEURAL_DATASET_FORMAT,
        "teacher": ORACLE_TEACHER_NAME,
        "trajectory": "oracle",
        "path": str(target),
        "games": cfg.games,
        "attempted": attempted,
        "samples": samples,
        "candidates": candidates,
        "pieces": pieces,
        "topouts": topouts,
        "skipped": dict(sorted(skipped.items())),
        "workers": resolved_workers,
        "config": {
            "games": cfg.games,
            "maxPieces": cfg.max_pieces,
            "seedBase": cfg.seed_base,
            "seedStep": cfg.seed_step,
            "maxCandidates": cfg.max_candidates,
            "oracle": {
                "beamWidth": cfg.oracle.beam_width,
                "depth": cfg.oracle.depth,
                "allow180": cfg.oracle.allow_180,
                "reachabilityNodeLimit": cfg.oracle.reachability_node_limit,
            },
        },
    }
    metadata_path = target.with_suffix(target.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_oracle_smoke(
    *,
    games: int = 1,
    max_pieces: int = 20,
    seed_base: int = 8_100_001,
    seed_step: int = 97,
    oracle: OracleConfig = OracleConfig(beam_width=64, depth=2),
) -> dict[str, object]:
    cfg = oracle.normalized()
    game_count = max(1, int(games))
    piece_limit = max(1, int(max_pieces))
    step = max(1, int(seed_step))
    game_states = [Game(int(seed_base) + index * step) for index in range(game_count)]

    no_choices = 0
    for game in game_states:
        while not game.game_over and game.pieces_placed < piece_limit:
            choice = search_oracle(game, cfg)
            if choice is None:
                no_choices += 1
                break
            apply_search_action(game, choice.action)

    pieces = sum(game.pieces_placed for game in game_states)
    return {
        "teacher": ORACLE_TEACHER_NAME,
        "games": game_count,
        "maxPieces": piece_limit,
        "pieces": pieces,
        "lines": sum(game.lines for game in game_states),
        "attack": sum(game.attack for game in game_states),
        "topouts": sum(int(game.game_over) for game in game_states),
        "completed": sum(
            int(not game.game_over and game.pieces_placed >= piece_limit)
            for game in game_states
        ),
        "noChoices": no_choices,
        "oracle": {
            "beamWidth": cfg.beam_width,
            "depth": cfg.depth,
            "allow180": cfg.allow_180,
            "reachabilityNodeLimit": cfg.reachability_node_limit,
        },
    }
