from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import random
from typing import Callable, Iterable, Sequence

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .neural import NeuralState, NeuralValueConfig, encode_game_state
from .neural_supervision import rollout_clean_attack_target, teacher_score_action
from .search import (
    SearchAction,
    SearchConfig,
    apply_search_action,
    choose_search_action,
    clone_game,
    rank_search_actions,
)

NEURAL_DATASET_FORMAT = "minoflux_neural_ranking_dataset_v1"
DEFAULT_DATASET_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.90,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
)


@dataclass(frozen=True, slots=True)
class NeuralDatasetConfig:
    games: int = 40
    max_pieces: int = 500
    seed_base: int = 3_000_001
    seed_step: int = 31
    max_candidates: int = 24
    sampling_mode: str = "diverse"
    hard_candidates: int = 8
    medium_candidates: int = 5
    random_candidates: int = 5
    bad_candidates: int = 5
    random_seed: int = 24_680
    teacher_lookahead: int = 0
    teacher_beam_width: int = 4
    teacher_acceptable_margin: float = 0.0
    # 0 preserves the legacy behavior and deep-scores every retained candidate.
    # Strong generation normally uses a representative subset because the
    # teacher's actual chosen action already comes from the full beam search.
    teacher_score_candidates: int = 0
    rollout_horizon: int = 0
    rollout_candidates: int = 0
    rollout_lookahead: int = 1
    rollout_beam_width: int = 4
    search_config: SearchConfig = DEFAULT_DATASET_SEARCH
    neural_config: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "NeuralDatasetConfig":
        sampling_mode = str(self.sampling_mode or "diverse").strip().lower()
        if sampling_mode not in {"diverse", "hard"}:
            raise ValueError("sampling_mode must be 'diverse' or 'hard'")
        return NeuralDatasetConfig(
            games=max(1, int(self.games)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=max(1, int(self.seed_step)),
            max_candidates=max(0, int(self.max_candidates)),
            sampling_mode=sampling_mode,
            hard_candidates=max(0, int(self.hard_candidates)),
            medium_candidates=max(0, int(self.medium_candidates)),
            random_candidates=max(0, int(self.random_candidates)),
            bad_candidates=max(0, int(self.bad_candidates)),
            random_seed=int(self.random_seed),
            teacher_lookahead=min(3, max(0, int(self.teacher_lookahead))),
            teacher_beam_width=min(128, max(1, int(self.teacher_beam_width))),
            teacher_acceptable_margin=max(0.0, float(self.teacher_acceptable_margin)),
            teacher_score_candidates=max(0, int(self.teacher_score_candidates)),
            rollout_horizon=max(0, int(self.rollout_horizon)),
            rollout_candidates=max(0, int(self.rollout_candidates)),
            rollout_lookahead=min(3, max(0, int(self.rollout_lookahead))),
            rollout_beam_width=min(128, max(1, int(self.rollout_beam_width))),
            search_config=self.search_config.normalized(),
            neural_config=self.neural_config.normalized(),
        )


@dataclass(frozen=True, slots=True)
class NeuralRankingCandidate:
    board_rows: tuple[int, ...]
    context: tuple[float, ...]
    move: tuple[object, ...]
    teacher_score: float | None = None
    target_value: float | None = None
    sampling_bucket: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "rows": list(self.board_rows),
            "context": list(self.context),
            "move": list(self.move),
        }
        if self.teacher_score is not None:
            result["teacherScore"] = float(self.teacher_score)
        if self.target_value is not None:
            result["targetValue"] = float(self.target_value)
        if self.sampling_bucket is not None:
            result["samplingBucket"] = self.sampling_bucket
        return result


@dataclass(frozen=True, slots=True)
class NeuralRankingSample:
    seed: int
    piece_index: int
    expert_index: int
    candidates: tuple[NeuralRankingCandidate, ...]
    expert_indices: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, object]:
        positives = self.expert_indices or (self.expert_index,)
        return {
            "format": NEURAL_DATASET_FORMAT,
            "seed": self.seed,
            "pieceIndex": self.piece_index,
            "expertIndex": self.expert_index,
            "expertIndices": list(positives),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def pack_board_rows(
    board: Sequence[float],
    config: NeuralValueConfig = NeuralValueConfig(),
) -> tuple[int, ...]:
    cfg = config.normalized()
    expected = cfg.board_height * cfg.board_width
    if len(board) != expected:
        raise ValueError(f"Expected {expected} board cells, got {len(board)}")
    rows: list[int] = []
    for y in range(cfg.board_height):
        mask = 0
        start = y * cfg.board_width
        for x in range(cfg.board_width):
            if float(board[start + x]) >= 0.5:
                mask |= 1 << x
        rows.append(mask)
    return tuple(rows)


def unpack_board_rows(
    rows: Sequence[int],
    config: NeuralValueConfig = NeuralValueConfig(),
) -> tuple[float, ...]:
    cfg = config.normalized()
    if len(rows) != cfg.board_height:
        raise ValueError(f"Expected {cfg.board_height} packed rows, got {len(rows)}")
    return tuple(
        1.0 if int(rows[y]) & (1 << x) else 0.0
        for y in range(cfg.board_height)
        for x in range(cfg.board_width)
    )


def _move_tuple(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        int(action.use_hold),
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
    )


def _action_key(action: SearchAction) -> tuple[object, ...]:
    return _move_tuple(action)


def _candidate_state(
    game: Game,
    action: SearchAction,
    neural_config: NeuralValueConfig,
    *,
    teacher_score: float | None = None,
    target_value: float | None = None,
    sampling_bucket: str | None = None,
) -> NeuralRankingCandidate:
    child = clone_game(game)
    apply_search_action(child, action)
    state: NeuralState = encode_game_state(child, neural_config)
    return NeuralRankingCandidate(
        board_rows=pack_board_rows(state.board, neural_config),
        context=state.context,
        move=_move_tuple(action),
        teacher_score=teacher_score,
        target_value=target_value,
        sampling_bucket=sampling_bucket,
    )


def _keep_hard_candidates(
    ranked: Sequence[tuple[SearchAction, object]],
    expert_action: SearchAction,
    max_candidates: int,
) -> tuple[tuple[SearchAction, object], ...]:
    if max_candidates <= 0 or len(ranked) <= max_candidates:
        return tuple(ranked)
    kept = list(ranked[:max_candidates])
    if any(action == expert_action for action, _ in kept):
        return tuple(kept)
    expert = next(item for item in ranked if item[0] == expert_action)
    kept[-1] = expert
    return tuple(kept)


def _sample_from_pool(
    pool: Sequence[int],
    count: int,
    rng: random.Random,
    selected: set[int],
) -> list[int]:
    available = [index for index in pool if index not in selected]
    if count <= 0 or not available:
        return []
    if len(available) <= count:
        return available
    return rng.sample(available, count)


def _select_diverse_candidates(
    ranked: Sequence[tuple[SearchAction, object]],
    expert_action: SearchAction,
    config: NeuralDatasetConfig,
    rng: random.Random,
) -> tuple[tuple[SearchAction, object, str], ...]:
    """Keep expert + hard, medium, random, and clearly bad negatives."""

    cfg = config.normalized()
    if cfg.sampling_mode == "hard":
        kept = _keep_hard_candidates(ranked, expert_action, cfg.max_candidates)
        expert_key = _action_key(expert_action)
        return tuple(
            (action, evaluation, "expert" if _action_key(action) == expert_key else "hard")
            for action, evaluation in kept
        )

    if not ranked:
        return ()
    if cfg.max_candidates <= 0 or len(ranked) <= cfg.max_candidates:
        expert_key = _action_key(expert_action)
        return tuple(
            (action, evaluation, "expert" if _action_key(action) == expert_key else "all")
            for action, evaluation in ranked
        )

    cap = max(1, cfg.max_candidates)
    expert_key = _action_key(expert_action)
    expert_rank = next(
        index for index, (action, _evaluation) in enumerate(ranked)
        if _action_key(action) == expert_key
    )
    selected: set[int] = {expert_rank}
    buckets: dict[int, str] = {expert_rank: "expert"}

    n = len(ranked)
    hard_pool = list(range(0, max(1, n // 3)))
    medium_pool = list(range(max(1, n // 3), max(2, (2 * n) // 3)))
    bad_pool = list(range(max(0, (2 * n) // 3), n))

    def take(pool: Sequence[int], count: int, bucket: str, *, randomize: bool) -> None:
        remaining = max(0, cap - len(selected))
        if remaining <= 0:
            return
        wanted = min(max(0, int(count)), remaining)
        if randomize:
            indices = _sample_from_pool(pool, wanted, rng, selected)
        else:
            indices = [index for index in pool if index not in selected][:wanted]
        for index in indices:
            selected.add(index)
            buckets[index] = bucket

    take(hard_pool, cfg.hard_candidates, "hard", randomize=False)
    take(medium_pool, cfg.medium_candidates, "medium", randomize=True)
    take(bad_pool[::-1], cfg.bad_candidates, "bad", randomize=False)

    remaining_pool = [index for index in range(n) if index not in selected]
    take(remaining_pool, cfg.random_candidates, "random", randomize=True)

    # Quotas are targets, not hard requirements. Fill an undersubscribed sample
    # by heuristic rank so every record still uses the requested candidate cap.
    if len(selected) < cap:
        for index in range(n):
            if index in selected:
                continue
            selected.add(index)
            buckets[index] = "fill"
            if len(selected) >= cap:
                break

    return tuple(
        (ranked[index][0], ranked[index][1], buckets[index])
        for index in sorted(selected)
    )


def _supervision_indices(
    scores: Sequence[float],
    count: int,
    rng: random.Random,
    *,
    required_index: int,
) -> tuple[int, ...]:
    """Pick a representative subset while always keeping the teacher action."""

    if not scores:
        return ()
    if required_index < 0 or required_index >= len(scores):
        raise ValueError("required supervision index is out of range")
    if count <= 0 or count >= len(scores):
        return tuple(range(len(scores)))

    count = max(1, min(len(scores), int(count)))
    ordered = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    selected: set[int] = {required_index}

    # Half of the budget stays on hard/high-value alternatives.
    hard_count = min(count, max(1, (count + 1) // 2))
    for index in ordered:
        if len(selected) >= hard_count:
            break
        selected.add(index)

    # One explicitly poor alternative anchors the low end of the value scale.
    if len(selected) < count:
        selected.add(ordered[-1])

    # The rest is sampled from the middle/remaining pool for diversity.
    remaining = [index for index in range(len(scores)) if index not in selected]
    need = count - len(selected)
    if need > 0:
        selected.update(rng.sample(remaining, min(need, len(remaining))))
    return tuple(sorted(selected))


def _generate_game_samples(
    cfg: NeuralDatasetConfig,
    weights: HeuristicWeights,
    game_index: int,
) -> Iterable[NeuralRankingSample]:
    seed = cfg.seed_base + game_index * cfg.seed_step
    game = Game(seed)
    while not game.game_over and game.pieces_placed < cfg.max_pieces:
        ranked = rank_search_actions(
            game,
            weights,
            cfg.search_config,
            limit=None,
        )
        if not ranked:
            break

        teacher_cfg = replace(
            cfg.search_config,
            lookahead_pieces=cfg.teacher_lookahead,
            beam_width=cfg.teacher_beam_width,
        ).normalized()
        if cfg.teacher_lookahead == 0:
            teacher_action = ranked[0][0]
        else:
            teacher = choose_search_action(game, weights, teacher_cfg)
            if teacher is None:
                break
            teacher_action = teacher.action

        state_rng = random.Random(
            cfg.random_seed
            ^ int(seed)
            ^ ((int(game.pieces_placed) + 1) * 0x9E3779B1)
        )
        selected = _select_diverse_candidates(ranked, teacher_action, cfg, state_rng)
        selected_pairs = tuple((action, evaluation) for action, evaluation, _bucket in selected)
        if not selected_pairs:
            break

        teacher_key = _action_key(teacher_action)
        primary_index = next(
            index
            for index, (action, _evaluation) in enumerate(selected_pairs)
            if _action_key(action) == teacher_key
        )
        immediate_scores = tuple(float(evaluation.score) for _action, evaluation in selected_pairs)

        teacher_scores: dict[int, float] = {}
        if cfg.teacher_lookahead == 0:
            # Immediate scores are already available, so keeping all of them is free.
            teacher_scores = {
                index: score for index, score in enumerate(immediate_scores)
            }
        else:
            teacher_indices = _supervision_indices(
                immediate_scores,
                cfg.teacher_score_candidates,
                state_rng,
                required_index=primary_index,
            )
            for index in teacher_indices:
                action, evaluation = selected_pairs[index]
                teacher_scores[index] = teacher_score_action(
                    game,
                    action,
                    evaluation,
                    weights,
                    cfg.search_config,
                    lookahead_pieces=cfg.teacher_lookahead,
                    beam_width=cfg.teacher_beam_width,
                )

        acceptable = (primary_index,)
        if cfg.teacher_acceptable_margin > 0.0:
            primary_score = teacher_scores.get(primary_index)
            if primary_score is not None:
                acceptable = tuple(
                    sorted(
                        index
                        for index, score in teacher_scores.items()
                        if primary_score - score <= cfg.teacher_acceptable_margin + 1e-12
                    )
                )
                if primary_index not in acceptable:
                    acceptable = tuple(sorted((*acceptable, primary_index)))

        rollout_targets: dict[int, float] = {}
        if cfg.rollout_horizon > 0 and cfg.rollout_candidates > 0:
            rollout_cfg = replace(
                cfg.search_config,
                lookahead_pieces=cfg.rollout_lookahead,
                beam_width=cfg.rollout_beam_width,
            ).normalized()
            rollout_scores = tuple(
                teacher_scores.get(index, immediate_scores[index])
                for index in range(len(selected_pairs))
            )
            rollout_indices = _supervision_indices(
                rollout_scores,
                cfg.rollout_candidates,
                state_rng,
                required_index=primary_index,
            )
            for index in rollout_indices:
                action, evaluation = selected_pairs[index]
                rollout_targets[index] = rollout_clean_attack_target(
                    game,
                    action,
                    evaluation,
                    weights,
                    rollout_cfg,
                    horizon=cfg.rollout_horizon,
                )

        candidates = tuple(
            _candidate_state(
                game,
                action,
                cfg.neural_config,
                teacher_score=teacher_scores.get(index),
                target_value=rollout_targets.get(index),
                sampling_bucket=bucket,
            )
            for index, (action, _evaluation, bucket) in enumerate(selected)
        )
        yield NeuralRankingSample(
            seed=seed,
            piece_index=game.pieces_placed,
            expert_index=primary_index,
            expert_indices=acceptable,
            candidates=candidates,
        )
        apply_search_action(game, selected_pairs[primary_index][0])


def generate_neural_ranking_samples(
    config: NeuralDatasetConfig = NeuralDatasetConfig(),
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
) -> Iterable[NeuralRankingSample]:
    cfg = config.normalized()
    for game_index in range(cfg.games):
        yield from _generate_game_samples(cfg, weights, game_index)


def _generate_game_task(
    task: tuple[NeuralDatasetConfig, HeuristicWeights, int],
) -> tuple[NeuralRankingSample, ...]:
    cfg, weights, game_index = task
    return tuple(_generate_game_samples(cfg, weights, game_index))


def _resolved_workers(requested: int, games: int) -> int:
    value = int(requested)
    if value > 0:
        return min(max(1, games), value)
    available = max(1, (os.cpu_count() or 1) - 1)
    return min(max(1, games), available)


def write_neural_ranking_dataset(
    path: str | Path,
    config: NeuralDatasetConfig = NeuralDatasetConfig(),
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    *,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 500,
    workers: int = 1,
) -> dict[str, object]:
    cfg = config.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    samples = 0
    candidates = 0
    rollout_targets = 0
    resolved_workers = _resolved_workers(workers, cfg.games)

    def write_sample(stream, sample: NeuralRankingSample) -> None:
        nonlocal samples, candidates, rollout_targets
        stream.write(json.dumps(sample.to_dict(), separators=(",", ":")) + "\n")
        samples += 1
        candidates += len(sample.candidates)
        rollout_targets += sum(candidate.target_value is not None for candidate in sample.candidates)
        if progress is not None and samples % max(1, int(progress_every)) == 0:
            progress(samples, candidates)

    with temporary.open("w", encoding="utf-8") as stream:
        if resolved_workers <= 1:
            for sample in generate_neural_ranking_samples(cfg, weights):
                write_sample(stream, sample)
        else:
            tasks = tuple((cfg, weights, game_index) for game_index in range(cfg.games))
            with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
                for game_samples in executor.map(_generate_game_task, tasks, chunksize=1):
                    for sample in game_samples:
                        write_sample(stream, sample)

    temporary.replace(target)
    result = {
        "format": NEURAL_DATASET_FORMAT,
        "path": str(target),
        "samples": samples,
        "candidates": candidates,
        "rolloutTargets": rollout_targets,
        "workers": resolved_workers,
        "config": {
            "games": cfg.games,
            "maxPieces": cfg.max_pieces,
            "seedBase": cfg.seed_base,
            "seedStep": cfg.seed_step,
            "maxCandidates": cfg.max_candidates,
            "samplingMode": cfg.sampling_mode,
            "hardCandidates": cfg.hard_candidates,
            "mediumCandidates": cfg.medium_candidates,
            "randomCandidates": cfg.random_candidates,
            "badCandidates": cfg.bad_candidates,
            "randomSeed": cfg.random_seed,
            "teacherLookahead": cfg.teacher_lookahead,
            "teacherBeamWidth": cfg.teacher_beam_width,
            "teacherAcceptableMargin": cfg.teacher_acceptable_margin,
            "teacherScoreCandidates": cfg.teacher_score_candidates,
            "rolloutHorizon": cfg.rollout_horizon,
            "rolloutCandidates": cfg.rollout_candidates,
            "rolloutLookahead": cfg.rollout_lookahead,
            "rolloutBeamWidth": cfg.rollout_beam_width,
            "searchConfig": cfg.search_config.to_dict(),
            "neuralConfig": asdict(cfg.neural_config),
        },
    }
    metadata_path = target.with_suffix(target.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
