from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from typing import Callable

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .neural import NeuralValueEvaluator
from .neural_dataset import (
    NEURAL_DATASET_FORMAT,
    NeuralDatasetConfig,
    NeuralRankingSample,
    _action_key,
    _candidate_state,
    _select_diverse_candidates,
    _supervision_indices,
)
from .neural_supervision import rollout_clean_attack_target, teacher_score_action
from .search import apply_search_action, choose_search_action, rank_search_actions


def write_neural_dagger_dataset(
    path: str | Path,
    evaluator: NeuralValueEvaluator,
    config: NeuralDatasetConfig,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    *,
    sample_rate: float = 0.25,
    uncertainty_margin: float = 0.08,
    danger_height: int = 12,
    danger_holes: int = 4,
    max_samples: int = 0,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 100,
) -> dict[str, object]:
    """Collect teacher labels on states visited by the neural learner itself.

    Disagreement, low-margin, and dangerous states are always retained. A random
    control sample is also kept so confident shared mistakes are not invisible.
    The learner action, not the teacher action, advances the game distribution.
    """

    cfg = config.normalized()
    rate = min(1.0, max(0.0, float(sample_rate)))
    uncertainty = max(0.0, float(uncertainty_margin))
    danger_height = max(1, int(danger_height))
    danger_holes = max(0, int(danger_holes))
    sample_limit = max(0, int(max_samples))
    rng = random.Random(cfg.random_seed ^ 0xDA66E7)
    teacher_cfg = replace(
        cfg.search_config,
        lookahead_pieces=cfg.teacher_lookahead,
        beam_width=cfg.teacher_beam_width,
    ).normalized()
    rollout_cfg = replace(
        cfg.search_config,
        lookahead_pieces=cfg.rollout_lookahead,
        beam_width=cfg.rollout_beam_width,
    ).normalized()

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    samples = 0
    candidates = 0
    rollout_targets = 0
    visited = 0
    topouts = 0

    with temporary.open("w", encoding="utf-8") as stream:
        for game_index in range(cfg.games):
            if sample_limit and samples >= sample_limit:
                break
            seed = cfg.seed_base + game_index * cfg.seed_step
            game = Game(seed)
            while not game.game_over and game.pieces_placed < cfg.max_pieces:
                learner_ranked = rank_search_actions(
                    game,
                    weights,
                    cfg.search_config,
                    limit=2,
                    scorer=evaluator,
                )
                if not learner_ranked:
                    break
                learner_action, learner_eval = learner_ranked[0]
                learner_margin = (
                    float("inf")
                    if len(learner_ranked) < 2
                    else float(learner_eval.score - learner_ranked[1][1].score)
                )
                teacher = choose_search_action(game, weights, teacher_cfg)
                if teacher is None:
                    break

                reasons: list[str] = []
                if _action_key(learner_action) != _action_key(teacher.action):
                    reasons.append("nn_teacher_disagree")
                if learner_margin <= uncertainty:
                    reasons.append("low_margin")
                board = learner_eval.features.board
                if board.max_height >= danger_height:
                    reasons.append("high_stack")
                if danger_holes > 0 and board.holes >= danger_holes:
                    reasons.append("holes")
                if rate > 0.0 and rng.random() < rate:
                    reasons.append("random_control")

                if reasons and (not sample_limit or samples < sample_limit):
                    ranked = rank_search_actions(
                        game,
                        weights,
                        cfg.search_config,
                        limit=None,
                    )
                    state_rng = random.Random(
                        cfg.random_seed
                        ^ int(seed)
                        ^ ((int(game.pieces_placed) + 1) * 0x9E3779B1)
                        ^ 0xDA66E7
                    )
                    selected = _select_diverse_candidates(ranked, teacher.action, cfg, state_rng)
                    selected_pairs = tuple(
                        (action, evaluation)
                        for action, evaluation, _bucket in selected
                    )
                    if selected_pairs:
                        teacher_key = _action_key(teacher.action)
                        primary_index = next(
                            index
                            for index, (action, _evaluation) in enumerate(selected_pairs)
                            if _action_key(action) == teacher_key
                        )
                        immediate_scores = tuple(
                            float(evaluation.score)
                            for _action, evaluation in selected_pairs
                        )

                        teacher_scores: dict[int, float] = {}
                        if cfg.teacher_lookahead == 0:
                            teacher_scores = {
                                index: score
                                for index, score in enumerate(immediate_scores)
                            }
                        else:
                            for index in _supervision_indices(
                                immediate_scores,
                                cfg.teacher_score_candidates,
                                state_rng,
                                required_index=primary_index,
                            ):
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

                        targets: dict[int, float] = {}
                        if cfg.rollout_horizon > 0 and cfg.rollout_candidates > 0:
                            rollout_scores = tuple(
                                teacher_scores.get(index, immediate_scores[index])
                                for index in range(len(selected_pairs))
                            )
                            for index in _supervision_indices(
                                rollout_scores,
                                cfg.rollout_candidates,
                                state_rng,
                                required_index=primary_index,
                            ):
                                action, evaluation = selected_pairs[index]
                                targets[index] = rollout_clean_attack_target(
                                    game,
                                    action,
                                    evaluation,
                                    weights,
                                    rollout_cfg,
                                    horizon=cfg.rollout_horizon,
                                )

                        ranking_candidates = tuple(
                            _candidate_state(
                                game,
                                action,
                                cfg.neural_config,
                                teacher_score=teacher_scores.get(index),
                                target_value=targets.get(index),
                                sampling_bucket=bucket,
                            )
                            for index, (action, _evaluation, bucket) in enumerate(selected)
                        )
                        sample = NeuralRankingSample(
                            seed=seed,
                            piece_index=game.pieces_placed,
                            expert_index=primary_index,
                            expert_indices=acceptable,
                            candidates=ranking_candidates,
                        )
                        payload = sample.to_dict()
                        payload["source"] = "dagger_teacher"
                        payload["daggerReasons"] = sorted(set(reasons))
                        payload["learnerMargin"] = (
                            None if learner_margin == float("inf") else learner_margin
                        )
                        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
                        samples += 1
                        candidates += len(ranking_candidates)
                        rollout_targets += sum(
                            candidate.target_value is not None
                            for candidate in ranking_candidates
                        )
                        if progress is not None and samples % max(1, int(progress_every)) == 0:
                            progress(samples, candidates)

                # DAgger: deliberately stay on the learner's state distribution.
                apply_search_action(game, learner_action)
                visited += 1
                if sample_limit and samples >= sample_limit:
                    break

            topouts += int(game.game_over)

    temporary.replace(target)
    summary = {
        "format": NEURAL_DATASET_FORMAT,
        "path": str(target),
        "source": "dagger_teacher",
        "samples": samples,
        "candidates": candidates,
        "rolloutTargets": rollout_targets,
        "visitedStates": visited,
        "topouts": topouts,
        "sampleRate": rate,
        "uncertaintyMargin": uncertainty,
        "dangerHeight": danger_height,
        "dangerHoles": danger_holes,
        "maxSamples": sample_limit,
        "config": {
            **asdict(cfg),
            "search_config": cfg.search_config.to_dict(),
            "neural_config": asdict(cfg.neural_config),
        },
    }
    target.with_suffix(target.suffix + ".meta.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
