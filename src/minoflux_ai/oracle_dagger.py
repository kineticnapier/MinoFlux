from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
from typing import Callable

from minoflux_engine import Game

from .heuristic import DEFAULT_WEIGHTS
from .neural import NeuralValueEvaluator
from .neural_dataset import NEURAL_DATASET_FORMAT, NeuralRankingSample
from .oracle import search_oracle
from .oracle_dataset import (
    ORACLE_TEACHER_NAME,
    OracleDatasetConfig,
    _action_key,
    _candidate_state,
    _selected_actions,
)
from .search import SearchConfig, apply_search_action, rank_search_actions


def write_oracle_dagger_dataset(
    path: str | Path,
    evaluator: NeuralValueEvaluator,
    config: OracleDatasetConfig = OracleDatasetConfig(),
    *,
    learner_search: SearchConfig = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        srs_reachable=True,
        allow_180=True,
        reachability_node_limit=8_000,
    ),
    sample_rate: float = 0.05,
    uncertainty_margin: float = 0.08,
    danger_height: int = 12,
    danger_holes: int = 4,
    max_samples: int = 500,
    progress: Callable[[int, int, int, int], None] | None = None,
    progress_every: int = 10,
) -> dict[str, object]:
    """Label selected neural self-play states with the expensive native oracle.

    Unlike the heuristic DAgger collector, this gates the teacher call itself.
    Low-margin, dangerous, and random-control states are selected first, then only
    those states pay for a native oracle search. The learner action always advances
    the game so the collected states stay on the learner's own distribution.
    """

    cfg = config.normalized()
    learner_cfg = learner_search.normalized()
    rate = min(1.0, max(0.0, float(sample_rate)))
    uncertainty = max(0.0, float(uncertainty_margin))
    danger_height = max(1, int(danger_height))
    danger_holes = max(0, int(danger_holes))
    sample_limit = max(0, int(max_samples))
    report_every = max(1, int(progress_every))
    rng = random.Random(int(cfg.seed_base) ^ 0x0DA66E7)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")

    samples = 0
    candidates = 0
    visited = 0
    oracle_queries = 0
    oracle_disagreements = 0
    topouts = 0
    completed = 0
    skipped: Counter[str] = Counter()

    with temporary.open("w", encoding="utf-8") as stream:
        for game_index in range(cfg.games):
            if sample_limit and samples >= sample_limit:
                break

            seed = cfg.seed_base + game_index * cfg.seed_step
            game = Game(seed)
            while not game.game_over and game.pieces_placed < cfg.max_pieces:
                learner_ranked = rank_search_actions(
                    game,
                    DEFAULT_WEIGHTS,
                    learner_cfg,
                    limit=2,
                    scorer=evaluator,
                )
                if not learner_ranked:
                    skipped["learner-no-choice"] += 1
                    break

                learner_action, learner_eval = learner_ranked[0]
                learner_margin = (
                    float("inf")
                    if len(learner_ranked) < 2
                    else float(learner_eval.score - learner_ranked[1][1].score)
                )
                board = learner_eval.features.board

                reasons: list[str] = []
                if learner_margin <= uncertainty:
                    reasons.append("low_margin")
                if board.max_height >= danger_height:
                    reasons.append("high_stack")
                if danger_holes > 0 and board.holes >= danger_holes:
                    reasons.append("holes")
                if rate > 0.0 and rng.random() < rate:
                    reasons.append("random_control")

                if reasons and (not sample_limit or samples < sample_limit):
                    oracle_queries += 1
                    choice = search_oracle(game, cfg.oracle)
                    if choice is None:
                        skipped["oracle-no-choice"] += 1
                    else:
                        disagreed = _action_key(learner_action) != _action_key(choice.action)
                        if disagreed:
                            oracle_disagreements += 1
                            reasons.append("nn_oracle_disagree")

                        selected = _selected_actions(game, choice.action, cfg)
                        if selected is None:
                            skipped["oracle-action-unmatched"] += 1
                        else:
                            actions, expert_index = selected
                            ranking_candidates = tuple(
                                _candidate_state(
                                    game,
                                    action,
                                    cfg.neural_config,
                                    sampling_bucket=(
                                        "expert" if index == expert_index else "negative"
                                    ),
                                )
                                for index, action in enumerate(actions)
                            )
                            sample = NeuralRankingSample(
                                seed=seed,
                                piece_index=game.pieces_placed,
                                expert_index=expert_index,
                                expert_indices=(expert_index,),
                                candidates=ranking_candidates,
                            )
                            payload = sample.to_dict()
                            payload["source"] = "oracle_dagger"
                            payload["teacher"] = ORACLE_TEACHER_NAME
                            payload["daggerReasons"] = sorted(set(reasons))
                            payload["learnerMargin"] = (
                                None if learner_margin == float("inf") else learner_margin
                            )
                            payload["learnerMatchedOracle"] = not disagreed
                            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
                            samples += 1
                            candidates += len(ranking_candidates)
                            if progress is not None and samples % report_every == 0:
                                progress(samples, candidates, visited, oracle_queries)

                apply_search_action(game, learner_action)
                visited += 1
                if sample_limit and samples >= sample_limit:
                    break

            topouts += int(game.game_over)
            completed += int(
                not game.game_over and game.pieces_placed >= cfg.max_pieces
            )

    temporary.replace(target)
    result: dict[str, object] = {
        "format": NEURAL_DATASET_FORMAT,
        "teacher": ORACLE_TEACHER_NAME,
        "trajectory": "neural",
        "source": "oracle_dagger",
        "path": str(target),
        "games": cfg.games,
        "samples": samples,
        "candidates": candidates,
        "visitedStates": visited,
        "oracleQueries": oracle_queries,
        "oracleDisagreements": oracle_disagreements,
        "topouts": topouts,
        "completed": completed,
        "skipped": dict(sorted(skipped.items())),
        "selection": {
            "sampleRate": rate,
            "uncertaintyMargin": uncertainty,
            "dangerHeight": danger_height,
            "dangerHoles": danger_holes,
            "maxSamples": sample_limit,
        },
        "config": {
            "games": cfg.games,
            "maxPieces": cfg.max_pieces,
            "seedBase": cfg.seed_base,
            "seedStep": cfg.seed_step,
            "maxCandidates": cfg.max_candidates,
            "learnerSearch": learner_cfg.to_dict(),
            "oracle": {
                "beamWidth": cfg.oracle.beam_width,
                "depth": cfg.oracle.depth,
                "allow180": cfg.oracle.allow_180,
                "reachabilityNodeLimit": cfg.oracle.reachability_node_limit,
            },
        },
    }
    target.with_suffix(target.suffix + ".meta.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
