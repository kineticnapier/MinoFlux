from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from .benchmark import BenchmarkResult, run_heuristic_benchmark
from .cem import (
    FITNESS_PROFILE_ATTACK_SPIN,
    FitnessProfile,
    benchmark_fitness,
    resolve_fitness_profile,
)
from .heuristic import HeuristicWeights, load_weights, save_weights
from .search import SearchConfig


@dataclass(frozen=True, slots=True)
class PromotionConfig:
    games: int = 10
    max_pieces: int = 1000
    seed_base: int = 2_000_033
    seed_step: int = 97
    minimum_fitness_gain: float = 0.0
    max_completion_loss: int = 1
    workers: int = 0

    def normalized(self) -> "PromotionConfig":
        return PromotionConfig(
            games=max(1, int(self.games)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=max(1, int(self.seed_step)),
            minimum_fitness_gain=float(self.minimum_fitness_gain),
            max_completion_loss=max(0, int(self.max_completion_loss)),
            workers=max(0, int(self.workers)),
        )


@dataclass(frozen=True, slots=True)
class PromotionResult:
    promoted: bool
    reason: str
    candidate_fitness: float
    champion_fitness: float | None
    fitness_gain: float | None
    completion_loss: int
    profile: FitnessProfile
    config: PromotionConfig
    candidate: BenchmarkResult
    champion: BenchmarkResult | None

    def to_dict(self) -> dict[str, object]:
        return {
            "promoted": self.promoted,
            "reason": self.reason,
            "candidateFitness": self.candidate_fitness,
            "championFitness": self.champion_fitness,
            "fitnessGain": self.fitness_gain,
            "completionLoss": self.completion_loss,
            "fitnessProfile": self.profile.to_dict(),
            "config": asdict(self.config),
            "candidate": self.candidate.to_dict(),
            "champion": self.champion.to_dict() if self.champion is not None else None,
        }


def compare_candidate_to_champion(
    candidate_weights: HeuristicWeights,
    champion_weights: HeuristicWeights | None,
    search_config: SearchConfig,
    *,
    fitness_profile: str | FitnessProfile = FITNESS_PROFILE_ATTACK_SPIN,
    config: PromotionConfig = PromotionConfig(),
) -> PromotionResult:
    cfg = config.normalized()
    profile = resolve_fitness_profile(fitness_profile)
    candidate = run_heuristic_benchmark(
        cfg.games,
        cfg.max_pieces,
        cfg.seed_base,
        cfg.seed_step,
        candidate_weights,
        search_config,
        workers=cfg.workers,
    )
    candidate_fitness = benchmark_fitness(candidate, profile)

    if champion_weights is None:
        return PromotionResult(
            promoted=True,
            reason="No champion model existed; candidate became the first champion.",
            candidate_fitness=candidate_fitness,
            champion_fitness=None,
            fitness_gain=None,
            completion_loss=0,
            profile=profile,
            config=cfg,
            candidate=candidate,
            champion=None,
        )

    champion = run_heuristic_benchmark(
        cfg.games,
        cfg.max_pieces,
        cfg.seed_base,
        cfg.seed_step,
        champion_weights,
        search_config,
        workers=cfg.workers,
    )
    champion_fitness = benchmark_fitness(champion, profile)
    gain = candidate_fitness - champion_fitness
    completion_loss = max(0, champion.completed - candidate.completed)

    if completion_loss > cfg.max_completion_loss:
        promoted = False
        reason = (
            f"Rejected: candidate completed {completion_loss} fewer game(s), "
            f"above the allowed loss of {cfg.max_completion_loss}."
        )
    elif gain <= cfg.minimum_fitness_gain:
        promoted = False
        reason = (
            f"Rejected: fitness gain {gain:.3f} did not exceed the required "
            f"{cfg.minimum_fitness_gain:.3f}."
        )
    else:
        promoted = True
        reason = (
            f"Promoted: fitness improved by {gain:.3f} with completion loss "
            f"{completion_loss}/{cfg.max_completion_loss}."
        )

    return PromotionResult(
        promoted=promoted,
        reason=reason,
        candidate_fitness=candidate_fitness,
        champion_fitness=champion_fitness,
        fitness_gain=gain,
        completion_loss=completion_loss,
        profile=profile,
        config=cfg,
        candidate=candidate,
        champion=champion,
    )


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")


def bootstrap_champion(
    champion_path: str | Path,
    *,
    recovery_path: str | Path | None = None,
    legacy_path: str | Path | None = None,
) -> Path | None:
    champion = Path(champion_path)
    if champion.is_file():
        return champion
    for source in (recovery_path, legacy_path):
        if source is None:
            continue
        candidate = Path(source)
        if candidate.is_file():
            return save_weights(champion, load_weights(candidate))
    return None


def evaluate_and_promote_model(
    candidate_weights: HeuristicWeights,
    search_config: SearchConfig,
    *,
    champion_path: str | Path,
    candidate_path: str | Path,
    history_dir: str | Path,
    compatibility_latest_path: str | Path | None = None,
    fitness_profile: str | FitnessProfile = FITNESS_PROFILE_ATTACK_SPIN,
    config: PromotionConfig = PromotionConfig(),
) -> PromotionResult:
    champion_file = Path(champion_path)
    candidate_file = save_weights(candidate_path, candidate_weights)
    history = Path(history_dir)
    history.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    save_weights(history / f"candidate-{stamp}.json", candidate_weights)

    champion_weights = load_weights(champion_file) if champion_file.is_file() else None
    result = compare_candidate_to_champion(
        candidate_weights,
        champion_weights,
        search_config,
        fitness_profile=fitness_profile,
        config=config,
    )

    if result.promoted:
        if champion_weights is not None:
            save_weights(history / f"champion-before-{stamp}.json", champion_weights)
        save_weights(champion_file, candidate_weights)
        if compatibility_latest_path is not None:
            save_weights(compatibility_latest_path, candidate_weights)

    report_path = history / f"promotion-{stamp}.json"
    report = result.to_dict()
    report["candidateModelPath"] = str(candidate_file.resolve())
    report["championModelPath"] = str(champion_file.resolve())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result
