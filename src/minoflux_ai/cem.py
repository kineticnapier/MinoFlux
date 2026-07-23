from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import ceil
import random
from statistics import fmean, pstdev
from typing import Callable

from .benchmark import BenchmarkResult, run_heuristic_benchmark
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights

TRAINABLE_WEIGHT_NAMES = tuple(item.name for item in fields(HeuristicWeights) if item.name != "game_over")


@dataclass(frozen=True, slots=True)
class CEMConfig:
    generations: int = 8
    population: int = 16
    elite_fraction: float = 0.25
    games_per_candidate: int = 3
    max_pieces: int = 200
    seed_base: int = 1
    seed_step: int = 31
    validation_games: int = 4
    initial_sigma: float = 0.35
    learning_rate: float = 0.7
    minimum_sigma: float = 0.01
    random_seed: int = 12345

    def normalized(self) -> "CEMConfig":
        return CEMConfig(
            generations=max(1, int(self.generations)),
            population=max(2, int(self.population)),
            elite_fraction=min(1.0, max(0.05, float(self.elite_fraction))),
            games_per_candidate=max(1, int(self.games_per_candidate)),
            max_pieces=max(1, int(self.max_pieces)),
            seed_base=int(self.seed_base),
            seed_step=int(self.seed_step),
            validation_games=max(1, int(self.validation_games)),
            initial_sigma=max(0.0, float(self.initial_sigma)),
            learning_rate=min(1.0, max(0.01, float(self.learning_rate))),
            minimum_sigma=max(1e-6, float(self.minimum_sigma)),
            random_seed=int(self.random_seed),
        )


@dataclass(frozen=True, slots=True)
class CEMGeneration:
    generation: int
    best_fitness: float
    mean_fitness: float
    elite_mean_fitness: float
    best_weights: dict[str, float]
    sigma: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CEMResult:
    config: CEMConfig
    best_weights: HeuristicWeights
    best_training_fitness: float
    validation_fitness: float
    validation: BenchmarkResult
    history: tuple[CEMGeneration, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "bestWeights": self.best_weights.to_dict(),
            "bestTrainingFitness": self.best_training_fitness,
            "validationFitness": self.validation_fitness,
            "validation": self.validation.to_dict(),
            "history": [item.to_dict() for item in self.history],
        }


def benchmark_fitness(result: BenchmarkResult) -> float:
    completion_bonus = (result.completed / result.games) * result.max_pieces * 0.25
    topout_penalty = (result.topouts / result.games) * result.max_pieces * 0.10
    return (
        result.mean_pieces
        + result.mean_lines * 2.0
        + result.mean_attack * 4.0
        + completion_bonus
        - topout_penalty
    )


def _candidate_from_values(base: HeuristicWeights, values: dict[str, float]) -> HeuristicWeights:
    merged = base.to_dict()
    merged.update(values)
    return HeuristicWeights.from_mapping(merged)


def train_cem(
    config: CEMConfig = CEMConfig(),
    initial_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    on_generation: Callable[[CEMGeneration], None] | None = None,
) -> CEMResult:
    cfg = config.normalized()
    rng = random.Random(cfg.random_seed)
    mean = {name: getattr(initial_weights, name) for name in TRAINABLE_WEIGHT_NAMES}
    sigma = {
        name: max(cfg.minimum_sigma, max(1.0, abs(value)) * cfg.initial_sigma)
        for name, value in mean.items()
    }
    best_weights = initial_weights
    baseline = run_heuristic_benchmark(
        cfg.games_per_candidate,
        cfg.max_pieces,
        cfg.seed_base,
        cfg.seed_step,
        initial_weights,
    )
    best_fitness = benchmark_fitness(baseline)
    history: list[CEMGeneration] = []

    for generation_index in range(cfg.generations):
        population: list[HeuristicWeights] = [_candidate_from_values(initial_weights, mean)]
        while len(population) < cfg.population:
            sampled = {
                name: rng.gauss(mean[name], sigma[name])
                for name in TRAINABLE_WEIGHT_NAMES
            }
            population.append(_candidate_from_values(initial_weights, sampled))

        scored: list[tuple[float, HeuristicWeights]] = []
        for candidate in population:
            benchmark = run_heuristic_benchmark(
                cfg.games_per_candidate,
                cfg.max_pieces,
                cfg.seed_base,
                cfg.seed_step,
                candidate,
            )
            scored.append((benchmark_fitness(benchmark), candidate))
        scored.sort(key=lambda item: item[0], reverse=True)

        elite_count = max(1, min(cfg.population, ceil(cfg.population * cfg.elite_fraction)))
        elites = scored[:elite_count]
        if scored[0][0] > best_fitness:
            best_fitness, best_weights = scored[0]

        elite_means: dict[str, float] = {}
        elite_sigmas: dict[str, float] = {}
        for name in TRAINABLE_WEIGHT_NAMES:
            values = [getattr(candidate, name) for _, candidate in elites]
            elite_means[name] = fmean(values)
            elite_sigmas[name] = max(cfg.minimum_sigma, pstdev(values) if len(values) > 1 else sigma[name] * 0.5)

        for name in TRAINABLE_WEIGHT_NAMES:
            mean[name] += (elite_means[name] - mean[name]) * cfg.learning_rate
            sigma[name] += (elite_sigmas[name] - sigma[name]) * cfg.learning_rate
            sigma[name] = max(cfg.minimum_sigma, sigma[name])

        generation = CEMGeneration(
            generation=generation_index + 1,
            best_fitness=scored[0][0],
            mean_fitness=fmean(score for score, _ in scored),
            elite_mean_fitness=fmean(score for score, _ in elites),
            best_weights=scored[0][1].to_dict(),
            sigma=dict(sigma),
        )
        history.append(generation)
        if on_generation is not None:
            on_generation(generation)

    validation_seed_base = cfg.seed_base + 1_000_003
    validation = run_heuristic_benchmark(
        cfg.validation_games,
        cfg.max_pieces,
        validation_seed_base,
        cfg.seed_step,
        best_weights,
        record_best_replay=True,
    )
    return CEMResult(
        config=cfg,
        best_weights=best_weights,
        best_training_fitness=best_fitness,
        validation_fitness=benchmark_fitness(validation),
        validation=validation,
        history=tuple(history),
    )
