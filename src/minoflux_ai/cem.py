from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from math import ceil
import os
import random
from statistics import fmean, pstdev
from time import perf_counter
from typing import Callable

from .benchmark import BenchmarkResult, run_heuristic_benchmark
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .search import SearchConfig

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
    workers: int = 0
    screen_games: int = 1
    screen_max_pieces: int = 60
    screen_fraction: float = 0.5
    allow_hold: bool = True
    lookahead_pieces: int = 0
    beam_width: int = 4
    lookahead_discount: float = 0.90

    def normalized(self) -> "CEMConfig":
        search = self.search_config()
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
            workers=max(0, int(self.workers)),
            screen_games=max(0, int(self.screen_games)),
            screen_max_pieces=max(0, int(self.screen_max_pieces)),
            screen_fraction=min(1.0, max(0.05, float(self.screen_fraction))),
            allow_hold=search.allow_hold,
            lookahead_pieces=search.lookahead_pieces,
            beam_width=search.beam_width,
            lookahead_discount=search.discount,
        )

    def search_config(self) -> SearchConfig:
        return SearchConfig(
            allow_hold=self.allow_hold,
            lookahead_pieces=self.lookahead_pieces,
            beam_width=self.beam_width,
            discount=self.lookahead_discount,
        ).normalized()

    def resolved_workers(self) -> int:
        if self.workers > 0:
            return min(self.population, self.workers)
        available = max(1, (os.cpu_count() or 1) - 1)
        return min(self.population, available)

    def screening_enabled(self) -> bool:
        return (
            self.screen_games > 0
            and self.screen_max_pieces > 0
            and self.screen_fraction < 1.0
        )


@dataclass(frozen=True, slots=True)
class CEMGeneration:
    generation: int
    best_fitness: float
    mean_fitness: float
    elite_mean_fitness: float
    best_weights: dict[str, float]
    sigma: dict[str, float]
    evaluated_candidates: int
    screened_out_candidates: int
    elapsed_seconds: float = field(compare=False)

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
    workers: int
    elapsed_seconds: float = field(compare=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "searchConfig": self.config.search_config().to_dict(),
            "workers": self.workers,
            "elapsedSeconds": self.elapsed_seconds,
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


def _weights_key(weights: HeuristicWeights) -> tuple[float, ...]:
    return tuple(getattr(weights, name) for name in TRAINABLE_WEIGHT_NAMES)


def _search_key(config: SearchConfig) -> tuple[object, ...]:
    cfg = config.normalized()
    return cfg.allow_hold, cfg.lookahead_pieces, cfg.beam_width, cfg.discount


def _evaluate_candidate_task(
    task: tuple[HeuristicWeights, int, int, int, int, SearchConfig],
) -> float:
    weights, games, max_pieces, seed_base, seed_step, search_config = task
    benchmark = run_heuristic_benchmark(
        games,
        max_pieces,
        seed_base,
        seed_step,
        weights,
        search_config,
    )
    return benchmark_fitness(benchmark)


def _score_candidates(
    candidates: list[HeuristicWeights],
    *,
    games: int,
    max_pieces: int,
    seed_base: int,
    seed_step: int,
    search_config: SearchConfig,
    executor: ProcessPoolExecutor | None,
    cache: dict[tuple[object, ...], float],
) -> list[tuple[float, HeuristicWeights]]:
    unique_missing: list[HeuristicWeights] = []
    missing_keys: list[tuple[object, ...]] = []
    seen_missing: set[tuple[object, ...]] = set()
    search_key = _search_key(search_config)
    for candidate in candidates:
        key = (games, max_pieces, seed_base, seed_step, *search_key, *_weights_key(candidate))
        if key not in cache and key not in seen_missing:
            unique_missing.append(candidate)
            missing_keys.append(key)
            seen_missing.add(key)

    tasks = [
        (candidate, games, max_pieces, seed_base, seed_step, search_config)
        for candidate in unique_missing
    ]
    if tasks:
        if executor is None:
            scores = map(_evaluate_candidate_task, tasks)
        else:
            scores = executor.map(_evaluate_candidate_task, tasks, chunksize=1)
        for key, score in zip(missing_keys, scores):
            cache[key] = score

    return [
        (
            cache[(games, max_pieces, seed_base, seed_step, *search_key, *_weights_key(candidate))],
            candidate,
        )
        for candidate in candidates
    ]


def train_cem(
    config: CEMConfig = CEMConfig(),
    initial_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    on_generation: Callable[[CEMGeneration], None] | None = None,
) -> CEMResult:
    started = perf_counter()
    cfg = config.normalized()
    search_config = cfg.search_config()
    workers = cfg.resolved_workers()
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
        search_config,
    )
    best_fitness = benchmark_fitness(baseline)
    history: list[CEMGeneration] = []
    cache: dict[tuple[object, ...], float] = {}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None

    try:
        for generation_index in range(cfg.generations):
            generation_started = perf_counter()
            population: list[HeuristicWeights] = [_candidate_from_values(initial_weights, mean)]
            while len(population) < cfg.population:
                sampled = {
                    name: rng.gauss(mean[name], sigma[name])
                    for name in TRAINABLE_WEIGHT_NAMES
                }
                population.append(_candidate_from_values(initial_weights, sampled))

            elite_count = max(1, min(cfg.population, ceil(cfg.population * cfg.elite_fraction)))
            screened_out = 0
            selected = population
            if cfg.screening_enabled() and len(population) > elite_count:
                screen_scores = _score_candidates(
                    population,
                    games=cfg.screen_games,
                    max_pieces=min(cfg.max_pieces, cfg.screen_max_pieces),
                    seed_base=cfg.seed_base,
                    seed_step=cfg.seed_step,
                    search_config=search_config,
                    executor=executor,
                    cache=cache,
                )
                screen_scores.sort(key=lambda item: item[0], reverse=True)
                keep_count = max(elite_count, ceil(cfg.population * cfg.screen_fraction))
                selected = [candidate for _, candidate in screen_scores[:keep_count]]
                screened_out = len(population) - len(selected)

            scored = _score_candidates(
                selected,
                games=cfg.games_per_candidate,
                max_pieces=cfg.max_pieces,
                seed_base=cfg.seed_base,
                seed_step=cfg.seed_step,
                search_config=search_config,
                executor=executor,
                cache=cache,
            )
            scored.sort(key=lambda item: item[0], reverse=True)
            elites = scored[:elite_count]
            if scored[0][0] > best_fitness:
                best_fitness, best_weights = scored[0]

            elite_means: dict[str, float] = {}
            elite_sigmas: dict[str, float] = {}
            for name in TRAINABLE_WEIGHT_NAMES:
                values = [getattr(candidate, name) for _, candidate in elites]
                elite_means[name] = fmean(values)
                elite_sigmas[name] = max(
                    cfg.minimum_sigma,
                    pstdev(values) if len(values) > 1 else sigma[name] * 0.5,
                )

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
                evaluated_candidates=len(scored),
                screened_out_candidates=screened_out,
                elapsed_seconds=round(perf_counter() - generation_started, 3),
            )
            history.append(generation)
            if on_generation is not None:
                on_generation(generation)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    validation_seed_base = cfg.seed_base + 1_000_003
    validation = run_heuristic_benchmark(
        cfg.validation_games,
        cfg.max_pieces,
        validation_seed_base,
        cfg.seed_step,
        best_weights,
        search_config,
        record_best_replay=True,
    )
    return CEMResult(
        config=cfg,
        best_weights=best_weights,
        best_training_fitness=best_fitness,
        validation_fitness=benchmark_fitness(validation),
        validation=validation,
        history=tuple(history),
        workers=workers,
        elapsed_seconds=round(perf_counter() - started, 3),
    )
