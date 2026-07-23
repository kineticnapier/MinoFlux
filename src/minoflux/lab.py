from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from minoflux_ai import (
    CEMConfig,
    DEFAULT_WEIGHTS,
    FITNESS_PROFILE_ATTACK_SPIN,
    FITNESS_PROFILE_NAMES,
    HeuristicWeights,
    PromotionConfig,
    SearchConfig,
    benchmark_fitness,
    bootstrap_champion,
    evaluate_and_promote_model,
    load_weights,
    resolve_fitness_profile,
    run_heuristic_benchmark,
    save_replay,
    save_weights,
    train_cem,
)

from .run_store import RunStore
from .simulation import run_smoke

LATEST_BENCHMARK_REPLAY = Path("data/replays/latest-benchmark.json")
LATEST_CEM_REPLAY = Path("data/replays/latest-cem.json")
CHAMPION_CEM_MODEL = Path("data/models/champion-cem.json")
CANDIDATE_CEM_MODEL = Path("data/models/candidate-cem.json")
LEGACY_LATEST_CEM_MODEL = Path("data/models/latest-cem.json")
MODEL_HISTORY_DIR = Path("data/models/history")
RECOVERED_ATTACK_MODEL = Path("presets/recovered-attack-20260723.json")

# Kept for compatibility with existing imports/tests. It now means champion.
LATEST_CEM_MODEL = CHAMPION_CEM_MODEL

MODEL_SOURCE_BUILTIN = "Built-in weights"
MODEL_SOURCE_LATEST = "Champion model"
MODEL_SOURCE_RECOVERED = "Recovered attack model (2026-07-23)"
MODEL_SOURCE_CUSTOM = "Custom model path"
MODEL_SOURCE_CHOICES = (
    MODEL_SOURCE_LATEST,
    MODEL_SOURCE_RECOVERED,
    MODEL_SOURCE_BUILTIN,
    MODEL_SOURCE_CUSTOM,
)


def _bootstrap_champion() -> Path | None:
    champion = bootstrap_champion(
        LATEST_CEM_MODEL,
        recovery_path=RECOVERED_ATTACK_MODEL,
        legacy_path=LEGACY_LATEST_CEM_MODEL,
    )
    if champion is not None:
        # Keep the old filename as a compatibility alias and overwrite stale models.
        save_weights(LEGACY_LATEST_CEM_MODEL, load_weights(champion))
    return champion


def _default_model_source() -> str:
    return MODEL_SOURCE_LATEST if LATEST_CEM_MODEL.is_file() else MODEL_SOURCE_BUILTIN


def _load_selected_weights(model_source: str, custom_model_path: str = "") -> tuple[HeuristicWeights, str]:
    source = str(model_source or MODEL_SOURCE_BUILTIN)
    if source == MODEL_SOURCE_BUILTIN:
        return DEFAULT_WEIGHTS, MODEL_SOURCE_BUILTIN

    if source == MODEL_SOURCE_LATEST:
        model_path = LATEST_CEM_MODEL
    elif source == MODEL_SOURCE_RECOVERED:
        model_path = RECOVERED_ATTACK_MODEL
    elif source == MODEL_SOURCE_CUSTOM:
        value = str(custom_model_path or "").strip()
        if not value:
            raise ValueError("Custom model path is empty.")
        model_path = Path(value).expanduser()
    else:
        raise ValueError(f"Unknown model source: {source!r}")

    if not model_path.is_file():
        raise ValueError(f"Model file does not exist: {model_path}")
    return load_weights(model_path), str(model_path.resolve())


def _search_config(
    allow_hold: bool,
    lookahead_pieces: int,
    beam_width: int,
    lookahead_discount: float,
) -> SearchConfig:
    return SearchConfig(
        allow_hold=bool(allow_hold),
        lookahead_pieces=int(lookahead_pieces),
        beam_width=int(beam_width),
        discount=float(lookahead_discount),
    ).normalized()


def run_experiment(games: int, max_pieces: int, seed_base: int, save: bool) -> tuple[str, dict[str, Any]]:
    result = run_smoke(max(1, int(games)), max(1, int(max_pieces)), int(seed_base))
    if save:
        run = RunStore().create("engine-smoke", {"games": games, "maxPieces": max_pieces, "seedBase": seed_base})
        run.save_result(result)
        run.append_metric({"type": "complete", "meanPieces": result["meanPieces"], "topouts": result["topouts"]})
        return f"Saved to `{run.path}`", result
    return "Completed without saving.", result


def run_heuristic_experiment(
    games: int,
    max_pieces: int,
    seed_base: int,
    seed_step: int,
    model_source: str,
    custom_model_path: str,
    allow_hold: bool,
    lookahead_pieces: int,
    beam_width: int,
    lookahead_discount: float,
    fitness_profile: str,
    save: bool,
) -> tuple[str, dict[str, Any], str]:
    weights, weight_source = _load_selected_weights(model_source, custom_model_path)
    search_config = _search_config(allow_hold, lookahead_pieces, beam_width, lookahead_discount)
    profile = resolve_fitness_profile(fitness_profile)
    benchmark = run_heuristic_benchmark(
        max(1, int(games)),
        max(1, int(max_pieces)),
        int(seed_base),
        int(seed_step),
        weights,
        search_config,
        record_best_replay=True,
    )
    result = benchmark.to_dict()
    result["weights"] = weights.to_dict()
    result["weightSource"] = weight_source
    result["fitnessProfile"] = profile.to_dict()
    result["fitness"] = benchmark_fitness(benchmark, profile)
    replay_path = ""
    if benchmark.best_replay is not None:
        replay_path = str(save_replay(LATEST_BENCHMARK_REPLAY, benchmark.best_replay).resolve())
    status = (
        f"Completed with `{weight_source}`; fitness {result['fitness']:.3f}; best replay `{replay_path}`"
        if replay_path
        else f"Completed with `{weight_source}`; fitness {result['fitness']:.3f}."
    )
    if save:
        config = {
            "games": games,
            "maxPieces": max_pieces,
            "seedBase": seed_base,
            "seedStep": seed_step,
            "weights": weights.to_dict(),
            "weightSource": weight_source,
            "searchConfig": search_config.to_dict(),
            "fitnessProfile": profile.to_dict(),
        }
        run = RunStore().create("heuristic-benchmark", config)
        if benchmark.best_replay is not None:
            replay_path = str(save_replay(run.path / "best_replay.json", benchmark.best_replay).resolve())
        result["bestReplayPath"] = replay_path
        run.save_result(result)
        run.append_metric({
            "type": "complete",
            "meanPieces": result["meanPieces"],
            "meanLines": result["meanLines"],
            "meanAttack": result["meanAttack"],
            "meanSpins": result["meanSpins"],
            "meanSpinLines": result["meanSpinLines"],
            "fitness": result["fitness"],
            "topouts": result["topouts"],
            "weightSource": weight_source,
            "searchConfig": search_config.to_dict(),
        })
        status = f"Saved to `{run.path}` using `{weight_source}`; fitness {result['fitness']:.3f}"
    result["bestReplayPath"] = replay_path
    return status, result, replay_path


def run_cem_experiment(
    generations: int,
    population: int,
    elite_fraction: float,
    games_per_candidate: int,
    max_pieces: int,
    validation_games: int,
    initial_sigma: float,
    learning_rate: float,
    seed_base: int,
    random_seed: int,
    workers: int,
    screen_games: int,
    screen_max_pieces: int,
    screen_fraction: float,
    model_source: str,
    custom_model_path: str,
    allow_hold: bool,
    lookahead_pieces: int,
    beam_width: int,
    lookahead_discount: float,
    fitness_profile: str,
    promotion_games: int,
    promotion_max_pieces: int,
    minimum_fitness_gain: float,
    max_completion_loss: int,
    save: bool,
) -> tuple[str, dict[str, Any], str, str]:
    _bootstrap_champion()
    initial_weights, weight_source = _load_selected_weights(model_source, custom_model_path)
    search_config = _search_config(allow_hold, lookahead_pieces, beam_width, lookahead_discount)
    profile = resolve_fitness_profile(fitness_profile)
    config = CEMConfig(
        generations=int(generations),
        population=int(population),
        elite_fraction=float(elite_fraction),
        games_per_candidate=int(games_per_candidate),
        max_pieces=int(max_pieces),
        validation_games=int(validation_games),
        initial_sigma=float(initial_sigma),
        learning_rate=float(learning_rate),
        seed_base=int(seed_base),
        random_seed=int(random_seed),
        workers=int(workers),
        screen_games=int(screen_games),
        screen_max_pieces=int(screen_max_pieces),
        screen_fraction=float(screen_fraction),
        allow_hold=search_config.allow_hold,
        lookahead_pieces=search_config.lookahead_pieces,
        beam_width=search_config.beam_width,
        lookahead_discount=search_config.discount,
        fitness_profile=profile.name,
    ).normalized()
    run = (
        RunStore().create(
            "cem-training",
            {
                "config": asdict(config),
                "initialWeights": initial_weights.to_dict(),
                "initialWeightSource": weight_source,
                "searchConfig": search_config.to_dict(),
                "fitnessProfile": profile.to_dict(),
            },
        )
        if save
        else None
    )

    def on_generation(generation) -> None:
        if run is not None:
            run.append_metric({"type": "generation", **generation.to_dict()})

    trained = train_cem(config, initial_weights, on_generation)
    elapsed = trained.elapsed_seconds
    replay_path = ""
    if trained.validation.best_replay is not None:
        replay_path = str(save_replay(LATEST_CEM_REPLAY, trained.validation.best_replay).resolve())

    promotion_config = PromotionConfig(
        games=int(promotion_games),
        max_pieces=int(promotion_max_pieces),
        seed_base=int(seed_base) + 2_000_033,
        seed_step=97,
        minimum_fitness_gain=float(minimum_fitness_gain),
        max_completion_loss=int(max_completion_loss),
        workers=0,
    )
    promotion = evaluate_and_promote_model(
        trained.best_weights,
        search_config,
        champion_path=LATEST_CEM_MODEL,
        candidate_path=CANDIDATE_CEM_MODEL,
        history_dir=MODEL_HISTORY_DIR,
        compatibility_latest_path=LEGACY_LATEST_CEM_MODEL,
        fitness_profile=profile,
        config=promotion_config,
    )
    model_path = LATEST_CEM_MODEL if promotion.promoted else CANDIDATE_CEM_MODEL

    result = trained.to_dict()
    result["initialWeightSource"] = weight_source
    result["candidateModelPath"] = str(CANDIDATE_CEM_MODEL.resolve())
    result["championModelPath"] = str(LATEST_CEM_MODEL.resolve())
    result["modelPath"] = str(model_path.resolve())
    result["bestReplayPath"] = replay_path
    result["promotion"] = promotion.to_dict()

    if run is not None:
        save_weights(run.path / "candidate_model.json", trained.best_weights)
        if trained.validation.best_replay is not None:
            replay_path = str(save_replay(run.path / "best_validation_replay.json", trained.validation.best_replay).resolve())
        result["bestReplayPath"] = replay_path
        run.save_result(result)
        run.append_metric({
            "type": "complete",
            "bestTrainingFitness": trained.best_training_fitness,
            "validationFitness": trained.validation_fitness,
            "workers": trained.workers,
            "elapsedSeconds": trained.elapsed_seconds,
            "initialWeightSource": weight_source,
            "searchConfig": search_config.to_dict(),
            "fitnessProfile": profile.to_dict(),
            "promoted": promotion.promoted,
            "promotionFitnessGain": promotion.fitness_gain,
        })

    status = (
        f"Completed in {elapsed}s. {promotion.reason} "
        f"Candidate: `{CANDIDATE_CEM_MODEL}`; champion: `{LATEST_CEM_MODEL}`"
    )
    return status, result, str(model_path.resolve()), replay_path


def launch_game() -> str:
    subprocess.Popen([sys.executable, "-m", "minoflux.game"], cwd=os.getcwd())
    return "Pygame client launched."


def launch_replay(path: str) -> str:
    replay_path = Path(path)
    if not path or not replay_path.is_file():
        return "Run a benchmark or CEM training first; no replay file is available."
    subprocess.Popen([sys.executable, "-m", "minoflux.replay", str(replay_path.resolve())], cwd=os.getcwd())
    return f"Replay launched: `{replay_path}`"


def build_app():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is not installed. Run: uv sync --extra ui") from error

    _bootstrap_champion()
    default_model_source = _default_model_source()
    champion_note = (
        f"Champion: `{LATEST_CEM_MODEL}`. A new CEM result replaces it only after winning a separate promotion benchmark."
        if LATEST_CEM_MODEL.is_file()
        else "No champion exists yet; the first candidate will be promoted automatically."
    )

    with gr.Blocks(title="MinoFlux Lab") as app:
        gr.Markdown("# MinoFlux Lab\nGame, attack/Spin training, champion promotion, replay, and beam-search experiments.")
        launch = gr.Button("Launch Pygame")
        launch_status = gr.Markdown()

        with gr.Tab("Heuristic benchmark"):
            gr.Markdown("Benchmarks a selected model. Attack/Spin fitness is shown directly in the result.")
            gr.Markdown(champion_note)
            with gr.Row():
                ai_games = gr.Number(label="Games", value=10, precision=0, minimum=1)
                ai_max_pieces = gr.Number(label="Max pieces", value=1000, precision=0, minimum=1)
                ai_seed_base = gr.Number(label="Seed base", value=1, precision=0, minimum=0)
                ai_seed_step = gr.Number(label="Seed step", value=31, precision=0, minimum=1)
            with gr.Row():
                ai_model_source = gr.Radio(
                    choices=list(MODEL_SOURCE_CHOICES), value=default_model_source, label="Weight model"
                )
                ai_custom_model = gr.Textbox(label="Custom model path", placeholder="data/models/example.json")
                ai_fitness_profile = gr.Dropdown(
                    choices=list(FITNESS_PROFILE_NAMES), value=FITNESS_PROFILE_ATTACK_SPIN, label="Fitness profile"
                )
            with gr.Row():
                ai_hold = gr.Checkbox(label="Allow Hold", value=True)
                ai_lookahead = gr.Number(label="Future lookahead pieces", value=1, precision=0, minimum=0, maximum=3)
                ai_beam = gr.Number(label="Beam width", value=4, precision=0, minimum=1, maximum=128)
                ai_discount = gr.Number(label="Lookahead discount", value=0.90, minimum=0, maximum=1)
            ai_save = gr.Checkbox(label="Save run files", value=True)
            ai_run = gr.Button("Run heuristic benchmark", variant="primary")
            ai_replay = gr.Button("Replay best game")
            ai_status = gr.Markdown()
            ai_replay_status = gr.Markdown()
            ai_result = gr.JSON(label="Benchmark result")
            ai_replay_path = gr.Textbox(label="Best replay path", interactive=False)
            gr.JSON(label="Built-in weights", value=DEFAULT_WEIGHTS.to_dict())

        with gr.Tab("CEM weight training"):
            gr.Markdown(
                "The default objective prioritizes Attack and Spin lines. Every trained model is saved as a candidate; "
                "the champion is replaced only when the candidate wins on separate unseen seeds."
            )
            gr.Markdown(champion_note)
            with gr.Row():
                cem_generations = gr.Number(label="Generations", value=10, precision=0, minimum=1)
                cem_population = gr.Number(label="Population", value=24, precision=0, minimum=2)
                cem_elite = gr.Number(label="Elite fraction", value=0.25, minimum=0.05, maximum=1.0)
            with gr.Row():
                cem_games = gr.Number(label="Full games / retained candidate", value=3, precision=0, minimum=1)
                cem_max_pieces = gr.Number(label="Full max pieces", value=300, precision=0, minimum=1)
                cem_validation = gr.Number(label="Validation games", value=6, precision=0, minimum=1)
            with gr.Row():
                cem_workers = gr.Number(label="Worker processes (0 = auto)", value=0, precision=0, minimum=0)
                cem_screen_games = gr.Number(label="Screen games (0 = off)", value=1, precision=0, minimum=0)
                cem_screen_pieces = gr.Number(label="Screen max pieces", value=60, precision=0, minimum=0)
                cem_screen_fraction = gr.Number(label="Fraction kept", value=0.5, minimum=0.05, maximum=1.0)
            with gr.Row():
                cem_sigma = gr.Number(label="Initial sigma", value=0.25, minimum=0)
                cem_learning_rate = gr.Number(label="Learning rate", value=0.7, minimum=0.01, maximum=1.0)
                cem_seed_base = gr.Number(label="Training seed base", value=1, precision=0)
                cem_random_seed = gr.Number(label="Sampler seed", value=12345, precision=0)
            with gr.Row():
                cem_model_source = gr.Radio(
                    choices=list(MODEL_SOURCE_CHOICES), value=default_model_source, label="Starting weight model"
                )
                cem_custom_model = gr.Textbox(
                    label="Custom starting model path", placeholder="data/models/example.json"
                )
                cem_fitness_profile = gr.Dropdown(
                    choices=list(FITNESS_PROFILE_NAMES), value=FITNESS_PROFILE_ATTACK_SPIN, label="Fitness profile"
                )
            with gr.Row():
                cem_hold = gr.Checkbox(label="Allow Hold", value=True)
                cem_lookahead = gr.Number(label="Future lookahead pieces", value=0, precision=0, minimum=0, maximum=3)
                cem_beam = gr.Number(label="Beam width", value=4, precision=0, minimum=1, maximum=128)
                cem_discount = gr.Number(label="Lookahead discount", value=0.90, minimum=0, maximum=1)
            with gr.Row():
                promotion_games = gr.Number(label="Promotion games", value=10, precision=0, minimum=1)
                promotion_pieces = gr.Number(label="Promotion max pieces", value=1000, precision=0, minimum=1)
                promotion_gain = gr.Number(label="Required fitness gain", value=0.0)
                promotion_completion_loss = gr.Number(label="Max completion loss", value=1, precision=0, minimum=0)
            cem_save = gr.Checkbox(label="Save run files", value=True)
            cem_run = gr.Button("Train candidate and challenge champion", variant="primary")
            cem_replay = gr.Button("Replay candidate validation game")
            cem_status = gr.Markdown()
            cem_replay_status = gr.Markdown()
            cem_result = gr.JSON(label="Training and promotion result")
            cem_model_path = gr.Textbox(label="Selected model path", interactive=False)
            cem_replay_path = gr.Textbox(label="Candidate replay path", interactive=False)

        with gr.Tab("Random smoke simulation"):
            with gr.Row():
                games = gr.Number(label="Games", value=4, precision=0, minimum=1)
                max_pieces = gr.Number(label="Max pieces", value=200, precision=0, minimum=1)
                seed_base = gr.Number(label="Seed base", value=1, precision=0, minimum=0)
            save = gr.Checkbox(label="Save run files", value=True)
            run_button = gr.Button("Run", variant="primary")
            status = gr.Markdown()
            result = gr.JSON(label="Result")

        launch.click(launch_game, outputs=launch_status)
        ai_run.click(
            run_heuristic_experiment,
            inputs=[
                ai_games, ai_max_pieces, ai_seed_base, ai_seed_step,
                ai_model_source, ai_custom_model,
                ai_hold, ai_lookahead, ai_beam, ai_discount,
                ai_fitness_profile, ai_save,
            ],
            outputs=[ai_status, ai_result, ai_replay_path],
        )
        ai_replay.click(launch_replay, inputs=ai_replay_path, outputs=ai_replay_status)
        cem_run.click(
            run_cem_experiment,
            inputs=[
                cem_generations, cem_population, cem_elite,
                cem_games, cem_max_pieces, cem_validation,
                cem_sigma, cem_learning_rate, cem_seed_base, cem_random_seed,
                cem_workers, cem_screen_games, cem_screen_pieces, cem_screen_fraction,
                cem_model_source, cem_custom_model,
                cem_hold, cem_lookahead, cem_beam, cem_discount,
                cem_fitness_profile,
                promotion_games, promotion_pieces, promotion_gain, promotion_completion_loss,
                cem_save,
            ],
            outputs=[cem_status, cem_result, cem_model_path, cem_replay_path],
        )
        cem_replay.click(launch_replay, inputs=cem_replay_path, outputs=cem_replay_status)
        run_button.click(run_experiment, inputs=[games, max_pieces, seed_base, save], outputs=[status, result])
    return app


def main() -> None:
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=int(os.environ.get("MINOFLUX_PORT", "7860")),
        inbrowser=True,
        share=False,
    )


if __name__ == "__main__":
    main()
