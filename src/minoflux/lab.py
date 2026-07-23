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
    HeuristicWeights,
    load_weights,
    run_heuristic_benchmark,
    save_replay,
    save_weights,
    train_cem,
)

from .run_store import RunStore
from .simulation import run_smoke

LATEST_BENCHMARK_REPLAY = Path("data/replays/latest-benchmark.json")
LATEST_CEM_REPLAY = Path("data/replays/latest-cem.json")
LATEST_CEM_MODEL = Path("data/models/latest-cem.json")

MODEL_SOURCE_BUILTIN = "Built-in weights"
MODEL_SOURCE_LATEST = "Previous latest-cem.json"
MODEL_SOURCE_CUSTOM = "Custom model path"
MODEL_SOURCE_CHOICES = (
    MODEL_SOURCE_BUILTIN,
    MODEL_SOURCE_LATEST,
    MODEL_SOURCE_CUSTOM,
)


def _default_model_source() -> str:
    return MODEL_SOURCE_LATEST if LATEST_CEM_MODEL.is_file() else MODEL_SOURCE_BUILTIN


def _load_selected_weights(model_source: str, custom_model_path: str = "") -> tuple[HeuristicWeights, str]:
    source = str(model_source or MODEL_SOURCE_BUILTIN)
    if source == MODEL_SOURCE_BUILTIN:
        return DEFAULT_WEIGHTS, MODEL_SOURCE_BUILTIN

    if source == MODEL_SOURCE_LATEST:
        model_path = LATEST_CEM_MODEL
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
    save: bool,
) -> tuple[str, dict[str, Any], str]:
    weights, weight_source = _load_selected_weights(model_source, custom_model_path)
    benchmark = run_heuristic_benchmark(
        max(1, int(games)),
        max(1, int(max_pieces)),
        int(seed_base),
        int(seed_step),
        weights,
        record_best_replay=True,
    )
    result = benchmark.to_dict()
    result["weights"] = weights.to_dict()
    result["weightSource"] = weight_source
    replay_path = ""
    if benchmark.best_replay is not None:
        replay_path = str(save_replay(LATEST_BENCHMARK_REPLAY, benchmark.best_replay).resolve())
    status = (
        f"Completed with `{weight_source}`. Best replay: `{replay_path}`"
        if replay_path
        else f"Completed with `{weight_source}` without a replay."
    )
    if save:
        config = {
            "games": games,
            "maxPieces": max_pieces,
            "seedBase": seed_base,
            "seedStep": seed_step,
            "weights": weights.to_dict(),
            "weightSource": weight_source,
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
            "topouts": result["topouts"],
            "weightSource": weight_source,
        })
        status = f"Saved to `{run.path}` using `{weight_source}`"
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
    save: bool,
) -> tuple[str, dict[str, Any], str, str]:
    initial_weights, weight_source = _load_selected_weights(model_source, custom_model_path)
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
    ).normalized()
    run = (
        RunStore().create(
            "cem-training",
            {
                "config": asdict(config),
                "initialWeights": initial_weights.to_dict(),
                "initialWeightSource": weight_source,
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
    model_path = save_weights(LATEST_CEM_MODEL, trained.best_weights).resolve()
    replay_path = ""
    if trained.validation.best_replay is not None:
        replay_path = str(save_replay(LATEST_CEM_REPLAY, trained.validation.best_replay).resolve())
    result = trained.to_dict()
    result["initialWeightSource"] = weight_source
    result["modelPath"] = str(model_path)
    result["bestReplayPath"] = replay_path
    status = (
        f"Completed in {elapsed}s using {trained.workers} worker(s), "
        f"starting from `{weight_source}`. Model: `{model_path}`"
    )
    if run is not None:
        model_path = save_weights(run.path / "model.json", trained.best_weights).resolve()
        if trained.validation.best_replay is not None:
            replay_path = str(save_replay(run.path / "best_validation_replay.json", trained.validation.best_replay).resolve())
        result["modelPath"] = str(model_path)
        result["bestReplayPath"] = replay_path
        run.save_result(result)
        run.append_metric({
            "type": "complete",
            "bestTrainingFitness": trained.best_training_fitness,
            "validationFitness": trained.validation_fitness,
            "workers": trained.workers,
            "elapsedSeconds": trained.elapsed_seconds,
            "initialWeightSource": weight_source,
        })
        status = (
            f"Saved to `{run.path}` in {elapsed}s using {trained.workers} worker(s), "
            f"starting from `{weight_source}`"
        )
    return status, result, str(model_path), replay_path


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

    default_model_source = _default_model_source()
    latest_model_note = (
        f"Previous model found at `{LATEST_CEM_MODEL}`; it is selected by default."
        if default_model_source == MODEL_SOURCE_LATEST
        else f"No previous model exists at `{LATEST_CEM_MODEL}`; built-in weights are selected."
    )

    with gr.Blocks(title="MinoFlux Lab") as app:
        gr.Markdown("# MinoFlux Lab\nPure-Python game, benchmark, replay, and CEM learning experiments.")
        launch = gr.Button("Launch Pygame")
        launch_status = gr.Markdown()
        with gr.Tab("Heuristic benchmark"):
            gr.Markdown("Scores every direct-drop legal placement and records the best game as a replay.")
            gr.Markdown(latest_model_note)
            with gr.Row():
                ai_games = gr.Number(label="Games", value=4, precision=0, minimum=1)
                ai_max_pieces = gr.Number(label="Max pieces", value=300, precision=0, minimum=1)
                ai_seed_base = gr.Number(label="Seed base", value=1, precision=0, minimum=0)
                ai_seed_step = gr.Number(label="Seed step", value=31, precision=0, minimum=1)
            with gr.Row():
                ai_model_source = gr.Radio(
                    choices=list(MODEL_SOURCE_CHOICES),
                    value=default_model_source,
                    label="Weight model",
                )
                ai_custom_model = gr.Textbox(
                    label="Custom model path",
                    placeholder="data/models/example.json",
                )
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
                "Uses board-only placement simulation, worker processes, and an optional short screening round before full evaluation."
            )
            gr.Markdown(latest_model_note)
            with gr.Row():
                cem_generations = gr.Number(label="Generations", value=5, precision=0, minimum=1)
                cem_population = gr.Number(label="Population", value=12, precision=0, minimum=2)
                cem_elite = gr.Number(label="Elite fraction", value=0.25, minimum=0.05, maximum=1.0)
            with gr.Row():
                cem_games = gr.Number(label="Full games / retained candidate", value=2, precision=0, minimum=1)
                cem_max_pieces = gr.Number(label="Full max pieces", value=150, precision=0, minimum=1)
                cem_validation = gr.Number(label="Validation games", value=4, precision=0, minimum=1)
            with gr.Row():
                cem_workers = gr.Number(label="Worker processes (0 = auto)", value=0, precision=0, minimum=0)
                cem_screen_games = gr.Number(label="Screen games (0 = off)", value=1, precision=0, minimum=0)
                cem_screen_pieces = gr.Number(label="Screen max pieces", value=60, precision=0, minimum=0)
                cem_screen_fraction = gr.Number(label="Fraction kept", value=0.5, minimum=0.05, maximum=1.0)
            with gr.Row():
                cem_sigma = gr.Number(label="Initial sigma", value=0.35, minimum=0)
                cem_learning_rate = gr.Number(label="Learning rate", value=0.7, minimum=0.01, maximum=1.0)
                cem_seed_base = gr.Number(label="Training seed base", value=1, precision=0)
                cem_random_seed = gr.Number(label="Sampler seed", value=12345, precision=0)
            with gr.Row():
                cem_model_source = gr.Radio(
                    choices=list(MODEL_SOURCE_CHOICES),
                    value=default_model_source,
                    label="Starting weight model",
                )
                cem_custom_model = gr.Textbox(
                    label="Custom starting model path",
                    placeholder="data/models/example.json",
                )
            cem_save = gr.Checkbox(label="Save run files", value=True)
            cem_run = gr.Button("Train weights", variant="primary")
            cem_replay = gr.Button("Replay best validation game")
            cem_status = gr.Markdown()
            cem_replay_status = gr.Markdown()
            cem_result = gr.JSON(label="Training result")
            cem_model_path = gr.Textbox(label="Model path", interactive=False)
            cem_replay_path = gr.Textbox(label="Best replay path", interactive=False)
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
                ai_games,
                ai_max_pieces,
                ai_seed_base,
                ai_seed_step,
                ai_model_source,
                ai_custom_model,
                ai_save,
            ],
            outputs=[ai_status, ai_result, ai_replay_path],
        )
        ai_replay.click(launch_replay, inputs=ai_replay_path, outputs=ai_replay_status)
        cem_run.click(
            run_cem_experiment,
            inputs=[
                cem_generations,
                cem_population,
                cem_elite,
                cem_games,
                cem_max_pieces,
                cem_validation,
                cem_sigma,
                cem_learning_rate,
                cem_seed_base,
                cem_random_seed,
                cem_workers,
                cem_screen_games,
                cem_screen_pieces,
                cem_screen_fraction,
                cem_model_source,
                cem_custom_model,
                cem_save,
            ],
            outputs=[cem_status, cem_result, cem_model_path, cem_replay_path],
        )
        cem_replay.click(launch_replay, inputs=cem_replay_path, outputs=cem_replay_status)
        run_button.click(run_experiment, inputs=[games, max_pieces, seed_base, save], outputs=[status, result])
    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("MINOFLUX_PORT", "7860")), inbrowser=True, share=False)


if __name__ == "__main__":
    main()
