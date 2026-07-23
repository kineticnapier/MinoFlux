from __future__ import annotations

import os
import subprocess
import sys
from time import perf_counter
from typing import Any

from minoflux_ai import DEFAULT_WEIGHTS, run_heuristic_benchmark

from .run_store import RunStore
from .simulation import run_smoke


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
    save: bool,
) -> tuple[str, dict[str, Any]]:
    started = perf_counter()
    benchmark = run_heuristic_benchmark(
        max(1, int(games)),
        max(1, int(max_pieces)),
        int(seed_base),
        int(seed_step),
        DEFAULT_WEIGHTS,
    )
    result = benchmark.to_dict()
    result["weights"] = DEFAULT_WEIGHTS.to_dict()
    result["elapsedSeconds"] = round(perf_counter() - started, 3)
    if save:
        config = {
            "games": games,
            "maxPieces": max_pieces,
            "seedBase": seed_base,
            "seedStep": seed_step,
            "weights": DEFAULT_WEIGHTS.to_dict(),
        }
        run = RunStore().create("heuristic-benchmark", config)
        run.save_result(result)
        run.append_metric({
            "type": "complete",
            "meanPieces": result["meanPieces"],
            "meanLines": result["meanLines"],
            "topouts": result["topouts"],
        })
        return f"Saved to `{run.path}`", result
    return "Completed without saving.", result


def launch_game() -> str:
    subprocess.Popen([sys.executable, "-m", "minoflux.game"], cwd=os.getcwd())
    return "Pygame client launched."


def build_app():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is not installed. Run: uv sync --extra ui") from error

    with gr.Blocks(title="MinoFlux Lab") as app:
        gr.Markdown("# MinoFlux Lab\nPure-Python game, simulation, and learning experiments.")
        launch = gr.Button("Launch Pygame", variant="primary")
        launch_status = gr.Markdown()
        with gr.Tab("Heuristic benchmark"):
            gr.Markdown("Scores every direct-drop legal placement using the built-in baseline weights.")
            with gr.Row():
                ai_games = gr.Number(label="Games", value=4, precision=0, minimum=1)
                ai_max_pieces = gr.Number(label="Max pieces", value=300, precision=0, minimum=1)
                ai_seed_base = gr.Number(label="Seed base", value=1, precision=0, minimum=0)
                ai_seed_step = gr.Number(label="Seed step", value=31, precision=0, minimum=1)
            ai_save = gr.Checkbox(label="Save run files", value=True)
            ai_run = gr.Button("Run heuristic benchmark", variant="primary")
            ai_status = gr.Markdown()
            ai_result = gr.JSON(label="Benchmark result")
            gr.JSON(label="Built-in weights", value=DEFAULT_WEIGHTS.to_dict())
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
            inputs=[ai_games, ai_max_pieces, ai_seed_base, ai_seed_step, ai_save],
            outputs=[ai_status, ai_result],
        )
        run_button.click(run_experiment, inputs=[games, max_pieces, seed_base, save], outputs=[status, result])
    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("MINOFLUX_PORT", "7860")), inbrowser=True, share=False)


if __name__ == "__main__":
    main()
