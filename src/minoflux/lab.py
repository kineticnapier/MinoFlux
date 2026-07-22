from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

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


def launch_game() -> str:
    subprocess.Popen([sys.executable, "-m", "minoflux.game"], cwd=os.getcwd())
    return "Pygame client launched."


def build_app():
    try:
        import gradio as gr
    except ImportError as error:
        raise RuntimeError("Gradio is not installed. Run: python -m pip install -e '.[ui]'") from error

    with gr.Blocks(title="MinoFlux Lab") as app:
        gr.Markdown("# MinoFlux Lab\nPure-Python game, simulation, and future learning experiments.")
        launch = gr.Button("Launch Pygame", variant="primary")
        launch_status = gr.Markdown()
        with gr.Tab("Smoke simulation"):
            with gr.Row():
                games = gr.Number(label="Games", value=4, precision=0, minimum=1)
                max_pieces = gr.Number(label="Max pieces", value=200, precision=0, minimum=1)
                seed_base = gr.Number(label="Seed base", value=1, precision=0, minimum=0)
            save = gr.Checkbox(label="Save run files", value=True)
            run_button = gr.Button("Run", variant="primary")
            status = gr.Markdown()
            result = gr.JSON(label="Result")
        launch.click(launch_game, outputs=launch_status)
        run_button.click(run_experiment, inputs=[games, max_pieces, seed_base, save], outputs=[status, result])
    return app


def main() -> None:
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=int(os.environ.get("MINOFLUX_PORT", "7860")), inbrowser=True, share=False)


if __name__ == "__main__":
    main()
