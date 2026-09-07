from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import statistics
import time

from minoflux_ai import (
    DEFAULT_WEIGHTS,
    FESTIVAL_SAFE_SCORER,
    NeuralValueConfig,
    NeuralValueEvaluator,
    SearchConfig,
    VersusSearchConfig,
    apply_search_action,
    build_neural_value_model,
    choose_search_action,
    extract_board_features,
    run_versus_benchmark,
)
from minoflux_engine import Game


SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=1,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
).normalized()


def run_solo(seed: int, max_pieces: int) -> dict[str, object]:
    game = Game(seed)
    quads = weak_clears = holes_peak = height_peak = 0
    started = time.perf_counter()
    while not game.game_over and game.pieces_placed < max_pieces:
        choice = choose_search_action(
            game,
            DEFAULT_WEIGHTS,
            SEARCH,
            scorer=FESTIVAL_SAFE_SCORER,
        )
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        quads += int(result.lines == 4)
        weak_clears += int(0 < result.lines < 4)
        board = extract_board_features(game.board)
        holes_peak = max(holes_peak, board.holes)
        height_peak = max(height_peak, board.max_height)
    elapsed = time.perf_counter() - started
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "lines": game.lines,
        "attack": game.attack,
        "quads": quads,
        "weakClears": weak_clears,
        "topout": game.game_over,
        "completed": not game.game_over and game.pieces_placed >= max_pieces,
        "peakHoles": holes_peak,
        "peakHeight": height_peak,
        "elapsedSeconds": elapsed,
        "pps": game.pieces_placed / max(elapsed, 1e-9),
    }


def main() -> None:
    smoke = run_solo(9_100_001, 1000)
    print("FESTIVAL_SMOKE=" + json.dumps(smoke, sort_keys=True))

    seeds = [9_200_001 + 97 * index for index in range(20)]
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        games = list(executor.map(run_solo, seeds, [500] * len(seeds), chunksize=1))
    elapsed = time.perf_counter() - started
    total_pieces = sum(int(item["pieces"]) for item in games)
    solo = {
        "games": 20,
        "maxPieces": 500,
        "pieces": total_pieces,
        "topouts": sum(bool(item["topout"]) for item in games),
        "completed": sum(bool(item["completed"]) for item in games),
        "quads": sum(int(item["quads"]) for item in games),
        "weakClears": sum(int(item["weakClears"]) for item in games),
        "attack": sum(int(item["attack"]) for item in games),
        "peakHoles": max(int(item["peakHoles"]) for item in games),
        "peakHeight": max(int(item["peakHeight"]) for item in games),
        "elapsedSeconds": elapsed,
        "pps": total_pieces / max(elapsed, 1e-9),
        "medianGamePps": statistics.median(float(item["pps"]) for item in games),
        "perGame": games,
    }
    print("FESTIVAL_SOLO20=" + json.dumps(solo, sort_keys=True))

    import torch

    default_neural = Path("data/models/neural-value-human.pt")
    if default_neural.is_file():
        neural = NeuralValueEvaluator.from_checkpoint(default_neural, device="cpu", precision="float32")
        neural_source = str(default_neural)
    else:
        torch.manual_seed(20260906)
        config = NeuralValueConfig()
        neural = NeuralValueEvaluator(
            build_neural_value_model(config),
            config,
            device="cpu",
            precision="float32",
        )
        neural_source = "deterministic-random-model (repo has no checkpoint)"

    versus_config = VersusSearchConfig(
        placement_search=SEARCH,
        candidate_width=6,
        opponent_reply_width=0,
    ).normalized()
    versus = run_versus_benchmark(
        4,
        max_turns=120,
        seed_base=9_300_001,
        seed_step=97,
        player_weights=DEFAULT_WEIGHTS,
        ai_weights=DEFAULT_WEIGHTS,
        player_config=versus_config,
        ai_config=versus_config,
        player_scorer=neural,
        ai_scorer=FESTIVAL_SAFE_SCORER,
        garbage_cap=8,
        progress=False,
        game_batch=1,
    ).to_dict()
    versus["playerNeuralSource"] = neural_source
    versus["aiPolicy"] = "festival-safe"
    print("FESTIVAL_VERSUS=" + json.dumps(versus, sort_keys=True))


if __name__ == "__main__":
    main()
