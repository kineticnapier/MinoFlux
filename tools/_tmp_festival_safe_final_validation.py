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


FESTIVAL_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=1,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
).normalized()

NEURAL_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
).normalized()


def run_solo(seed: int, max_pieces: int) -> dict[str, object]:
    game = Game(seed)
    quads = 0
    weak_clears = 0
    peak_holes = 0
    peak_height = 0
    started = time.perf_counter()
    while not game.game_over and game.pieces_placed < max_pieces:
        choice = choose_search_action(
            game,
            DEFAULT_WEIGHTS,
            FESTIVAL_SEARCH,
            scorer=FESTIVAL_SAFE_SCORER,
        )
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        quads += int(result.lines == 4)
        weak_clears += int(0 < result.lines < 4)
        features = extract_board_features(game.board)
        peak_holes = max(peak_holes, features.holes)
        peak_height = max(peak_height, features.max_height)
    elapsed = time.perf_counter() - started
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "lines": game.lines,
        "attack": game.attack,
        "attackPerPiece": game.attack / max(1, game.pieces_placed),
        "quads": quads,
        "weakClears": weak_clears,
        "topout": game.game_over,
        "completed": not game.game_over and game.pieces_placed >= max_pieces,
        "peakHoles": peak_holes,
        "peakHeight": peak_height,
        "elapsedSeconds": elapsed,
        "piecesPerSecond": game.pieces_placed / max(elapsed, 1e-9),
    }


def main() -> None:
    smoke = run_solo(9_100_001, 1000)
    print("FESTIVAL_FINAL_SMOKE=" + json.dumps(smoke, sort_keys=True), flush=True)
    if not smoke["completed"]:
        raise SystemExit("festival-safe final 1000-piece smoke toped out")

    seeds = [9_200_001 + 97 * index for index in range(20)]
    batch_started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        games = list(executor.map(run_solo, seeds, [500] * len(seeds), chunksize=1))
    batch_elapsed = time.perf_counter() - batch_started
    total_pieces = sum(int(item["pieces"]) for item in games)
    total_attack = sum(int(item["attack"]) for item in games)
    solo = {
        "games": 20,
        "maxPieces": 500,
        "pieces": total_pieces,
        "attack": total_attack,
        "attackPerPiece": total_attack / max(1, total_pieces),
        "topouts": sum(bool(item["topout"]) for item in games),
        "completed": sum(bool(item["completed"]) for item in games),
        "quads": sum(int(item["quads"]) for item in games),
        "weakClears": sum(int(item["weakClears"]) for item in games),
        "peakHoles": max(int(item["peakHoles"]) for item in games),
        "peakHeight": max(int(item["peakHeight"]) for item in games),
        "elapsedSeconds": batch_elapsed,
        "aggregatePiecesPerSecond": total_pieces / max(batch_elapsed, 1e-9),
        "medianGamePiecesPerSecond": statistics.median(
            float(item["piecesPerSecond"]) for item in games
        ),
        "perGame": games,
    }
    print("FESTIVAL_FINAL_SOLO20=" + json.dumps(solo, sort_keys=True), flush=True)

    # A committed trained neural checkpoint is intentionally not required for
    # the fallback. If one exists, use it; otherwise use a deterministic random
    # neural model strictly as an integration/versus plumbing check.
    import torch

    checkpoint = Path("data/models/neural-value-human.pt")
    if checkpoint.is_file():
        neural = NeuralValueEvaluator.from_checkpoint(
            checkpoint,
            device="cpu",
            precision="float32",
        )
        neural_source = str(checkpoint)
    else:
        torch.manual_seed(20260906)
        neural_config = NeuralValueConfig()
        neural = NeuralValueEvaluator(
            build_neural_value_model(neural_config),
            neural_config,
            device="cpu",
            precision="float32",
        )
        neural_source = "deterministic-random-neural (no trained checkpoint in repository)"

    festival_versus = VersusSearchConfig(
        placement_search=FESTIVAL_SEARCH,
        candidate_width=8,
        opponent_reply_width=0,
    ).normalized()
    neural_versus = VersusSearchConfig(
        placement_search=NEURAL_SEARCH,
        candidate_width=8,
        opponent_reply_width=0,
    ).normalized()
    versus = run_versus_benchmark(
        4,
        max_turns=80,
        seed_base=9_300_001,
        seed_step=97,
        player_weights=DEFAULT_WEIGHTS,
        ai_weights=DEFAULT_WEIGHTS,
        player_config=neural_versus,
        ai_config=festival_versus,
        player_scorer=neural,
        ai_scorer=FESTIVAL_SAFE_SCORER,
        garbage_cap=8,
        progress=False,
        game_batch=1,
    ).to_dict()
    versus["playerNeuralSource"] = neural_source
    versus["aiPolicy"] = "festival-safe"
    print("FESTIVAL_FINAL_VERSUS=" + json.dumps(versus, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
