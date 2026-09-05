from __future__ import annotations

import hashlib
import json
import os
import statistics
import time

import torch

from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator, build_neural_value_model
from minoflux_ai.neural_search_fast import choose_search_actions_batch
from minoflux_ai.reachability import ReachabilityProfile, clear_reachability_cache, collect_reachability_profile
from minoflux_ai.reachability_native import clear_native_record_cache
from minoflux_ai.search import SearchConfig, apply_search_action
from minoflux_engine import Game

CFG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
).normalized()
NCFG = NeuralValueConfig().normalized()
SEEDS = tuple(8100001 + i * 97 for i in range(20))
MAX_PIECES = 120
WORKERS = (1, 2, 4, 6, 8)


def make_evaluator(state_dict):
    model = build_neural_value_model(NCFG)
    model.load_state_dict(state_dict)
    return NeuralValueEvaluator(model, NCFG, device="cpu")


def run(state_dict, workers: int, *, profile: bool = False):
    os.environ["MINOFLUX_NATIVE_REACHABILITY_THREADS"] = str(workers)
    evaluator = make_evaluator(state_dict)
    games = [Game(seed) for seed in SEEDS]
    clear_reachability_cache()
    clear_native_record_cache()
    reach = ReachabilityProfile() if profile else None
    digest = hashlib.sha256()
    started = time.perf_counter()
    while True:
        active = tuple(
            game
            for game in games
            if not game.game_over and game.pieces_placed < MAX_PIECES
        )
        if not active:
            break
        if reach is None:
            choices = choose_search_actions_batch(active, config=CFG, scorer=evaluator)
        else:
            with collect_reachability_profile(reach):
                choices = choose_search_actions_batch(active, config=CFG, scorer=evaluator)
        for game, choice in zip(active, choices):
            if choice is None:
                game.game_over = True
                continue
            p = choice.action.placement
            digest.update(
                repr(
                    (
                        choice.action.use_hold,
                        p.piece,
                        p.x,
                        p.y,
                        p.rotation,
                        p.last_move_was_rotation,
                        p.rotation_kick_index,
                        p.rotation_from,
                        p.rotation_to,
                        choice.score,
                    )
                ).encode()
            )
            apply_search_action(game, choice.action)
    elapsed = time.perf_counter() - started
    pieces = sum(game.pieces_placed for game in games)
    result = {
        "workers": workers,
        "signature": digest.hexdigest(),
        "pieces": pieces,
        "attack": sum(game.attack for game in games),
        "topouts": sum(int(game.game_over) for game in games),
        "completed": sum(
            int(not game.game_over and game.pieces_placed >= MAX_PIECES)
            for game in games
        ),
        "elapsed": elapsed,
        "pps": pieces / elapsed,
    }
    if reach is not None:
        result["reachability"] = reach.to_dict()
    return result


def main():
    # Reduce unrelated CPU model-thread noise; this benchmark is for root-SRS
    # scheduling rather than model throughput.
    torch.set_num_threads(1)
    torch.manual_seed(20260905)
    template = build_neural_value_model(NCFG)
    state_dict = {
        name: tensor.detach().clone()
        for name, tensor in template.state_dict().items()
    }

    results = {}
    reference = None
    for workers in WORKERS:
        profile = run(state_dict, workers, profile=True)
        times = [run(state_dict, workers)["elapsed"] for _ in range(5)]
        entry = {
            "profile": profile,
            "times": times,
            "median": statistics.median(times),
        }
        if reference is None:
            reference = profile
        else:
            assert profile["signature"] == reference["signature"]
            for key in ("pieces", "attack", "topouts", "completed"):
                assert profile[key] == reference[key], (
                    workers,
                    key,
                    reference[key],
                    profile[key],
                )
        results[str(workers)] = entry

    baseline = results["1"]["median"]
    for workers in WORKERS:
        results[str(workers)]["speedupVs1"] = baseline / results[str(workers)]["median"]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
