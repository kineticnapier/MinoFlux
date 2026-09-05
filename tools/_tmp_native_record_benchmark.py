from __future__ import annotations

import hashlib
import json
import statistics
import time

import torch

from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator, build_neural_value_model
from minoflux_ai.neural_search_fast import (
    _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH,
    choose_search_actions_batch as choose_native_record_actions_batch,
)
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


def make_evaluator(state_dict):
    model = build_neural_value_model(NCFG)
    model.load_state_dict(state_dict)
    return NeuralValueEvaluator(model, NCFG, device="cpu")


def run(which: str, state_dict, *, profile: bool = False):
    evaluator = make_evaluator(state_dict)
    games = [Game(seed) for seed in SEEDS]
    choose = (
        _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH
        if which == "placement"
        else choose_native_record_actions_batch
    )
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
            choices = choose(active, config=CFG, scorer=evaluator)
        else:
            with collect_reachability_profile(reach):
                choices = choose(active, config=CFG, scorer=evaluator)
        for game, choice in zip(active, choices):
            if choice is None:
                game.game_over = True
                continue
            p = choice.action.placement
            digest.update(
                repr((
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
                )).encode()
            )
            apply_search_action(game, choice.action)
    elapsed = time.perf_counter() - started
    summary = {
        "signature": digest.hexdigest(),
        "pieces": sum(g.pieces_placed for g in games),
        "attack": sum(g.attack for g in games),
        "topouts": sum(int(g.game_over) for g in games),
        "completed": sum(int(not g.game_over and g.pieces_placed >= MAX_PIECES) for g in games),
        "elapsed": elapsed,
        "pps": sum(g.pieces_placed for g in games) / elapsed,
    }
    if reach is not None:
        summary["reachability"] = reach.to_dict()
    return summary


def main():
    torch.manual_seed(20260905)
    template = build_neural_value_model(NCFG)
    state_dict = {name: tensor.detach().clone() for name, tensor in template.state_dict().items()}

    placement_profile = run("placement", state_dict, profile=True)
    record_profile = run("record", state_dict, profile=True)
    assert placement_profile["signature"] == record_profile["signature"]
    assert {k: placement_profile[k] for k in ("pieces", "attack", "topouts", "completed")} == {
        k: record_profile[k] for k in ("pieces", "attack", "topouts", "completed")
    }

    placement_times = []
    record_times = []
    for _ in range(3):
        placement_times.append(run("placement", state_dict)["elapsed"])
        record_times.append(run("record", state_dict)["elapsed"])

    output = {
        "placementProfile": placement_profile,
        "recordProfile": record_profile,
        "placementTimes": placement_times,
        "recordTimes": record_times,
        "placementMedian": statistics.median(placement_times),
        "recordMedian": statistics.median(record_times),
        "speedup": statistics.median(placement_times) / statistics.median(record_times),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
