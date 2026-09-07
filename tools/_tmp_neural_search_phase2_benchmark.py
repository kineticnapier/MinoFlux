from __future__ import annotations

import hashlib
import json
import statistics
import struct
import time
from heapq import nlargest

import torch

from minoflux_ai import neural_search_fast as fast
from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator, build_neural_value_model
from minoflux_ai.reachability import clear_reachability_cache
from minoflux_ai.reachability_native import NativePlacementRecords, clear_native_record_cache, reachable_placement_records_native
from minoflux_ai.search import SearchConfig, apply_search_action
from minoflux_engine import Game


BATCH = 20
STEPS = 80
SEED_BASE = 8_100_001
SEED_STEP = 31
CFG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
).normalized()


def _old_rank(branches, limit):
    total = 0
    score_values = []
    for _use_hold, _game, batch, values in branches:
        if len(values) != len(batch):
            raise ValueError
        total += len(batch)
        score_values.extend(float(value) for value in values)
    if total == 0:
        return ()
    count = max(0, int(limit))
    if count == 0:
        return ()
    cutoff = nlargest(count, score_values)[-1] if count < len(score_values) else float("-inf")
    materialized = []
    for use_hold, branch_game, batch, values in branches:
        placements = []
        kept_values = []
        for index, value in enumerate(values):
            score = float(value)
            if score >= cutoff:
                placements.append(batch.materialize(index))
                kept_values.append(score)
        if placements:
            materialized.append((use_hold, branch_game, tuple(placements), tuple(kept_values)))
    return fast._search._rank_precomputed_actions(tuple(materialized), count)


def choose_old(games, *, scorer):
    cfg = CFG
    score_records = fast._native_record_scorer(scorer)
    scorer_config = fast._scorer_config(scorer)
    assert score_records is not None and scorer_config is not None
    queue_length = scorer_config.queue_length
    prepared = []
    groups = []
    group_keys = []
    for index, game in enumerate(games):
        if game.game_over:
            prepared.append((NativePlacementRecords.empty(game.current), None, None))
            continue
        held = fast._search._held_search_game(game) if cfg.allow_hold else None
        direct = reachable_placement_records_native(
            game, allow_180=cfg.allow_180, max_nodes=cfg.reachability_node_limit
        )
        assert direct is not None
        held_records = (
            reachable_placement_records_native(
                held,
                allow_180=cfg.allow_180,
                max_nodes=cfg.reachability_node_limit,
                rows=direct.rows,
            )
            if held is not None else None
        )
        prepared.append((direct, held, held_records))
        if direct:
            groups.append((game, direct)); group_keys.append((index, False))
        if held is not None and held_records:
            groups.append((held, held_records)); group_keys.append((index, True))
    grouped_values = score_records(tuple(groups)) if groups else ()
    values_by_key = {key: values for key, values in zip(group_keys, grouped_values)}
    choices = []
    for index, game in enumerate(games):
        direct, held, held_records = prepared[index]
        branches = []
        if direct:
            branches.append((False, game, direct, values_by_key[(index, False)]))
        if held is not None and held_records:
            branches.append((True, held, held_records, values_by_key[(index, True)]))
        ranked = _old_rank(tuple(branches), 1)
        if not ranked:
            choices.append(None); continue
        action, evaluation = ranked[0]
        choices.append(fast._search.SearchChoice(action, evaluation.score, evaluation, (action,)))
    return tuple(choices)


def _choice_signature(choice):
    if choice is None:
        return None
    p = choice.action.placement
    f = choice.immediate.features
    return (
        choice.action.use_hold, p.piece, p.x, p.y, p.rotation, p.cells, p.path,
        p.last_move_was_rotation, p.rotation_kick_index, p.rotation_from, p.rotation_to,
        choice.score, choice.immediate.score,
        f.board.aggregate_height, f.board.max_height, f.board.holes, f.board.hole_depth,
        f.board.bumpiness, f.board.wells, f.board.t_spin_slots, f.board.occupied_cells,
        f.new_holes, f.lines, f.attack, f.spin_lines, f.perfect_clear, f.game_over,
        f.spin, f.t_spin_slot_delta,
    )


def _digest_update(h, sig):
    h.update(repr(sig).encode("utf-8")); h.update(b"\n")


def run(chooser, evaluator):
    games = [Game(SEED_BASE + i * SEED_STEP) for i in range(BATCH)]
    generations = [0] * BATCH
    h = hashlib.sha256()
    pieces = attack = 0
    search_seconds = 0.0
    for _ in range(STEPS):
        for i, game in enumerate(games):
            if game.game_over:
                generations[i] += 1
                games[i] = Game(SEED_BASE + i * SEED_STEP + generations[i] * 1_000_003)
        t = time.perf_counter()
        choices = chooser(tuple(games), scorer=evaluator)
        search_seconds += time.perf_counter() - t
        for i, choice in enumerate(choices):
            sig = _choice_signature(choice)
            _digest_update(h, sig)
            if choice is None:
                continue
            before_pieces = games[i].pieces_placed
            before_attack = games[i].attack
            apply_search_action(games[i], choice.action)
            pieces += games[i].pieces_placed - before_pieces
            attack += games[i].attack - before_attack
    return {
        "signature": h.hexdigest(),
        "pieces": pieces,
        "attack": attack,
        "searchSeconds": search_seconds,
        "searchPps": pieces / search_seconds,
    }


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    cfg = NeuralValueConfig().normalized()
    model = build_neural_value_model(cfg)
    evaluator = NeuralValueEvaluator(model, cfg, device="cpu", precision="float32")

    # Warm torch kernels and native tables outside measured runs.
    warm_games = tuple(Game(7_000_001 + i * 31) for i in range(BATCH))
    for _ in range(8):
        choices = fast.choose_search_actions_batch(warm_games, config=CFG, scorer=evaluator)
        for game, choice in zip(warm_games, choices):
            if choice is not None and not game.game_over:
                apply_search_action(game, choice.action)

    def optimized(games, *, scorer):
        return fast.choose_search_actions_batch(games, config=CFG, scorer=scorer)

    results = {"old": [], "optimized": []}
    order = ("old", "optimized", "optimized", "old", "old", "optimized")
    for name in order:
        clear_reachability_cache(); clear_native_record_cache()
        result = run(choose_old if name == "old" else optimized, evaluator)
        results[name].append(result)

    old_sig = {item["signature"] for item in results["old"]}
    new_sig = {item["signature"] for item in results["optimized"]}
    assert len(old_sig) == 1 and old_sig == new_sig, (old_sig, new_sig)
    for old, new in zip(results["old"], results["optimized"]):
        assert old["pieces"] == new["pieces"]
        assert old["attack"] == new["attack"]

    summary = {
        "results": results,
        "medianOldPps": statistics.median(x["searchPps"] for x in results["old"]),
        "medianOptimizedPps": statistics.median(x["searchPps"] for x in results["optimized"]),
        "signature": next(iter(old_sig)),
    }
    summary["speedup"] = summary["medianOptimizedPps"] / summary["medianOldPps"]
    print("PHASE2_BENCH=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
