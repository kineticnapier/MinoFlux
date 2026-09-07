from __future__ import annotations

import cProfile
import hashlib
import json
import pstats
import struct
import time
from collections import defaultdict
from io import StringIO

import torch

from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator, build_neural_value_model
from minoflux_ai import neural_fast, neural_search_fast
from minoflux_ai.reachability import ReachabilityProfile, clear_reachability_cache, collect_reachability_profile
from minoflux_ai.reachability_native import NativePlacementRecords, clear_native_record_cache
from minoflux_ai.search import SearchConfig, apply_search_action
from minoflux_engine import Game


BATCH_SIZE = 20
WARMUP_STEPS = 20
PROFILE_STEPS = 120
SEED_BASE = 8_100_001
SEED_STEP = 31


def _choice_bytes(choice) -> bytes:
    if choice is None:
        return b"NONE"
    p = choice.action.placement
    return b"|".join(
        (
            str(int(choice.action.use_hold)).encode(),
            p.piece.encode(),
            str(p.x).encode(),
            str(p.y).encode(),
            str(p.rotation).encode(),
            str(int(p.last_move_was_rotation)).encode(),
            str(-1 if p.rotation_kick_index is None else p.rotation_kick_index).encode(),
            str(-1 if p.rotation_from is None else p.rotation_from).encode(),
            str(-1 if p.rotation_to is None else p.rotation_to).encode(),
            struct.pack("<d", float(choice.score)),
        )
    )


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20260907)
    config = NeuralValueConfig().normalized()
    model = build_neural_value_model(config)
    evaluator = NeuralValueEvaluator(model, config, device="cpu", precision="float32")
    search_config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        discount=0.9,
        srs_reachable=True,
        allow_180=False,
        reachability_node_limit=8_000,
    ).normalized()

    generations = [0] * BATCH_SIZE
    games = [Game(SEED_BASE + i * SEED_STEP) for i in range(BATCH_SIZE)]

    def recycle_finished() -> None:
        for i, game in enumerate(games):
            if game.game_over:
                generations[i] += 1
                games[i] = Game(SEED_BASE + i * SEED_STEP + generations[i] * 1_000_003)

    def step(*, digest=None) -> int:
        recycle_finished()
        choices = neural_search_fast.choose_search_actions_batch(
            tuple(games),
            config=search_config,
            scorer=evaluator,
        )
        states = 0
        for i, choice in enumerate(choices):
            if digest is not None:
                digest.update(_choice_bytes(choice))
            if choice is not None:
                apply_search_action(games[i], choice.action)
                states += 1
        return states

    for _ in range(WARMUP_STEPS):
        step()
    clear_reachability_cache()
    clear_native_record_cache()

    timers = defaultdict(float)
    counts = defaultdict(int)

    original_score_buffers = neural_fast._score_packed_buffers
    def profiled_score_buffers(evaluator_, board_bytes, context_bytes, state_count):
        if state_count <= 0:
            return ()
        torch_ = evaluator_._torch

        t = time.perf_counter()
        board_cpu = torch_.frombuffer(board_bytes, dtype=torch_.uint8).reshape(
            state_count, 1, evaluator_.config.board_height, evaluator_.config.board_width
        )
        timers["scorer.frombufferBoard"] += time.perf_counter() - t

        t = time.perf_counter()
        context_cpu = torch_.frombuffer(context_bytes, dtype=torch_.float32).reshape(
            state_count, evaluator_.config.context_size
        )
        timers["scorer.frombufferContext"] += time.perf_counter() - t

        t = time.perf_counter()
        boards = board_cpu.to(device=evaluator_.device, dtype=torch_.float32)
        timers["scorer.boardToDeviceAndDtype"] += time.perf_counter() - t

        t = time.perf_counter()
        contexts = context_cpu.to(device=evaluator_.device)
        timers["scorer.contextToDevice"] += time.perf_counter() - t

        t = time.perf_counter()
        with torch_.inference_mode(), evaluator_._autocast_context():
            values = evaluator_.model(boards, contexts).reshape(-1)
        timers["scorer.forward"] += time.perf_counter() - t

        t = time.perf_counter()
        cpu_values = values.detach().cpu()
        timers["scorer.gpuToCpu"] += time.perf_counter() - t

        t = time.perf_counter()
        listed = cpu_values.tolist()
        timers["scorer.tolist"] += time.perf_counter() - t

        t = time.perf_counter()
        result = tuple(listed)
        timers["scorer.tupleScores"] += time.perf_counter() - t
        counts["scorer.states"] += state_count
        counts["scorer.calls"] += 1
        return result
    neural_fast._score_packed_buffers = profiled_score_buffers

    original_encode = neural_fast._native.encode_packed_record_group
    def profiled_encode(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_encode(*args, **kwargs)
        finally:
            timers["scorer.nativeEncode"] += time.perf_counter() - t
            counts["scorer.nativeEncodeCalls"] += 1
    neural_fast._native.encode_packed_record_group = profiled_encode

    original_prefixes = neural_fast._group_prefixes
    def profiled_prefixes(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_prefixes(*args, **kwargs)
        finally:
            timers["scorer.contextPrefixes"] += time.perf_counter() - t
            counts["scorer.contextPrefixCalls"] += 1
    neural_fast._group_prefixes = profiled_prefixes

    original_rows = neural_fast.board_row_masks
    def profiled_rows(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_rows(*args, **kwargs)
        finally:
            timers["scorer.boardRowMasks"] += time.perf_counter() - t
            counts["scorer.boardRowMaskCalls"] += 1
    neural_fast.board_row_masks = profiled_rows

    original_score_groups = NeuralValueEvaluator.score_native_record_groups
    def profiled_score_groups(self, groups):
        t = time.perf_counter()
        try:
            return original_score_groups(self, groups)
        finally:
            timers["scorer.total"] += time.perf_counter() - t
            counts["scorer.groupCalls"] += 1
            counts["scorer.groups"] += len(groups)
    NeuralValueEvaluator.score_native_record_groups = profiled_score_groups

    original_reach = neural_search_fast.reachable_placement_records_native
    def profiled_reach(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_reach(*args, **kwargs)
        finally:
            timers["search.reachBoundary"] += time.perf_counter() - t
            counts["search.reachBoundaryCalls"] += 1
    neural_search_fast.reachable_placement_records_native = profiled_reach

    original_rank = neural_search_fast._rank_native_record_branches
    def profiled_rank(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_rank(*args, **kwargs)
        finally:
            timers["search.rankWinners"] += time.perf_counter() - t
            counts["search.rankCalls"] += 1
    neural_search_fast._rank_native_record_branches = profiled_rank

    original_held = neural_search_fast._search._held_search_game
    def profiled_held(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_held(*args, **kwargs)
        finally:
            timers["search.heldGame"] += time.perf_counter() - t
            counts["search.heldGameCalls"] += 1
    neural_search_fast._search._held_search_game = profiled_held

    original_rank_precomputed = neural_search_fast._search._rank_precomputed_actions
    def profiled_rank_precomputed(*args, **kwargs):
        t = time.perf_counter()
        try:
            return original_rank_precomputed(*args, **kwargs)
        finally:
            timers["search.rankPrecomputed"] += time.perf_counter() - t
            counts["search.rankPrecomputedCalls"] += 1
    neural_search_fast._search._rank_precomputed_actions = profiled_rank_precomputed

    original_materialize = NativePlacementRecords.materialize
    def profiled_materialize(self, index):
        t = time.perf_counter()
        try:
            return original_materialize(self, index)
        finally:
            timers["search.materializeWinners"] += time.perf_counter() - t
            counts["search.materializeCalls"] += 1
    NativePlacementRecords.materialize = profiled_materialize

    reach_profile = ReachabilityProfile()
    digest = hashlib.sha256()
    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    with collect_reachability_profile(reach_profile):
        for _ in range(PROFILE_STEPS):
            step(digest=digest)
    profiler.disable()
    elapsed = time.perf_counter() - t0

    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(35)

    scorer_known = sum(
        timers[key]
        for key in (
            "scorer.nativeEncode",
            "scorer.contextPrefixes",
            "scorer.boardRowMasks",
            "scorer.frombufferBoard",
            "scorer.frombufferContext",
            "scorer.boardToDeviceAndDtype",
            "scorer.contextToDevice",
            "scorer.forward",
            "scorer.gpuToCpu",
            "scorer.tolist",
            "scorer.tupleScores",
        )
    )
    timers["scorer.unaccounted"] = max(0.0, timers["scorer.total"] - scorer_known)
    search_known = timers["scorer.total"] + timers["search.reachBoundary"] + timers["search.rankWinners"] + timers["search.heldGame"]
    timers["search.approxUnaccounted"] = max(0.0, elapsed - search_known)

    payload = {
        "environment": {
            "device": evaluator.device,
            "torch": torch.__version__,
            "batchSize": BATCH_SIZE,
            "warmupSteps": WARMUP_STEPS,
            "profileSteps": PROFILE_STEPS,
        },
        "elapsedSeconds": elapsed,
        "timers": dict(sorted(timers.items())),
        "counts": dict(sorted(counts.items())),
        "reachability": reach_profile.to_dict(),
        "actionSignature": digest.hexdigest(),
        "cprofile": stream.getvalue(),
    }
    print("PHASE1_PROFILE=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
