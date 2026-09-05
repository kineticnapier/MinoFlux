from __future__ import annotations

from array import array
import statistics
import time

from minoflux_ai.bitboard import ROW_OCCUPANCY_BYTES, board_row_masks
from minoflux_ai.neural import (
    NeuralValueConfig,
    NeuralValueEvaluator,
    _context_prefix,
    _encode_placement_result_compact,
    build_neural_value_model,
)
from minoflux_ai.search import SearchAction, SearchConfig, _branch_groups, apply_search_action
from minoflux_engine import Game

CFG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
).normalized()
NCFG = NeuralValueConfig().normalized()


def collect_groups():
    games = [Game(8100001 + i * 97) for i in range(20)]
    groups = []
    total = 0
    for _step in range(40):
        batch = []
        for game in games:
            if game.game_over:
                continue
            direct, held, hold = _branch_groups(game, CFG, include_paths=False)
            if direct:
                batch.append((game, direct))
            if held is not None and hold:
                batch.append((held, hold))
            chosen = direct[0] if direct else (hold[0] if hold else None)
            if chosen is not None:
                apply_search_action(game, SearchAction(False if direct else True, chosen))
            total += len(direct) + len(hold)
        groups.append(tuple(batch))
    return groups, total


def encode(groups):
    states = []
    for batch in groups:
        for game, placements in batch:
            source_rows = board_row_masks(game.board)
            source_queue = tuple(game.queue)
            locked = _context_prefix(
                current=game.current,
                hold_piece=game.hold_piece,
                queue=source_queue,
                config=NCFG,
            )
            normal = (
                _context_prefix(
                    current=source_queue[0],
                    hold_piece=game.hold_piece,
                    queue=source_queue[1:],
                    config=NCFG,
                )
                if len(source_queue) >= NCFG.queue_length + 1
                else None
            )
            for placement in placements:
                state = _encode_placement_result_compact(
                    game,
                    placement,
                    NCFG,
                    source_rows=source_rows,
                    source_queue=source_queue,
                    normal_context_prefix=normal,
                    locked_context_prefix=locked,
                )
                if state is not None:
                    states.append(state)
    return states


def old_pack(states):
    board = bytearray().join(
        ROW_OCCUPANCY_BYTES[mask]
        for state in states
        for mask in state.rows
    )
    contexts = array("f")
    extend = contexts.extend
    for state in states:
        extend(state.context)
    return board, contexts


def row_array_pack(states):
    rows = array("H")
    contexts = array("f")
    extend_rows = rows.extend
    extend_context = contexts.extend
    for state in states:
        extend_rows(state.rows)
        extend_context(state.context)
    return rows, contexts


def main():
    groups, candidates = collect_groups()
    started = time.perf_counter()
    states = encode(groups)
    encode_seconds = time.perf_counter() - started
    print("candidates", candidates, "states", len(states), "encode", encode_seconds)

    old_times = []
    row_times = []
    for _ in range(7):
        started = time.perf_counter()
        old_pack(states)
        old_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        row_array_pack(states)
        row_times.append(time.perf_counter() - started)
    print("old_pack_median", statistics.median(old_times))
    print("row_array_pack_median", statistics.median(row_times))
    print(
        "python_pack_speedup_if_native_expand_free",
        statistics.median(old_times) / statistics.median(row_times),
    )

    model = build_neural_value_model(NCFG)
    evaluator = NeuralValueEvaluator(model, NCFG, device="cpu", precision="float32")
    scorer_times = []
    scorer_states = 0
    for batch in groups[:10]:
        scorer_states += sum(len(p) for _g, p in batch)
        started = time.perf_counter()
        evaluator.score_placement_groups(batch)
        scorer_times.append(time.perf_counter() - started)
    print("score_10_batches_states", scorer_states)
    print("score_10_batches_seconds", sum(scorer_times))


if __name__ == "__main__":
    main()
