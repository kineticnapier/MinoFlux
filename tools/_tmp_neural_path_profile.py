from __future__ import annotations

from array import array
import statistics
import time

from minoflux_ai import _neural_native
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


def encode_batch(batch):
    states = []
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


def native_pack(states):
    rows, contexts = row_array_pack(states)
    board = _neural_native.expand_row_masks(rows, NCFG.board_width)
    return board, contexts


def main():
    games = [Game(8100001 + i * 97) for i in range(20)]
    model = build_neural_value_model(NCFG)
    evaluator = NeuralValueEvaluator(model, NCFG, device="cpu", precision="float32")

    all_states = []
    candidate_count = 0
    encode_seconds = 0.0
    score_seconds = 0.0
    score_states = 0

    for step in range(40):
        batch = []
        chosen = []
        for game in games:
            if game.game_over:
                chosen.append((game, None))
                continue
            direct, held, hold = _branch_groups(game, CFG, include_paths=False)
            if direct:
                batch.append((game, direct))
            if held is not None and hold:
                batch.append((held, hold))
            candidate_count += len(direct) + len(hold)
            pick = direct[0] if direct else (hold[0] if hold else None)
            chosen.append((game, SearchAction(False if direct else True, pick) if pick is not None else None))

        started = time.perf_counter()
        states = encode_batch(tuple(batch))
        encode_seconds += time.perf_counter() - started
        all_states.extend(states)

        if step < 10:
            score_states += sum(len(p) for _g, p in batch)
            started = time.perf_counter()
            evaluator.score_placement_groups(tuple(batch))
            score_seconds += time.perf_counter() - started

        for game, action in chosen:
            if action is not None and not game.game_over:
                apply_search_action(game, action)

    print("candidates", candidate_count, "states", len(all_states), "encode", encode_seconds)

    old_times = []
    row_times = []
    native_times = []
    for _ in range(9):
        started = time.perf_counter()
        old_board, old_context = old_pack(all_states)
        old_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        row_array_pack(all_states)
        row_times.append(time.perf_counter() - started)
        started = time.perf_counter()
        new_board, new_context = native_pack(all_states)
        native_times.append(time.perf_counter() - started)
        assert bytes(old_board) == new_board
        assert old_context.tobytes() == new_context.tobytes()
    old_median = statistics.median(old_times)
    row_median = statistics.median(row_times)
    native_median = statistics.median(native_times)
    print("old_pack_median", old_median)
    print("row_array_pack_median", row_median)
    print("native_pack_median", native_median)
    print("native_pack_speedup", old_median / native_median)
    print("score_10_batches_states", score_states)
    print("score_10_batches_seconds", score_seconds)


if __name__ == "__main__":
    main()
