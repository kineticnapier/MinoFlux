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
from minoflux_engine.pieces import SHAPES

CFG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
).normalized()
NCFG = NeuralValueConfig().normalized()
_neural_native.register_shapes(SHAPES)


def prefixes(game):
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
    return source_queue, normal, locked


def encode_batch(batch):
    states = []
    for game, placements in batch:
        source_rows = board_row_masks(game.board)
        source_queue, normal, locked = prefixes(game)
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
    return bytes(board), contexts.tobytes()


def native_encode_batch(batch):
    board_chunks = []
    context_chunks = []
    for game, placements in batch:
        source_rows = board_row_masks(game.board)
        source_queue, normal, locked = prefixes(game)
        if normal is None:
            raise RuntimeError("temporary benchmark expects full queue")
        board, context = _neural_native.encode_placement_group(
            source_rows,
            placements,
            game.current,
            source_queue[0],
            game.width,
            game.height,
            game.hidden_rows,
            game.combo,
            game.back_to_back,
            game.b2b_chain,
            normal,
            locked,
        )
        board_chunks.append(board)
        context_chunks.append(context)
    return b"".join(board_chunks), b"".join(context_chunks)


def main():
    games = [Game(8100001 + i * 97) for i in range(20)]
    model = build_neural_value_model(NCFG)
    evaluator = NeuralValueEvaluator(model, NCFG, device="cpu", precision="float32")

    candidate_count = 0
    python_encode_pack_seconds = 0.0
    native_encode_seconds = 0.0
    score_seconds = 0.0
    score_states = 0
    compared_states = 0

    for step in range(80):
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
        python_board, python_context = old_pack(states)
        python_encode_pack_seconds += time.perf_counter() - started

        started = time.perf_counter()
        native_board, native_context = native_encode_batch(tuple(batch))
        native_encode_seconds += time.perf_counter() - started

        assert python_board == native_board
        assert python_context == native_context
        compared_states += len(states)

        if step < 10:
            score_states += sum(len(p) for _g, p in batch)
            started = time.perf_counter()
            evaluator.score_placement_groups(tuple(batch))
            score_seconds += time.perf_counter() - started

        for game, action in chosen:
            if action is not None and not game.game_over:
                apply_search_action(game, action)

    print("candidates", candidate_count)
    print("compared_states", compared_states)
    print("python_encode_pack_seconds", python_encode_pack_seconds)
    print("native_encode_seconds", native_encode_seconds)
    print("native_encode_speedup", python_encode_pack_seconds / native_encode_seconds)
    print("score_10_batches_states", score_states)
    print("score_10_batches_seconds", score_seconds)


if __name__ == "__main__":
    main()
