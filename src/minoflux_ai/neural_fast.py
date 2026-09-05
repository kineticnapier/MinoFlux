from __future__ import annotations

import os
from typing import TYPE_CHECKING, Sequence

from minoflux_engine import Game, Placement
from minoflux_engine.pieces import SHAPES

from .bitboard import board_row_masks
from .neural import NeuralValueEvaluator, _context_prefix

if TYPE_CHECKING:
    from typing import Any

try:
    from . import _neural_native as _native
except ImportError:
    _native = None

_NATIVE_DISABLED_ENV = "MINOFLUX_DISABLE_NATIVE_NEURAL"
_ORIGINAL_SCORE_PLACEMENT_GROUPS = NeuralValueEvaluator.score_placement_groups
_NATIVE_READY = False


def native_neural_available() -> bool:
    return _native is not None


def _ensure_native_ready() -> bool:
    global _NATIVE_READY
    if _native is None:
        return False
    if not _NATIVE_READY:
        _native.register_shapes(SHAPES)
        _NATIVE_READY = True
    return True


def _score_packed_buffers(
    evaluator: NeuralValueEvaluator,
    board_bytes: bytearray,
    context_bytes: bytearray,
    state_count: int,
) -> tuple[float, ...]:
    if state_count <= 0:
        return ()
    torch = evaluator._torch
    board_cpu = torch.frombuffer(board_bytes, dtype=torch.uint8).reshape(
        state_count,
        1,
        evaluator.config.board_height,
        evaluator.config.board_width,
    )
    context_cpu = torch.frombuffer(context_bytes, dtype=torch.float32).reshape(
        state_count,
        evaluator.config.context_size,
    )
    boards = board_cpu.to(device=evaluator.device, dtype=torch.float32)
    contexts = context_cpu.to(device=evaluator.device)
    with torch.inference_mode(), evaluator._autocast_context():
        values = evaluator.model(boards, contexts).reshape(-1)
    return tuple(values.detach().cpu().tolist())


def _native_score_placement_groups(
    self: NeuralValueEvaluator,
    groups: Sequence[tuple[Game, Sequence[Placement]]],
) -> tuple[tuple[float, ...], ...]:
    if os.environ.get(_NATIVE_DISABLED_ENV) or not _ensure_native_ready():
        return _ORIGINAL_SCORE_PLACEMENT_GROUPS(self, groups)

    config = self.config
    if config.board_width > 16 or config.board_height > 64:
        return _ORIGINAL_SCORE_PLACEMENT_GROUPS(self, groups)

    sizes: list[int] = []
    board_chunks: list[bytes] = []
    context_chunks: list[bytes] = []
    state_count = 0
    queue_length = config.queue_length

    for game, placements in groups:
        size = len(placements)
        sizes.append(size)
        if size == 0:
            continue

        source_queue = tuple(game.queue)
        if len(source_queue) < queue_length + 1:
            return _ORIGINAL_SCORE_PLACEMENT_GROUPS(self, groups)

        source_rows = board_row_masks(game.board)
        locked_context_prefix = _context_prefix(
            current=game.current,
            hold_piece=game.hold_piece,
            queue=source_queue,
            config=config,
        )
        normal_context_prefix = _context_prefix(
            current=source_queue[0],
            hold_piece=game.hold_piece,
            queue=source_queue[1:],
            config=config,
        )
        board_chunk, context_chunk = _native.encode_placement_group(
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
            normal_context_prefix,
            locked_context_prefix,
        )
        board_chunks.append(board_chunk)
        context_chunks.append(context_chunk)
        state_count += size

    board_bytes = bytearray().join(board_chunks)
    context_bytes = bytearray().join(context_chunks)
    values = _score_packed_buffers(self, board_bytes, context_bytes, state_count)

    output: list[tuple[float, ...]] = []
    offset = 0
    for size in sizes:
        output.append(values[offset : offset + size])
        offset += size
    return tuple(output)


def install_neural_fast_path() -> None:
    NeuralValueEvaluator.score_placement_groups = _native_score_placement_groups
