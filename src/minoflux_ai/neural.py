from __future__ import annotations

from array import array
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from minoflux_engine import Game, Placement
from minoflux_engine.b2b import resolve_b2b_charging
from minoflux_engine.spin import base_attack, is_difficult_clear, t_spin_event

from .bitboard import (
    ROW_OCCUPANCY_BYTES,
    board_row_masks,
    classify_t_spin_row_masks,
    collides_row_masks,
    hidden_rows_occupied,
    place_and_clear_row_masks,
)
from .heuristic import PlacementEvaluation

NEURAL_VALUE_FORMAT = "minoflux_neural_value_v1"
NEURAL_BOARD_HEIGHT = 24
NEURAL_BOARD_WIDTH = 10
NEURAL_QUEUE_LENGTH = 5
PIECES = ("I", "O", "T", "S", "Z", "J", "L")
_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}
_ZERO_PIECE = (0.0,) * len(PIECES)
_ZERO_HOLD = (0.0,) * (len(PIECES) + 1)
_PIECE_ONE_HOT = {
    piece: tuple(1.0 if index == piece_index else 0.0 for index in range(len(PIECES)))
    for piece, piece_index in _PIECE_INDEX.items()
}
_HOLD_ONE_HOT = {
    piece: tuple(
        1.0 if index == piece_index else 0.0
        for index in range(len(PIECES) + 1)
    )
    for piece, piece_index in _PIECE_INDEX.items()
}
_HOLD_ONE_HOT[None] = tuple(
    1.0 if index == len(PIECES) else 0.0
    for index in range(len(PIECES) + 1)
)


@dataclass(frozen=True, slots=True)
class NeuralValueConfig:
    board_height: int = NEURAL_BOARD_HEIGHT
    board_width: int = NEURAL_BOARD_WIDTH
    queue_length: int = NEURAL_QUEUE_LENGTH
    conv_channels: tuple[int, int] = (16, 32)
    hidden_size: int = 256
    value_hidden_size: int = 128

    def normalized(self) -> "NeuralValueConfig":
        board_height = int(self.board_height)
        board_width = int(self.board_width)
        queue_length = int(self.queue_length)
        if board_height != NEURAL_BOARD_HEIGHT or board_width != NEURAL_BOARD_WIDTH:
            raise ValueError(
                f"Neural evaluator currently expects a {NEURAL_BOARD_HEIGHT}x{NEURAL_BOARD_WIDTH} board"
            )
        if queue_length < 1 or queue_length > 7:
            raise ValueError("queue_length must be between 1 and 7")
        channels = tuple(max(1, int(value)) for value in self.conv_channels)
        if len(channels) != 2:
            raise ValueError("conv_channels must contain exactly two values")
        return NeuralValueConfig(
            board_height=board_height,
            board_width=board_width,
            queue_length=queue_length,
            conv_channels=(channels[0], channels[1]),
            hidden_size=max(8, int(self.hidden_size)),
            value_hidden_size=max(8, int(self.value_hidden_size)),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "NeuralValueConfig":
        channels = values.get("conv_channels", values.get("convChannels", (16, 32)))
        if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
            raise ValueError("conv_channels must be an array")
        return cls(
            board_height=int(values.get("board_height", values.get("boardHeight", NEURAL_BOARD_HEIGHT))),
            board_width=int(values.get("board_width", values.get("boardWidth", NEURAL_BOARD_WIDTH))),
            queue_length=int(values.get("queue_length", values.get("queueLength", NEURAL_QUEUE_LENGTH))),
            conv_channels=tuple(int(value) for value in channels),
            hidden_size=int(values.get("hidden_size", values.get("hiddenSize", 256))),
            value_hidden_size=int(values.get("value_hidden_size", values.get("valueHiddenSize", 128))),
        ).normalized()

    @property
    def context_size(self) -> int:
        return len(PIECES) + (len(PIECES) + 1) + self.queue_length * len(PIECES) + 9


@dataclass(frozen=True, slots=True)
class NeuralState:
    board: tuple[float, ...]
    context: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _CompactNeuralState:
    rows: tuple[int, ...]
    context: tuple[float, ...]


def _one_hot_piece(piece: str | None, *, include_none: bool = False) -> tuple[float, ...]:
    if include_none:
        if piece is None:
            return _HOLD_ONE_HOT[None]
        return _HOLD_ONE_HOT.get(str(piece).upper(), _ZERO_HOLD)
    if piece is None:
        return _ZERO_PIECE
    return _PIECE_ONE_HOT.get(str(piece).upper(), _ZERO_PIECE)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _context_prefix(
    *,
    current: str,
    hold_piece: str | None,
    queue: Sequence[str | None],
    config: NeuralValueConfig,
) -> tuple[float, ...]:
    """Encode the placement-invariant categorical part of neural context."""

    context: list[float] = []
    context.extend(_one_hot_piece(current))
    context.extend(_one_hot_piece(hold_piece, include_none=True))
    queue_length = config.queue_length
    for index in range(queue_length):
        piece = queue[index] if index < len(queue) else None
        context.extend(_one_hot_piece(piece))
    expected = config.context_size - 9
    if len(context) != expected:
        raise AssertionError(f"Neural context prefix size mismatch: {len(context)} != {expected}")
    return tuple(context)


def _context_values(
    *,
    current: str,
    hold_piece: str | None,
    queue: Sequence[str | None],
    combo: int,
    back_to_back: bool,
    b2b_chain: int,
    surge_charge: int,
    game_over: bool,
    last_lines: int,
    last_attack: int,
    last_spin: bool,
    last_perfect_clear: bool,
    config: NeuralValueConfig,
    prefix: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    head = (
        prefix
        if prefix is not None
        else _context_prefix(
            current=current,
            hold_piece=hold_piece,
            queue=queue,
            config=config,
        )
    )
    context = head + (
        _clip01((combo + 1) / 16.0),
        float(bool(back_to_back)),
        _clip01(b2b_chain / 20.0),
        _clip01(surge_charge / 20.0),
        float(bool(game_over)),
        _clip01(last_lines / 4.0),
        _clip01(last_attack / 20.0),
        float(bool(last_spin)),
        float(bool(last_perfect_clear)),
    )
    if len(context) != config.context_size:
        raise AssertionError(f"Neural context size mismatch: {len(context)} != {config.context_size}")
    return context


def _encode_components(
    board_rows: Sequence[Sequence[object | None]],
    *,
    current: str,
    hold_piece: str | None,
    queue: Sequence[str | None],
    combo: int,
    back_to_back: bool,
    b2b_chain: int,
    surge_charge: int,
    game_over: bool,
    last_lines: int,
    last_attack: int,
    last_spin: bool,
    last_perfect_clear: bool,
    config: NeuralValueConfig,
) -> NeuralState:
    cfg = config.normalized()
    if len(board_rows) != cfg.board_height or any(len(row) != cfg.board_width for row in board_rows):
        raise ValueError(
            f"Expected board shape {cfg.board_height}x{cfg.board_width}, "
            f"got {len(board_rows)}x{len(board_rows[0]) if board_rows else 0}"
        )
    board = tuple(0.0 if cell is None else 1.0 for row in board_rows for cell in row)
    return NeuralState(
        board=board,
        context=_context_values(
            current=current,
            hold_piece=hold_piece,
            queue=queue,
            combo=combo,
            back_to_back=back_to_back,
            b2b_chain=b2b_chain,
            surge_charge=surge_charge,
            game_over=game_over,
            last_lines=last_lines,
            last_attack=last_attack,
            last_spin=last_spin,
            last_perfect_clear=last_perfect_clear,
            config=cfg,
        ),
    )


def encode_game_state(
    game: Game,
    config: NeuralValueConfig = NeuralValueConfig(),
) -> NeuralState:
    last = game.last_lock
    return _encode_components(
        game.board,
        current=game.current,
        hold_piece=game.hold_piece,
        queue=tuple(game.queue),
        combo=game.combo,
        back_to_back=game.back_to_back,
        b2b_chain=game.b2b_chain,
        surge_charge=game.surge_charge,
        game_over=game.game_over,
        last_lines=last.lines if last is not None else 0,
        last_attack=last.attack if last is not None else 0,
        last_spin=bool(last is not None and last.spin is not None),
        last_perfect_clear=bool(last is not None and last.perfect_clear),
        config=config,
    )


def _encode_placement_result_compact(
    game: Game,
    placement: Placement,
    config: NeuralValueConfig,
    *,
    source_rows: tuple[int, ...] | None = None,
    source_queue: tuple[str, ...] | None = None,
    normal_context_prefix: tuple[float, ...] | None = None,
    locked_context_prefix: tuple[float, ...] | None = None,
) -> _CompactNeuralState | None:
    if placement.piece != game.current:
        raise ValueError(f"Placement is for {placement.piece}, current piece is {game.current}")
    rows = source_rows if source_rows is not None else board_row_masks(game.board)
    queue = source_queue if source_queue is not None else tuple(game.queue)

    spin_kind = classify_t_spin_row_masks(
        rows,
        piece=placement.piece,
        x=placement.x,
        y=placement.y,
        rotation=placement.rotation,
        last_move_was_rotation=placement.last_move_was_rotation,
        rotation_kick_index=placement.rotation_kick_index,
        width=game.width,
    )
    after_rows, lines, topped_out = place_and_clear_row_masks(
        rows,
        placement,
        width=game.width,
    )
    spin = t_spin_event(spin_kind, lines)
    perfect_clear = not any(after_rows)
    difficult = is_difficult_clear(lines, spin)
    b2b = resolve_b2b_charging(
        active=game.back_to_back,
        chain=game.b2b_chain,
        difficult=difficult,
        lines=lines,
        perfect_clear=perfect_clear and lines > 0,
    )
    combo = game.combo + 1 if lines else -1
    sent = base_attack(lines, spin) + b2b.attack_bonus
    if lines and combo > 0:
        sent += min(4, combo // 2 + 1)
    if perfect_clear and lines:
        sent += 10
    total_sent = sent + b2b.released

    locked_out = topped_out or hidden_rows_occupied(after_rows, game.hidden_rows)
    if locked_out:
        current = game.current
        next_queue: Sequence[str | None] = queue
        game_over = True
        context_prefix = locked_context_prefix
    else:
        if len(queue) < config.queue_length + 1:
            return None
        current = queue[0]
        next_queue = queue[1:]
        game_over = collides_row_masks(
            after_rows,
            current,
            3,
            1,
            0,
            width=game.width,
        )
        context_prefix = normal_context_prefix

    context = _context_values(
        current=current,
        hold_piece=game.hold_piece,
        queue=next_queue,
        combo=combo,
        back_to_back=b2b.active,
        b2b_chain=b2b.chain,
        surge_charge=b2b.charge,
        game_over=game_over,
        last_lines=lines,
        last_attack=total_sent,
        last_spin=spin is not None,
        last_perfect_clear=perfect_clear,
        config=config,
        prefix=context_prefix,
    )
    return _CompactNeuralState(rows=after_rows, context=context)


def encode_placement_result(
    game: Game,
    placement: Placement,
    config: NeuralValueConfig = NeuralValueConfig(),
) -> NeuralState | None:
    """Encode a post-placement state without cloning or mutating ``game``."""

    cfg = config.normalized()
    compact = _encode_placement_result_compact(game, placement, cfg)
    if compact is None:
        return None
    board = tuple(
        float(bool(mask & (1 << x)))
        for mask in compact.rows
        for x in range(cfg.board_width)
    )
    return NeuralState(board=board, context=compact.context)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for neural evaluation. Install the optional ML dependencies with "
            "`uv sync --extra ml`."
        ) from error
    return torch, nn


def build_neural_value_model(config: NeuralValueConfig = NeuralValueConfig()) -> Any:
    cfg = config.normalized()
    torch, nn = _require_torch()
    first_channels, second_channels = cfg.conv_channels

    class NeuralValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(1, first_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(first_channels, second_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            flattened = second_channels * (cfg.board_height // 2) * (cfg.board_width // 2)
            self.value_head = nn.Sequential(
                nn.Linear(flattened + cfg.context_size, cfg.hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.hidden_size, cfg.value_hidden_size),
                nn.ReLU(),
                nn.Linear(cfg.value_hidden_size, 1),
            )

        def forward(self, board: Any, context: Any) -> Any:
            encoded = self.board_encoder(board).flatten(start_dim=1)
            return self.value_head(torch.cat((encoded, context), dim=1)).squeeze(1)

    return NeuralValueModel()


def _resolve_device(torch: Any, device: str | None) -> str:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for neural evaluation, but no CUDA device is available")
    return requested


def _resolve_precision(device: str, precision: str | None) -> str:
    requested = "float32" if precision is None else str(precision).strip().lower()
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    requested = aliases.get(requested, requested)
    if requested == "auto":
        return "float16" if device.startswith("cuda") else "float32"
    if requested not in {"float32", "float16", "bfloat16"}:
        raise ValueError("precision must be float32, float16, bfloat16, or auto")
    if requested == "float16" and not device.startswith("cuda"):
        raise ValueError("float16 neural inference currently requires CUDA")
    return requested


class NeuralValueEvaluator:
    """Batched leaf evaluator that can replace heuristic placement scores in search."""

    def __init__(
        self,
        model: Any,
        config: NeuralValueConfig = NeuralValueConfig(),
        *,
        device: str | None = "auto",
        precision: str | None = "float32",
        compile_model: bool = False,
    ) -> None:
        torch, _ = _require_torch()
        self._torch = torch
        self.config = config.normalized()
        self.device = _resolve_device(torch, device)
        self.precision = _resolve_precision(self.device, precision)
        self.model = model.to(self.device)
        self.model.eval()
        self.compiled = False
        if compile_model:
            compile_fn = getattr(torch, "compile", None)
            if not callable(compile_fn):
                raise RuntimeError("This PyTorch build does not provide torch.compile")
            self.model = compile_fn(self.model, mode="reduce-overhead")
            self.compiled = True

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = "auto",
        precision: str | None = "float32",
        compile_model: bool = False,
    ) -> "NeuralValueEvaluator":
        torch, _ = _require_torch()
        target_device = _resolve_device(torch, device)
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != NEURAL_VALUE_FORMAT:
            found = payload.get("format") if isinstance(payload, dict) else None
            raise ValueError(f"Unsupported neural value model format: {found!r}")
        raw_config = payload.get("config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("Neural value checkpoint has no config object")
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Neural value checkpoint has no state_dict")
        config = NeuralValueConfig.from_mapping(raw_config)
        model = build_neural_value_model(config)
        model.load_state_dict(state_dict)
        return cls(
            model,
            config,
            device=target_device,
            precision=precision,
            compile_model=compile_model,
        )

    def _autocast_context(self):
        if self.precision == "float32":
            return nullcontext()
        dtype = (
            self._torch.float16
            if self.precision == "float16"
            else self._torch.bfloat16
        )
        return self._torch.autocast(
            device_type=self.device.split(":", 1)[0],
            dtype=dtype,
        )

    def _score_states(self, states: Sequence[NeuralState]) -> tuple[float, ...]:
        if not states:
            return ()
        torch = self._torch
        board_values = array("f")
        context_values = array("f")
        extend_board = board_values.extend
        extend_context = context_values.extend
        for state in states:
            extend_board(state.board)
            extend_context(state.context)
        boards = torch.frombuffer(board_values, dtype=torch.float32).reshape(
            len(states), 1, self.config.board_height, self.config.board_width
        ).to(self.device)
        contexts = torch.frombuffer(context_values, dtype=torch.float32).reshape(
            len(states), self.config.context_size
        ).to(self.device)
        with torch.inference_mode(), self._autocast_context():
            values = self.model(boards, contexts).reshape(-1)
        return tuple(values.detach().cpu().tolist())

    def _score_compact_states(self, states: Sequence[_CompactNeuralState]) -> tuple[float, ...]:
        if not states:
            return ()
        torch = self._torch
        occupancy_bytes = ROW_OCCUPANCY_BYTES
        board_bytes = bytearray().join(
            occupancy_bytes[mask]
            for state in states
            for mask in state.rows
        )
        context_values = array("f")
        extend_context = context_values.extend
        for state in states:
            extend_context(state.context)

        board_cpu = torch.frombuffer(board_bytes, dtype=torch.uint8).reshape(
            len(states), 1, self.config.board_height, self.config.board_width
        )
        context_cpu = torch.frombuffer(context_values, dtype=torch.float32).reshape(
            len(states), self.config.context_size
        )
        boards = board_cpu.to(device=self.device, dtype=torch.float32)
        contexts = context_cpu.to(device=self.device)
        with torch.inference_mode(), self._autocast_context():
            values = self.model(boards, contexts).reshape(-1)
        return tuple(values.detach().cpu().tolist())

    def score_placement_groups(
        self,
        groups: Sequence[tuple[Game, Sequence[Placement]]],
    ) -> tuple[tuple[float, ...], ...]:
        """Score several game/placement groups in one compact GPU forward pass."""

        states: list[_CompactNeuralState] = []
        sizes: list[int] = []
        config = self.config
        queue_length = config.queue_length
        for game, placements in groups:
            sizes.append(len(placements))
            source_rows = board_row_masks(game.board)
            source_queue = tuple(game.queue)
            locked_context_prefix = _context_prefix(
                current=game.current,
                hold_piece=game.hold_piece,
                queue=source_queue,
                config=config,
            )
            normal_context_prefix = (
                _context_prefix(
                    current=source_queue[0],
                    hold_piece=game.hold_piece,
                    queue=source_queue[1:],
                    config=config,
                )
                if len(source_queue) >= queue_length + 1
                else None
            )
            for placement in placements:
                encoded = _encode_placement_result_compact(
                    game,
                    placement,
                    config,
                    source_rows=source_rows,
                    source_queue=source_queue,
                    normal_context_prefix=normal_context_prefix,
                    locked_context_prefix=locked_context_prefix,
                )
                if encoded is None:
                    from .search import clone_game

                    child = clone_game(game)
                    child.place(placement)
                    state = encode_game_state(child, config)
                    encoded = _CompactNeuralState(
                        rows=board_row_masks(child.board),
                        context=state.context,
                    )
                states.append(encoded)

        values = self._score_compact_states(states)
        output: list[tuple[float, ...]] = []
        offset = 0
        for size in sizes:
            output.append(values[offset : offset + size])
            offset += size
        return tuple(output)

    def score_placements(
        self,
        game: Game,
        placements: Sequence[Placement],
    ) -> tuple[float, ...]:
        return self.score_placement_groups(((game, placements),))[0]

    def score_many(
        self,
        game: Game,
        evaluations: Sequence[PlacementEvaluation],
    ) -> tuple[float, ...]:
        return self.score_placements(
            game,
            tuple(evaluation.placement for evaluation in evaluations),
        )


def save_neural_value_checkpoint(
    path: str | Path,
    model: Any,
    config: NeuralValueConfig = NeuralValueConfig(),
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    torch, _ = _require_torch()
    cfg = config.normalized()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "format": NEURAL_VALUE_FORMAT,
            "config": asdict(cfg),
            "state_dict": state_dict,
            "metadata": dict(metadata or {}),
        },
        target,
    )
    return target
