from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from minoflux_engine import Game, Placement
from minoflux_engine.b2b import resolve_b2b_charging
from minoflux_engine.pieces import SHAPES
from minoflux_engine.spin import base_attack, classify_t_spin, is_difficult_clear, t_spin_event

from .heuristic import PlacementEvaluation

NEURAL_VALUE_FORMAT = "minoflux_neural_value_v1"
NEURAL_BOARD_HEIGHT = 24
NEURAL_BOARD_WIDTH = 10
NEURAL_QUEUE_LENGTH = 5
PIECES = ("I", "O", "T", "S", "Z", "J", "L")
_PIECE_INDEX = {piece: index for index, piece in enumerate(PIECES)}


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
        # current piece + hold/none + queue + scalar game context
        return len(PIECES) + (len(PIECES) + 1) + self.queue_length * len(PIECES) + 9


@dataclass(frozen=True, slots=True)
class NeuralState:
    board: tuple[float, ...]
    context: tuple[float, ...]


def _one_hot_piece(piece: str | None, *, include_none: bool = False) -> tuple[float, ...]:
    size = len(PIECES) + int(include_none)
    values = [0.0] * size
    if piece is None:
        if include_none:
            values[-1] = 1.0
        return tuple(values)
    index = _PIECE_INDEX.get(str(piece).upper())
    if index is not None:
        values[index] = 1.0
    elif include_none:
        values[-1] = 1.0
    return tuple(values)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


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

    board = tuple(
        0.0 if cell is None else 1.0
        for row in board_rows
        for cell in row
    )
    next_queue = list(queue)[: cfg.queue_length]
    while len(next_queue) < cfg.queue_length:
        next_queue.append(None)

    context: list[float] = []
    context.extend(_one_hot_piece(current))
    context.extend(_one_hot_piece(hold_piece, include_none=True))
    for piece in next_queue:
        context.extend(_one_hot_piece(piece))
    context.extend(
        (
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
    )
    if len(context) != cfg.context_size:
        raise AssertionError(f"Neural context size mismatch: {len(context)} != {cfg.context_size}")
    return NeuralState(board=board, context=tuple(context))


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


def _board_collides(
    board: Sequence[Sequence[object | None]],
    piece: str,
    x: int,
    y: int,
    rotation: int,
) -> bool:
    height = len(board)
    width = len(board[0]) if board else 0
    for dx, dy in SHAPES[piece][rotation % 4]:
        cell_x, cell_y = x + dx, y + dy
        if cell_x < 0 or cell_x >= width or cell_y >= height:
            return True
        if cell_y >= 0 and board[cell_y][cell_x] is not None:
            return True
    return False


def encode_placement_result(
    game: Game,
    placement: Placement,
    config: NeuralValueConfig = NeuralValueConfig(),
) -> NeuralState | None:
    """Encode a post-placement state without cloning or mutating ``game``.

    Returns ``None`` only when the configured queue preview would require drawing
    fresh bag pieces. The normal five-piece neural preview is always available
    from a regular engine state (including the lightweight Hold search state).
    """

    cfg = config.normalized()
    if placement.piece != game.current:
        raise ValueError(f"Placement is for {placement.piece}, current piece is {game.current}")

    spin_kind = classify_t_spin(
        game.board,
        piece=placement.piece,
        x=placement.x,
        y=placement.y,
        rotation=placement.rotation,
        last_move_was_rotation=placement.last_move_was_rotation,
        rotation_kick_index=placement.rotation_kick_index,
    )

    board: list[Sequence[object | None]] = list(game.board)
    copied_rows: set[int] = set()
    topped_out = False
    for cell_x, cell_y in placement.cells:
        if cell_y < 0:
            topped_out = True
            continue
        if cell_y >= game.height:
            topped_out = True
            continue
        if cell_y not in copied_rows:
            board[cell_y] = list(game.board[cell_y])
            copied_rows.add(cell_y)
        row = board[cell_y]
        assert isinstance(row, list)
        row[cell_x] = placement.piece

    full_rows = [index for index, row in enumerate(board) if all(cell is not None for cell in row)]
    lines = len(full_rows)
    if full_rows:
        full_set = set(full_rows)
        board = [[None] * game.width for _ in full_rows] + [
            row for index, row in enumerate(board) if index not in full_set
        ]

    spin = t_spin_event(spin_kind, lines)
    perfect_clear = all(cell is None for row in board for cell in row)
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

    hidden_occupied = any(
        cell is not None
        for row in board[: game.hidden_rows]
        for cell in row
    )
    locked_out = topped_out or hidden_occupied
    source_queue = tuple(game.queue)

    if locked_out:
        current = game.current
        next_queue: Sequence[str | None] = source_queue
        game_over = True
    else:
        if len(source_queue) < cfg.queue_length + 1:
            return None
        current = source_queue[0]
        next_queue = source_queue[1:]
        game_over = _board_collides(board, current, 3, 1, 0)

    return _encode_components(
        board,
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
        config=cfg,
    )


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


class NeuralValueEvaluator:
    """Batched leaf evaluator that can replace heuristic placement scores in search."""

    def __init__(
        self,
        model: Any,
        config: NeuralValueConfig = NeuralValueConfig(),
        *,
        device: str | None = "auto",
    ) -> None:
        torch, _ = _require_torch()
        self._torch = torch
        self.config = config.normalized()
        self.device = _resolve_device(torch, device)
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | None = "auto",
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
        return cls(model, config, device=target_device)

    def _score_states(self, states: Sequence[NeuralState]) -> tuple[float, ...]:
        if not states:
            return ()
        torch = self._torch
        flat_boards = [value for state in states for value in state.board]
        flat_contexts = [value for state in states for value in state.context]
        boards = torch.tensor(
            flat_boards,
            dtype=torch.float32,
            device=self.device,
        ).reshape(
            len(states),
            1,
            self.config.board_height,
            self.config.board_width,
        )
        contexts = torch.tensor(
            flat_contexts,
            dtype=torch.float32,
            device=self.device,
        ).reshape(len(states), self.config.context_size)
        with torch.inference_mode():
            values = self.model(boards, contexts).reshape(-1)
        return tuple(float(value) for value in values.detach().cpu().tolist())

    def score_placement_groups(
        self,
        groups: Sequence[tuple[Game, Sequence[Placement]]],
    ) -> tuple[tuple[float, ...], ...]:
        """Score several game/placement groups in one GPU forward pass."""

        states: list[NeuralState] = []
        sizes: list[int] = []
        for game, placements in groups:
            sizes.append(len(placements))
            for placement in placements:
                encoded = encode_placement_result(game, placement, self.config)
                if encoded is None:
                    # Rare fallback for non-default queue preview lengths that need
                    # a fresh bag draw. Normal five-piece inference never hits this.
                    from .search import clone_game

                    child = clone_game(game)
                    child.place(placement)
                    encoded = encode_game_state(child, self.config)
                states.append(encoded)

        values = self._score_states(states)
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
