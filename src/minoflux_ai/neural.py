from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from minoflux_engine import Game

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


def encode_game_state(
    game: Game,
    config: NeuralValueConfig = NeuralValueConfig(),
) -> NeuralState:
    cfg = config.normalized()
    if len(game.board) != cfg.board_height or any(len(row) != cfg.board_width for row in game.board):
        raise ValueError(
            f"Expected board shape {cfg.board_height}x{cfg.board_width}, "
            f"got {len(game.board)}x{len(game.board[0]) if game.board else 0}"
        )

    board = tuple(
        0.0 if cell is None else 1.0
        for row in game.board
        for cell in row
    )

    queue = list(game.queue)[: cfg.queue_length]
    while len(queue) < cfg.queue_length:
        queue.append(None)

    context: list[float] = []
    context.extend(_one_hot_piece(game.current))
    context.extend(_one_hot_piece(game.hold_piece, include_none=True))
    for piece in queue:
        context.extend(_one_hot_piece(piece))

    last = game.last_lock
    context.extend(
        (
            _clip01((game.combo + 1) / 16.0),
            float(bool(game.back_to_back)),
            _clip01(game.b2b_chain / 20.0),
            _clip01(game.surge_charge / 20.0),
            float(bool(game.game_over)),
            _clip01((last.lines if last is not None else 0) / 4.0),
            _clip01((last.attack if last is not None else 0) / 20.0),
            float(bool(last is not None and last.spin is not None)),
            float(bool(last is not None and last.perfect_clear)),
        )
    )
    if len(context) != cfg.context_size:
        raise AssertionError(f"Neural context size mismatch: {len(context)} != {cfg.context_size}")
    return NeuralState(board=board, context=tuple(context))


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

    def score_many(
        self,
        game: Game,
        evaluations: Sequence[PlacementEvaluation],
    ) -> tuple[float, ...]:
        if not evaluations:
            return ()
        # Import here so simply importing minoflux_ai does not make torch a core dependency.
        torch, _ = _require_torch()
        from .search import clone_game

        states: list[NeuralState] = []
        for evaluation in evaluations:
            child = clone_game(game)
            child.place(evaluation.placement)
            states.append(encode_game_state(child, self.config))

        boards = torch.tensor(
            [state.board for state in states],
            dtype=torch.float32,
            device=self.device,
        ).reshape(
            len(states),
            1,
            self.config.board_height,
            self.config.board_width,
        )
        contexts = torch.tensor(
            [state.context for state in states],
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            values = self.model(boards, contexts).reshape(-1)
        return tuple(float(value) for value in values.detach().cpu().tolist())


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
