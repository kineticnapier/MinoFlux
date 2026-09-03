from __future__ import annotations

from array import array
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Literal, Mapping, Sequence

from minoflux_engine import VersusMatch

from .bitboard import ROW_OCCUPANCY_BYTES, board_row_masks
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights
from .search import SearchScorer, apply_search_action
from .versus_search import (
    DEFAULT_VERSUS_SEARCH_CONFIG,
    DEFAULT_VERSUS_WEIGHTS,
    SideName,
    VersusSearchConfig,
    VersusStateScorer,
    VersusWeights,
    choose_versus_action,
    score_versus_state,
)

VERSUS_VALUE_FORMAT = "minoflux_versus_value_v1"
VERSUS_SELFPLAY_FORMAT = "minoflux_versus_selfplay_v1"
VERSUS_BOARD_HEIGHT = 24
VERSUS_BOARD_WIDTH = 10
VERSUS_QUEUE_LENGTH = 5
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
_SIDE_NUMERIC_SIZE = 11


@dataclass(frozen=True, slots=True)
class VersusValueConfig:
    board_height: int = VERSUS_BOARD_HEIGHT
    board_width: int = VERSUS_BOARD_WIDTH
    queue_length: int = VERSUS_QUEUE_LENGTH
    conv_channels: tuple[int, int] = (24, 48)
    hidden_size: int = 384
    value_hidden_size: int = 192

    def normalized(self) -> "VersusValueConfig":
        height = int(self.board_height)
        width = int(self.board_width)
        queue_length = int(self.queue_length)
        if height != VERSUS_BOARD_HEIGHT or width != VERSUS_BOARD_WIDTH:
            raise ValueError(f"Versus value expects a {VERSUS_BOARD_HEIGHT}x{VERSUS_BOARD_WIDTH} board")
        if queue_length < 1 or queue_length > 7:
            raise ValueError("queue_length must be between 1 and 7")
        channels = tuple(max(1, int(value)) for value in self.conv_channels)
        if len(channels) != 2:
            raise ValueError("conv_channels must contain exactly two values")
        return VersusValueConfig(
            board_height=height,
            board_width=width,
            queue_length=queue_length,
            conv_channels=(channels[0], channels[1]),
            hidden_size=max(8, int(self.hidden_size)),
            value_hidden_size=max(8, int(self.value_hidden_size)),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "VersusValueConfig":
        channels = values.get("conv_channels", values.get("convChannels", (24, 48)))
        if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
            raise ValueError("conv_channels must be an array")
        return cls(
            board_height=int(values.get("board_height", values.get("boardHeight", VERSUS_BOARD_HEIGHT))),
            board_width=int(values.get("board_width", values.get("boardWidth", VERSUS_BOARD_WIDTH))),
            queue_length=int(values.get("queue_length", values.get("queueLength", VERSUS_QUEUE_LENGTH))),
            conv_channels=tuple(int(value) for value in channels),
            hidden_size=int(values.get("hidden_size", values.get("hiddenSize", 384))),
            value_hidden_size=int(values.get("value_hidden_size", values.get("valueHiddenSize", 192))),
        ).normalized()

    @property
    def side_context_size(self) -> int:
        return len(PIECES) + len(PIECES) + 1 + self.queue_length * len(PIECES) + _SIDE_NUMERIC_SIZE

    @property
    def context_size(self) -> int:
        return self.side_context_size * 2


@dataclass(frozen=True, slots=True)
class VersusNeuralState:
    own_rows: tuple[int, ...]
    opponent_rows: tuple[int, ...]
    context: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class VersusTrainConfig:
    epochs: int = 6
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    validation_fraction: float = 0.10
    teacher_weight: float = 0.25
    seed: int = 20260903
    device: str = "auto"


@dataclass(frozen=True, slots=True)
class VersusSelfPlayConfig:
    games: int = 50
    max_turns: int = 500
    seed_base: int = 6_000_001
    seed_step: int = 31
    garbage_cap: int = 8
    search_config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _piece_hot(piece: str | None) -> tuple[float, ...]:
    if piece is None:
        return _ZERO_PIECE
    return _PIECE_ONE_HOT.get(str(piece).upper(), _ZERO_PIECE)


def _hold_hot(piece: str | None) -> tuple[float, ...]:
    if piece is None:
        return _HOLD_ONE_HOT[None]
    return _HOLD_ONE_HOT.get(str(piece).upper(), _ZERO_HOLD)


def _side_context(side, config: VersusValueConfig) -> tuple[float, ...]:
    game = side.game
    values: list[float] = []
    values.extend(_piece_hot(game.current))
    values.extend(_hold_hot(game.hold_piece))
    queue = tuple(game.queue)
    for index in range(config.queue_length):
        values.extend(_piece_hot(queue[index] if index < len(queue) else None))
    values.extend(
        (
            _clip01((game.combo + 1) / 16.0),
            float(bool(game.back_to_back)),
            _clip01(game.b2b_chain / 20.0),
            _clip01(game.surge_charge / 20.0),
            _clip01(side.pending.pending_lines / 20.0),
            _clip01(side.sent / 100.0),
            _clip01(side.received / 100.0),
            _clip01(side.canceled / 100.0),
            _clip01(side.garbage_applied / 100.0),
            _clip01(game.pieces_placed / 500.0),
            float(bool(game.game_over)),
        )
    )
    if len(values) != config.side_context_size:
        raise AssertionError(f"Versus side context size mismatch: {len(values)} != {config.side_context_size}")
    return tuple(values)


def encode_versus_state(
    match: VersusMatch,
    root_side: SideName,
    config: VersusValueConfig = VersusValueConfig(),
) -> VersusNeuralState:
    cfg = config.normalized()
    own = match.side(root_side)
    opponent = match.opponent(root_side)
    own_rows = board_row_masks(own.game.board)
    opponent_rows = board_row_masks(opponent.game.board)
    if len(own_rows) != cfg.board_height or len(opponent_rows) != cfg.board_height:
        raise ValueError("Unexpected versus board height")
    context = _side_context(own, cfg) + _side_context(opponent, cfg)
    return VersusNeuralState(own_rows, opponent_rows, context)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for versus value learning. Install the ML dependencies with `uv sync --extra ml`."
        ) from error
    return torch, nn


def _resolve_device(torch: Any, device: str | None) -> str:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for versus value evaluation, but CUDA is unavailable")
    return requested


def build_versus_value_model(config: VersusValueConfig = VersusValueConfig()) -> Any:
    cfg = config.normalized()
    torch, nn = _require_torch()
    first_channels, second_channels = cfg.conv_channels

    class VersusValueModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.board_encoder = nn.Sequential(
                nn.Conv2d(2, first_channels, kernel_size=3, padding=1),
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
                nn.Tanh(),
            )

        def forward(self, board: Any, context: Any) -> Any:
            encoded = self.board_encoder(board).flatten(start_dim=1)
            return self.value_head(torch.cat((encoded, context), dim=1)).squeeze(1)

    return VersusValueModel()


def save_versus_value_checkpoint(
    path: str | Path,
    model: Any,
    config: VersusValueConfig = VersusValueConfig(),
    *,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    torch, _ = _require_torch()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": VERSUS_VALUE_FORMAT,
            "config": asdict(config.normalized()),
            "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
            "metadata": dict(metadata or {}),
        },
        target,
    )
    return target


class VersusValueEvaluator(VersusStateScorer):
    def __init__(
        self,
        model: Any,
        config: VersusValueConfig = VersusValueConfig(),
        *,
        device: str | None = "auto",
        compile_model: bool = False,
    ) -> None:
        torch, _ = _require_torch()
        self._torch = torch
        self.config = config.normalized()
        self.device = _resolve_device(torch, device)
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
        compile_model: bool = False,
    ) -> "VersusValueEvaluator":
        torch, _ = _require_torch()
        target_device = _resolve_device(torch, device)
        payload = torch.load(Path(path), map_location=target_device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("format") != VERSUS_VALUE_FORMAT:
            found = payload.get("format") if isinstance(payload, dict) else None
            raise ValueError(f"Unsupported versus value model format: {found!r}")
        raw_config = payload.get("config")
        state_dict = payload.get("state_dict")
        if not isinstance(raw_config, Mapping) or not isinstance(state_dict, Mapping):
            raise ValueError("Invalid versus value checkpoint")
        config = VersusValueConfig.from_mapping(raw_config)
        model = build_versus_value_model(config)
        model.load_state_dict(state_dict)
        return cls(model, config, device=target_device, compile_model=compile_model)

    def score_states(self, states: Sequence[VersusNeuralState]) -> tuple[float, ...]:
        if not states:
            return ()
        torch = self._torch
        board_bytes = bytearray()
        context_values = array("f")
        occupancy = ROW_OCCUPANCY_BYTES
        for state in states:
            for mask in state.own_rows:
                board_bytes.extend(occupancy[mask])
            for mask in state.opponent_rows:
                board_bytes.extend(occupancy[mask])
            context_values.extend(state.context)
        boards = torch.frombuffer(board_bytes, dtype=torch.uint8).reshape(
            len(states), 2, self.config.board_height, self.config.board_width
        ).to(device=self.device, dtype=torch.float32)
        contexts = torch.frombuffer(context_values, dtype=torch.float32).reshape(
            len(states), self.config.context_size
        ).to(self.device)
        with torch.inference_mode():
            values = self.model(boards, contexts).reshape(-1)
        return tuple(values.detach().cpu().tolist())

    def score_match(self, match: VersusMatch, root_side: SideName) -> float:
        return self.score_states((encode_versus_state(match, root_side, self.config),))[0]


def _state_record(
    state: VersusNeuralState,
    *,
    game_index: int,
    ply: int,
    side: SideName,
    seed: int,
    teacher_value: float,
) -> dict[str, object]:
    return {
        "format": VERSUS_SELFPLAY_FORMAT,
        "game": int(game_index),
        "ply": int(ply),
        "side": side,
        "seed": int(seed),
        "ownRows": list(state.own_rows),
        "opponentRows": list(state.opponent_rows),
        "context": list(state.context),
        "teacherValue": float(teacher_value),
    }


def _outcome(winner: str, side: SideName) -> float:
    if winner == "draw":
        return 0.0
    return 1.0 if winner == side else -1.0


def generate_versus_selfplay_dataset(
    path: str | Path,
    solo_scorer: SearchScorer,
    config: VersusSelfPlayConfig = VersusSelfPlayConfig(),
    *,
    heuristic_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    versus_weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS,
    value_scorer: VersusStateScorer | None = None,
    value_config: VersusValueConfig = VersusValueConfig(),
) -> dict[str, object]:
    cfg = VersusSelfPlayConfig(
        games=max(1, int(config.games)),
        max_turns=max(1, int(config.max_turns)),
        seed_base=int(config.seed_base),
        seed_step=int(config.seed_step),
        garbage_cap=max(1, int(config.garbage_cap)),
        search_config=config.search_config.normalized(),
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    wins = {"player": 0, "ai": 0, "draw": 0}
    total_turns = 0

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for game_index in range(cfg.games):
            seed = cfg.seed_base + game_index * cfg.seed_step
            match = VersusMatch(seed, garbage_cap=cfg.garbage_cap)
            turn_side: SideName = "player" if game_index % 2 == 0 else "ai"
            pending_records: list[tuple[dict[str, object], SideName]] = []
            turns = 0
            while match.winner is None and turns < cfg.max_turns:
                for perspective in ("player", "ai"):
                    state = encode_versus_state(match, perspective, value_config)
                    teacher_raw = score_versus_state(match, perspective, weights=versus_weights)
                    teacher_value = math.tanh(teacher_raw / 64.0)
                    pending_records.append(
                        (
                            _state_record(
                                state,
                                game_index=game_index,
                                ply=turns,
                                side=perspective,
                                seed=seed,
                                teacher_value=teacher_value,
                            ),
                            perspective,
                        )
                    )
                choice = choose_versus_action(
                    match,
                    turn_side,
                    heuristic_weights,
                    cfg.search_config,
                    versus_weights,
                    scorer=solo_scorer,
                    opponent_scorer=solo_scorer,
                    state_scorer=value_scorer,
                )
                if choice is None:
                    match.side(turn_side).game.game_over = True
                    match._update_winner()
                    break
                result = apply_search_action(match.side(turn_side).game, choice.action)
                match.resolve_lock(turn_side, result)
                turns += 1
                turn_side = "ai" if turn_side == "player" else "player"

            winner = match.winner or "draw"
            wins[winner] += 1
            total_turns += turns
            for record, perspective in pending_records:
                record["outcome"] = _outcome(winner, perspective)
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                total_records += 1

    return {
        "format": VERSUS_SELFPLAY_FORMAT,
        "path": str(target),
        "games": cfg.games,
        "records": total_records,
        "playerWins": wins["player"],
        "aiWins": wins["ai"],
        "draws": wins["draw"],
        "meanTurns": total_turns / cfg.games,
        "searchConfig": cfg.search_config.to_dict(),
    }


def _load_selfplay_records(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or value.get("format") != VERSUS_SELFPLAY_FORMAT:
                raise ValueError(f"Invalid versus self-play record at line {line_number}")
            records.append(value)
    if not records:
        raise ValueError("Versus self-play dataset is empty")
    return records


def train_versus_value_model(
    dataset_path: str | Path,
    output_path: str | Path,
    train_config: VersusTrainConfig = VersusTrainConfig(),
    model_config: VersusValueConfig = VersusValueConfig(),
) -> dict[str, object]:
    torch, _ = _require_torch()
    cfg = model_config.normalized()
    train_cfg = VersusTrainConfig(
        epochs=max(1, int(train_config.epochs)),
        batch_size=max(1, int(train_config.batch_size)),
        learning_rate=max(1e-7, float(train_config.learning_rate)),
        weight_decay=max(0.0, float(train_config.weight_decay)),
        validation_fraction=min(0.5, max(0.0, float(train_config.validation_fraction))),
        teacher_weight=max(0.0, float(train_config.teacher_weight)),
        seed=int(train_config.seed),
        device=str(train_config.device),
    )
    records = _load_selfplay_records(dataset_path)
    rng = random.Random(train_cfg.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    validation_count = int(round(len(order) * train_cfg.validation_fraction))
    validation_count = min(max(0, validation_count), max(0, len(order) - 1))
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]
    device = _resolve_device(torch, train_cfg.device)
    model = build_versus_value_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    def pack(indices: Sequence[int]) -> tuple[Any, Any, Any, Any]:
        board_bytes = bytearray()
        context_values = array("f")
        outcomes = array("f")
        teachers = array("f")
        occupancy = ROW_OCCUPANCY_BYTES
        for index in indices:
            record = records[index]
            own_rows = record.get("ownRows")
            opponent_rows = record.get("opponentRows")
            context = record.get("context")
            if not isinstance(own_rows, list) or not isinstance(opponent_rows, list) or not isinstance(context, list):
                raise ValueError("Malformed versus self-play state")
            for mask in own_rows:
                board_bytes.extend(occupancy[int(mask)])
            for mask in opponent_rows:
                board_bytes.extend(occupancy[int(mask)])
            context_values.extend(float(value) for value in context)
            outcomes.append(float(record.get("outcome", 0.0)))
            teachers.append(float(record.get("teacherValue", 0.0)))
        count = len(indices)
        boards = torch.frombuffer(board_bytes, dtype=torch.uint8).reshape(
            count, 2, cfg.board_height, cfg.board_width
        ).to(device=device, dtype=torch.float32)
        contexts = torch.frombuffer(context_values, dtype=torch.float32).reshape(
            count, cfg.context_size
        ).to(device)
        outcome_tensor = torch.frombuffer(outcomes, dtype=torch.float32).to(device)
        teacher_tensor = torch.frombuffer(teachers, dtype=torch.float32).to(device)
        return boards, contexts, outcome_tensor, teacher_tensor

    def evaluate(indices: Sequence[int]) -> float:
        if not indices:
            return 0.0
        model.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for start in range(0, len(indices), train_cfg.batch_size):
                batch = indices[start : start + train_cfg.batch_size]
                boards, contexts, outcomes, teachers = pack(batch)
                predictions = model(boards, contexts)
                loss = torch.mean((predictions - outcomes) ** 2)
                if train_cfg.teacher_weight:
                    loss = loss + train_cfg.teacher_weight * torch.mean((predictions - teachers) ** 2)
                total += float(loss.detach().cpu()) * len(batch)
                count += len(batch)
        return total / max(1, count)

    history: list[dict[str, float | int]] = []
    for epoch in range(1, train_cfg.epochs + 1):
        rng.shuffle(train_indices)
        model.train()
        total_loss = 0.0
        seen = 0
        for start in range(0, len(train_indices), train_cfg.batch_size):
            batch = train_indices[start : start + train_cfg.batch_size]
            boards, contexts, outcomes, teachers = pack(batch)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(boards, contexts)
            loss = torch.mean((predictions - outcomes) ** 2)
            if train_cfg.teacher_weight:
                loss = loss + train_cfg.teacher_weight * torch.mean((predictions - teachers) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        history.append(
            {
                "epoch": epoch,
                "trainLoss": total_loss / max(1, seen),
                "validationLoss": evaluate(validation_indices),
            }
        )

    save_versus_value_checkpoint(
        output_path,
        model,
        cfg,
        metadata={
            "dataset": str(dataset_path),
            "records": len(records),
            "trainConfig": asdict(train_cfg),
            "history": history,
        },
    )
    return {
        "format": VERSUS_VALUE_FORMAT,
        "output": str(output_path),
        "device": device,
        "records": len(records),
        "trainRecords": len(train_indices),
        "validationRecords": len(validation_indices),
        "history": history,
        "config": asdict(cfg),
    }
