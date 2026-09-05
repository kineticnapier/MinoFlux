from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

from minoflux_engine import Game, LockResult
from minoflux_engine.pieces import SHAPES

from .bitboard import board_row_masks
from .neural import NeuralValueConfig, encode_placement_result
from .neural_dataset import (
    NEURAL_DATASET_FORMAT,
    NeuralRankingCandidate,
    NeuralRankingSample,
    pack_board_rows,
)
from .reachability import reachable_placements
from .search import SearchAction, _held_search_game

HUMAN_PLACEMENT_FORMAT = "minoflux_human_placement_v1"
DEFAULT_HUMAN_PLACEMENT_LOG = Path("data/human/placements.jsonl")
DEFAULT_HUMAN_RANKING_DATASET = Path("data/neural/human-placement-ranking.jsonl")


@dataclass(frozen=True, slots=True)
class HumanDatasetConfig:
    allow_hold: bool = True
    allow_180: bool = False
    reachability_node_limit: int = 8_000
    max_candidates: int = 0
    random_seed: int = 26_090_905
    neural: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "HumanDatasetConfig":
        return HumanDatasetConfig(
            allow_hold=bool(self.allow_hold),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
            max_candidates=max(0, int(self.max_candidates)),
            random_seed=int(self.random_seed),
            neural=self.neural.normalized(),
        )

    def to_dict(self) -> dict[str, object]:
        cfg = self.normalized()
        return {
            "allowHold": cfg.allow_hold,
            "allow180": cfg.allow_180,
            "reachabilityNodeLimit": cfg.reachability_node_limit,
            "maxCandidates": cfg.max_candidates,
            "randomSeed": cfg.random_seed,
            "neural": asdict(cfg.neural),
        }


def _capture_turn_state(game: Game) -> dict[str, object]:
    return {
        "rows": list(board_row_masks(game.board)),
        "current": game.current,
        "holdPiece": game.hold_piece,
        "queue": list(game.queue),
        "combo": int(game.combo),
        "backToBack": bool(game.back_to_back),
        "b2bChain": int(game.b2b_chain),
        "surgeCharge": int(game.surge_charge),
    }


def _capture_choice(game: Game, *, hard_drop: bool) -> dict[str, object]:
    landing_y = game.ghost_y() if hard_drop else game.y
    return {
        "hold": bool(game.hold_used),
        "piece": game.current,
        "x": int(game.x),
        "y": int(landing_y),
        "rotation": int(game.rotation),
        "lastMoveWasRotation": bool(game.last_move_was_rotation),
        "rotationKickIndex": game.last_rotation_kick_index,
        "rotationFrom": game.last_rotation_from,
        "rotationTo": game.last_rotation_to,
    }


class HumanPlacementRecorder:
    """Append lightweight local-play placement demonstrations as JSONL."""

    def __init__(
        self,
        path: str | Path = DEFAULT_HUMAN_PLACEMENT_LOG,
        *,
        session_id: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.session_id = (
            int(session_id)
            if session_id is not None
            else int(time.time_ns() & ((1 << 62) - 1))
        )
        self.game_index = -1
        self.recorded = 0
        self._turn_state: dict[str, object] | None = None
        self._piece_index = 0

    def begin_game(self, game: Game) -> None:
        self.game_index += 1
        self._piece_index = int(game.pieces_placed)
        self._turn_state = _capture_turn_state(game)

    def capture_choice(self, game: Game, *, hard_drop: bool) -> dict[str, object]:
        return _capture_choice(game, hard_drop=hard_drop)

    def record_lock(
        self,
        choice: Mapping[str, object],
        result: LockResult,
        game_after: Game,
    ) -> None:
        if self._turn_state is None:
            return
        record = {
            "format": HUMAN_PLACEMENT_FORMAT,
            "sessionId": self.session_id,
            "gameIndex": self.game_index,
            "pieceIndex": self._piece_index,
            "state": self._turn_state,
            "choice": dict(choice),
            "result": {
                "lines": int(result.lines),
                "attack": int(result.attack),
                "spin": result.spin,
                "perfectClear": bool(result.perfect_clear),
                "combo": int(result.combo),
                "backToBack": bool(result.back_to_back),
                "gameOver": bool(result.game_over),
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self.recorded += 1
        self._piece_index = int(game_after.pieces_placed)
        self._turn_state = None if game_after.game_over else _capture_turn_state(game_after)


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an array")
    return value


def _game_from_state(state: Mapping[str, object]) -> Game:
    rows_raw = _require_sequence(state.get("rows"), "state.rows")
    if len(rows_raw) != 24:
        raise ValueError("state.rows must contain 24 packed rows")

    game = Game(0)
    masks = tuple(int(value) for value in rows_raw)
    game.board = [
        ["G" if mask & (1 << x) else None for x in range(game.width)]
        for mask in masks
    ]

    current = str(state.get("current", "")).upper()
    if current not in {"I", "O", "T", "S", "Z", "J", "L"}:
        raise ValueError("state.current is invalid")
    game.current = current
    game.x, game.y, game.rotation = 3, 1, 0

    raw_hold = state.get("holdPiece")
    game.hold_piece = None if raw_hold is None else str(raw_hold).upper()
    queue_raw = _require_sequence(state.get("queue"), "state.queue")
    game.queue = deque(str(piece).upper() for piece in queue_raw)
    if len(game.queue) < 6:
        raise ValueError("state.queue must contain at least 6 pieces")

    game.hold_used = False
    game.combo = int(state.get("combo", -1))
    game.back_to_back = bool(state.get("backToBack", False))
    game.b2b_chain = int(state.get("b2bChain", 0))
    game.surge_charge = int(state.get("surgeCharge", 0))
    game.score = 0
    game.lines = 0
    game.attack = 0
    game.pieces_placed = 0
    game.last_lock = None
    game.last_action = None
    game.last_move_was_rotation = False
    game.last_rotation_kick_index = None
    game.last_rotation_from = None
    game.last_rotation_to = None
    game.paused = False
    game.game_over = game._collides(game.current, game.x, game.y, game.rotation)
    return game


def _legal_actions(
    game: Game,
    cfg: HumanDatasetConfig,
) -> tuple[tuple[SearchAction, Game], ...]:
    actions: list[tuple[SearchAction, Game]] = []
    direct = reachable_placements(
        game,
        allow_180=cfg.allow_180,
        max_nodes=cfg.reachability_node_limit,
        include_paths=False,
    )
    actions.extend((SearchAction(False, placement), game) for placement in direct)

    if cfg.allow_hold:
        held = _held_search_game(game)
        if held is not None:
            held_placements = reachable_placements(
                held,
                allow_180=cfg.allow_180,
                max_nodes=cfg.reachability_node_limit,
                include_paths=False,
            )
            actions.extend((SearchAction(True, placement), held) for placement in held_placements)

    return tuple(actions)


def _choice_key(choice: Mapping[str, object]) -> tuple[object, ...]:
    kick = choice.get("rotationKickIndex")
    return (
        bool(choice.get("hold", False)),
        str(choice.get("piece", "")).upper(),
        int(choice.get("x", 0)),
        int(choice.get("y", 0)),
        int(choice.get("rotation", 0)) % 4,
        bool(choice.get("lastMoveWasRotation", False)),
        None if kick is None else int(kick),
    )


def _action_key(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        bool(action.use_hold),
        placement.piece,
        int(placement.x),
        int(placement.y),
        int(placement.rotation) % 4,
        bool(placement.last_move_was_rotation),
        placement.rotation_kick_index,
    )


def _choice_geometry_key(choice: Mapping[str, object]) -> tuple[object, ...]:
    piece = str(choice.get("piece", "")).upper()
    shapes = SHAPES.get(piece)
    if shapes is None:
        raise ValueError("choice.piece is invalid")
    x = int(choice.get("x", 0))
    y = int(choice.get("y", 0))
    rotation = int(choice.get("rotation", 0)) % 4
    cells = tuple(sorted((x + dx, y + dy) for dx, dy in shapes[rotation]))
    return (bool(choice.get("hold", False)), piece, cells)


def _action_geometry_key(action: SearchAction) -> tuple[object, ...]:
    placement = action.placement
    return (
        bool(action.use_hold),
        placement.piece,
        tuple(sorted(placement.cells)),
    )


def _seed_for_record(record: Mapping[str, object]) -> int:
    session_id = int(record.get("sessionId", 0))
    game_index = int(record.get("gameIndex", 0))
    return (session_id ^ (game_index * 0x9E3779B1)) & ((1 << 63) - 1)


def _select_indices(
    count: int,
    expert_indices: Sequence[int],
    max_candidates: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if max_candidates <= 0 or count <= max_candidates:
        return tuple(range(count))
    experts = tuple(sorted(set(int(index) for index in expert_indices)))
    if len(experts) >= max_candidates:
        return experts[:max_candidates]
    expert_set = set(experts)
    pool = [index for index in range(count) if index not in expert_set]
    wanted = min(len(pool), max_candidates - len(experts))
    chosen = rng.sample(pool, wanted)
    return tuple(sorted((*experts, *chosen)))


def _candidate(
    branch: Game,
    action: SearchAction,
    config: NeuralValueConfig,
) -> NeuralRankingCandidate | None:
    state = encode_placement_result(branch, action.placement, config)
    if state is None:
        return None
    placement = action.placement
    return NeuralRankingCandidate(
        board_rows=pack_board_rows(state.board, config),
        context=state.context,
        move=(
            int(action.use_hold),
            placement.piece,
            placement.x,
            placement.y,
            placement.rotation,
        ),
        sampling_bucket="human",
    )


def _load_raw_records(path: str | Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, dict) or value.get("format") != HUMAN_PLACEMENT_FORMAT:
                raise ValueError(f"Invalid human placement record at line {line_number}")
            records.append(value)
    return tuple(records)


def write_human_ranking_dataset(
    input_path: str | Path,
    output_path: str | Path = DEFAULT_HUMAN_RANKING_DATASET,
    config: HumanDatasetConfig = HumanDatasetConfig(),
) -> dict[str, object]:
    cfg = config.normalized()
    records = _load_raw_records(input_path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    candidates_written = 0
    skipped: dict[str, int] = {}
    expert_matches = {"exact": 0, "geometry": 0}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            try:
                state = _require_mapping(record.get("state"), "state")
                choice = _require_mapping(record.get("choice"), "choice")
                game = _game_from_state(state)
            except (TypeError, ValueError):
                skip("invalid-state")
                continue
            if game.game_over:
                skip("spawn-collision")
                continue

            action_branches = _legal_actions(game, cfg)
            if not action_branches:
                skip("no-candidates")
                continue

            try:
                wanted_key = _choice_key(choice)
                wanted_geometry = _choice_geometry_key(choice)
            except (TypeError, ValueError):
                skip("invalid-choice")
                continue

            raw_experts = [
                index
                for index, (action, _branch) in enumerate(action_branches)
                if _action_key(action) == wanted_key
            ]
            if raw_experts:
                expert_matches["exact"] += 1
            else:
                raw_experts = [
                    index
                    for index, (action, _branch) in enumerate(action_branches)
                    if _action_geometry_key(action) == wanted_geometry
                ]
                if raw_experts:
                    expert_matches["geometry"] += 1
            if not raw_experts:
                skip("expert-not-reachable")
                continue

            prepared: list[tuple[int, NeuralRankingCandidate]] = []
            for raw_index, (action, branch) in enumerate(action_branches):
                candidate = _candidate(branch, action, cfg.neural)
                if candidate is not None:
                    prepared.append((raw_index, candidate))
            if not prepared:
                skip("no-encodable-candidates")
                continue

            raw_to_prepared = {
                raw_index: prepared_index
                for prepared_index, (raw_index, _candidate_value) in enumerate(prepared)
            }
            prepared_experts = [
                raw_to_prepared[index]
                for index in raw_experts
                if index in raw_to_prepared
            ]
            if not prepared_experts:
                skip("expert-not-encodable")
                continue

            rng = random.Random(
                cfg.random_seed
                ^ _seed_for_record(record)
                ^ ((int(record.get("pieceIndex", 0)) + 1) * 0x85EBCA6B)
            )
            selected = _select_indices(
                len(prepared),
                prepared_experts,
                cfg.max_candidates,
                rng,
            )
            selected_map = {
                old_index: new_index for new_index, old_index in enumerate(selected)
            }
            expert_indices = tuple(
                sorted(selected_map[index] for index in prepared_experts if index in selected_map)
            )
            if not expert_indices:
                skip("expert-dropped")
                continue

            candidates = tuple(prepared[index][1] for index in selected)
            sample = NeuralRankingSample(
                seed=_seed_for_record(record),
                piece_index=int(record.get("pieceIndex", 0)),
                expert_index=expert_indices[0],
                expert_indices=expert_indices,
                candidates=candidates,
            ).to_dict()
            sample["teacher"] = "human-local"
            sample["humanSource"] = {
                "sessionId": int(record.get("sessionId", 0)),
                "gameIndex": int(record.get("gameIndex", 0)),
            }
            raw_candidates = sample.get("candidates")
            if isinstance(raw_candidates, list):
                for index in expert_indices:
                    candidate = raw_candidates[index]
                    if isinstance(candidate, dict):
                        candidate["samplingBucket"] = "human-expert"
            stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
            written += 1
            candidates_written += len(candidates)

    return {
        "format": NEURAL_DATASET_FORMAT,
        "teacher": "human-local",
        "input": str(input_path),
        "output": str(target),
        "records": len(records),
        "samples": written,
        "candidates": candidates_written,
        "skipped": dict(sorted(skipped.items())),
        "expertMatches": expert_matches,
        "datasetConfig": cfg.to_dict(),
    }
