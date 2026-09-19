from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from hashlib import blake2b
import json
from pathlib import Path
import random
from typing import Mapping, Sequence

from minoflux_engine import Game

from .neural import NeuralValueConfig, encode_placement_result
from .neural_dataset import (
    NEURAL_DATASET_FORMAT,
    NeuralRankingCandidate,
    NeuralRankingSample,
    pack_board_rows,
)
from .reachability import reachable_placements
from .search import SearchAction, _held_search_game
from .tetrio_alignment import ALIGNMENT_FORMAT, CaptureAlignment
from .tetrio_capture import CAPTURE_DATASET_FORMAT, CaptureSample, normalize_board

_PIECES = {"I", "O", "T", "S", "Z", "J", "L"}


@dataclass(frozen=True, slots=True)
class TetrioDistillConfig:
    max_candidates: int = 24
    allow_180: bool = True
    reachability_node_limit: int = 8_000
    random_seed: int = 26_090_916
    neural: NeuralValueConfig = NeuralValueConfig()

    def normalized(self) -> "TetrioDistillConfig":
        return TetrioDistillConfig(
            max_candidates=max(0, int(self.max_candidates)),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
            random_seed=int(self.random_seed),
            neural=self.neural.normalized(),
        )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _load_capture_samples(path: str | Path) -> tuple[CaptureSample, ...]:
    samples: list[CaptureSample] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping) or value.get("format") != CAPTURE_DATASET_FORMAT:
                raise ValueError(f"Invalid TETR.IO capture sample at line {line_number}")
            before_raw = value.get("board_before")
            after_raw = value.get("board_after")
            if after_raw is None:
                raise ValueError(f"Missing board_after at line {line_number}")
            samples.append(
                CaptureSample(
                    group_id=str(value.get("group_id", "")),
                    split=str(value.get("split", "train")),
                    username=str(value.get("username", "unknown")),
                    game_id=_optional_int(value.get("game_id")),
                    round=max(1, int(value.get("round", 1))),
                    sequence=int(value.get("sequence", 0)),
                    frame=int(value.get("frame", 0)),
                    frame_delta=_optional_int(value.get("frame_delta")),
                    piece_index=_optional_int(value.get("piece_index")),
                    piece=str(value.get("piece", "")).upper(),
                    x=_optional_float(value.get("x")),
                    y=_optional_float(value.get("y")),
                    rotation=int(value.get("rotation", 0)) % 4,
                    hold_before=None if value.get("hold_before") is None else str(value.get("hold_before")).upper(),
                    hold_after=None if value.get("hold_after") is None else str(value.get("hold_after")).upper(),
                    used_hold=_optional_bool(value.get("used_hold")),
                    next_placed_piece=None if value.get("next_placed_piece") is None else str(value.get("next_placed_piece")).upper(),
                    operations=tuple(str(item) for item in value.get("operations", ())),
                    board_before=None if before_raw is None else normalize_board(before_raw, height=24),
                    board_after=normalize_board(after_raw, height=24),
                    estimated_lines=_optional_int(value.get("estimated_lines")),
                    transition_confidence=str(value.get("transition_confidence", "unknown")),
                    seed=_optional_int(value.get("seed")),
                    hold_locked=_optional_bool(value.get("hold_locked")),
                    next_queue=tuple(str(item).upper() for item in value.get("next_queue", ())),
                )
            )
    samples.sort(key=lambda item: (item.group_id, item.sequence, item.frame))
    return tuple(samples)


def _load_alignments(path: str | Path) -> tuple[CaptureAlignment, ...]:
    alignments: list[CaptureAlignment] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            value = json.loads(text)
            if not isinstance(value, Mapping) or value.get("format") != ALIGNMENT_FORMAT:
                raise ValueError(f"Invalid TETR.IO alignment at line {line_number}")
            alignments.append(
                CaptureAlignment(
                    group_id=str(value.get("group_id", "")),
                    sequence=int(value.get("sequence", 0)),
                    piece=str(value.get("piece", "")).upper(),
                    status=str(value.get("status", "unmatched")),
                    candidate_count=int(value.get("candidate_count", 0)),
                    x=_optional_int(value.get("x")),
                    y=_optional_int(value.get("y")),
                    rotation=_optional_int(value.get("rotation")),
                    path=tuple(str(item) for item in value.get("path", ())),
                    last_move_was_rotation=bool(value.get("last_move_was_rotation", False)),
                    rotation_kick_index=_optional_int(value.get("rotation_kick_index")),
                    rotation_from=_optional_int(value.get("rotation_from")),
                    rotation_to=_optional_int(value.get("rotation_to")),
                )
            )
    return tuple(alignments)


def _turn_current(sample: CaptureSample) -> str | None:
    if sample.used_hold:
        hold_after = sample.hold_after
        return hold_after if hold_after in _PIECES else None
    return sample.piece if sample.piece in _PIECES else None


def _future_queue(items: Sequence[CaptureSample], index: int, count: int) -> tuple[str, ...]:
    """Reconstruct queue pops after this turn from later observed turns."""

    pieces: list[str] = []
    for following in items[index + 1 :]:
        current = _turn_current(following)
        if current is None:
            return ()
        pieces.append(current)
        if len(pieces) >= count:
            break
        if following.used_hold and following.hold_before is None:
            if following.piece not in _PIECES:
                return ()
            pieces.append(following.piece)
            if len(pieces) >= count:
                break
    return tuple(pieces[:count])


def _queue_for_sample(
    items: Sequence[CaptureSample],
    index: int,
    queue_length: int,
) -> tuple[str, ...]:
    """Reconstruct the queue at the start of an observed turn.

    Encoding a placement needs one next current plus ``queue_length`` preview
    pieces. An empty-Hold candidate consumes one additional queue piece before
    the lock, so empty-Hold states need one extra reconstructed pop.
    """

    sample = items[index]
    after_lock = queue_length + 1
    if sample.hold_before is None and sample.used_hold:
        if sample.piece not in _PIECES:
            return ()
        future = _future_queue(items, index, after_lock)
        if len(future) < after_lock:
            return ()
        return (sample.piece, *future)

    needed = after_lock + 1 if sample.hold_before is None else after_lock
    future = _future_queue(items, index, needed)
    return future if len(future) >= needed else ()


def _game_for_sample(sample: CaptureSample, queue: Sequence[str]) -> Game | None:
    if sample.board_before is None:
        return None
    current = _turn_current(sample)
    if current is None:
        return None
    if sample.hold_before is not None and sample.hold_before not in _PIECES:
        return None
    game = Game(0)
    game.board = [list(row) for row in sample.board_before]
    game.current = current
    game.x, game.y, game.rotation = 3, 1, 0
    game.hold_piece = sample.hold_before
    game.queue = deque(queue)
    game.hold_used = False
    game.combo = -1
    game.back_to_back = False
    game.b2b_chain = 0
    game.surge_charge = 0
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
    return None if game.game_over else game


def _legal_actions(
    game: Game,
    cfg: TetrioDistillConfig,
) -> tuple[tuple[SearchAction, Game], ...]:
    actions: list[tuple[SearchAction, Game]] = []
    direct = reachable_placements(
        game,
        allow_180=cfg.allow_180,
        max_nodes=cfg.reachability_node_limit,
        include_paths=False,
    )
    actions.extend((SearchAction(False, placement), game) for placement in direct)

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


def _placement_key(placement) -> tuple[object, ...]:
    return (
        placement.piece,
        int(placement.x),
        int(placement.y),
        int(placement.rotation) % 4,
        bool(placement.last_move_was_rotation),
        placement.rotation_kick_index,
    )


def _action_key(action: SearchAction) -> tuple[object, ...]:
    return (bool(action.use_hold), *_placement_key(action.placement))


def _alignment_key(alignment: CaptureAlignment, *, used_hold: bool) -> tuple[object, ...]:
    return (
        bool(used_hold),
        alignment.piece,
        alignment.x,
        alignment.y,
        None if alignment.rotation is None else alignment.rotation % 4,
        bool(alignment.last_move_was_rotation),
        alignment.rotation_kick_index,
    )


def _stable_seed(group_id: str, sequence: int, random_seed: int) -> int:
    digest = blake2b(f"{group_id}|{sequence}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") ^ int(random_seed)


def _select_indices(count: int, expert: int, maximum: int, rng: random.Random) -> tuple[int, ...]:
    if maximum <= 0 or count <= maximum:
        return tuple(range(count))
    others = [index for index in range(count) if index != expert]
    chosen = rng.sample(others, min(len(others), maximum - 1))
    return tuple(sorted((expert, *chosen)))


def write_tetrio_ranking_dataset(
    capture_path: str | Path,
    alignment_path: str | Path,
    output_path: str | Path,
    config: TetrioDistillConfig = TetrioDistillConfig(),
) -> dict[str, object]:
    """Write neural ranking data from aligned TETR.IO captures.

    The pre-turn current, Hold slot, and short queue are reconstructed from
    neighboring observations. Direct and Hold branches are then enumerated
    together so the observed Hold choice is part of the ranking target.
    """

    cfg = config.normalized()
    samples = _load_capture_samples(capture_path)
    alignments = _load_alignments(alignment_path)
    alignment_by_key = {(item.group_id, item.sequence): item for item in alignments}

    groups: dict[str, list[CaptureSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id, []).append(sample)
    for items in groups.values():
        items.sort(key=lambda item: (item.sequence, item.frame))

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    candidates_written = 0
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for group_id, items in sorted(groups.items()):
            for index, sample in enumerate(items):
                if sample.board_before is None:
                    skip("no-before-board")
                    continue
                if sample.used_hold is None:
                    skip("unknown-hold")
                    continue
                alignment = alignment_by_key.get((sample.group_id, sample.sequence))
                if alignment is None or alignment.status not in {"exact", "ambiguous"}:
                    skip("unaligned")
                    continue

                queue = _queue_for_sample(items, index, cfg.neural.queue_length)
                if not queue:
                    skip("insufficient-future-queue")
                    continue
                game = _game_for_sample(sample, queue)
                if game is None:
                    skip("invalid-state")
                    continue

                action_branches = _legal_actions(game, cfg)
                if not action_branches:
                    skip("no-candidates")
                    continue
                wanted = _alignment_key(alignment, used_hold=sample.used_hold)
                expert_indices = [
                    action_index
                    for action_index, (action, _branch) in enumerate(action_branches)
                    if _action_key(action) == wanted
                ]
                if not expert_indices:
                    skip("expert-not-reachable")
                    continue
                expert_raw = expert_indices[0]

                prepared: list[NeuralRankingCandidate] = []
                prepared_raw_indices: list[int] = []
                for raw_index, (action, branch) in enumerate(action_branches):
                    placement = action.placement
                    state = encode_placement_result(branch, placement, cfg.neural)
                    if state is None:
                        continue
                    prepared.append(
                        NeuralRankingCandidate(
                            board_rows=pack_board_rows(state.board, cfg.neural),
                            context=state.context,
                            move=(
                                int(action.use_hold),
                                placement.piece,
                                placement.x,
                                placement.y,
                                placement.rotation,
                            ),
                            sampling_bucket="tetrio-expert" if raw_index == expert_raw else "tetrio-negative",
                        )
                    )
                    prepared_raw_indices.append(raw_index)
                if expert_raw not in prepared_raw_indices:
                    skip("expert-not-encodable")
                    continue
                expert_prepared = prepared_raw_indices.index(expert_raw)

                rng = random.Random(_stable_seed(group_id, sample.sequence, cfg.random_seed))
                selected = _select_indices(len(prepared), expert_prepared, cfg.max_candidates, rng)
                selected_map = {old: new for new, old in enumerate(selected)}
                if expert_prepared not in selected_map:
                    skip("expert-dropped")
                    continue
                candidates = tuple(prepared[item] for item in selected)
                expert_index = selected_map[expert_prepared]
                seed = _stable_seed(group_id, 0, 0) & ((1 << 63) - 1)
                record = NeuralRankingSample(
                    seed=seed,
                    piece_index=sample.piece_index if sample.piece_index is not None else sample.sequence,
                    expert_index=expert_index,
                    expert_indices=(expert_index,),
                    candidates=candidates,
                ).to_dict()
                record["teacher"] = "tetrio-capture"
                record["tetrioSource"] = {
                    "groupId": sample.group_id,
                    "sequence": sample.sequence,
                    "username": sample.username,
                    "split": sample.split,
                    "usedHold": bool(sample.used_hold),
                }
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                candidates_written += len(candidates)

    return {
        "format": NEURAL_DATASET_FORMAT,
        "teacher": "tetrio-capture",
        "input": str(capture_path),
        "alignment": str(alignment_path),
        "output": str(target),
        "records": len(samples),
        "samples": written,
        "candidates": candidates_written,
        "skipped": dict(sorted(skipped.items())),
        "datasetConfig": {
            "maxCandidates": cfg.max_candidates,
            "allow180": cfg.allow_180,
            "reachabilityNodeLimit": cfg.reachability_node_limit,
            "randomSeed": cfg.random_seed,
            "queueLength": cfg.neural.queue_length,
            "holdDecisions": "included",
        },
    }
