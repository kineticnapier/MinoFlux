from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import blake2b
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

CAPTURE_DATASET_FORMAT = "minoflux_tetrio_capture_v1"
BOARD_WIDTH = 10
MINOFLUX_BOARD_HEIGHT = 24

Board = tuple[tuple[str | None, ...], ...]


def _cell(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def normalize_board(value: object, *, height: int | None = None) -> Board:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Capture board must be an array")
    rows: list[tuple[str | None, ...]] = []
    for raw_row in value:
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, (str, bytes)):
            raise ValueError("Capture board rows must be arrays")
        row = tuple(_cell(item) for item in raw_row)
        if len(row) != BOARD_WIDTH:
            raise ValueError(f"Capture board row width must be {BOARD_WIDTH}, got {len(row)}")
        rows.append(row)
    if height is not None:
        target = max(1, int(height))
        if len(rows) < target:
            rows = [(None,) * BOARD_WIDTH for _ in range(target - len(rows))] + rows
        else:
            rows = rows[-target:]
    return tuple(rows)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _operations(value: Mapping[str, object]) -> tuple[str, ...]:
    for key in ("operations", "inputs", "path", "moves"):
        raw = value.get(key)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return tuple(str(item) for item in raw)
    return ()


def _split_name(group_id: str) -> str:
    bucket = int.from_bytes(blake2b(group_id.encode("utf-8"), digest_size=2).digest(), "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _count_cells(board: Board) -> int:
    return sum(cell is not None for row in board for cell in row)


def _count_garbage(board: Board) -> int:
    return sum(cell == "g" for row in board for cell in row)


@dataclass(frozen=True, slots=True)
class CapturePlacement:
    username: str
    game_id: int | None
    round: int
    sequence: int
    piece_index: int | None
    piece: str
    x: float | None
    y: float | None
    rotation: int
    frame: int
    hold: str | None
    captured_at: int | None
    board40: Board
    operations: tuple[str, ...] = ()

    @property
    def group_id(self) -> str:
        game = "none" if self.game_id is None else str(self.game_id)
        return f"{self.username}|r{self.round}|g{game}"

    @property
    def board24(self) -> Board:
        return normalize_board(self.board40, height=MINOFLUX_BOARD_HEIGHT)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "CapturePlacement":
        piece = str(value.get("piece", "")).strip().upper()
        if piece not in {"I", "O", "T", "S", "Z", "J", "L"}:
            raise ValueError(f"Unsupported capture piece: {piece!r}")
        hold_raw = value.get("hold")
        hold = None if hold_raw is None else str(hold_raw).strip().upper() or None
        return cls(
            username=str(value.get("username", "unknown")),
            game_id=_optional_int(value.get("gameid", value.get("gameId"))),
            round=max(1, int(value.get("round", 1))),
            sequence=int(value.get("sequence", 0)),
            piece_index=_optional_int(value.get("pieceIndex", value.get("piece_index"))),
            piece=piece,
            x=_optional_float(value.get("x")),
            y=_optional_float(value.get("y")),
            rotation=int(value.get("rotation", 0)) % 4,
            frame=int(value.get("frame", 0)),
            hold=hold,
            captured_at=_optional_int(value.get("capturedAt", value.get("captured_at"))),
            board40=normalize_board(value.get("board")),
            operations=_operations(value),
        )


@dataclass(frozen=True, slots=True)
class CaptureSample:
    group_id: str
    split: str
    username: str
    game_id: int | None
    round: int
    sequence: int
    frame: int
    frame_delta: int | None
    piece_index: int | None
    piece: str
    x: float | None
    y: float | None
    rotation: int
    hold_before: str | None
    hold_after: str | None
    used_hold: bool | None
    next_placed_piece: str | None
    operations: tuple[str, ...]
    board_before: Board | None
    board_after: Board
    estimated_lines: int | None
    transition_confidence: str

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["format"] = CAPTURE_DATASET_FORMAT
        value["operations"] = list(self.operations)
        value["board_before"] = None if self.board_before is None else [list(row) for row in self.board_before]
        value["board_after"] = [list(row) for row in self.board_after]
        return value


def load_tetrio_capture(path: str | Path) -> tuple[CapturePlacement, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw = payload.get("placements", payload.get("records", payload.get("data")))
    else:
        raw = payload
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("TETR.IO capture must contain a placements array")
    placements = [
        CapturePlacement.from_mapping(item)
        for item in raw
        if isinstance(item, Mapping)
    ]
    placements.sort(key=lambda item: (item.group_id, item.sequence, item.frame, item.piece_index or -1))
    return tuple(placements)


def _transition_info(before: Board | None, after: Board) -> tuple[int | None, str]:
    if before is None:
        return None, "first-observation"
    before_cells = _count_cells(before)
    after_cells = _count_cells(after)
    garbage_delta = _count_garbage(after) - _count_garbage(before)
    numerator = before_cells + 4 - after_cells
    if garbage_delta == 0 and numerator >= 0 and numerator % BOARD_WIDTH == 0:
        lines = numerator // BOARD_WIDTH
        if 0 <= lines <= 4:
            return lines, "clean-count-inference"
    if garbage_delta > 0:
        return None, "external-garbage-change"
    return None, "unresolved-board-change"


def build_capture_samples(
    placements: Iterable[CapturePlacement],
    *,
    username: str | None = None,
) -> tuple[CaptureSample, ...]:
    filtered = [
        item for item in placements
        if username is None or item.username.casefold() == username.casefold()
    ]
    groups: dict[str, list[CapturePlacement]] = {}
    for item in filtered:
        groups.setdefault(item.group_id, []).append(item)

    samples: list[CaptureSample] = []
    for group_id, items in sorted(groups.items()):
        items.sort(key=lambda item: (item.sequence, item.frame, item.piece_index or -1))
        split = _split_name(group_id)
        for index, item in enumerate(items):
            previous = items[index - 1] if index else None
            following = items[index + 1] if index + 1 < len(items) else None
            board_before = previous.board24 if previous is not None else None
            board_after = item.board24
            estimated_lines, confidence = _transition_info(board_before, board_after)
            hold_before = previous.hold if previous is not None else None
            used_hold = None if previous is None else hold_before != item.hold
            samples.append(
                CaptureSample(
                    group_id=group_id,
                    split=split,
                    username=item.username,
                    game_id=item.game_id,
                    round=item.round,
                    sequence=item.sequence,
                    frame=item.frame,
                    frame_delta=None if previous is None else max(0, item.frame - previous.frame),
                    piece_index=item.piece_index,
                    piece=item.piece,
                    x=item.x,
                    y=item.y,
                    rotation=item.rotation,
                    hold_before=hold_before,
                    hold_after=item.hold,
                    used_hold=used_hold,
                    next_placed_piece=following.piece if following is not None else None,
                    operations=item.operations,
                    board_before=board_before,
                    board_after=board_after,
                    estimated_lines=estimated_lines,
                    transition_confidence=confidence,
                )
            )
    samples.sort(key=lambda item: (item.group_id, item.sequence, item.frame))
    return tuple(samples)


def capture_summary(samples: Sequence[CaptureSample]) -> dict[str, object]:
    splits = {name: sum(item.split == name for item in samples) for name in ("train", "validation", "test")}
    confidence: dict[str, int] = {}
    users: set[str] = set()
    groups: set[str] = set()
    operation_samples = 0
    for item in samples:
        users.add(item.username)
        groups.add(item.group_id)
        confidence[item.transition_confidence] = confidence.get(item.transition_confidence, 0) + 1
        operation_samples += int(bool(item.operations))
    return {
        "format": CAPTURE_DATASET_FORMAT,
        "samples": len(samples),
        "groups": len(groups),
        "users": sorted(users),
        "splits": splits,
        "transitionConfidence": dict(sorted(confidence.items())),
        "samplesWithOperations": operation_samples,
    }


def save_capture_dataset(
    output: str | Path,
    samples: Sequence[CaptureSample],
    *,
    summary_path: str | Path | None = None,
) -> tuple[Path, Path]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for sample in samples:
            stream.write(json.dumps(sample.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_target = Path(summary_path) if summary_path is not None else target.with_suffix(".summary.json")
    summary_target.parent.mkdir(parents=True, exist_ok=True)
    summary_target.write_text(
        json.dumps(capture_summary(samples), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target, summary_target
