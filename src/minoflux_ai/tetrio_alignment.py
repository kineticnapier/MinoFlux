from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

from minoflux_engine import Game

from .reachability import reachable_placements
from .search import clone_game
from .tetrio_capture import Board, CaptureSample

ALIGNMENT_FORMAT = "minoflux_tetrio_alignment_v1"


@dataclass(frozen=True, slots=True)
class CaptureAlignment:
    group_id: str
    sequence: int
    piece: str
    status: str
    candidate_count: int
    x: int | None = None
    y: int | None = None
    rotation: int | None = None
    path: tuple[str, ...] = ()
    last_move_was_rotation: bool = False
    rotation_kick_index: int | None = None
    rotation_from: int | None = None
    rotation_to: int | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["format"] = ALIGNMENT_FORMAT
        value["path"] = list(self.path)
        return value


def _board_tuple(game: Game) -> Board:
    return tuple(tuple(None if cell is None else str(cell).lower() for cell in row) for row in game.board)


def _capture_game(sample: CaptureSample) -> Game | None:
    if sample.board_before is None:
        return None
    game = Game(0)
    game.board = [list(row) for row in sample.board_before]
    game.current = sample.piece
    game.x, game.y, game.rotation = 3, 1, 0
    game.hold_piece = sample.hold_before
    game.hold_used = False
    game.game_over = game._collides(game.current, game.x, game.y, game.rotation)
    game.paused = False
    game.combo = -1
    game.back_to_back = False
    game.b2b_chain = 0
    game.surge_charge = 0
    game.last_lock = None
    return None if game.game_over else game


def align_capture_sample(
    sample: CaptureSample,
    *,
    allow_180: bool = True,
    max_nodes: int = 8_000,
) -> CaptureAlignment:
    game = _capture_game(sample)
    if game is None:
        return CaptureAlignment(sample.group_id, sample.sequence, sample.piece, "no-before-board", 0)

    matches = []
    for placement in reachable_placements(game, allow_180=allow_180, max_nodes=max_nodes):
        simulated = clone_game(game)
        try:
            simulated.place(placement)
        except (RuntimeError, ValueError):
            continue
        if _board_tuple(simulated) == sample.board_after:
            matches.append(placement)

    if not matches:
        return CaptureAlignment(sample.group_id, sample.sequence, sample.piece, "unmatched", 0)
    matches.sort(
        key=lambda placement: (
            len(placement.path),
            int(not placement.last_move_was_rotation),
            placement.rotation,
            placement.x,
            placement.y,
        )
    )
    placement = matches[0]
    status = "exact" if len(matches) == 1 else "ambiguous"
    return CaptureAlignment(
        group_id=sample.group_id,
        sequence=sample.sequence,
        piece=sample.piece,
        status=status,
        candidate_count=len(matches),
        x=placement.x,
        y=placement.y,
        rotation=placement.rotation,
        path=placement.path,
        last_move_was_rotation=placement.last_move_was_rotation,
        rotation_kick_index=placement.rotation_kick_index,
        rotation_from=placement.rotation_from,
        rotation_to=placement.rotation_to,
    )


def align_capture_samples(
    samples: Sequence[CaptureSample],
    *,
    allow_180: bool = True,
    max_nodes: int = 8_000,
) -> tuple[CaptureAlignment, ...]:
    return tuple(
        align_capture_sample(sample, allow_180=allow_180, max_nodes=max_nodes)
        for sample in samples
    )


def alignment_summary(alignments: Sequence[CaptureAlignment]) -> dict[str, object]:
    statuses: dict[str, int] = {}
    for item in alignments:
        statuses[item.status] = statuses.get(item.status, 0) + 1
    aligned = sum(item.status in ("exact", "ambiguous") for item in alignments)
    return {
        "format": ALIGNMENT_FORMAT,
        "samples": len(alignments),
        "aligned": aligned,
        "alignmentRate": aligned / len(alignments) if alignments else 0.0,
        "statuses": dict(sorted(statuses.items())),
    }


def save_alignments(
    output: str | Path,
    alignments: Sequence[CaptureAlignment],
) -> tuple[Path, Path]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for item in alignments:
            stream.write(json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = target.with_suffix(".summary.json")
    summary.write_text(
        json.dumps(alignment_summary(alignments), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target, summary
