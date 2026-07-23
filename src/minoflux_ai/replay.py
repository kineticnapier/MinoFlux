from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from minoflux_engine import Game, LockResult, Placement

REPLAY_FORMAT = "minoflux_replay_v3"
LEGACY_REPLAY_FORMAT = "minoflux_replay_v1"
LEGACY_REPLAY_FORMAT_V2 = "minoflux_replay_v2"
SUPPORTED_REPLAY_FORMATS = (REPLAY_FORMAT, LEGACY_REPLAY_FORMAT_V2, LEGACY_REPLAY_FORMAT)


@dataclass(frozen=True, slots=True)
class ReplayStep:
    piece: str
    x: int
    y: int
    rotation: int
    lines: int
    attack: int
    score: int
    total_lines: int
    total_attack: int
    hold: bool = False
    spin: str | None = None
    perfect_clear: bool = False
    path: tuple[str, ...] = ()
    last_move_was_rotation: bool = False
    rotation_kick_index: int | None = None
    rotation_from: int | None = None
    rotation_to: int | None = None

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["path"] = list(self.path)
        return value

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReplayStep":
        raw_spin = value.get("spin")
        raw_path = value.get("path", ())
        if not isinstance(raw_path, Sequence) or isinstance(raw_path, (str, bytes)):
            raw_path = ()
        return cls(
            piece=str(value["piece"]),
            x=int(value["x"]),
            y=int(value["y"]),
            rotation=int(value["rotation"]),
            lines=int(value.get("lines", 0)),
            attack=int(value.get("attack", 0)),
            score=int(value.get("score", 0)),
            total_lines=int(value.get("total_lines", value.get("totalLines", 0))),
            total_attack=int(value.get("total_attack", value.get("totalAttack", 0))),
            hold=bool(value.get("hold", False)),
            spin=None if raw_spin is None else str(raw_spin),
            perfect_clear=bool(value.get("perfect_clear", value.get("perfectClear", False))),
            path=tuple(str(item) for item in raw_path),
            last_move_was_rotation=bool(
                value.get("last_move_was_rotation", value.get("lastMoveWasRotation", False))
            ),
            rotation_kick_index=(
                None
                if value.get("rotation_kick_index", value.get("rotationKickIndex")) is None
                else int(value.get("rotation_kick_index", value.get("rotationKickIndex")))
            ),
            rotation_from=(
                None
                if value.get("rotation_from", value.get("rotationFrom")) is None
                else int(value.get("rotation_from", value.get("rotationFrom")))
            ),
            rotation_to=(
                None
                if value.get("rotation_to", value.get("rotationTo")) is None
                else int(value.get("rotation_to", value.get("rotationTo")))
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplaySummary:
    pieces: int
    lines: int
    attack: int
    score: int
    topout: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ReplaySummary":
        return cls(
            pieces=int(value["pieces"]),
            lines=int(value["lines"]),
            attack=int(value["attack"]),
            score=int(value["score"]),
            topout=bool(value["topout"]),
        )


@dataclass(frozen=True, slots=True)
class Replay:
    seed: int
    max_pieces: int
    weights: dict[str, float]
    steps: tuple[ReplayStep, ...]
    final: ReplaySummary
    search_config: dict[str, object] | None = None
    format: str = REPLAY_FORMAT

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "format": REPLAY_FORMAT,
            "seed": self.seed,
            "maxPieces": self.max_pieces,
            "weights": dict(self.weights),
            "steps": [step.to_dict() for step in self.steps],
            "final": self.final.to_dict(),
        }
        if self.search_config is not None:
            result["searchConfig"] = dict(self.search_config)
        return result

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Replay":
        replay_format = str(value.get("format", ""))
        if replay_format not in SUPPORTED_REPLAY_FORMATS:
            raise ValueError(f"Unsupported replay format: {replay_format!r}")
        raw_weights = value.get("weights", {})
        raw_steps = value.get("steps", ())
        raw_final = value.get("final")
        raw_search = value.get("searchConfig")
        if not isinstance(raw_weights, Mapping):
            raise ValueError("Replay weights must be an object")
        if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
            raise ValueError("Replay steps must be an array")
        if not isinstance(raw_final, Mapping):
            raise ValueError("Replay final summary must be an object")
        if raw_search is not None and not isinstance(raw_search, Mapping):
            raise ValueError("Replay searchConfig must be an object")
        return cls(
            seed=int(value["seed"]),
            max_pieces=int(value.get("maxPieces", len(raw_steps))),
            weights={str(key): float(item) for key, item in raw_weights.items()},
            steps=tuple(
                ReplayStep.from_mapping(step)
                for step in raw_steps
                if isinstance(step, Mapping)
            ),
            final=ReplaySummary.from_mapping(raw_final),
            search_config=(
                {str(key): item for key, item in raw_search.items()}
                if isinstance(raw_search, Mapping)
                else None
            ),
            format=replay_format,
        )


def save_replay(path: str | Path, replay: Replay) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(replay.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def load_replay(path: str | Path) -> Replay:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Replay root must be an object")
    return Replay.from_mapping(payload)


def apply_replay_step(game: Game, step: ReplayStep, *, strict: bool = True) -> LockResult:
    if step.hold and not game.hold():
        raise ValueError("Replay requested an unavailable Hold")
    if game.current != step.piece:
        raise ValueError(f"Replay expected {step.piece}, but engine produced {game.current}")
    cells = game.cells(step.piece, step.x, step.y, step.rotation)
    placement = Placement(
        step.piece,
        step.x,
        step.y,
        step.rotation,
        cells,
        path=step.path,
        last_move_was_rotation=step.last_move_was_rotation,
        rotation_kick_index=step.rotation_kick_index,
        rotation_from=step.rotation_from,
        rotation_to=step.rotation_to,
    )
    result = game.place(placement)
    if strict:
        actual = (result.lines, result.attack, game.score, game.lines, game.attack)
        expected = (step.lines, step.attack, step.score, step.total_lines, step.total_attack)
        if actual != expected:
            raise ValueError(f"Replay state mismatch: expected {expected}, got {actual}")
        if result.spin != step.spin:
            raise ValueError(f"Replay spin mismatch: expected {step.spin!r}, got {result.spin!r}")
        if result.perfect_clear != step.perfect_clear:
            raise ValueError(
                f"Replay perfect-clear mismatch: expected {step.perfect_clear}, got {result.perfect_clear}"
            )
    return result


def replay_to_game(replay: Replay, steps: int | None = None, *, strict: bool = True) -> Game:
    game = Game(replay.seed)
    count = len(replay.steps) if steps is None else max(0, min(len(replay.steps), int(steps)))
    exact = strict and replay.format == REPLAY_FORMAT
    for step in replay.steps[:count]:
        apply_replay_step(game, step, strict=exact)
    return game
