from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import os


DEFAULT_BINDINGS: dict[str, str] = {
    "left": "left",
    "right": "right",
    "soft_drop": "down",
    "rotate_ccw": "z",
    "rotate_cw": "x",
    "rotate_180": "a",
    "hold": "c",
    "hard_drop": "space",
    "pause": "p",
    "restart": "r",
}

ACTION_LABELS: dict[str, str] = {
    "left": "Move left",
    "right": "Move right",
    "soft_drop": "Soft drop",
    "rotate_ccw": "Rotate CCW",
    "rotate_cw": "Rotate CW",
    "rotate_180": "Rotate 180",
    "hold": "Hold",
    "hard_drop": "Hard drop",
    "pause": "Pause",
    "restart": "Restart",
}


@dataclass(slots=True)
class GameSettings:
    """Player handling and one-key-per-action bindings.

    Values are milliseconds. ``arr_ms == 0`` means instant horizontal
    movement after DAS. ``soft_drop_ms == 0`` means instant soft drop to the
    floor without locking.
    """

    das_ms: int = 133
    arr_ms: int = 16
    soft_drop_ms: int = 25
    bindings: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))

    def normalize(self) -> "GameSettings":
        self.das_ms = max(0, min(1000, int(self.das_ms)))
        self.arr_ms = max(0, min(500, int(self.arr_ms)))
        self.soft_drop_ms = max(0, min(500, int(self.soft_drop_ms)))
        normalized = dict(DEFAULT_BINDINGS)
        for action in DEFAULT_BINDINGS:
            value = self.bindings.get(action)
            if isinstance(value, str) and value.strip():
                normalized[action] = value.strip().lower()
        self.bindings = normalized
        return self

    def to_json(self) -> dict[str, Any]:
        return asdict(self.normalize())

    @classmethod
    def from_json(cls, value: object) -> "GameSettings":
        if not isinstance(value, dict):
            return cls()
        bindings = value.get("bindings")
        return cls(
            das_ms=value.get("das_ms", 133),
            arr_ms=value.get("arr_ms", 16),
            soft_drop_ms=value.get("soft_drop_ms", 25),
            bindings=dict(bindings) if isinstance(bindings, dict) else dict(DEFAULT_BINDINGS),
        ).normalize()


def settings_path() -> Path:
    override = os.environ.get("MINOFLUX_SETTINGS")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".minoflux" / "settings.json"


def load_settings(path: Path | None = None) -> GameSettings:
    target = path or settings_path()
    try:
        return GameSettings.from_json(json.loads(target.read_text(encoding="utf-8")))
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        return GameSettings()


def save_settings(settings: GameSettings, path: Path | None = None) -> Path:
    target = path or settings_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(settings.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
