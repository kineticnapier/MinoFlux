from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4
import json


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(slots=True)
class RunDirectory:
    path: Path

    def save_result(self, value: Any, filename: str = "result.json") -> Path:
        target = self.path / filename
        _write_json(target, value)
        return target

    def append_metric(self, metric: Mapping[str, Any]) -> Path:
        target = self.path / "metrics.jsonl"
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **dict(metric)}
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return target


class RunStore:
    def __init__(self, root: str | Path = "data/runs") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, kind: str, config: Mapping[str, Any]) -> RunDirectory:
        safe_kind = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in kind).strip("-") or "run"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = self.root / f"{stamp}-{safe_kind}-{uuid4().hex[:6]}"
        path.mkdir(parents=True, exist_ok=False)
        _write_json(path / "config.json", dict(config))
        return RunDirectory(path)
