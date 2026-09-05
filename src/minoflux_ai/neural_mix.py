from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .neural_dataset import NEURAL_DATASET_FORMAT


def _record_key(record: Mapping[str, object]) -> tuple[int, int]:
    return int(record.get("seed", 0)), int(record.get("pieceIndex", 0))


def merge_neural_datasets(
    output_path: str | Path,
    input_paths: Sequence[str | Path],
    *,
    deduplicate: bool = True,
) -> dict[str, object]:
    """Merge ranking JSONL files; later files replace duplicate positions."""

    if not input_paths:
        raise ValueError("At least one input dataset is required")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    read_records = 0
    if deduplicate:
        records: dict[tuple[int, int], dict[str, object]] = {}
        for raw_path in input_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or value.get("format") != NEURAL_DATASET_FORMAT:
                        raise ValueError(f"Invalid neural dataset record in {path} at line {line_number}")
                    read_records += 1
                    records[_record_key(value)] = value
        merged: Iterable[dict[str, object]] = records.values()
        written = len(records)
    else:
        collected: list[dict[str, object]] = []
        for raw_path in input_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or value.get("format") != NEURAL_DATASET_FORMAT:
                        raise ValueError(f"Invalid neural dataset record in {path} at line {line_number}")
                    read_records += 1
                    collected.append(value)
        merged = collected
        written = len(collected)

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in merged:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    temporary.replace(output)

    result = {
        "format": NEURAL_DATASET_FORMAT,
        "path": str(output),
        "inputs": [str(Path(path)) for path in input_paths],
        "readRecords": read_records,
        "writtenRecords": written,
        "deduplicated": bool(deduplicate),
        "duplicatesRemoved": read_records - written,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
