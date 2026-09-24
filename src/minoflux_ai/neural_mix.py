from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .neural_dataset import NEURAL_DATASET_FORMAT


def _record_key(record: Mapping[str, object]) -> tuple[int, int]:
    return int(record.get("seed", 0)), int(record.get("pieceIndex", 0))


def _normalized_input_weights(
    input_paths: Sequence[str | Path],
    input_weights: Sequence[float] | None,
) -> tuple[float, ...] | None:
    if input_weights is None:
        return None
    if len(input_weights) != len(input_paths):
        raise ValueError("input_weights must match the number of input datasets")
    weights = tuple(float(weight) for weight in input_weights)
    if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
        raise ValueError("input_weights must be finite and non-negative")
    return weights


def _apply_input_weight(record: dict[str, object], weight: float | None) -> dict[str, object]:
    if weight is None:
        return record
    existing = float(record.get("trainingWeight", 1.0))
    if not math.isfinite(existing) or existing < 0.0:
        raise ValueError("trainingWeight must be finite and non-negative")
    weighted = dict(record)
    weighted["trainingWeight"] = existing * weight
    return weighted


def merge_neural_datasets(
    output_path: str | Path,
    input_paths: Sequence[str | Path],
    *,
    deduplicate: bool = True,
    input_weights: Sequence[float] | None = None,
) -> dict[str, object]:
    """Merge ranking JSONL files; later files replace duplicate positions.

    Optional input weights are stored as per-record ``trainingWeight`` values.
    Nested weighted mixes compose multiplicatively, while unweighted mixes preserve
    any existing record weight unchanged.
    """

    if not input_paths:
        raise ValueError("At least one input dataset is required")
    normalized_weights = _normalized_input_weights(input_paths, input_weights)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    read_records = 0
    if deduplicate:
        records: dict[tuple[int, int], dict[str, object]] = {}
        for input_index, raw_path in enumerate(input_paths):
            path = Path(raw_path)
            weight = None if normalized_weights is None else normalized_weights[input_index]
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or value.get("format") != NEURAL_DATASET_FORMAT:
                        raise ValueError(f"Invalid neural dataset record in {path} at line {line_number}")
                    read_records += 1
                    value = _apply_input_weight(value, weight)
                    records[_record_key(value)] = value
        merged: Iterable[dict[str, object]] = records.values()
        written = len(records)
    else:
        collected: list[dict[str, object]] = []
        for input_index, raw_path in enumerate(input_paths):
            path = Path(raw_path)
            weight = None if normalized_weights is None else normalized_weights[input_index]
            with path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or value.get("format") != NEURAL_DATASET_FORMAT:
                        raise ValueError(f"Invalid neural dataset record in {path} at line {line_number}")
                    read_records += 1
                    collected.append(_apply_input_weight(value, weight))
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
        "inputWeights": None if normalized_weights is None else list(normalized_weights),
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
