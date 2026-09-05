from __future__ import annotations

import json

import pytest

from minoflux_ai.human_placement import (
    HumanDatasetConfig,
    HumanPlacementRecorder,
    write_human_ranking_dataset,
)
from minoflux_ai.neural import encode_game_state
from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT, pack_board_rows
from minoflux_engine import Game


def _read_one(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    return json.loads(lines[0])


@pytest.mark.parametrize("use_hold", [False, True])
def test_local_human_play_converts_to_exact_ranking_sample(tmp_path, use_hold: bool) -> None:
    raw = tmp_path / "human.jsonl"
    dataset = tmp_path / "ranking.jsonl"
    game = Game(12345)
    recorder = HumanPlacementRecorder(raw, session_id=777)
    recorder.begin_game(game)

    if use_hold:
        assert game.hold()

    choice = recorder.capture_choice(game, hard_drop=True)
    result = game.hard_drop()
    recorder.record_lock(choice, result, game)

    summary = write_human_ranking_dataset(raw, dataset)
    assert summary["records"] == 1
    assert summary["samples"] == 1
    assert summary["skipped"] == {}

    record = _read_one(dataset)
    assert record["format"] == NEURAL_DATASET_FORMAT
    assert record["teacher"] == "human-local"
    assert record["humanSource"] == {"sessionId": 777, "gameIndex": 0}
    assert record["expertIndex"] in record["expertIndices"]

    expert = record["candidates"][record["expertIndex"]]
    assert bool(expert["move"][0]) is use_hold
    assert expert["samplingBucket"] == "human-expert"

    actual = encode_game_state(game)
    assert expert["rows"] == list(pack_board_rows(actual.board))
    assert expert["context"] == pytest.approx(actual.context)


def test_human_candidate_cap_keeps_expert(tmp_path) -> None:
    raw = tmp_path / "human.jsonl"
    dataset = tmp_path / "ranking.jsonl"
    game = Game(4567)
    recorder = HumanPlacementRecorder(raw, session_id=888)
    recorder.begin_game(game)

    choice = recorder.capture_choice(game, hard_drop=True)
    result = game.hard_drop()
    recorder.record_lock(choice, result, game)

    summary = write_human_ranking_dataset(
        raw,
        dataset,
        HumanDatasetConfig(max_candidates=8),
    )
    assert summary["samples"] == 1
    record = _read_one(dataset)
    assert len(record["candidates"]) <= 8
    assert record["expertIndex"] in record["expertIndices"]
    assert record["candidates"][record["expertIndex"]]["samplingBucket"] == "human-expert"
