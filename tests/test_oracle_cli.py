from __future__ import annotations

import json

from minoflux.neural_dispatch_cli import main
from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT


def test_oracle_smoke_cli_runs_native_policy(capsys) -> None:
    code = main(
        [
            "oracle-smoke",
            "--games",
            "1",
            "--max-pieces",
            "3",
            "--beam",
            "16",
            "--depth",
            "1",
            "--no-180",
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["teacher"] == "minoflux-native-oracle"
    assert result["games"] == 1
    assert result["pieces"] == 3
    assert result["topouts"] == 0


def test_oracle_profile_cli_reports_native_hotspots(capsys) -> None:
    code = main(
        [
            "oracle-profile",
            "--seed",
            "8100001",
            "--beam",
            "16",
            "--depth",
            "2",
            "--no-180",
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["teacher"] == "minoflux-native-oracle"
    assert result["searchSeconds"] > 0.0
    assert 0.0 <= result["reachabilitySeconds"] <= result["searchSeconds"]
    assert result["movegenCalls"] > 0
    assert result["generatedMoves"] > 0
    assert result["featureCalls"] > 0
    assert result["featureSeconds"] >= 0.0
    assert result["dedupSeconds"] >= 0.0
    assert result["pruneSeconds"] >= 0.0
    assert result["expandedChildren"] > 0
    assert result["reachability"]["bfsSeconds"] >= 0.0
    assert result["reachability"]["landingSeconds"] >= 0.0


def test_oracle_dataset_cli_writes_trainable_ranking_jsonl(tmp_path, capsys) -> None:
    target = tmp_path / "oracle.jsonl"

    code = main(
        [
            "oracle-dataset",
            "--output",
            str(target),
            "--games",
            "1",
            "--max-pieces",
            "2",
            "--beam",
            "16",
            "--depth",
            "1",
            "--max-candidates",
            "8",
            "--workers",
            "1",
            "--no-180",
        ]
    )

    assert code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["teacher"] == "minoflux-native-oracle"
    assert summary["samples"] == 2
    assert summary["candidates"] >= 2

    records = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["format"] == NEURAL_DATASET_FORMAT
        assert 0 <= record["expertIndex"] < len(record["candidates"])
        assert record["expertIndices"] == [record["expertIndex"]]

    metadata = json.loads(
        target.with_suffix(target.suffix + ".meta.json").read_text(encoding="utf-8")
    )
    assert metadata["teacher"] == "minoflux-native-oracle"
    assert metadata["config"]["oracle"]["beamWidth"] == 16
    assert metadata["config"]["oracle"]["depth"] == 1
