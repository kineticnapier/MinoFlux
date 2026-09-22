from __future__ import annotations

import json
from pathlib import Path
import tempfile

from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT
from minoflux_ai.oracle import OracleConfig
from minoflux_ai.oracle_dagger import write_oracle_dagger_dataset
from minoflux_ai.oracle_dataset import ORACLE_TEACHER_NAME, OracleDatasetConfig
from minoflux_ai.search import SearchConfig


class _HeuristicLikeScorer:
    def score_many(self, game, evaluations):
        return [evaluation.score for evaluation in evaluations]


def test_oracle_dagger_labels_learner_states_with_native_oracle() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oracle-dagger.jsonl"
        result = write_oracle_dagger_dataset(
            path,
            _HeuristicLikeScorer(),
            OracleDatasetConfig(
                games=1,
                max_pieces=3,
                max_candidates=4,
                oracle=OracleConfig(
                    beam_width=16,
                    depth=1,
                    allow_180=False,
                    reachability_node_limit=8000,
                ),
            ),
            learner_search=SearchConfig(
                allow_hold=True,
                lookahead_pieces=0,
                beam_width=1,
                srs_reachable=True,
                allow_180=False,
                reachability_node_limit=8000,
            ),
            sample_rate=1.0,
            max_samples=1,
            progress_every=1,
        )

        assert result["teacher"] == ORACLE_TEACHER_NAME
        assert result["trajectory"] == "neural"
        assert result["source"] == "oracle_dagger"
        assert result["samples"] == 1
        assert result["oracleQueries"] == 1
        assert result["visitedStates"] == 1
        assert result["candidates"] >= 1

        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        assert record["format"] == NEURAL_DATASET_FORMAT
        assert record["source"] == "oracle_dagger"
        assert record["teacher"] == ORACLE_TEACHER_NAME
        assert "random_control" in record["daggerReasons"]
        assert 0 <= record["expertIndex"] < len(record["candidates"])
        assert record["expertIndices"] == [record["expertIndex"]]

        metadata = json.loads(
            path.with_suffix(path.suffix + ".meta.json").read_text(encoding="utf-8")
        )
        assert metadata["oracleQueries"] == 1
        assert metadata["selection"]["sampleRate"] == 1.0
        assert metadata["config"]["oracle"]["beamWidth"] == 16
