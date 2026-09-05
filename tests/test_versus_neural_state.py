from __future__ import annotations

import json

from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_neural import (
    VERSUS_SELFPLAY_FORMAT,
    VersusSelfPlayConfig,
    VersusValueConfig,
    encode_versus_state,
    generate_versus_selfplay_dataset,
)
from minoflux_ai.versus_search import VersusSearchConfig
from minoflux_engine import VersusMatch


class _FlatScorer:
    def score_many(self, game, evaluations):
        return tuple(0.0 for _evaluation in evaluations)


def test_versus_state_encoding_has_two_perspectives_and_turn_context() -> None:
    match = VersusMatch(20260903)
    match.player.pending.enqueue(4, 2)
    match.ai.pending.enqueue(2, 7)
    config = VersusValueConfig()

    player = encode_versus_state(match, "player", config, "player")
    ai = encode_versus_state(match, "ai", config, "player")
    player_reply_turn = encode_versus_state(match, "player", config, "ai")

    assert len(player.own_rows) == config.board_height
    assert len(player.opponent_rows) == config.board_height
    assert len(player.context) == config.context_size
    assert player.own_rows == ai.opponent_rows
    assert player.opponent_rows == ai.own_rows
    assert player.context != ai.context
    assert player.context[-1] == 1.0
    assert ai.context[-1] == 0.0
    assert player_reply_turn.context[-1] == 0.0
    assert player.context != player_reply_turn.context


def test_tiny_selfplay_dataset_contains_outcomes_and_terminal_states(tmp_path) -> None:
    output = tmp_path / "versus.jsonl"
    result = generate_versus_selfplay_dataset(
        output,
        _FlatScorer(),
        VersusSelfPlayConfig(
            games=1,
            max_turns=2,
            search_config=VersusSearchConfig(
                placement_search=SearchConfig(
                    allow_hold=False,
                    lookahead_pieces=0,
                    beam_width=1,
                    srs_reachable=False,
                ),
                candidate_width=2,
                opponent_reply_width=0,
            ),
        ),
    )

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["records"] == len(records)
    assert records
    assert all(record["format"] == VERSUS_SELFPLAY_FORMAT for record in records)
    assert all(record["outcome"] in (-1.0, 0.0, 1.0) for record in records)
    assert sum(bool(record["terminal"]) for record in records) == 2
    assert all(len(record["context"]) == VersusValueConfig().context_size for record in records)
