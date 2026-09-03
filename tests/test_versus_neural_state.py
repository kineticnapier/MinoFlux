from __future__ import annotations

from minoflux_ai.versus_neural import VersusValueConfig, encode_versus_state
from minoflux_engine import VersusMatch


def test_versus_state_encoding_has_two_perspectives() -> None:
    match = VersusMatch(20260903)
    match.player.pending.enqueue(4, 2)
    match.ai.pending.enqueue(2, 7)
    config = VersusValueConfig()

    player = encode_versus_state(match, "player", config)
    ai = encode_versus_state(match, "ai", config)

    assert len(player.own_rows) == config.board_height
    assert len(player.opponent_rows) == config.board_height
    assert len(player.context) == config.context_size
    assert player.own_rows == ai.opponent_rows
    assert player.opponent_rows == ai.own_rows
    assert player.context != ai.context
