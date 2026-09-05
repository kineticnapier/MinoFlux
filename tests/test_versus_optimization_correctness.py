from __future__ import annotations

import json
from typing import Sequence

from minoflux_ai.reachability import reachable_placements
from minoflux_ai.search import (
    SearchAction,
    SearchConfig,
    apply_search_action,
    clone_game,
    rank_search_actions,
)
from minoflux_ai.versus_search import (
    VersusSearchConfig,
    VersusSearchRequest,
    _simulate_action,
    choose_versus_action,
    choose_versus_actions_batch,
    clone_versus_match,
)
from minoflux_ai.versus_neural import (
    VERSUS_SELFPLAY_FORMAT,
    VersusSelfPlayConfig,
    VersusValueConfig,
    generate_versus_selfplay_dataset,
)
from minoflux_engine import Game, VersusMatch


class _DeterministicPlacementScorer:
    """Exercise both serial and grouped scoring without floating-point reductions."""

    @staticmethod
    def _scores(placements) -> tuple[float, ...]:
        return tuple(
            float(placement.x)
            + 0.01 * float(placement.rotation)
            + 0.0001 * float(placement.y)
            for placement in placements
        )

    def score_placements(self, game, placements):
        return self._scores(placements)

    def score_placement_groups(self, groups):
        return tuple(self._scores(placements) for _game, placements in groups)

    def score_many(self, game, evaluations):
        return self._scores(tuple(evaluation.placement for evaluation in evaluations))


class _DeterministicMatchScorer:
    @staticmethod
    def score_match(match, root_side, to_move=None) -> float:
        own = match.side(root_side)
        opponent = match.opponent(root_side)
        turn_term = 0.125 if to_move == root_side else -0.125
        return (
            0.01 * float(own.sent - opponent.sent)
            + 0.001 * float(opponent.pending.pending_lines - own.pending.pending_lines)
            + turn_term
        )

    def score_matches(self, entries):
        return tuple(self.score_match(*entry) for entry in entries)


class _UnexpectedReplyScorer:
    def score_placements(self, game, placements):
        raise AssertionError("opponent scorer must not run when reply width is zero")

    def score_placement_groups(self, groups):
        raise AssertionError("opponent scorer must not run when reply width is zero")

    def score_many(self, game, evaluations):
        raise AssertionError("opponent scorer must not run when reply width is zero")


def _search_config(*, reply_width: int = 4) -> VersusSearchConfig:
    return VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=1,
            srs_reachable=True,
        ),
        candidate_width=16,
        opponent_reply_width=reply_width,
    )


def _matches() -> tuple[VersusMatch, ...]:
    matches = (VersusMatch(811), VersusMatch(829))
    for index, match in enumerate(matches):
        match.player.pending.enqueue(index + 1, 3 + index)
    return matches


def _requests(
    matches: Sequence[VersusMatch],
    *,
    config: VersusSearchConfig,
    scorer,
    state_scorer,
    opponent_scorer=None,
) -> tuple[VersusSearchRequest, ...]:
    return tuple(
        VersusSearchRequest(
            match=match,
            side_name="player",
            config=config,
            scorer=scorer,
            opponent_scorer=(scorer if opponent_scorer is None else opponent_scorer),
            state_scorer=state_scorer,
        )
        for match in matches
    )


def _choice_signature(choice):
    if choice is None:
        return None
    return (
        choice.action.to_dict(),
        choice.score,
        choice.immediate,
        choice.resolution,
        None if choice.opponent_reply is None else choice.opponent_reply.to_dict(),
    )


def _side_signature(side) -> tuple[object, ...]:
    game = side.game
    return (
        game.snapshot(queue_size=len(game.queue)),
        game.last_action,
        tuple(game.queue),
        tuple(game._bag._queue),
        game._bag._rng.getstate(),
        tuple((packet.lines, packet.hole) for packet in side.pending.packets),
        side.sent,
        side.received,
        side.canceled,
        side.garbage_applied,
    )


def _match_signature(match: VersusMatch) -> tuple[object, ...]:
    return (
        match.seed,
        match.garbage_cap,
        match.winner,
        _side_signature(match.player),
        _side_signature(match.ai),
        match._garbage_rng.getstate(),
    )


def _execute_path(game: Game, action: SearchAction):
    if action.use_hold:
        assert game.hold()
    placement = action.placement
    assert placement.path
    assert placement.path[-1] == "hard_drop"
    for command in placement.path[:-1]:
        if command == "left":
            succeeded = game.move_left()
        elif command == "right":
            succeeded = game.move_right()
        elif command == "down":
            succeeded = game.soft_drop()
        elif command == "cw":
            succeeded = game.rotate_cw()
        elif command == "ccw":
            succeeded = game.rotate_ccw()
        elif command == "180":
            succeeded = game.rotate_180()
        else:
            raise AssertionError(f"unknown reachability command: {command}")
        assert succeeded

    assert game.current == placement.piece
    assert game.x == placement.x
    assert game.rotation == placement.rotation
    assert game.ghost_y() == placement.y
    assert game.last_move_was_rotation == placement.last_move_was_rotation
    assert game.last_rotation_kick_index == placement.rotation_kick_index
    return game.hard_drop()


def test_exact_srs_width_16_reply_4_matches_serial_and_repeats() -> None:
    config = _search_config()

    serial_matches = _matches()
    serial_scorer = _DeterministicPlacementScorer()
    serial_state = _DeterministicMatchScorer()
    serial = tuple(
        choose_versus_action(
            match,
            "player",
            config=config,
            scorer=serial_scorer,
            opponent_scorer=serial_scorer,
            state_scorer=serial_state,
        )
        for match in serial_matches
    )

    def batched_run():
        scorer = _DeterministicPlacementScorer()
        state_scorer = _DeterministicMatchScorer()
        matches = _matches()
        choices = choose_versus_actions_batch(
            _requests(
                matches,
                config=config,
                scorer=scorer,
                state_scorer=state_scorer,
            )
        )
        return matches, choices

    first_matches, first = batched_run()
    _second_matches, second = batched_run()

    assert tuple(map(_choice_signature, first)) == tuple(map(_choice_signature, serial))
    assert tuple(map(_choice_signature, second)) == tuple(map(_choice_signature, first))
    assert all(choice is not None for choice in first)
    assert all(choice.opponent_reply is not None for choice in first if choice is not None)
    assert any(choice.action.use_hold for choice in first if choice is not None)

    for match, choice in zip(first_matches, first):
        assert choice is not None
        replayed = clone_game(match.player.game)
        replay_result = _execute_path(replayed, choice.action)
        direct = clone_game(match.player.game)
        direct_result = apply_search_action(direct, choice.action)
        assert replay_result == direct_result
        assert replayed.board == direct.board
        assert replayed.current == direct.current
        assert replayed.hold_piece == direct.hold_piece
        assert tuple(replayed.queue) == tuple(direct.queue)

        assert choice.opponent_reply is not None
        reply_replayed = clone_game(match.ai.game)
        reply_result = _execute_path(reply_replayed, choice.opponent_reply)
        reply_direct = clone_game(match.ai.game)
        direct_reply_result = apply_search_action(reply_direct, choice.opponent_reply)
        assert reply_result == direct_reply_result
        assert reply_replayed.board == reply_direct.board


def test_compact_neural_ranking_matches_full_metadata_and_restores_choice() -> None:
    match = VersusMatch(1871)
    scorer = _DeterministicPlacementScorer()
    placement_config = _search_config().placement_search
    full = rank_search_actions(
        match.player.game,
        config=placement_config,
        limit=16,
        scorer=scorer,
    )
    compact = rank_search_actions(
        match.player.game,
        config=placement_config,
        limit=16,
        scorer=scorer,
        _ranking_only=True,
    )

    assert tuple(action for action, _evaluation in compact) == tuple(
        action for action, _evaluation in full
    )
    assert tuple(evaluation.score for _action, evaluation in compact) == tuple(
        evaluation.score for _action, evaluation in full
    )
    choice = choose_versus_action(
        match,
        "player",
        config=_search_config(),
        scorer=scorer,
        opponent_scorer=scorer,
        state_scorer=_DeterministicMatchScorer(),
    )
    assert choice is not None
    full_evaluation = next(
        evaluation for action, evaluation in full if action == choice.action
    )
    assert choice.immediate == full_evaluation


def test_reply_width_zero_matches_serial_and_skips_opponent_ranking() -> None:
    config = _search_config(reply_width=0)
    matches = _matches()
    scorer = _DeterministicPlacementScorer()
    state_scorer = _DeterministicMatchScorer()
    serial = tuple(
        choose_versus_action(
            match,
            "player",
            config=config,
            scorer=scorer,
            opponent_scorer=_UnexpectedReplyScorer(),
            state_scorer=state_scorer,
        )
        for match in matches
    )

    batched = choose_versus_actions_batch(
        _requests(
            _matches(),
            config=config,
            scorer=_DeterministicPlacementScorer(),
            opponent_scorer=_UnexpectedReplyScorer(),
            state_scorer=_DeterministicMatchScorer(),
        )
    )

    assert tuple(map(_choice_signature, batched)) == tuple(map(_choice_signature, serial))
    assert all(choice is not None for choice in batched)
    assert all(choice.opponent_reply is None for choice in batched if choice is not None)


def test_early_no_action_is_none_in_serial_and_real_batch_paths() -> None:
    config = _search_config()
    stopped = VersusMatch(1901)
    stopped.player.game.game_over = True
    stopped._update_winner()
    live = VersusMatch(1902)
    scorer = _DeterministicPlacementScorer()
    state_scorer = _DeterministicMatchScorer()
    before = _match_signature(stopped)

    assert choose_versus_action(
        stopped,
        "player",
        config=config,
        scorer=scorer,
        opponent_scorer=scorer,
        state_scorer=state_scorer,
    ) is None
    choices = choose_versus_actions_batch(
        _requests(
            (stopped, live),
            config=config,
            scorer=scorer,
            state_scorer=state_scorer,
        )
    )
    assert choices[0] is None
    assert choices[1] is not None
    assert _match_signature(stopped) == before


def test_simulated_topout_matches_full_clone_and_does_not_mutate_source() -> None:
    match = VersusMatch(77)
    match.player.game.board[4][0] = "G"
    match.player.pending.enqueue(1, 4)
    before = _match_signature(match)
    ranked = rank_search_actions(
        match.player.game,
        config=_search_config().placement_search,
        limit=16,
    )
    action = next(action for action, evaluation in ranked if evaluation.features.lines == 0)

    expected = clone_versus_match(match)
    expected_lock = apply_search_action(expected.player.game, action)
    expected_resolution = expected.resolve_lock("player", expected_lock)
    actual, actual_resolution = _simulate_action(match, "player", action)

    assert actual_resolution == expected_resolution
    assert _match_signature(actual) == _match_signature(expected)
    assert actual_resolution.garbage_applied == 1
    assert actual.player.game.game_over
    assert actual.winner == "ai"
    assert _match_signature(match) == before


def test_simulation_preserves_cancel_send_apply_and_garbage_hole_sequence() -> None:
    match = VersusMatch(3)
    assert match.player.game.current == "I"
    for y in range(match.player.game.height - 4, match.player.game.height):
        match.player.game.board[y] = ["G"] * match.player.game.width
        match.player.game.board[y][4] = None
    match.player.pending.enqueue(2, 1)
    before = _match_signature(match)
    placement = next(
        placement
        for placement in reachable_placements(match.player.game)
        if {x for x, _y in placement.cells} == {4}
    )
    root_action = SearchAction(False, placement)

    expected_root = clone_versus_match(match)
    expected_lock = apply_search_action(expected_root.player.game, root_action)
    expected_root_resolution = expected_root.resolve_lock("player", expected_lock)
    actual_root, actual_root_resolution = _simulate_action(match, "player", root_action)

    assert actual_root_resolution == expected_root_resolution
    assert _match_signature(actual_root) == _match_signature(expected_root)
    assert actual_root_resolution.canceled_lines == 2
    assert actual_root_resolution.sent_lines == 12
    assert tuple((packet.lines, packet.hole) for packet in actual_root.ai.pending.packets) == ((12, 1),)

    reply = SearchAction(False, reachable_placements(expected_root.ai.game)[0])
    expected_reply = clone_versus_match(expected_root)
    expected_reply_lock = apply_search_action(expected_reply.ai.game, reply)
    expected_reply_resolution = expected_reply.resolve_lock("ai", expected_reply_lock)
    actual_reply, actual_reply_resolution = _simulate_action(actual_root, "ai", reply)

    assert actual_reply_resolution == expected_reply_resolution
    assert _match_signature(actual_reply) == _match_signature(expected_reply)
    assert actual_reply_resolution.garbage_applied == 8
    assert tuple((packet.lines, packet.hole) for packet in actual_reply.ai.pending.packets) == ((4, 1),)
    assert _match_signature(match) == before


def test_selfplay_records_repeat_and_preserve_field_semantics(tmp_path) -> None:
    config = VersusSelfPlayConfig(
        games=2,
        max_turns=3,
        seed_base=7101,
        seed_step=31,
        search_config=_search_config(),
        game_batch=2,
    )
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    first = generate_versus_selfplay_dataset(
        first_path,
        _DeterministicPlacementScorer(),
        config,
        value_scorer=_DeterministicMatchScorer(),
    )
    second = generate_versus_selfplay_dataset(
        second_path,
        _DeterministicPlacementScorer(),
        config,
        value_scorer=_DeterministicMatchScorer(),
    )

    assert first_path.read_bytes() == second_path.read_bytes()
    assert first == {**second, "path": str(first_path)}
    records = [
        json.loads(line)
        for line in first_path.read_text(encoding="utf-8").splitlines()
    ]
    required_keys = {
        "format",
        "game",
        "seed",
        "ply",
        "side",
        "toMove",
        "terminal",
        "ownRows",
        "opponentRows",
        "context",
        "teacherValue",
        "outcome",
    }
    value_config = VersusValueConfig()
    assert all(set(record) == required_keys for record in records)
    assert all(record["format"] == VERSUS_SELFPLAY_FORMAT for record in records)

    for game_index in range(config.games):
        game_records = [record for record in records if record["game"] == game_index]
        assert game_records
        assert {record["seed"] for record in game_records} == {
            config.seed_base + game_index * config.seed_step
        }
        terminal_ply = max(record["ply"] for record in game_records)
        assert len(game_records) == 2 * (terminal_ply + 1)
        expected_order = [
            (ply, side)
            for ply in range(terminal_ply + 1)
            for side in ("player", "ai")
        ]
        assert [(record["ply"], record["side"]) for record in game_records] == expected_order

        for ply in range(terminal_ply + 1):
            player, ai = game_records[2 * ply : 2 * ply + 2]
            assert player["ownRows"] == ai["opponentRows"]
            assert player["opponentRows"] == ai["ownRows"]
            assert len(player["ownRows"]) == value_config.board_height
            assert len(ai["ownRows"]) == value_config.board_height
            assert len(player["context"]) == value_config.context_size
            assert len(ai["context"]) == value_config.context_size
            if ply == terminal_ply:
                assert player["terminal"] and ai["terminal"]
                assert player["toMove"] is None and ai["toMove"] is None
                assert player["teacherValue"] == player["outcome"]
                assert ai["teacherValue"] == ai["outcome"]
            else:
                expected_to_move = (
                    ("player" if game_index % 2 == 0 else "ai")
                    if ply % 2 == 0
                    else ("ai" if game_index % 2 == 0 else "player")
                )
                assert not player["terminal"] and not ai["terminal"]
                assert player["toMove"] == expected_to_move
                assert ai["toMove"] == expected_to_move
                assert player["context"][-1] == float(expected_to_move == "player")
                assert ai["context"][-1] == float(expected_to_move == "ai")
        assert player["outcome"] == -ai["outcome"]
