from __future__ import annotations

import random
import unittest

from minoflux_ai.search import (
    SearchAction,
    SearchConfig,
    _clone_random,
    apply_search_action,
    clone_game,
    rank_search_actions,
)
from minoflux_ai.versus_search import _simulate_action, clone_versus_match
from minoflux_engine import GARBAGE_CELL, Placement, VersusMatch


def _game_state(game) -> tuple[object, ...]:
    return (
        game.snapshot(queue_size=len(game.queue)),
        tuple(game.queue),
        tuple(game._bag._queue),
        game._bag._rng.getstate(),
        game.last_action,
        game.last_move_was_rotation,
        game.last_rotation_kick_index,
        game.last_rotation_from,
        game.last_rotation_to,
    )


def _match_state(match: VersusMatch) -> tuple[object, ...]:
    def side_state(side) -> tuple[object, ...]:
        return (
            _game_state(side.game),
            tuple((packet.lines, packet.hole) for packet in side.pending.packets),
            side.sent,
            side.received,
            side.canceled,
            side.garbage_applied,
        )

    return (
        match.seed,
        match.garbage_cap,
        match.winner,
        side_state(match.player),
        side_state(match.ai),
        match._garbage_rng.getstate(),
    )


def _reference_simulation(
    match: VersusMatch,
    side_name: str,
    action: SearchAction,
):
    simulated = clone_versus_match(match)
    side = simulated.player if side_name == "player" else simulated.ai
    result = apply_search_action(side.game, action)
    return simulated, simulated.resolve_lock(side_name, result)


def _vertical_tetris_action() -> SearchAction:
    return SearchAction(
        False,
        Placement(
            piece="I",
            x=2,
            y=20,
            rotation=1,
            cells=((4, 20), (4, 21), (4, 22), (4, 23)),
            path=("rotate_cw", "hard_drop"),
        ),
    )


class FastCloneSimulationTests(unittest.TestCase):
    def assert_simulation_matches_reference(
        self,
        match: VersusMatch,
        side_name: str,
        action: SearchAction,
    ):
        before = _match_state(match)
        expected, expected_resolution = _reference_simulation(match, side_name, action)
        self.assertEqual(_match_state(match), before)

        actual, actual_resolution = _simulate_action(match, side_name, action)
        self.assertEqual(_match_state(match), before)
        self.assertEqual(actual_resolution, expected_resolution)
        self.assertEqual(_match_state(actual), _match_state(expected))
        return actual, actual_resolution

    def test_random_clone_preserves_state_without_aliasing(self) -> None:
        source = random.Random(1201)
        for _ in range(17):
            source.randrange(1000)
        before = source.getstate()

        cloned = _clone_random(source)
        reference = random.Random()
        reference.setstate(before)

        self.assertIsNot(cloned, source)
        self.assertEqual(cloned.getstate(), before)
        self.assertEqual(
            [cloned.randrange(1000) for _ in range(64)],
            [reference.randrange(1000) for _ in range(64)],
        )
        self.assertEqual(source.getstate(), before)

    def test_game_clone_preserves_bag_refill_and_source_rng(self) -> None:
        game = VersusMatch(2301).player.game
        game._bag._queue.clear()
        before = _game_state(game)
        cloned = clone_game(game)

        self.assertIsNot(cloned._bag._rng, game._bag._rng)
        self.assertEqual(_game_state(cloned), before)
        cloned_draws = tuple(cloned._bag.pop() for _ in range(14))

        reference = clone_game(game)
        reference_draws = tuple(reference._bag.pop() for _ in range(14))
        self.assertEqual(cloned_draws, reference_draws)
        self.assertEqual(_game_state(game), before)

    def test_attack_cancel_and_garbage_hole_match_reference(self) -> None:
        match = VersusMatch(3401)
        game = match.player.game
        game.current = "I"
        for row in game.board[-4:]:
            row[:] = [GARBAGE_CELL] * game.width
            row[4] = None
        match.player.pending.enqueue(2, 1)

        actual, resolution = self.assert_simulation_matches_reference(
            match,
            "player",
            _vertical_tetris_action(),
        )
        self.assertEqual(resolution.canceled_lines, 2)
        self.assertGreater(resolution.sent_lines, 0)
        self.assertGreater(actual.ai.pending.pending_lines, 0)

    def test_no_clear_applies_same_pending_garbage(self) -> None:
        match = VersusMatch(4501, garbage_cap=8)
        match.player.pending.enqueue(3, 2)
        match.player.pending.enqueue(3, 7)
        ranked = rank_search_actions(
            match.player.game,
            config=SearchConfig(
                allow_hold=False,
                lookahead_pieces=0,
                srs_reachable=True,
            ),
            limit=1,
        )
        self.assertTrue(ranked)

        actual, resolution = self.assert_simulation_matches_reference(
            match,
            "player",
            ranked[0][0],
        )
        self.assertEqual(resolution.garbage_applied, 6)
        self.assertEqual(actual.player.garbage_applied, 6)

    def test_empty_hold_with_bag_refill_matches_reference(self) -> None:
        match = VersusMatch(5601)
        game = match.player.game
        game.hold_piece = None
        game._bag._queue.clear()
        ranked = rank_search_actions(
            game,
            config=SearchConfig(
                allow_hold=True,
                lookahead_pieces=0,
                srs_reachable=True,
            ),
            limit=None,
        )
        action = next(action for action, _evaluation in ranked if action.use_hold)

        actual, _resolution = self.assert_simulation_matches_reference(
            match,
            "player",
            action,
        )
        self.assertIsNotNone(actual.player.game.hold_piece)

    def test_root_then_reply_preserves_rng_and_garbage_sequence(self) -> None:
        match = VersusMatch(6701)
        game = match.player.game
        game.current = "I"
        for row in game.board[-4:]:
            row[:] = [GARBAGE_CELL] * game.width
            row[4] = None

        after_root, _resolution = self.assert_simulation_matches_reference(
            match,
            "player",
            _vertical_tetris_action(),
        )
        ranked = rank_search_actions(
            after_root.ai.game,
            config=SearchConfig(
                allow_hold=True,
                lookahead_pieces=0,
                srs_reachable=True,
            ),
            limit=1,
        )
        self.assertTrue(ranked)
        self.assert_simulation_matches_reference(after_root, "ai", ranked[0][0])


if __name__ == "__main__":
    unittest.main()
