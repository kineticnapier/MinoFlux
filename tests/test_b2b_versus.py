from __future__ import annotations

import unittest

from minoflux_engine import (
    GarbageQueue,
    LockResult,
    VersusMatch,
    resolve_b2b_charging,
    split_surge,
)


class B2BChargingTests(unittest.TestCase):
    def test_surge_is_split_into_three_ordered_packets(self) -> None:
        self.assertEqual(split_surge(4), (2, 1, 1))
        self.assertEqual(split_surge(5), (2, 2, 1))
        self.assertEqual(split_surge(8), (3, 3, 2))

    def test_chain_charges_at_displayed_b2b_four_and_releases_on_break(self) -> None:
        active = False
        chain = 0
        outcome = None
        for expected_chain in (0, 1, 2, 3, 4):
            outcome = resolve_b2b_charging(
                active=active,
                chain=chain,
                difficult=True,
                lines=4,
            )
            active, chain = outcome.active, outcome.chain
            self.assertEqual(chain, expected_chain)
        assert outcome is not None
        self.assertEqual(outcome.charge, 4)
        self.assertEqual(outcome.attack_bonus, 1)

        broken = resolve_b2b_charging(
            active=active,
            chain=chain,
            difficult=False,
            lines=1,
        )
        self.assertFalse(broken.active)
        self.assertEqual(broken.released, 4)
        self.assertEqual(broken.charge, 0)

    def test_no_clear_preserves_charge(self) -> None:
        outcome = resolve_b2b_charging(
            active=True,
            chain=9,
            difficult=False,
            lines=0,
        )
        self.assertTrue(outcome.active)
        self.assertEqual(outcome.chain, 9)
        self.assertEqual(outcome.charge, 9)
        self.assertEqual(outcome.released, 0)

    def test_perfect_clear_adds_exactly_two_levels(self) -> None:
        outcome = resolve_b2b_charging(
            active=True,
            chain=3,
            difficult=True,
            lines=4,
            perfect_clear=True,
        )
        self.assertEqual(outcome.chain, 5)
        self.assertEqual(outcome.charge, 5)


class GarbageQueueTests(unittest.TestCase):
    def test_fifo_cancel_and_take(self) -> None:
        queue = GarbageQueue()
        queue.enqueue(3, 2)
        queue.enqueue(4, 7)
        remaining, canceled = queue.cancel(5)
        self.assertEqual((remaining, canceled), (0, 5))
        self.assertEqual(queue.pending_lines, 2)
        packets = queue.take(1)
        self.assertEqual([(packet.lines, packet.hole) for packet in packets], [(1, 7)])
        self.assertEqual(queue.pending_lines, 1)


class VersusResolutionTests(unittest.TestCase):
    def test_outgoing_packets_cancel_before_being_sent(self) -> None:
        match = VersusMatch(123)
        match.player.pending.enqueue(3, 4)
        result = LockResult(
            lines=4,
            attack=5,
            spin=None,
            perfect_clear=False,
            combo=0,
            back_to_back=True,
            game_over=False,
            attack_packets=(2, 3),
        )
        resolution = match.resolve_lock("player", result)
        self.assertEqual(resolution.canceled_lines, 3)
        self.assertEqual(resolution.sent_packets, (2,))
        self.assertEqual(match.player.pending.pending_lines, 0)
        self.assertEqual(match.ai.pending.pending_lines, 2)

    def test_no_clear_accepts_only_the_garbage_cap(self) -> None:
        match = VersusMatch(456, garbage_cap=8)
        match.player.pending.enqueue(10, 5)
        result = LockResult(
            lines=0,
            attack=0,
            spin=None,
            perfect_clear=False,
            combo=-1,
            back_to_back=False,
            game_over=False,
        )
        resolution = match.resolve_lock("player", result)
        self.assertEqual(resolution.garbage_applied, 8)
        self.assertEqual(match.player.pending.pending_lines, 2)
        self.assertEqual(sum(cell == "G" for row in match.player.game.board for cell in row), 72)

    def test_line_clear_blocks_pending_garbage_for_the_turn(self) -> None:
        match = VersusMatch(789, garbage_cap=8)
        match.player.pending.enqueue(6, 3)
        result = LockResult(
            lines=1,
            attack=0,
            spin=None,
            perfect_clear=False,
            combo=0,
            back_to_back=False,
            game_over=False,
        )
        resolution = match.resolve_lock("player", result)
        self.assertEqual(resolution.garbage_applied, 0)
        self.assertEqual(match.player.pending.pending_lines, 6)


if __name__ == "__main__":
    unittest.main()
