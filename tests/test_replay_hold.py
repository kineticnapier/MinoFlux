from __future__ import annotations

import unittest

from minoflux_ai import (
    LEGACY_REPLAY_FORMAT,
    REPLAY_FORMAT,
    Replay,
    ReplayStep,
    ReplaySummary,
    replay_to_game,
)
from minoflux_engine import Game


class HoldReplayTests(unittest.TestCase):
    def test_hold_step_replays_exactly(self) -> None:
        source = Game(seed=20)
        self.assertTrue(source.hold())
        placement = source.legal_placements()[0]
        result = source.place(placement)
        step = ReplayStep(
            piece=placement.piece,
            x=placement.x,
            y=placement.y,
            rotation=placement.rotation,
            lines=result.lines,
            attack=result.attack,
            score=source.score,
            total_lines=source.lines,
            total_attack=source.attack,
            hold=True,
        )
        replay = Replay(
            seed=20,
            max_pieces=1,
            weights={},
            steps=(step,),
            final=ReplaySummary(
                pieces=source.pieces_placed,
                lines=source.lines,
                attack=source.attack,
                score=source.score,
                topout=source.game_over,
            ),
            search_config={"allow_hold": True, "lookahead_pieces": 1, "beam_width": 4, "discount": 0.9},
        )
        rebuilt = replay_to_game(replay)
        self.assertEqual(rebuilt.snapshot(), source.snapshot())
        self.assertEqual(replay.to_dict()["format"], REPLAY_FORMAT)

    def test_v1_replay_loads_with_hold_false(self) -> None:
        game = Game(seed=21)
        placement = game.legal_placements()[0]
        result = game.place(placement)
        payload = {
            "format": LEGACY_REPLAY_FORMAT,
            "seed": 21,
            "maxPieces": 1,
            "weights": {},
            "steps": [{
                "piece": placement.piece,
                "x": placement.x,
                "y": placement.y,
                "rotation": placement.rotation,
                "lines": result.lines,
                "attack": result.attack,
                "score": game.score,
                "total_lines": game.lines,
                "total_attack": game.attack,
            }],
            "final": {
                "pieces": game.pieces_placed,
                "lines": game.lines,
                "attack": game.attack,
                "score": game.score,
                "topout": game.game_over,
            },
        }
        replay = Replay.from_mapping(payload)
        self.assertFalse(replay.steps[0].hold)
        self.assertEqual(replay.format, REPLAY_FORMAT)
        rebuilt = replay_to_game(replay)
        self.assertEqual(rebuilt.snapshot(), game.snapshot())


if __name__ == "__main__":
    unittest.main()
