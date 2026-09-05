from __future__ import annotations

import unittest

from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator, build_neural_value_model
from minoflux_ai.neural_fast import native_neural_available
from minoflux_ai.neural_search_fast import choose_search_actions_batch
from minoflux_ai.reachability import ReachabilityProfile, clear_reachability_cache, collect_reachability_profile
from minoflux_ai.reachability_native import clear_native_record_cache, native_pathless_available
from minoflux_ai.search import SearchConfig
from minoflux_engine import Game
from minoflux.neural_cli import _ProfiledNeuralScorer
import minoflux.neural_cli as neural_cli

try:
    import torch
except ImportError:
    torch = None


CFG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
).normalized()
NCFG = NeuralValueConfig().normalized()


def _signature(choice):
    if choice is None:
        return None
    p = choice.action.placement
    return (
        choice.action.use_hold,
        p.piece,
        p.x,
        p.y,
        p.rotation,
        p.last_move_was_rotation,
        p.rotation_kick_index,
        p.rotation_from,
        p.rotation_to,
        choice.score,
    )


@unittest.skipUnless(
    torch is not None and native_pathless_available() and native_neural_available(),
    "native ML path unavailable",
)
class NativeRecordProfileTests(unittest.TestCase):
    def test_cli_binding_and_profile_wrapper_use_native_record_path(self) -> None:
        self.assertIs(neural_cli.choose_search_actions_batch, choose_search_actions_batch)

        torch.manual_seed(20260905)
        evaluator = NeuralValueEvaluator(
            build_neural_value_model(NCFG),
            NCFG,
            device="cpu",
        )
        profiled = _ProfiledNeuralScorer(evaluator)
        games = tuple(Game(8100001 + index * 97) for index in range(4))

        clear_reachability_cache()
        clear_native_record_cache()
        expected = choose_search_actions_batch(games, config=CFG, scorer=evaluator)

        clear_reachability_cache()
        clear_native_record_cache()
        reach = ReachabilityProfile()
        with collect_reachability_profile(reach):
            actual = choose_search_actions_batch(games, config=CFG, scorer=profiled)

        self.assertEqual(
            tuple(_signature(choice) for choice in actual),
            tuple(_signature(choice) for choice in expected),
        )
        self.assertEqual(profiled.calls, 1)
        self.assertGreater(profiled.groups, 0)
        self.assertGreater(profiled.states, 0)
        self.assertGreater(profiled.seconds, 0.0)
        self.assertGreater(reach.calls, 0)
        self.assertGreater(reach.placements, 0)


if __name__ == "__main__":
    unittest.main()
