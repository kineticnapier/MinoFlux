from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from minoflux_ai.neural import NeuralValueEvaluator, NeuralValueConfig, build_neural_value_model, _context_prefix
from minoflux_ai.neural_fast import native_neural_available
from minoflux_ai.neural_search_fast import (
    _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH,
    choose_search_actions_batch as choose_native_record_actions_batch,
)
from minoflux_ai.reachability import clear_reachability_cache
from minoflux_ai.reachability_native import (
    clear_native_record_cache,
    native_pathless_available,
    reachable_placement_records_native,
)
from minoflux_ai.search import SearchAction, SearchConfig, _branch_groups, apply_search_action
from minoflux_ai.bitboard import board_row_masks
from minoflux_engine import Game
from minoflux_engine.pieces import SHAPES

try:
    from minoflux_ai import _neural_native
except ImportError:
    _neural_native = None

try:
    import torch
except ImportError:
    torch = None


_CONFIG = NeuralValueConfig().normalized()
_SEARCH_CONFIG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
).normalized()


def _prefixes(game: Game):
    queue = tuple(game.queue)
    locked = _context_prefix(
        current=game.current,
        hold_piece=game.hold_piece,
        queue=queue,
        config=_CONFIG,
    )
    normal = _context_prefix(
        current=queue[0],
        hold_piece=game.hold_piece,
        queue=queue[1:],
        config=_CONFIG,
    )
    return queue, normal, locked


def _record_buffers(game: Game, batch):
    queue, normal, locked = _prefixes(game)
    return _neural_native.encode_record_group(
        board_row_masks(game.board),
        batch.records,
        game.current,
        queue[0],
        game.width,
        game.height,
        game.hidden_rows,
        game.combo,
        game.back_to_back,
        game.b2b_chain,
        normal,
        locked,
    )


def _placement_buffers(game: Game, placements):
    queue, normal, locked = _prefixes(game)
    return _neural_native.encode_placement_group(
        board_row_masks(game.board),
        placements,
        game.current,
        queue[0],
        game.width,
        game.height,
        game.hidden_rows,
        game.combo,
        game.back_to_back,
        game.b2b_chain,
        normal,
        locked,
    )


def _choice_signature(choice):
    if choice is None:
        return None
    placement = choice.action.placement
    features = choice.immediate.features
    return (
        choice.action.use_hold,
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
        placement.cells,
        placement.path,
        placement.last_move_was_rotation,
        placement.rotation_kick_index,
        placement.rotation_from,
        placement.rotation_to,
        choice.score,
        choice.immediate.score,
        features.attack,
        features.spin_lines,
        features.lines,
        features.board.holes,
        features.board.max_height,
        features.perfect_clear,
        features.game_over,
        features.spin,
    )


@unittest.skipUnless(
    native_pathless_available() and native_neural_available(),
    "native reachability/neural extensions unavailable",
)
class NativeRecordPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _neural_native.register_shapes(SHAPES)

    def setUp(self) -> None:
        clear_reachability_cache()
        clear_native_record_cache()

    def test_raw_records_materialize_to_exact_ordered_placements(self) -> None:
        games = [Game(8100001 + index * 97) for index in range(8)]
        compared = 0
        for _step in range(30):
            for game in games:
                if game.game_over:
                    continue
                clear_reachability_cache()
                clear_native_record_cache()
                direct, held, hold = _branch_groups(
                    game,
                    _SEARCH_CONFIG,
                    include_paths=False,
                )
                raw_direct = reachable_placement_records_native(
                    game,
                    allow_180=False,
                    max_nodes=8_000,
                )
                self.assertIsNotNone(raw_direct)
                assert raw_direct is not None
                self.assertEqual(
                    tuple(raw_direct.materialize(i) for i in range(len(raw_direct))),
                    direct,
                )
                compared += len(direct)

                if held is not None:
                    raw_hold = reachable_placement_records_native(
                        held,
                        allow_180=False,
                        max_nodes=8_000,
                    )
                    self.assertIsNotNone(raw_hold)
                    assert raw_hold is not None
                    self.assertEqual(
                        tuple(raw_hold.materialize(i) for i in range(len(raw_hold))),
                        hold,
                    )
                    compared += len(hold)

                placement = direct[0] if direct else (hold[0] if hold else None)
                if placement is not None:
                    apply_search_action(
                        game,
                        SearchAction(False if direct else True, placement),
                    )
        self.assertGreater(compared, 2_000)

    def test_record_encoder_is_byte_identical_to_placement_encoder(self) -> None:
        game = Game(8100001)
        compared = 0
        for _step in range(30):
            if game.game_over:
                break
            clear_reachability_cache()
            clear_native_record_cache()
            direct, held, hold = _branch_groups(
                game,
                _SEARCH_CONFIG,
                include_paths=False,
            )
            raw_direct = reachable_placement_records_native(game)
            assert raw_direct is not None
            self.assertEqual(
                _record_buffers(game, raw_direct),
                _placement_buffers(game, direct),
            )
            compared += len(direct)
            if held is not None and hold:
                raw_hold = reachable_placement_records_native(held)
                assert raw_hold is not None
                self.assertEqual(
                    _record_buffers(held, raw_hold),
                    _placement_buffers(held, hold),
                )
                compared += len(hold)
            if direct:
                apply_search_action(game, SearchAction(False, direct[0]))
        self.assertGreater(compared, 500)

    @unittest.skipUnless(torch is not None, "PyTorch unavailable")
    def test_batch_action_stream_matches_existing_placement_path(self) -> None:
        torch.manual_seed(20260905)
        model_old = build_neural_value_model(_CONFIG)
        model_new = build_neural_value_model(_CONFIG)
        model_new.load_state_dict(model_old.state_dict())
        old_evaluator = NeuralValueEvaluator(model_old, _CONFIG, device="cpu")
        new_evaluator = NeuralValueEvaluator(model_new, _CONFIG, device="cpu")

        old_games = [Game(8100001 + index * 97) for index in range(12)]
        new_games = [deepcopy(game) for game in old_games]
        compared = 0
        for _step in range(40):
            old_active = tuple(game for game in old_games if not game.game_over)
            new_active = tuple(game for game in new_games if not game.game_over)
            self.assertEqual(len(old_active), len(new_active))
            if not old_active:
                break
            clear_reachability_cache()
            clear_native_record_cache()
            old_choices = _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                old_active,
                config=_SEARCH_CONFIG,
                scorer=old_evaluator,
            )
            clear_reachability_cache()
            clear_native_record_cache()
            new_choices = choose_native_record_actions_batch(
                new_active,
                config=_SEARCH_CONFIG,
                scorer=new_evaluator,
            )
            self.assertEqual(
                tuple(_choice_signature(choice) for choice in new_choices),
                tuple(_choice_signature(choice) for choice in old_choices),
            )
            compared += len(old_choices)
            for old_game, old_choice, new_game, new_choice in zip(
                old_active,
                old_choices,
                new_active,
                new_choices,
            ):
                if old_choice is not None:
                    apply_search_action(old_game, old_choice.action)
                if new_choice is not None:
                    apply_search_action(new_game, new_choice.action)
                self.assertEqual(old_game.pieces_placed, new_game.pieces_placed)
                self.assertEqual(old_game.attack, new_game.attack)
                self.assertEqual(old_game.game_over, new_game.game_over)
        self.assertGreater(compared, 100)

    def test_native_neural_disable_uses_existing_fallback(self) -> None:
        game = Game(8100001)
        with patch.dict("os.environ", {"MINOFLUX_DISABLE_NATIVE_NEURAL": "1"}):
            # A dummy scorer forces the generic path without needing torch.
            class Scorer:
                def score_placement_groups(self, groups):
                    return tuple(tuple(0.0 for _ in placements) for _g, placements in groups)

                def score_placements(self, game, placements):
                    return tuple(0.0 for _ in placements)

            expected = _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                (game,),
                config=_SEARCH_CONFIG,
                scorer=Scorer(),
            )
            actual = choose_native_record_actions_batch(
                (game,),
                config=_SEARCH_CONFIG,
                scorer=Scorer(),
            )
            self.assertEqual(
                tuple(_choice_signature(choice) for choice in actual),
                tuple(_choice_signature(choice) for choice in expected),
            )


if __name__ == "__main__":
    unittest.main()
