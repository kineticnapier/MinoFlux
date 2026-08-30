from __future__ import annotations

from minoflux_ai.neural import encode_game_state, encode_placement_result
from minoflux_ai.reachability import reachable_placements
from minoflux_ai.search import _held_search_game, clone_game
from minoflux_engine import Game


def _assert_fast_state_matches_clone(game: Game, placements) -> None:
    for placement in tuple(placements)[:12]:
        fast = encode_placement_result(game, placement)
        assert fast is not None
        child = clone_game(game)
        child.place(placement)
        assert fast == encode_game_state(child)


def test_direct_post_placement_encoder_matches_engine_clone() -> None:
    for seed in (101, 202, 303):
        game = Game(seed)
        for _ in range(8):
            placements = reachable_placements(game)
            assert placements
            _assert_fast_state_matches_clone(game, placements)
            game.place(placements[len(placements) // 2])
            if game.game_over:
                break


def test_lightweight_hold_branch_matches_real_hold_for_neural_preview() -> None:
    for seed in (404, 505):
        game = Game(seed)
        for _ in range(4):
            held = _held_search_game(game)
            exact = clone_game(game)
            assert exact.hold()
            assert held is not None
            assert held.current == exact.current
            assert held.hold_piece == exact.hold_piece
            assert tuple(held.queue)[:6] == tuple(exact.queue)[:6]

            placements = reachable_placements(held)
            assert placements
            for placement in placements[:10]:
                fast = encode_placement_result(held, placement)
                assert fast is not None
                exact_child = clone_game(exact)
                exact_child.place(placement)
                assert fast == encode_game_state(exact_child)

            direct = reachable_placements(game)
            assert direct
            game.place(direct[0])
            if game.game_over:
                break
