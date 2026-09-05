from minoflux.handling import HandlingController, RepeatTimer
from minoflux_engine import Game


def test_arr_zero_stays_charged_until_release() -> None:
    timer = RepeatTimer()
    timer.press(10.0, 100)

    assert timer.poll(10.05, 0).instant is False
    assert timer.poll(10.10, 0).instant is True
    assert timer.poll(10.20, 0).instant is True

    timer.release()
    assert timer.poll(10.30, 0).instant is False


def test_horizontal_arr_zero_survives_after_first_wall_shift() -> None:
    handling = HandlingController()
    handling.press_horizontal(1, 1.0, 100)

    direction, first = handling.poll_horizontal(1.10, 0)
    assert direction == 1
    assert first.instant is True

    direction, later = handling.poll_horizontal(1.20, 0)
    assert direction == 1
    assert later.instant is True


def _empty_i_game(rotation: int) -> Game:
    game = Game(123)
    game.board = [[None] * game.width for _ in range(game.height)]
    game.current = "I"
    game.x = 3
    game.y = 4
    game.rotation = rotation
    game.game_over = False
    game.paused = False
    return game


def test_i_piece_can_touch_both_side_walls_in_every_rotation() -> None:
    for rotation in range(4):
        left = _empty_i_game(rotation)
        while left.move_left():
            pass
        assert min(x for x, _y in left.cells()) == 0

        right = _empty_i_game(rotation)
        while right.move_right():
            pass
        assert max(x for x, _y in right.cells()) == right.width - 1
