from __future__ import annotations

from minoflux_engine.pieces import kick_tests


def test_tetrio_srs_plus_jlstz_180_kicks() -> None:
    expected = {
        0: ((0, 0), (0, -1), (1, -1), (-1, -1), (1, 0), (-1, 0)),
        1: ((0, 0), (1, 0), (1, -2), (1, -1), (0, -2), (0, -1)),
        2: ((0, 0), (0, 1), (-1, 1), (1, 1), (-1, 0), (1, 0)),
        3: ((0, 0), (-1, 0), (-1, -2), (-1, -1), (0, -2), (0, -1)),
    }

    for piece in ("J", "L", "S", "T", "Z"):
        for source, kicks in expected.items():
            assert kick_tests(piece, source, source + 2) == kicks


def test_tetrio_srs_plus_i_180_kicks() -> None:
    expected = {
        0: ((0, 0), (0, -1)),
        1: ((0, 0), (1, 0)),
        2: ((0, 0), (0, 1)),
        3: ((0, 0), (-1, 0)),
    }

    for source, kicks in expected.items():
        assert kick_tests("I", source, source + 2) == kicks


def test_tetrio_srs_plus_i_90_kicks_are_symmetric() -> None:
    assert kick_tests("I", 0, 1) == (
        (0, 0),
        (1, 0),
        (-2, 0),
        (-2, 1),
        (1, -2),
    )
    assert kick_tests("I", 0, 3) == (
        (0, 0),
        (-1, 0),
        (2, 0),
        (2, 1),
        (-1, -2),
    )


def test_o_rotation_remains_stationary() -> None:
    assert kick_tests("O", 0, 2) == ((0, 0),)
