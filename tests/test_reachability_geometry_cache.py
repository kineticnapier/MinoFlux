from __future__ import annotations

from minoflux_ai.reachability import _geometry_key
from minoflux_engine.pieces import SHAPES


def _reference_geometry_key(piece: str, x: int, y: int, rotation: int, width: int = 10) -> int:
    key = 0
    for dx, dy in SHAPES[piece][rotation % 4]:
        cell_x = x + dx
        cell_y = y + dy
        if cell_y < 0:
            return -1
        key |= 1 << (cell_y * width + cell_x)
    return key


def test_cached_geometry_key_matches_reference() -> None:
    _geometry_key.cache_clear()
    checked: list[tuple[str, int, int, int]] = []
    for piece, rotations in SHAPES.items():
        for rotation, shape in enumerate(rotations):
            for x in range(-2, 10):
                if any(x + dx < 0 or x + dx >= 10 for dx, _dy in shape):
                    continue
                for y in (-4, -2, -1, 0, 1, 7, 16, 20):
                    assert _geometry_key(piece, x, y, rotation, 10) == _reference_geometry_key(
                        piece, x, y, rotation, 10
                    )
                    checked.append((piece, x, y, rotation))

    misses = _geometry_key.cache_info().misses
    for piece, x, y, rotation in checked:
        _geometry_key(piece, x, y, rotation, 10)
    info = _geometry_key.cache_info()
    assert info.misses == misses
    assert info.hits >= len(checked)
