from __future__ import annotations

import random

from minoflux_ai.bitboard import board_row_masks, collides_row_masks
from minoflux_ai.features import BoardFeatures, extract_board_features_from_masks
from minoflux_ai.reachability import reachable_placements
from minoflux_engine import Game
from minoflux_engine.pieces import SHAPES


def _placement_signature(placement):
    return (
        placement.piece,
        placement.x,
        placement.y,
        placement.rotation,
        placement.cells,
        placement.last_move_was_rotation,
        placement.rotation_kick_index,
        placement.rotation_from,
        placement.rotation_to,
    )


def _legacy_occupied_or_wall(board, x: int, y: int) -> bool:
    height = len(board)
    width = len(board[0]) if board else 0
    return x < 0 or x >= width or y < 0 or y >= height or board[y][x] is not None


def _legacy_empty(board, x: int, y: int) -> bool:
    height = len(board)
    width = len(board[0]) if board else 0
    return 0 <= x < width and 0 <= y < height and board[y][x] is None


def _legacy_t_spin_slots(board, start_y: int) -> int:
    height = len(board)
    width = len(board[0])
    orientations = (
        ((0, -1), (-1, 0), (0, 0), (1, 0)),
        ((0, -1), (0, 0), (1, 0), (0, 1)),
        ((-1, 0), (0, 0), (1, 0), (0, 1)),
        ((0, -1), (-1, 0), (0, 0), (0, 1)),
    )
    slots = 0
    for pivot_y in range(max(0, start_y), height):
        for pivot_x in range(width):
            if board[pivot_y][pivot_x] is not None:
                continue
            corners = (
                _legacy_occupied_or_wall(board, pivot_x - 1, pivot_y - 1),
                _legacy_occupied_or_wall(board, pivot_x + 1, pivot_y - 1),
                _legacy_occupied_or_wall(board, pivot_x - 1, pivot_y + 1),
                _legacy_occupied_or_wall(board, pivot_x + 1, pivot_y + 1),
            )
            if sum(corners) < 3:
                continue
            if any(
                all(_legacy_empty(board, pivot_x + dx, pivot_y + dy) for dx, dy in cells)
                for cells in orientations
            ):
                slots += 1
    return slots


def _legacy_features(board) -> BoardFeatures:
    height = len(board)
    width = len(board[0])
    heights: list[int] = []
    for x in range(width):
        top = height
        for y, row in enumerate(board):
            if row[x] is not None:
                top = y
                break
        heights.append(height - top)

    holes = 0
    hole_depth = 0
    occupied_cells = 0
    for x in range(width):
        seen_block = False
        blocks_above = 0
        for y in range(height):
            occupied = board[y][x] is not None
            if occupied:
                occupied_cells += 1
                seen_block = True
                blocks_above += 1
            elif seen_block:
                holes += 1
                hole_depth += blocks_above

    bumpiness = sum(abs(left - right) for left, right in zip(heights, heights[1:]))
    wells = 0
    for x in range(width):
        depth = 0
        for y in range(height):
            if board[y][x] is not None:
                depth = 0
                continue
            left_filled = x == 0 or board[y][x - 1] is not None
            right_filled = x == width - 1 or board[y][x + 1] is not None
            if left_filled and right_filled:
                depth += 1
                wells += depth
            else:
                depth = 0

    max_height = max(heights, default=0)
    slot_start_y = height if max_height == 0 else max(0, height - max_height - 1)
    return BoardFeatures(
        aggregate_height=sum(heights),
        max_height=max_height,
        holes=holes,
        hole_depth=hole_depth,
        bumpiness=bumpiness,
        wells=wells,
        t_spin_slots=_legacy_t_spin_slots(board, slot_start_y),
        occupied_cells=occupied_cells,
    )


def _reference_collision(rows, piece: str, x: int, y: int, rotation: int) -> bool:
    width = 10
    height = len(rows)
    for dx, dy in SHAPES[piece][rotation % 4]:
        cell_x = x + dx
        cell_y = y + dy
        if cell_x < 0 or cell_x >= width or cell_y >= height:
            return True
        if cell_y >= 0 and rows[cell_y] & (1 << cell_x):
            return True
    return False


def test_negative_srs_anchor_bitboard_collision() -> None:
    rows = (0,) * 24
    # Vertical I has all occupied cells at dx=2, so x=-2 is a legal anchor.
    assert not collides_row_masks(rows, "I", -2, 1, 1)
    blocked = list(rows)
    blocked[1] = 1
    assert collides_row_masks(tuple(blocked), "I", -2, 1, 1)


def test_precomputed_collision_masks_match_reference() -> None:
    rng = random.Random(90210)
    boards = [(0,) * 24]
    for _ in range(8):
        boards.append(tuple(rng.randrange(1 << 10) for _ in range(24)))

    for rows in boards:
        for piece in SHAPES:
            for rotation in range(4):
                for x in range(-4, 14):
                    for y in (-4, -1, 0, 1, 10, 20, 23, 24):
                        assert collides_row_masks(rows, piece, x, y, rotation) == _reference_collision(
                            rows, piece, x, y, rotation
                        )


def test_mask_feature_extractor_matches_legacy_reference() -> None:
    rng = random.Random(712367)
    for _ in range(30):
        board = [
            ["G" if rng.random() < 0.28 else None for _x in range(10)]
            for _y in range(24)
        ]
        expected = _legacy_features(board)
        actual = extract_board_features_from_masks(
            board_row_masks(board),
            width=10,
        )
        assert actual == expected


def test_pathless_reachability_keeps_geometry_and_spin_metadata() -> None:
    for seed in (11, 29, 47, 83):
        game = Game(seed)
        for _ in range(6):
            with_paths = reachable_placements(game, include_paths=True)
            pathless = reachable_placements(game, include_paths=False)
            assert with_paths
            assert {_placement_signature(item) for item in pathless} == {
                _placement_signature(item) for item in with_paths
            }
            assert all(not item.path for item in pathless)
            game.place(with_paths[len(with_paths) // 2])
            if game.game_over:
                break
