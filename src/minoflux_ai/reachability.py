from __future__ import annotations

from collections import deque

from minoflux_engine import Game, Placement
from minoflux_engine.pieces import SHAPES, kick_tests

from .bitboard import (
    board_row_masks,
    classify_t_spin_row_masks,
    collides_row_masks,
    placement_cells,
)

MOVE_LEFT = "left"
MOVE_RIGHT = "right"
MOVE_DOWN = "down"
ROTATE_CW = "cw"
ROTATE_CCW = "ccw"
ROTATE_180 = "180"
HARD_DROP = "hard_drop"

_CMD_NONE = -1
_CMD_LEFT = 0
_CMD_RIGHT = 1
_CMD_DOWN = 2
_CMD_CW = 3
_CMD_CCW = 4
_CMD_180 = 5
_COMMAND_NAMES = (MOVE_LEFT, MOVE_RIGHT, MOVE_DOWN, ROTATE_CW, ROTATE_CCW, ROTATE_180)
_NO_STATE = -1
_NO_LANDING = -10_000
_X_MARGIN = 4
_Y_MIN = -4


def _state_layout(game: Game) -> tuple[int, int, int, int, int]:
    x_min = -_X_MARGIN
    x_max = game.width + _X_MARGIN - 1
    x_count = x_max - x_min + 1
    y_min = _Y_MIN
    y_count = game.height - y_min
    state_count = x_count * y_count * 4
    return x_min, x_max, x_count, y_min, state_count


def _pack_state(
    x: int,
    y: int,
    rotation: int,
    *,
    x_min: int,
    x_max: int,
    x_count: int,
    y_min: int,
    height: int,
) -> int:
    if x < x_min or x > x_max or y < y_min or y >= height:
        return _NO_STATE
    return ((((y - y_min) * x_count) + (x - x_min)) << 2) | (rotation & 3)


def _path(
    parents: list[int],
    commands: list[int],
    node_index: int,
) -> tuple[str, ...]:
    result: list[str] = []
    index = node_index
    while index >= 0:
        command = commands[index]
        if command != _CMD_NONE:
            result.append(_COMMAND_NAMES[command])
        index = parents[index]
    result.reverse()
    result.append(HARD_DROP)
    return tuple(result)


def _geometry_key(piece: str, x: int, y: int, rotation: int, width: int) -> int:
    key = 0
    for dx, dy in SHAPES[piece][rotation % 4]:
        cell_x, cell_y = x + dx, y + dy
        if cell_y < 0:
            return -1
        key |= 1 << (cell_y * width + cell_x)
    return key


def reachable_placements(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
    include_paths: bool = True,
) -> tuple[Placement, ...]:
    """Enumerate exact-SRS placements using packed states and bitboard collision.

    ``include_paths=False`` is intended for neural evaluation: placement geometry
    and T-spin metadata are preserved, but parent paths are not materialized.
    Replay/interactive callers keep the historical path behavior by default.
    """

    if game.game_over or game.paused:
        return ()

    rows = board_row_masks(game.board)
    piece = game.current
    x_min, x_max, x_count, y_min, packed_count = _state_layout(game)

    def pack(x: int, y: int, rotation: int) -> int:
        return _pack_state(
            x,
            y,
            rotation,
            x_min=x_min,
            x_max=x_max,
            x_count=x_count,
            y_min=y_min,
            height=game.height,
        )

    def collides(x: int, y: int, rotation: int) -> bool:
        return collides_row_masks(rows, piece, x, y, rotation, width=game.width)

    start_state = pack(game.x, game.y, game.rotation)
    if start_state < 0 or collides(game.x, game.y, game.rotation):
        return ()

    # Cache hard-drop destinations only for states that are actually queried.
    # Each vertical chain is filled in one pass, so later nodes reuse the result.
    landing = [_NO_LANDING] * packed_count

    def landing_y_for(x: int, y: int, rotation: int) -> int:
        state_id = pack(x, y, rotation)
        if state_id < 0:
            return _NO_LANDING
        cached = landing[state_id]
        if cached != _NO_LANDING:
            return cached

        trail: list[int] = []
        current_y = y
        result = _NO_LANDING
        while True:
            current_id = pack(x, current_y, rotation)
            if current_id < 0:
                break
            cached = landing[current_id]
            if cached != _NO_LANDING:
                result = cached
                break
            trail.append(current_id)
            if collides(x, current_y + 1, rotation):
                result = current_y
                break
            current_y += 1

        if result != _NO_LANDING:
            for cached_id in trail:
                landing[cached_id] = result
        return result

    # Parallel primitive arrays avoid allocating a dataclass and tuple hash key
    # for every reachable geometry.
    xs: list[int] = [game.x]
    ys: list[int] = [game.y]
    rotations: list[int] = [game.rotation & 3]
    parents: list[int] = [-1]
    commands: list[int] = [_CMD_NONE]
    depths: list[int] = [0]
    last_rotations: list[bool] = [False]
    kick_indices: list[int] = [-1]
    rotation_froms: list[int] = [-1]
    rotation_tos: list[int] = [-1]

    frontier: deque[int] = deque([0])
    state_nodes = [_NO_STATE] * packed_count
    rotation_nodes = [_NO_STATE] * packed_count
    state_nodes[start_state] = 0
    reachable_count = 1
    budget = max(1, int(max_nodes))

    def append_node(
        x: int,
        y: int,
        rotation: int,
        parent: int,
        command: int,
        depth: int,
        *,
        last_rotation: bool = False,
        kick_index: int = -1,
        rotation_from: int = -1,
        rotation_to: int = -1,
    ) -> int:
        index = len(xs)
        xs.append(x)
        ys.append(y)
        rotations.append(rotation & 3)
        parents.append(parent)
        commands.append(command)
        depths.append(depth)
        last_rotations.append(last_rotation)
        kick_indices.append(kick_index)
        rotation_froms.append(rotation_from)
        rotation_tos.append(rotation_to)
        return index

    while frontier and reachable_count <= budget:
        node_index = frontier.popleft()
        x = xs[node_index]
        y = ys[node_index]
        rotation = rotations[node_index]
        depth = depths[node_index]

        for dx, command in ((-1, _CMD_LEFT), (1, _CMD_RIGHT)):
            target_x = x + dx
            state_id = pack(target_x, y, rotation)
            if state_id < 0 or state_nodes[state_id] != _NO_STATE:
                continue
            if collides(target_x, y, rotation):
                continue
            successor = append_node(
                target_x,
                y,
                rotation,
                node_index,
                command,
                depth + 1,
            )
            state_nodes[state_id] = successor
            reachable_count += 1
            frontier.append(successor)

        target_y = y + 1
        state_id = pack(x, target_y, rotation)
        if (
            state_id >= 0
            and state_nodes[state_id] == _NO_STATE
            and not collides(x, target_y, rotation)
        ):
            successor = append_node(
                x,
                target_y,
                rotation,
                node_index,
                _CMD_DOWN,
                depth + 1,
            )
            state_nodes[state_id] = successor
            reachable_count += 1
            frontier.append(successor)

        if piece != "O":
            rotation_specs = [(1, _CMD_CW), (-1, _CMD_CCW)]
            if allow_180:
                rotation_specs.append((2, _CMD_180))
            for direction, command in rotation_specs:
                target_rotation = (rotation + direction) & 3
                rotated: tuple[int, int, int] | None = None
                for kick_index, (kick_x, kick_y) in enumerate(
                    kick_tests(piece, rotation, target_rotation)
                ):
                    target_x = x + kick_x
                    target_y = y + kick_y
                    if target_y < y_min:
                        continue
                    state_id = pack(target_x, target_y, target_rotation)
                    if state_id < 0:
                        continue
                    if not collides(target_x, target_y, target_rotation):
                        rotated = (target_x, target_y, kick_index)
                        break
                if rotated is None:
                    continue

                target_x, target_y, kick_index = rotated
                state_id = pack(target_x, target_y, target_rotation)
                previous_rotation = rotation_nodes[state_id]
                new_depth = depth + 1
                improves_rotation = (
                    previous_rotation == _NO_STATE
                    or new_depth < depths[previous_rotation]
                )
                adds_geometry = state_nodes[state_id] == _NO_STATE
                if not improves_rotation and not adds_geometry:
                    continue

                successor = append_node(
                    target_x,
                    target_y,
                    target_rotation,
                    node_index,
                    command,
                    new_depth,
                    last_rotation=True,
                    kick_index=kick_index,
                    rotation_from=rotation,
                    rotation_to=target_rotation,
                )
                if improves_rotation:
                    rotation_nodes[state_id] = successor
                if adds_geometry:
                    state_nodes[state_id] = successor
                    reachable_count += 1
                    frontier.append(successor)

        if reachable_count > budget:
            break

    # key -> (preference, x, landing_y, rotation, node, last_rotation,
    #         kick, rotation_from, rotation_to)
    best: dict[int, tuple[tuple[int, int, int], int, int, int, int, bool, int, int, int]] = {}

    def emit(node_index: int) -> None:
        x = xs[node_index]
        y = ys[node_index]
        rotation = rotations[node_index]
        landing_y = landing_y_for(x, y, rotation)
        if landing_y == _NO_LANDING:
            return
        key = _geometry_key(piece, x, landing_y, rotation, game.width)
        if key < 0:
            return
        last_rotation = last_rotations[node_index]
        spin_kind = (
            classify_t_spin_row_masks(
                rows,
                piece=piece,
                x=x,
                y=landing_y,
                rotation=rotation,
                last_move_was_rotation=True,
                rotation_kick_index=(
                    kick_indices[node_index] if kick_indices[node_index] >= 0 else None
                ),
                width=game.width,
            )
            if last_rotation
            else None
        )
        preference = (
            int(spin_kind is not None),
            int(spin_kind == "full"),
            -depths[node_index],
        )
        previous = best.get(key)
        if previous is None or preference > previous[0]:
            best[key] = (
                preference,
                x,
                landing_y,
                rotation,
                node_index,
                last_rotation,
                kick_indices[node_index] if last_rotation else -1,
                rotation_froms[node_index] if last_rotation else -1,
                rotation_tos[node_index] if last_rotation else -1,
            )

    if include_paths:
        # Preserve historical shortest-hard-drop routes for replay/UI callers.
        for node_index in state_nodes:
            if node_index != _NO_STATE:
                emit(node_index)
    else:
        # For pure evaluation, every non-rotation hard-drop geometry is already
        # represented by its reachable grounded state. Skip all higher duplicates.
        for node_index in state_nodes:
            if node_index == _NO_STATE:
                continue
            y = ys[node_index]
            if landing_y_for(xs[node_index], y, rotations[node_index]) == y:
                emit(node_index)

    # A hard drop does not clear the preceding rotation metadata, so rotation-ending
    # nodes must still be considered even when they are above the landing position.
    for node_index in rotation_nodes:
        if node_index != _NO_STATE:
            emit(node_index)

    placements: list[Placement] = []
    for (
        _preference,
        x,
        landing_y,
        rotation,
        node_index,
        last_rotation,
        kick_index,
        rotation_from,
        rotation_to,
    ) in best.values():
        placements.append(
            Placement(
                piece=piece,
                x=x,
                y=landing_y,
                rotation=rotation,
                cells=placement_cells(piece, x, landing_y, rotation),
                path=_path(parents, commands, node_index) if include_paths else (),
                last_move_was_rotation=last_rotation,
                rotation_kick_index=kick_index if kick_index >= 0 else None,
                rotation_from=rotation_from if rotation_from >= 0 else None,
                rotation_to=rotation_to if rotation_to >= 0 else None,
            )
        )

    placements.sort(key=lambda item: (item.rotation, item.x, item.y, len(item.path)))
    return tuple(placements)
