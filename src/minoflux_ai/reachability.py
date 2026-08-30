from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from minoflux_engine import Game, Placement
from minoflux_engine.pieces import SHAPES, kick_tests
from minoflux_engine.spin import classify_t_spin

MOVE_LEFT = "left"
MOVE_RIGHT = "right"
MOVE_DOWN = "down"
ROTATE_CW = "cw"
ROTATE_CCW = "ccw"
ROTATE_180 = "180"
HARD_DROP = "hard_drop"


@dataclass(slots=True)
class _Node:
    x: int
    y: int
    rotation: int
    parent: int | None
    command: str | None
    depth: int
    last_move_was_rotation: bool = False
    kick_index: int | None = None
    rotation_from: int | None = None
    rotation_to: int | None = None

    def geometry_key(self) -> tuple[int, int, int]:
        return self.x, self.y, self.rotation


@dataclass(slots=True)
class _Candidate:
    x: int
    y: int
    rotation: int
    cells: tuple[tuple[int, int], ...]
    node_index: int
    last_move_was_rotation: bool
    kick_index: int | None
    rotation_from: int | None
    rotation_to: int | None
    spin_kind: str | None
    depth: int

    def preference(self) -> tuple[int, int, int]:
        return int(self.spin_kind is not None), int(self.spin_kind == "full"), -self.depth


def _collides(game: Game, x: int, y: int, rotation: int) -> bool:
    """Hot-path collision test without allocating ``Game.cells()`` tuples."""

    board = game.board
    width = game.width
    height = game.height
    for dx, dy in SHAPES[game.current][rotation % 4]:
        cell_x, cell_y = x + dx, y + dy
        if cell_x < 0 or cell_x >= width or cell_y >= height:
            return True
        if cell_y >= 0 and board[cell_y][cell_x] is not None:
            return True
    return False


def _cells(game: Game, x: int, y: int, rotation: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (x + dx, y + dy)
        for dx, dy in SHAPES[game.current][rotation % 4]
    )


def _landing_y(
    game: Game,
    node: _Node,
    cache: dict[tuple[int, int, int], int],
) -> int:
    key = node.geometry_key()
    cached = cache.get(key)
    if cached is not None:
        return cached
    y = node.y
    while not _collides(game, node.x, y + 1, node.rotation):
        y += 1
    cache[key] = y
    return y


def _path(nodes: list[_Node], node_index: int) -> tuple[str, ...]:
    commands: list[str] = []
    index: int | None = node_index
    while index is not None:
        node = nodes[index]
        if node.command is not None:
            commands.append(node.command)
        index = node.parent
    commands.reverse()
    commands.append(HARD_DROP)
    return tuple(commands)


def _emit_candidate(
    game: Game,
    nodes: list[_Node],
    node_index: int,
    best_by_cells: dict[tuple[tuple[int, int], ...], _Candidate],
    landing_cache: dict[tuple[int, int, int], int],
) -> None:
    node = nodes[node_index]
    landing_y = _landing_y(game, node, landing_cache)
    cells = _cells(game, node.x, landing_y, node.rotation)
    if any(cell_y < 0 for _, cell_y in cells):
        return
    spin_kind = classify_t_spin(
        game.board,
        piece=game.current,
        x=node.x,
        y=landing_y,
        rotation=node.rotation,
        last_move_was_rotation=node.last_move_was_rotation,
        rotation_kick_index=node.kick_index,
    )
    candidate = _Candidate(
        x=node.x,
        y=landing_y,
        rotation=node.rotation,
        cells=cells,
        node_index=node_index,
        last_move_was_rotation=node.last_move_was_rotation,
        kick_index=node.kick_index if node.last_move_was_rotation else None,
        rotation_from=node.rotation_from if node.last_move_was_rotation else None,
        rotation_to=node.rotation_to if node.last_move_was_rotation else None,
        spin_kind=spin_kind,
        depth=node.depth,
    )
    key = tuple(sorted(cells))
    previous = best_by_cells.get(key)
    if previous is None or candidate.preference() > previous.preference():
        best_by_cells[key] = candidate


def _rotation_node(
    game: Game,
    node: _Node,
    node_index: int,
    direction: int,
    command: str,
) -> _Node | None:
    if game.current == "O":
        return None
    target = (node.rotation + direction) % 4
    for kick_index, (kick_x, kick_y) in enumerate(kick_tests(game.current, node.rotation, target)):
        x, y = node.x + kick_x, node.y + kick_y
        if y < -4:
            continue
        if not _collides(game, x, y, target):
            return _Node(
                x=x,
                y=y,
                rotation=target,
                parent=node_index,
                command=command,
                depth=node.depth + 1,
                last_move_was_rotation=True,
                kick_index=kick_index,
                rotation_from=node.rotation,
                rotation_to=target,
            )
    return None


def reachable_placements(
    game: Game,
    *,
    allow_180: bool = False,
    max_nodes: int = 8_000,
) -> tuple[Placement, ...]:
    """Enumerate placements reachable by movement, exact SRS, and hard drop.

    Geometry is explored once per ``(x, y, rotation)`` state. A separate shortest
    rotation-ending route is retained for each geometry so exact T-spin metadata
    is not lost when the same state is also reachable by movement. Lock-delay
    timing is intentionally not simulated.
    """

    if game.game_over or game.paused:
        return ()
    start = _Node(game.x, game.y, game.rotation, None, None, 0)
    if _collides(game, start.x, start.y, start.rotation):
        return ()

    nodes: list[_Node] = [start]
    frontier: deque[int] = deque([0])
    state_nodes: dict[tuple[int, int, int], int] = {start.geometry_key(): 0}
    rotation_nodes: dict[tuple[int, int, int], int] = {}
    budget = max(1, int(max_nodes))

    while frontier and len(state_nodes) <= budget:
        node_index = frontier.popleft()
        node = nodes[node_index]

        for dx, command in ((-1, MOVE_LEFT), (1, MOVE_RIGHT)):
            x = node.x + dx
            key = (x, node.y, node.rotation)
            if key in state_nodes or _collides(game, x, node.y, node.rotation):
                continue
            successor_index = len(nodes)
            nodes.append(_Node(x, node.y, node.rotation, node_index, command, node.depth + 1))
            state_nodes[key] = successor_index
            frontier.append(successor_index)

        down_y = node.y + 1
        key = (node.x, down_y, node.rotation)
        if key not in state_nodes and not _collides(game, node.x, down_y, node.rotation):
            successor_index = len(nodes)
            nodes.append(_Node(node.x, down_y, node.rotation, node_index, MOVE_DOWN, node.depth + 1))
            state_nodes[key] = successor_index
            frontier.append(successor_index)

        rotations = [
            _rotation_node(game, node, node_index, 1, ROTATE_CW),
            _rotation_node(game, node, node_index, -1, ROTATE_CCW),
        ]
        if allow_180:
            rotations.append(_rotation_node(game, node, node_index, 2, ROTATE_180))
        for successor in rotations:
            if successor is None:
                continue
            successor_index = len(nodes)
            nodes.append(successor)
            key = successor.geometry_key()
            previous_rotation = rotation_nodes.get(key)
            if previous_rotation is None or successor.depth < nodes[previous_rotation].depth:
                rotation_nodes[key] = successor_index
            if key not in state_nodes:
                state_nodes[key] = successor_index
                frontier.append(successor_index)

        if len(state_nodes) > budget:
            break

    best_by_cells: dict[tuple[tuple[int, int], ...], _Candidate] = {}
    landing_cache: dict[tuple[int, int, int], int] = {}
    for node_index in state_nodes.values():
        _emit_candidate(game, nodes, node_index, best_by_cells, landing_cache)
    for node_index in rotation_nodes.values():
        _emit_candidate(game, nodes, node_index, best_by_cells, landing_cache)

    placements = [
        Placement(
            piece=game.current,
            x=candidate.x,
            y=candidate.y,
            rotation=candidate.rotation,
            cells=candidate.cells,
            path=_path(nodes, candidate.node_index),
            last_move_was_rotation=candidate.last_move_was_rotation,
            rotation_kick_index=candidate.kick_index,
            rotation_from=candidate.rotation_from,
            rotation_to=candidate.rotation_to,
        )
        for candidate in best_by_cells.values()
    ]
    placements.sort(key=lambda item: (item.rotation, item.x, item.y, len(item.path)))
    return tuple(placements)
