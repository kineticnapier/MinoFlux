from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Mapping

from minoflux_engine import Game, Placement

from . import reachability as _reachability_reference
from . import reachability_pathless as _reachability_pathless
from .bitboard import board_row_masks, placement_cells
from .reachability_native import _pack_masks
from .search import SearchAction, choose_search_action, clone_game

try:
    from . import _oracle_native as _native
except (ImportError, OSError):
    _native = None


@dataclass(frozen=True, slots=True)
class OracleConfig:
    beam_width: int = 2_000
    depth: int = 18
    allow_180: bool = True
    reachability_node_limit: int = 8_000

    def normalized(self) -> "OracleConfig":
        return OracleConfig(
            beam_width=max(1, int(self.beam_width)),
            depth=max(1, int(self.depth)),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(
                50_000,
                max(100, int(self.reachability_node_limit)),
            ),
        )


@dataclass(frozen=True, slots=True)
class OracleChoice:
    action: SearchAction
    score: float


def oracle_native_available() -> bool:
    return _native is not None and int(_native.api_version()) == 1


@cache
def _ensure_native_reachability_tables(
    allow_180: bool,
    width: int,
    height: int,
) -> None:
    if _native is None:
        raise RuntimeError("MinoFlux native oracle extension is unavailable")
    if width != 10 or height != 24:
        raise ValueError("native oracle currently supports only a 10x24 board")

    for piece in "IJLOSTZ":
        tables = _reachability_reference._state_tables(piece, width, height)
        collision_invalid, collision_masks = _pack_masks(tables.collision_mask)
        geometry_invalid, geometry_masks = _pack_masks(tables.geometry_mask)
        rotation_transitions = _reachability_pathless._rotation_kick_groups(
            piece,
            bool(allow_180),
            width,
            height,
        )
        _native.register_reachability_table(
            piece,
            bool(allow_180),
            width,
            height,
            tables.x_min,
            tables.x_max,
            tables.x_count,
            tables.y_min,
            tables.state_x,
            tables.state_y,
            tables.left_state,
            tables.right_state,
            tables.down_state,
            collision_invalid,
            collision_masks,
            geometry_invalid,
            geometry_masks,
            rotation_transitions,
        )


def _optional_index(value: object) -> int | None:
    index = int(value)
    return None if index < 0 else index


def _choice_from_native(value: Mapping[str, object]) -> OracleChoice:
    piece = str(value["piece"])
    x = int(value["x"])
    y = int(value["y"])
    rotation = int(value["rotation"])
    kick_index = _optional_index(value["kickIndex"])
    rotation_from = _optional_index(value["rotationFrom"])
    rotation_to = _optional_index(value["rotationTo"])
    placement = Placement(
        piece=piece,
        x=x,
        y=y,
        rotation=rotation,
        cells=placement_cells(piece, x, y, rotation),
        path=(),
        last_move_was_rotation=bool(value["lastMoveWasRotation"]),
        rotation_kick_index=kick_index,
        rotation_from=rotation_from,
        rotation_to=rotation_to,
    )
    return OracleChoice(
        action=SearchAction(bool(value["holdUsed"]), placement),
        score=float(value["score"]),
    )


def search_oracle(
    game: Game,
    config: OracleConfig = OracleConfig(),
) -> OracleChoice | None:
    if not oracle_native_available():
        raise RuntimeError("MinoFlux native oracle extension is unavailable")
    if game.game_over or game.paused:
        return None

    cfg = config.normalized()
    _ensure_native_reachability_tables(
        cfg.allow_180,
        game.width,
        game.height,
    )
    request_game = clone_game(game)
    request_game._fill_queue(cfg.depth + 2)
    value = _native.search(
        board_row_masks(request_game.board),
        request_game.current,
        request_game.hold_piece,
        tuple(request_game.queue),
        int(request_game.combo),
        bool(request_game.back_to_back),
        int(request_game.b2b_chain),
        not bool(request_game.hold_used),
        cfg.beam_width,
        cfg.depth,
        cfg.allow_180,
        cfg.reachability_node_limit,
    )
    if value is None:
        return None
    return _choice_from_native(value)
