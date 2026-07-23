from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import asdict, dataclass
from heapq import nlargest

from minoflux_engine import Game, LockResult, Placement

from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights, PlacementEvaluation, rank_placements
from .reachability import reachable_placements


class _PreviewBag:
    __slots__ = ()

    def pop(self) -> str:
        return "T"


_PREVIEW_BAG = _PreviewBag()


def clone_game(game: Game) -> Game:
    cloned = copy(game)
    cloned.board = [row.copy() for row in game.board]
    cloned.queue = deque(game.queue)
    cloned._bag = _PREVIEW_BAG
    return cloned


@dataclass(frozen=True, slots=True)
class SearchConfig:
    allow_hold: bool = True
    lookahead_pieces: int = 1
    beam_width: int = 4
    discount: float = 0.90
    srs_reachable: bool = True
    allow_180: bool = False
    reachability_node_limit: int = 8_000

    def normalized(self) -> "SearchConfig":
        return SearchConfig(
            allow_hold=bool(self.allow_hold),
            lookahead_pieces=min(3, max(0, int(self.lookahead_pieces))),
            beam_width=min(128, max(1, int(self.beam_width))),
            discount=min(1.0, max(0.0, float(self.discount))),
            srs_reachable=bool(self.srs_reachable),
            allow_180=bool(self.allow_180),
            reachability_node_limit=min(50_000, max(100, int(self.reachability_node_limit))),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())


DEFAULT_SEARCH_CONFIG = SearchConfig()
DIRECT_SEARCH_CONFIG = SearchConfig(
    allow_hold=False,
    lookahead_pieces=0,
    beam_width=1,
    srs_reachable=False,
)


@dataclass(frozen=True, slots=True)
class SearchAction:
    use_hold: bool
    placement: Placement

    def to_dict(self) -> dict[str, object]:
        return {
            "hold": self.use_hold,
            "piece": self.placement.piece,
            "x": self.placement.x,
            "y": self.placement.y,
            "rotation": self.placement.rotation,
            "path": list(self.placement.path),
            "lastMoveWasRotation": self.placement.last_move_was_rotation,
            "rotationKickIndex": self.placement.rotation_kick_index,
        }


@dataclass(frozen=True, slots=True)
class SearchChoice:
    action: SearchAction
    score: float
    immediate: PlacementEvaluation
    path: tuple[SearchAction, ...]


@dataclass(slots=True)
class _BeamNode:
    game: Game
    score: float
    path: tuple[SearchAction, ...]
    first_evaluation: PlacementEvaluation


def _candidate_key(item: tuple[SearchAction, PlacementEvaluation]) -> tuple[float, int, int, int, int, int, int, int]:
    action, evaluation = item
    features = evaluation.features
    return (
        evaluation.score,
        features.attack,
        features.spin_lines,
        features.lines,
        -features.board.holes,
        -features.board.max_height,
        -int(action.use_hold),
        -action.placement.rotation * 100 - action.placement.x,
    )


def _placements_for_game(game: Game, config: SearchConfig) -> tuple[Placement, ...]:
    if not config.srs_reachable:
        return game.legal_placements()
    return reachable_placements(
        game,
        allow_180=config.allow_180,
        max_nodes=config.reachability_node_limit,
    )


def rank_search_actions(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
    *,
    limit: int | None = None,
) -> tuple[tuple[SearchAction, PlacementEvaluation], ...]:
    cfg = config.normalized()
    branch_limit = None if limit is None else max(1, int(limit))
    direct_placements = _placements_for_game(game, cfg)
    candidates: list[tuple[SearchAction, PlacementEvaluation]] = [
        (SearchAction(False, evaluation.placement), evaluation)
        for evaluation in rank_placements(
            game,
            weights,
            placements=direct_placements,
            limit=branch_limit,
        )
    ]

    if cfg.allow_hold and not game.hold_used and not game.game_over:
        held = clone_game(game)
        if held.hold():
            held_placements = _placements_for_game(held, cfg)
            candidates.extend(
                (SearchAction(True, evaluation.placement), evaluation)
                for evaluation in rank_placements(
                    held,
                    weights,
                    placements=held_placements,
                    limit=branch_limit,
                )
            )

    if limit is not None:
        count = max(0, int(limit))
        if count == 0:
            return ()
        if count < len(candidates):
            return tuple(nlargest(count, candidates, key=_candidate_key))
    candidates.sort(key=_candidate_key, reverse=True)
    return tuple(candidates)


def apply_search_action(game: Game, action: SearchAction) -> LockResult:
    if action.use_hold and not game.hold():
        raise ValueError("Search action requested an unavailable Hold")
    if game.current != action.placement.piece:
        raise ValueError(
            f"Search action expected {action.placement.piece}, but engine produced {game.current}"
        )
    return game.place(action.placement)


def _node_key(node: _BeamNode) -> tuple[float, int, int, int, int, int]:
    first = node.path[0]
    return (
        node.score,
        node.game.attack,
        node.game.lines,
        node.game.pieces_placed,
        -int(first.use_hold),
        -first.placement.rotation * 100 - first.placement.x,
    )


def choose_search_action(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> SearchChoice | None:
    cfg = config.normalized()
    root_limit = 1 if cfg.lookahead_pieces == 0 else cfg.beam_width
    ranked_root = rank_search_actions(game, weights, cfg, limit=root_limit)
    if not ranked_root:
        return None

    if cfg.lookahead_pieces == 0:
        action, evaluation = ranked_root[0]
        return SearchChoice(action, evaluation.score, evaluation, (action,))

    frontier: list[_BeamNode] = []
    for action, evaluation in ranked_root:
        child = clone_game(game)
        apply_search_action(child, action)
        frontier.append(
            _BeamNode(
                game=child,
                score=evaluation.score,
                path=(action,),
                first_evaluation=evaluation,
            )
        )
    frontier.sort(key=_node_key, reverse=True)
    frontier = frontier[: cfg.beam_width]

    for depth in range(1, cfg.lookahead_pieces + 1):
        expanded: list[_BeamNode] = []
        future_weight = cfg.discount ** depth
        for node in frontier:
            if node.game.game_over:
                expanded.append(node)
                continue
            for action, evaluation in rank_search_actions(
                node.game,
                weights,
                cfg,
                limit=cfg.beam_width,
            ):
                child = clone_game(node.game)
                apply_search_action(child, action)
                expanded.append(
                    _BeamNode(
                        game=child,
                        score=node.score + future_weight * evaluation.score,
                        path=(*node.path, action),
                        first_evaluation=node.first_evaluation,
                    )
                )
        if not expanded:
            break
        if len(expanded) > cfg.beam_width:
            frontier = nlargest(cfg.beam_width, expanded, key=_node_key)
        else:
            expanded.sort(key=_node_key, reverse=True)
            frontier = expanded

    best = max(frontier, key=_node_key)
    return SearchChoice(
        action=best.path[0],
        score=best.score,
        immediate=best.first_evaluation,
        path=best.path,
    )
