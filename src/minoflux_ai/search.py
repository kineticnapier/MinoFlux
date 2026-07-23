from __future__ import annotations

from collections import deque
from copy import copy
from dataclasses import asdict, dataclass
import random

from minoflux_engine import Game, LockResult, Placement

from .heuristic import (
    DEFAULT_WEIGHTS,
    HeuristicWeights,
    PlacementEvaluation,
    rank_placements,
)


def clone_game(game: Game) -> Game:
    """Clone engine state without recursively copying immutable shape tables."""

    cloned = copy(game)
    cloned.board = [row.copy() for row in game.board]
    cloned.queue = deque(game.queue)
    cloned._bag = copy(game._bag)
    cloned._bag._queue = deque(game._bag._queue)
    cloned_rng = random.Random()
    cloned_rng.setstate(game._bag._rng.getstate())
    cloned._bag._rng = cloned_rng
    return cloned


@dataclass(frozen=True, slots=True)
class SearchConfig:
    """Configuration for Hold-aware beam search.

    ``lookahead_pieces`` counts future pieces beyond the current placement:
    0 is greedy one-ply search, 1 considers the next piece, and so on.
    """

    allow_hold: bool = True
    lookahead_pieces: int = 1
    beam_width: int = 4
    discount: float = 0.90

    def normalized(self) -> "SearchConfig":
        return SearchConfig(
            allow_hold=bool(self.allow_hold),
            lookahead_pieces=min(3, max(0, int(self.lookahead_pieces))),
            beam_width=min(128, max(1, int(self.beam_width))),
            discount=min(1.0, max(0.0, float(self.discount))),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self.normalized())


DEFAULT_SEARCH_CONFIG = SearchConfig()
DIRECT_SEARCH_CONFIG = SearchConfig(allow_hold=False, lookahead_pieces=0, beam_width=1)


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


def _candidate_key(item: tuple[SearchAction, PlacementEvaluation]) -> tuple[float, int, int, int, int, int, int]:
    action, evaluation = item
    features = evaluation.features
    return (
        evaluation.score,
        features.attack,
        features.lines,
        -features.board.holes,
        -features.board.max_height,
        -int(action.use_hold),
        -action.placement.rotation * 100 - action.placement.x,
    )


def rank_search_actions(
    game: Game,
    weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: SearchConfig = DEFAULT_SEARCH_CONFIG,
) -> tuple[tuple[SearchAction, PlacementEvaluation], ...]:
    """Rank legal direct placements and optional Hold placements."""

    cfg = config.normalized()
    candidates: list[tuple[SearchAction, PlacementEvaluation]] = [
        (SearchAction(False, evaluation.placement), evaluation)
        for evaluation in rank_placements(game, weights)
    ]

    if cfg.allow_hold and not game.hold_used and not game.game_over:
        held = clone_game(game)
        if held.hold():
            candidates.extend(
                (SearchAction(True, evaluation.placement), evaluation)
                for evaluation in rank_placements(held, weights)
            )

    candidates.sort(key=_candidate_key, reverse=True)
    return tuple(candidates)


def apply_search_action(game: Game, action: SearchAction) -> LockResult:
    """Apply a selected Hold/placement action to a real or simulated game."""

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
    """Choose an action with Hold-aware discounted beam search.

    Greedy search (lookahead 0) avoids child-game copies. Deeper search keeps
    only ``beam_width`` states after each ply, making one-piece lookahead much
    cheaper than exhaustive two-ply search.
    """

    cfg = config.normalized()
    ranked_root = rank_search_actions(game, weights, cfg)
    if not ranked_root:
        return None

    if cfg.lookahead_pieces == 0:
        action, evaluation = ranked_root[0]
        return SearchChoice(action, evaluation.score, evaluation, (action,))

    frontier: list[_BeamNode] = []
    for action, evaluation in ranked_root[: cfg.beam_width]:
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

    for _ in range(cfg.lookahead_pieces):
        expanded: list[_BeamNode] = []
        for node in frontier:
            if node.game.game_over:
                expanded.append(node)
                continue
            for action, evaluation in rank_search_actions(node.game, weights, cfg)[: cfg.beam_width]:
                child = clone_game(node.game)
                apply_search_action(child, action)
                expanded.append(
                    _BeamNode(
                        game=child,
                        score=node.score * (1.0 - cfg.discount) + evaluation.score * cfg.discount,
                        path=(*node.path, action),
                        first_evaluation=node.first_evaluation,
                    )
                )
        if not expanded:
            break
        expanded.sort(key=_node_key, reverse=True)
        frontier = expanded[: cfg.beam_width]

    best = max(frontier, key=_node_key)
    return SearchChoice(
        action=best.path[0],
        score=best.score,
        immediate=best.first_evaluation,
        path=best.path,
    )
