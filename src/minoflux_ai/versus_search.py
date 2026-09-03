from __future__ import annotations

from copy import copy
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, Sequence

from minoflux_engine import GarbagePacket, GarbageQueue, VersusMatch, VersusResolution, VersusSide

from .features import extract_board_features
from .heuristic import DEFAULT_WEIGHTS, HeuristicWeights, PlacementEvaluation
from .search import (
    DEFAULT_SEARCH_CONFIG,
    SearchAction,
    SearchConfig,
    SearchScorer,
    apply_search_action,
    clone_game,
    rank_search_actions,
)

SideName = Literal["player", "ai"]


class VersusStateScorer(Protocol):
    """Root-side value scorer for complete versus match states."""

    def score_match(
        self,
        match: VersusMatch,
        root_side: SideName,
        to_move: SideName | None = None,
    ) -> float: ...


@dataclass(frozen=True, slots=True)
class VersusWeights:
    solo_evaluation: float = 1.0
    state_value: float = 24.0
    sent_lines: float = 8.0
    canceled_lines: float = 6.0
    garbage_applied: float = -8.0
    own_pending: float = -5.0
    opponent_pending: float = 3.0
    own_max_height: float = -2.4
    opponent_max_height: float = 1.4
    own_holes: float = -5.0
    opponent_holes: float = 1.5
    own_b2b: float = 0.8
    opponent_b2b: float = -0.5
    own_surge: float = 1.4
    opponent_surge: float = -0.8
    input_cost: float = -0.025
    win: float = 1_000_000.0
    loss: float = -1_000_000.0

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VersusSearchConfig:
    placement_search: SearchConfig = field(default_factory=lambda: DEFAULT_SEARCH_CONFIG)
    candidate_width: int = 8
    opponent_reply_width: int = 2

    def normalized(self) -> "VersusSearchConfig":
        return VersusSearchConfig(
            placement_search=self.placement_search.normalized(),
            candidate_width=min(64, max(1, int(self.candidate_width))),
            opponent_reply_width=min(16, max(0, int(self.opponent_reply_width))),
        )

    def to_dict(self) -> dict[str, object]:
        cfg = self.normalized()
        return {
            "placementSearch": cfg.placement_search.to_dict(),
            "candidateWidth": cfg.candidate_width,
            "opponentReplyWidth": cfg.opponent_reply_width,
        }


DEFAULT_VERSUS_WEIGHTS = VersusWeights()
DEFAULT_VERSUS_SEARCH_CONFIG = VersusSearchConfig()


@dataclass(frozen=True, slots=True)
class VersusChoice:
    action: SearchAction
    score: float
    immediate: PlacementEvaluation
    resolution: VersusResolution
    opponent_reply: SearchAction | None = None


def _clone_queue(queue: GarbageQueue) -> GarbageQueue:
    cloned = GarbageQueue(queue.width)
    cloned.packets.extend(GarbagePacket(packet.lines, packet.hole) for packet in queue.packets)
    return cloned


def _clone_side(side: VersusSide) -> VersusSide:
    return VersusSide(
        game=clone_game(side.game),
        pending=_clone_queue(side.pending),
        sent=side.sent,
        received=side.received,
        canceled=side.canceled,
        garbage_applied=side.garbage_applied,
    )


def clone_versus_match(match: VersusMatch) -> VersusMatch:
    cloned = copy(match)
    cloned.player = _clone_side(match.player)
    cloned.ai = _clone_side(match.ai)
    cloned._garbage_rng = copy(match._garbage_rng)
    cloned._garbage_rng.setstate(match._garbage_rng.getstate())
    return cloned


def _side(match: VersusMatch, name: SideName) -> VersusSide:
    return match.player if name == "player" else match.ai


def _opponent_name(name: SideName) -> SideName:
    return "ai" if name == "player" else "player"


def score_versus_state(
    match: VersusMatch,
    root_side: SideName,
    *,
    weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS,
    resolution: VersusResolution | None = None,
    solo_score: float = 0.0,
    state_value: float = 0.0,
    path_length: int = 0,
    action_side: SideName | None = None,
) -> float:
    own = _side(match, root_side)
    opponent = _side(match, _opponent_name(root_side))
    if match.winner == root_side:
        return weights.win
    if match.winner == _opponent_name(root_side):
        return weights.loss
    if match.winner == "draw":
        return 0.0

    own_board = extract_board_features(own.game.board)
    opponent_board = extract_board_features(opponent.game.board)
    input_direction = 1.0 if action_side in (None, root_side) else -1.0
    score = (
        solo_score * weights.solo_evaluation
        + state_value * weights.state_value
        + own.pending.pending_lines * weights.own_pending
        + opponent.pending.pending_lines * weights.opponent_pending
        + own_board.max_height * weights.own_max_height
        + opponent_board.max_height * weights.opponent_max_height
        + own_board.holes * weights.own_holes
        + opponent_board.holes * weights.opponent_holes
        + own.game.b2b_chain * weights.own_b2b
        + opponent.game.b2b_chain * weights.opponent_b2b
        + own.game.surge_charge * weights.own_surge
        + opponent.game.surge_charge * weights.opponent_surge
        + input_direction * max(0, int(path_length)) * weights.input_cost
    )
    if resolution is not None:
        direction = 1.0 if resolution.side == root_side else -1.0
        score += direction * resolution.sent_lines * weights.sent_lines
        score += direction * resolution.canceled_lines * weights.canceled_lines
        score += direction * resolution.garbage_applied * weights.garbage_applied
    return score


def _simulate_action(
    match: VersusMatch,
    side_name: SideName,
    action: SearchAction,
) -> tuple[VersusMatch, VersusResolution]:
    simulated = clone_versus_match(match)
    result = apply_search_action(_side(simulated, side_name).game, action)
    resolution = simulated.resolve_lock(side_name, result)
    return simulated, resolution


def _state_values(
    scorer: VersusStateScorer | None,
    entries: Sequence[tuple[VersusMatch, SideName, SideName | None]],
) -> tuple[float, ...]:
    if not entries:
        return ()
    if scorer is None:
        return (0.0,) * len(entries)
    score_matches = getattr(scorer, "score_matches", None)
    if callable(score_matches):
        values = tuple(float(value) for value in score_matches(entries))
        if len(values) != len(entries):
            raise ValueError("Versus state scorer returned the wrong number of values")
        return values
    return tuple(float(scorer.score_match(match, root_side, to_move)) for match, root_side, to_move in entries)


def choose_versus_action(
    match: VersusMatch,
    side_name: SideName,
    heuristic_weights: HeuristicWeights = DEFAULT_WEIGHTS,
    config: VersusSearchConfig = DEFAULT_VERSUS_SEARCH_CONFIG,
    versus_weights: VersusWeights = DEFAULT_VERSUS_WEIGHTS,
    *,
    scorer: SearchScorer | None = None,
    opponent_scorer: SearchScorer | None = None,
    opponent_heuristic_weights: HeuristicWeights | None = None,
    state_scorer: VersusStateScorer | None = None,
) -> VersusChoice | None:
    """Choose a versus action with neural candidate, reply, and state-value scoring.

    ``opponent_scorer=None`` deliberately means heuristic opponent replies. Callers
    doing neural self-play should explicitly pass the same neural scorer for both
    ``scorer`` and ``opponent_scorer``.
    """

    cfg = config.normalized()
    own = _side(match, side_name)
    ranked = rank_search_actions(
        own.game,
        heuristic_weights,
        cfg.placement_search,
        limit=cfg.candidate_width,
        scorer=scorer,
    )
    if not ranked:
        return None

    opponent_name = _opponent_name(side_name)
    reply_scorer = opponent_scorer
    reply_weights = heuristic_weights if opponent_heuristic_weights is None else opponent_heuristic_weights

    root_candidates: list[
        tuple[SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]
    ] = []
    for action, evaluation in ranked:
        after, resolution = _simulate_action(match, side_name, action)
        root_candidates.append((action, evaluation, after, resolution))
    root_state_values = _state_values(
        state_scorer,
        tuple((after, side_name, opponent_name) for _action, _evaluation, after, _resolution in root_candidates),
    )

    best: VersusChoice | None = None
    for (action, evaluation, after, resolution), root_state_value in zip(
        root_candidates,
        root_state_values,
    ):
        base_score = score_versus_state(
            after,
            side_name,
            weights=versus_weights,
            resolution=resolution,
            solo_score=evaluation.score,
            state_value=root_state_value,
            path_length=len(action.placement.path),
            action_side=side_name,
        )
        reply_action: SearchAction | None = None
        score = base_score

        if (
            cfg.opponent_reply_width > 0
            and after.winner is None
            and not _side(after, opponent_name).game.game_over
        ):
            replies = rank_search_actions(
                _side(after, opponent_name).game,
                reply_weights,
                cfg.placement_search,
                limit=cfg.opponent_reply_width,
                scorer=reply_scorer,
            )
            if replies:
                reply_candidates: list[
                    tuple[SearchAction, PlacementEvaluation, VersusMatch, VersusResolution]
                ] = []
                for reply, reply_evaluation in replies:
                    replied, reply_resolution = _simulate_action(after, opponent_name, reply)
                    reply_candidates.append((reply, reply_evaluation, replied, reply_resolution))
                reply_state_values = _state_values(
                    state_scorer,
                    tuple(
                        (replied, side_name, side_name)
                        for _reply, _reply_evaluation, replied, _reply_resolution in reply_candidates
                    ),
                )
                worst_score: float | None = None
                worst_action: SearchAction | None = None
                for (
                    reply,
                    reply_evaluation,
                    replied,
                    reply_resolution,
                ), reply_state_value in zip(reply_candidates, reply_state_values):
                    reply_score = score_versus_state(
                        replied,
                        side_name,
                        weights=versus_weights,
                        resolution=reply_resolution,
                        solo_score=-reply_evaluation.score,
                        state_value=reply_state_value,
                        path_length=len(reply.placement.path),
                        action_side=opponent_name,
                    )
                    if worst_score is None or reply_score < worst_score:
                        worst_score = reply_score
                        worst_action = reply
                if worst_score is not None:
                    score = worst_score
                    reply_action = worst_action

        choice = VersusChoice(
            action=action,
            score=score,
            immediate=evaluation,
            resolution=resolution,
            opponent_reply=reply_action,
        )
        if best is None or (
            choice.score,
            choice.resolution.sent_lines,
            choice.resolution.canceled_lines,
            -len(choice.action.placement.path),
            -int(choice.action.use_hold),
        ) > (
            best.score,
            best.resolution.sent_lines,
            best.resolution.canceled_lines,
            -len(best.action.placement.path),
            -int(best.action.use_hold),
        ):
            best = choice
    return best
