from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from minoflux_engine import Game, Placement

from .bitboard import board_row_masks, hidden_rows_occupied, place_and_clear_row_masks
from .features import BoardFeatures, extract_board_features_from_masks
from .heuristic import PlacementEvaluation


FESTIVAL_SAFE_MODEL = "festival-safe"


@dataclass(frozen=True, slots=True)
class FestivalSafeConfig:
    """Conservative 9-0/Quad fallback policy for live demonstrations.

    The scorer intentionally has no learned parameters. It evaluates every legal
    exact-SRS placement supplied by the search layer and strongly prefers clean,
    low boards with an open right-side Quad well. When the stack is dangerous or
    garbage/holes are present, survival and digging override the Quad discipline.
    """

    danger_height: int = 12
    critical_height: int = 16
    holes: float = -105.0
    new_holes: float = -240.0
    removed_holes: float = 150.0
    hole_depth: float = -11.0
    aggregate_height: float = -1.35
    max_height: float = -7.0
    danger_height_quadratic: float = -8.0
    critical_height_penalty: float = -180.0
    stack_bumpiness: float = -4.5
    center_peak: float = -7.0
    right_well_depth: float = 18.0
    deep_right_well: float = 3.0
    right_well_fill: float = -70.0
    right_overstack: float = -22.0
    garbage_well_scale: float = 0.12
    perfect_clear: float = 220.0
    topout: float = -1_000_000_000.0


FESTIVAL_SAFE_CONFIG = FestivalSafeConfig()


def _column_heights(rows: Sequence[int], width: int) -> tuple[int, ...]:
    height = len(rows)
    result: list[int] = []
    for x in range(width):
        bit = 1 << x
        top = next((y for y, row in enumerate(rows) if row & bit), height)
        result.append(0 if top == height else height - top)
    return tuple(result)


def _has_garbage(game: Game) -> bool:
    return any(cell == "G" for row in game.board for cell in row)


def _line_clear_score(lines: int, *, danger: bool, garbage: bool) -> float:
    if lines <= 0:
        return 0.0
    if lines >= 4:
        return 300.0 if not danger and not garbage else 245.0
    if danger:
        return (0.0, 70.0, 125.0, 185.0)[lines]
    if garbage:
        return (0.0, 55.0, 105.0, 165.0)[lines]
    # On a healthy clean board, preserve the stack for a Quad instead of
    # cashing out weak line clears.
    return (0.0, -38.0, -24.0, -12.0)[lines]


def _score_after_rows(
    *,
    game: Game,
    placement: Placement,
    before: BoardFeatures,
    after_rows: Sequence[int],
    after: BoardFeatures,
    lines: int,
    topped_out: bool,
    garbage: bool,
    config: FestivalSafeConfig,
) -> float:
    game_over = topped_out or hidden_rows_occupied(after_rows, game.hidden_rows)
    if game_over:
        return config.topout

    new_holes = max(0, after.holes - before.holes)
    removed_holes = max(0, before.holes - after.holes)
    danger = before.max_height >= config.danger_height or before.holes >= 3

    score = (
        after.holes * config.holes
        + new_holes * config.new_holes
        + removed_holes * config.removed_holes
        + after.hole_depth * config.hole_depth
        + after.aggregate_height * config.aggregate_height
        + after.max_height * config.max_height
    )

    danger_excess = max(0, after.max_height - 9)
    score += danger_excess * danger_excess * config.danger_height_quadratic
    critical_excess = max(0, after.max_height - config.critical_height + 1)
    score += critical_excess * config.critical_height_penalty

    heights = _column_heights(after_rows, game.width)
    if len(heights) >= 2:
        # Keep the first width-1 columns smooth. The final 8->9 height jump is
        # deliberately excluded here because it is the intended Quad well.
        stack = heights[:-1]
        stack_bumpiness = sum(
            abs(left - right) for left, right in zip(stack, stack[1:])
        )
        score += stack_bumpiness * config.stack_bumpiness

        if len(stack) >= 5:
            center = stack[2:-2] or stack
            shoulders = stack[:2] + stack[-2:]
            center_peak = max(0, max(center) - max(shoulders, default=0))
            score += center_peak * config.center_peak

        right_height = heights[-1]
        left_floor = min(stack, default=0)
        well_depth = max(0, left_floor - right_height)
        well_score = (
            min(4, well_depth) * config.right_well_depth
            + max(0, well_depth - 4) * config.deep_right_well
        )
        if garbage or danger:
            well_score *= config.garbage_well_scale
        score += well_score

        right_overstack = max(0, right_height - left_floor)
        score += right_overstack * config.right_overstack

        fills_right = any(cell_x == game.width - 1 for cell_x, _cell_y in placement.cells)
        if fills_right and lines < 4 and not garbage and not danger:
            score += config.right_well_fill

    score += _line_clear_score(lines, danger=danger, garbage=garbage)
    if after.occupied_cells == 0 and lines > 0:
        score += config.perfect_clear
    return score


def score_festival_safe_placement(
    game: Game,
    placement: Placement,
    *,
    config: FestivalSafeConfig = FESTIVAL_SAFE_CONFIG,
    source_rows: Sequence[int] | None = None,
    before: BoardFeatures | None = None,
    garbage: bool | None = None,
) -> float:
    """Score one already-legal placement without mutating ``game``."""

    rows = tuple(source_rows) if source_rows is not None else board_row_masks(game.board)
    before_features = before or extract_board_features_from_masks(rows, width=game.width)
    garbage_mode = (
        bool(garbage)
        if garbage is not None
        else (_has_garbage(game) or before_features.holes > 0)
    )
    after_rows, lines, topped_out = place_and_clear_row_masks(
        rows,
        placement,
        width=game.width,
    )
    after = extract_board_features_from_masks(after_rows, width=game.width)
    return _score_after_rows(
        game=game,
        placement=placement,
        before=before_features,
        after_rows=after_rows,
        after=after,
        lines=lines,
        topped_out=topped_out,
        garbage=garbage_mode,
        config=config,
    )


@dataclass(frozen=True, slots=True)
class FestivalSafeScorer:
    config: FestivalSafeConfig = FESTIVAL_SAFE_CONFIG

    def score_placements(
        self,
        game: Game,
        placements: Sequence[Placement],
    ) -> tuple[float, ...]:
        if not placements:
            return ()
        rows = board_row_masks(game.board)
        before = extract_board_features_from_masks(rows, width=game.width)
        garbage = _has_garbage(game) or before.holes > 0
        return tuple(
            score_festival_safe_placement(
                game,
                placement,
                config=self.config,
                source_rows=rows,
                before=before,
                garbage=garbage,
            )
            for placement in placements
        )

    def score_many(
        self,
        game: Game,
        evaluations: Sequence[PlacementEvaluation],
    ) -> tuple[float, ...]:
        return self.score_placements(
            game,
            tuple(evaluation.placement for evaluation in evaluations),
        )

    def score_placement_groups(
        self,
        groups: Sequence[tuple[Game, Sequence[Placement]]],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(self.score_placements(game, placements) for game, placements in groups)


FESTIVAL_SAFE_SCORER = FestivalSafeScorer()
