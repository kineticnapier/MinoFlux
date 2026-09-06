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
    real garbage is present, survival and digging override the Quad discipline.
    """

    danger_height: int = 10
    critical_height: int = 14
    holes: float = -160.0
    new_holes: float = -400.0
    removed_holes: float = 220.0
    hole_depth: float = -20.0
    aggregate_height: float = -0.90
    max_height: float = -11.0
    danger_height_quadratic: float = -12.0
    critical_height_penalty: float = -350.0
    stack_bumpiness: float = -7.0
    center_peak: float = -11.0
    right_well_depth: float = 13.0
    deep_right_well: float = -4.0
    right_well_fill: float = -45.0
    right_overstack: float = -35.0
    garbage_well_scale: float = 0.08
    garbage_bumpiness_scale: float = 2.5
    garbage_center_peak_scale: float = 4.0

    # Generic board holes do not describe the intentional gap in a garbage row.
    # Explicit excavation terms stop the policy from building a clean-looking
    # mountain over the channel instead of digging toward it.
    garbage_channel_blockers: float = -180.0
    new_garbage_channel_blockers: float = -320.0
    removed_garbage_channel_blockers: float = 180.0
    garbage_cover_cells: float = -8.0
    garbage_stack_height: float = -55.0
    garbage_rows_removed: float = 650.0
    garbage_cells_removed: float = 28.0

    perfect_clear: float = 220.0
    topout: float = -1_000_000_000.0


FESTIVAL_SAFE_CONFIG = FestivalSafeConfig()


@dataclass(frozen=True, slots=True)
class GarbageExcavationMetrics:
    rows: int = 0
    cells: int = 0
    channel_blockers: int = 0
    cover_cells: int = 0
    stack_height: int = 0


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


def _garbage_excavation_metrics(
    board: Sequence[Sequence[str | None]],
) -> GarbageExcavationMetrics:
    """Measure how buried the currently applied garbage is.

    Each live garbage row normally has an intentional empty gap. That gap is not
    a generic board hole, so count occupied cells above each available gap and
    penalize burying the route needed to excavate the garbage stack.
    """

    garbage_rows = [
        (y, row)
        for y, row in enumerate(board)
        if any(cell == "G" for cell in row)
    ]
    if not garbage_rows:
        return GarbageExcavationMetrics()

    garbage_cells = sum(
        1
        for _y, row in garbage_rows
        for cell in row
        if cell == "G"
    )
    channel_blockers = 0
    for y, row in garbage_rows:
        gaps = [x for x, cell in enumerate(row) if cell is None]
        if not gaps:
            # A full row should vanish on lock. Treat malformed/manual states
            # conservatively instead of declaring them easy to excavate.
            channel_blockers += len(row)
            continue
        channel_blockers += min(
            sum(board[above_y][x] is not None for above_y in range(y))
            for x in gaps
        )

    top_garbage_y = min(y for y, _row in garbage_rows)
    cover_cells = sum(
        cell is not None and cell != "G"
        for row in board[:top_garbage_y]
        for cell in row
    )
    non_garbage_rows_above = [
        y
        for y, row in enumerate(board[:top_garbage_y])
        if any(cell is not None and cell != "G" for cell in row)
    ]
    stack_height = (
        0
        if not non_garbage_rows_above
        else top_garbage_y - min(non_garbage_rows_above)
    )
    return GarbageExcavationMetrics(
        rows=len(garbage_rows),
        cells=garbage_cells,
        channel_blockers=channel_blockers,
        cover_cells=cover_cells,
        stack_height=stack_height,
    )


def _board_after_placement(
    game: Game,
    placement: Placement,
) -> tuple[list[list[str | None]], int, bool]:
    """Simulate one legal placement while preserving garbage-cell identity."""

    board = [row.copy() for row in game.board]
    topped_out = False
    for cell_x, cell_y in placement.cells:
        if cell_y < 0:
            topped_out = True
        elif 0 <= cell_y < game.height:
            board[cell_y][cell_x] = placement.piece

    full_rows = [
        index
        for index, row in enumerate(board)
        if all(cell is not None for cell in row)
    ]
    for index in reversed(full_rows):
        del board[index]
    for _ in full_rows:
        board.insert(0, [None] * game.width)
    return board, len(full_rows), topped_out


def _line_clear_score(lines: int, *, danger: bool, garbage: bool) -> float:
    if lines <= 0:
        return 0.0
    if lines >= 4:
        return 350.0 if not danger and not garbage else 300.0
    if danger:
        return (0.0, 100.0, 180.0, 260.0)[lines]
    if garbage:
        return (0.0, 80.0, 150.0, 220.0)[lines]
    # On a healthy clean board, preserve the stack for a Quad instead of
    # cashing out weak line clears.
    return (0.0, -25.0, -15.0, -8.0)[lines]


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
    garbage_before: GarbageExcavationMetrics | None = None,
    garbage_after: GarbageExcavationMetrics | None = None,
) -> float:
    game_over = topped_out or hidden_rows_occupied(after_rows, game.hidden_rows)
    if game_over:
        return config.topout

    new_holes = max(0, after.holes - before.holes)
    removed_holes = max(0, before.holes - after.holes)
    danger = before.max_height >= config.danger_height or before.holes > 0

    score = (
        after.holes * config.holes
        + new_holes * config.new_holes
        + removed_holes * config.removed_holes
        + after.hole_depth * config.hole_depth
        + after.aggregate_height * config.aggregate_height
        + after.max_height * config.max_height
    )

    danger_excess = max(0, after.max_height - 8)
    score += danger_excess * danger_excess * config.danger_height_quadratic
    critical_excess = max(0, after.max_height - config.critical_height + 1)
    score += critical_excess * config.critical_height_penalty

    heights = _column_heights(after_rows, game.width)
    if len(heights) >= 2:
        # Garbage mode has no privileged Quad-well edge: smooth the full surface
        # so the old 9-0 exemption cannot turn into a post-garbage mountain.
        surface = heights if garbage else heights[:-1]
        stack_bumpiness = sum(
            abs(left - right) for left, right in zip(surface, surface[1:])
        )
        bumpiness_scale = config.garbage_bumpiness_scale if garbage else 1.0
        score += stack_bumpiness * config.stack_bumpiness * bumpiness_scale

        if len(surface) >= 5:
            center = surface[2:-2] or surface
            shoulders = surface[:2] + surface[-2:]
            center_peak = max(0, max(center) - max(shoulders, default=0))
            center_scale = config.garbage_center_peak_scale if garbage else 1.0
            score += center_peak * config.center_peak * center_scale

        stack = heights[:-1]
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

    if garbage and garbage_before is not None and garbage_after is not None:
        new_blockers = max(
            0,
            garbage_after.channel_blockers - garbage_before.channel_blockers,
        )
        removed_blockers = max(
            0,
            garbage_before.channel_blockers - garbage_after.channel_blockers,
        )
        removed_rows = max(0, garbage_before.rows - garbage_after.rows)
        removed_cells = max(0, garbage_before.cells - garbage_after.cells)
        score += (
            garbage_after.channel_blockers * config.garbage_channel_blockers
            + new_blockers * config.new_garbage_channel_blockers
            + removed_blockers * config.removed_garbage_channel_blockers
            + garbage_after.cover_cells * config.garbage_cover_cells
            + garbage_after.stack_height * config.garbage_stack_height
            + removed_rows * config.garbage_rows_removed
            + removed_cells * config.garbage_cells_removed
        )

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
    garbage_before: GarbageExcavationMetrics | None = None,
) -> float:
    """Score one already-legal placement without mutating ``game``."""

    rows = tuple(source_rows) if source_rows is not None else board_row_masks(game.board)
    before_features = before or extract_board_features_from_masks(rows, width=game.width)
    garbage_mode = bool(garbage) if garbage is not None else _has_garbage(game)
    after_rows, lines, topped_out = place_and_clear_row_masks(
        rows,
        placement,
        width=game.width,
    )
    after = extract_board_features_from_masks(after_rows, width=game.width)

    before_excavation = garbage_before
    after_excavation = None
    if garbage_mode:
        before_excavation = before_excavation or _garbage_excavation_metrics(game.board)
        after_board, board_lines, board_topped_out = _board_after_placement(game, placement)
        if board_lines != lines or board_topped_out != topped_out:
            raise AssertionError("Garbage-preserving placement simulation diverged")
        after_excavation = _garbage_excavation_metrics(after_board)

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
        garbage_before=before_excavation,
        garbage_after=after_excavation,
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
        garbage = _has_garbage(game)
        garbage_before = _garbage_excavation_metrics(game.board) if garbage else None
        return tuple(
            score_festival_safe_placement(
                game,
                placement,
                config=self.config,
                source_rows=rows,
                before=before,
                garbage=garbage,
                garbage_before=garbage_before,
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
