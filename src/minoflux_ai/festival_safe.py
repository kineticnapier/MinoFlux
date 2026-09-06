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

    # Garbage excavation is deliberately different from keeping a permanent
    # open well. A block above the current garbage gap is acceptable when its
    # row is almost ready to clear; what is dangerous is creating sparse new
    # layers that turn the stack into a mountain.
    garbage_channel_blockers: float = -18.0
    new_garbage_channel_blockers: float = -28.0
    removed_garbage_channel_blockers: float = 14.0
    garbage_channel_debt: float = -55.0
    garbage_clear_debt: float = -42.0
    garbage_surface_spread: float = -90.0
    garbage_surface_variation: float = -30.0
    garbage_cover_cells: float = -10.0
    garbage_stack_height: float = -80.0
    garbage_rows_removed: float = 900.0
    garbage_cells_removed: float = 30.0

    perfect_clear: float = 220.0
    topout: float = -1_000_000_000.0


FESTIVAL_SAFE_CONFIG = FestivalSafeConfig()


@dataclass(frozen=True, slots=True)
class GarbageExcavationMetrics:
    rows: int = 0
    cells: int = 0
    channel_blockers: int = 0
    channel_debt: int = 0
    clear_debt: int = 0
    cover_cells: int = 0
    stack_height: int = 0
    surface_spread: int = 0
    surface_variation: int = 0


def _column_heights(rows: Sequence[int], width: int) -> tuple[int, ...]:
    height = len(rows)
    result: list[int] = []
    for x in range(width):
        bit = 1 << x
        top = next((y for y, row in enumerate(rows) if row & bit), height)
        result.append(0 if top == height else height - top)
    return tuple(result)


def _board_column_heights(board: Sequence[Sequence[str | None]]) -> tuple[int, ...]:
    if not board:
        return ()
    height = len(board)
    width = len(board[0])
    result: list[int] = []
    for x in range(width):
        top = next((y for y, row in enumerate(board) if row[x] is not None), height)
        result.append(0 if top == height else height - top)
    return tuple(result)


def _has_garbage(game: Game) -> bool:
    return any(cell == "G" for row in game.board for cell in row)


def _garbage_excavation_metrics(
    board: Sequence[Sequence[str | None]],
) -> GarbageExcavationMetrics:
    """Measure whether the current garbage can be excavated without a mountain.

    Only the gap of the topmost live garbage row is the immediate excavation
    target. Keeping that entire column permanently empty is *not* required:
    ordinary rows above it may need a block in that column in order to clear.
    Instead, ``channel_debt`` charges such a blocker according to how many cells
    its row still needs before it can disappear, while ``clear_debt`` and the
    non-channel surface shape punish sparse new layers.
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
    top_garbage_y, top_garbage_row = min(garbage_rows, key=lambda item: item[0])
    gaps = [x for x, cell in enumerate(top_garbage_row) if cell is None]

    target_gap: int | None = None
    channel_blockers = 0
    channel_debt = 0
    if gaps:
        blocker_counts = {
            x: sum(board[y][x] is not None for y in range(top_garbage_y))
            for x in gaps
        }
        target_gap = min(gaps, key=lambda x: (blocker_counts[x], x))
        channel_blockers = blocker_counts[target_gap]
        for y in range(top_garbage_y):
            if board[y][target_gap] is None:
                continue
            # A channel block in a nearly-complete row is cheap because the row
            # is about to vanish. A block in a sparse mountain layer is costly.
            channel_debt += sum(cell is None for cell in board[y])
    else:
        # Malformed/manual full garbage rows should normally have cleared.
        channel_blockers = len(top_garbage_row)
        channel_debt = len(top_garbage_row) * len(top_garbage_row)

    active_rows = [
        row
        for row in board[:top_garbage_y]
        if any(cell is not None for cell in row)
    ]
    clear_debt = sum(sum(cell is None for cell in row) for row in active_rows)
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

    heights = _board_column_heights(board)
    support_heights = [
        height
        for x, height in enumerate(heights)
        if target_gap is None or x != target_gap
    ]
    if len(support_heights) >= 2:
        surface_spread = max(support_heights) - min(support_heights)
        ordered = sorted(support_heights)
        median = ordered[len(ordered) // 2]
        surface_variation = sum(abs(height - median) for height in support_heights)
    else:
        surface_spread = 0
        surface_variation = 0

    return GarbageExcavationMetrics(
        rows=len(garbage_rows),
        cells=garbage_cells,
        channel_blockers=channel_blockers,
        channel_debt=channel_debt,
        clear_debt=clear_debt,
        cover_cells=cover_cells,
        stack_height=stack_height,
        surface_spread=surface_spread,
        surface_variation=surface_variation,
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
            + garbage_after.channel_debt * config.garbage_channel_debt
            + garbage_after.clear_debt * config.garbage_clear_debt
            + garbage_after.surface_spread * config.garbage_surface_spread
            + garbage_after.surface_variation * config.garbage_surface_variation
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
