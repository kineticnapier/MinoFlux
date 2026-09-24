from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

from minoflux_ai.features import extract_board_features_from_masks
from minoflux_ai.neural import NeuralState, NeuralValueEvaluator, unpack_board_rows
from minoflux_ai.neural_train import _load_jsonl


Record = dict[str, object]


def _sequence(value: object) -> Sequence[object] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return None


def _record_key(record: Mapping[str, object]) -> tuple[int, int]:
    return int(record.get("seed", 0)), int(record.get("pieceIndex", 0))


def _reasons(record: Mapping[str, object]) -> tuple[str, ...]:
    raw = _sequence(record.get("daggerReasons"))
    if raw is None:
        return ()
    return tuple(sorted({str(value) for value in raw}))


def _is_disagreement(record: Mapping[str, object]) -> bool:
    return record.get("learnerMatchedOracle") is False or "nn_oracle_disagree" in _reasons(record)


def _margin(record: Mapping[str, object]) -> float:
    raw = record.get("learnerMargin")
    if raw is None:
        return float("-inf")
    value = float(raw)
    return value if math.isfinite(value) else float("-inf")


def _candidates(record: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = _sequence(record.get("candidates"))
    if raw is None:
        return ()
    return tuple(candidate for candidate in raw if isinstance(candidate, Mapping))


def _expert_index(record: Mapping[str, object], candidate_count: int) -> int:
    index = int(record.get("expertIndex", -1))
    return index if 0 <= index < candidate_count else -1


def _candidate_rows(candidate: Mapping[str, object]) -> tuple[int, ...]:
    raw = _sequence(candidate.get("rows"))
    if raw is None:
        return ()
    return tuple(int(value) for value in raw)


def _candidate_context(candidate: Mapping[str, object]) -> tuple[float, ...]:
    raw = _sequence(candidate.get("context"))
    if raw is None:
        return ()
    return tuple(float(value) for value in raw)


def _move_parts(candidate: Mapping[str, object]) -> tuple[bool, str, int, int, int] | None:
    move = candidate.get("move")
    if isinstance(move, Mapping):
        return (
            bool(move.get("hold")),
            str(move.get("piece", "?")),
            int(move.get("x", 0)),
            int(move.get("y", 0)),
            int(move.get("rotation", 0)),
        )
    seq = _sequence(move)
    if seq is not None and len(seq) >= 5:
        hold, piece, x, y, rotation = seq[:5]
        return bool(int(hold)), str(piece), int(x), int(y), int(rotation)
    return None


def _move_text(candidate: Mapping[str, object]) -> str:
    move = _move_parts(candidate)
    if move is None:
        return "?"
    hold, piece, x, y, rotation = move
    return f"{piece}@({x},{y})r{rotation}" + (" hold" if hold else "")


def _candidate_snapshot(candidate: Mapping[str, object]) -> dict[str, object]:
    rows = _candidate_rows(candidate)
    result: dict[str, object] = {"move": _move_text(candidate)}
    if rows:
        features = extract_board_features_from_masks(rows, width=10)
        result.update(
            {
                "max_height": features.max_height,
                "holes": features.holes,
                "aggregate_height": features.aggregate_height,
                "bumpiness": features.bumpiness,
                "t_spin_slots": features.t_spin_slots,
            }
        )

    context = _candidate_context(candidate)
    if len(context) >= 9:
        tail = context[-9:]
        result.update(
            {
                "combo": int(round(tail[0] * 16.0)) - 1,
                "b2b": tail[1] >= 0.5,
                "b2b_chain": int(round(tail[2] * 20.0)),
                "surge": int(round(tail[3] * 20.0)),
                "game_over": tail[4] >= 0.5,
                "lines": int(round(tail[5] * 4.0)),
                "attack": int(round(tail[6] * 20.0)),
                "spin": tail[7] >= 0.5,
                "perfect_clear": tail[8] >= 0.5,
            }
        )
    return result


def _state_for_candidate(
    candidate: Mapping[str, object],
    evaluator: NeuralValueEvaluator,
) -> NeuralState:
    rows = _candidate_rows(candidate)
    context = _candidate_context(candidate)
    if len(rows) != evaluator.config.board_height:
        raise ValueError(
            f"Candidate has {len(rows)} board rows; model expects {evaluator.config.board_height}"
        )
    if len(context) != evaluator.config.context_size:
        raise ValueError(
            f"Candidate has context size {len(context)}; model expects {evaluator.config.context_size}"
        )
    return NeuralState(
        board=unpack_board_rows(rows, evaluator.config),
        context=context,
    )


def _score_candidates(
    record: Mapping[str, object],
    evaluator: NeuralValueEvaluator | None,
) -> tuple[float, ...]:
    candidates = _candidates(record)
    if evaluator is None or not candidates:
        return ()
    states = tuple(_state_for_candidate(candidate, evaluator) for candidate in candidates)
    return tuple(float(value) for value in evaluator._score_states(states))


def _best_index(scores: Sequence[float], candidate_count: int) -> int:
    if not scores or candidate_count <= 0:
        return -1
    limit = min(len(scores), candidate_count)
    return max(range(limit), key=lambda index: float(scores[index]))


def _sort_records(records: Sequence[Record], mode: str) -> list[Record]:
    if mode == "piece":
        return sorted(records, key=lambda record: (int(record.get("pieceIndex", 0)), int(record.get("seed", 0))))
    if mode == "seed":
        return sorted(records, key=lambda record: (int(record.get("seed", 0)), int(record.get("pieceIndex", 0))))
    return sorted(records, key=lambda record: (_margin(record), int(record.get("pieceIndex", 0))), reverse=True)


def _visible_records(
    records: Sequence[Record],
    *,
    disagreements_only: bool,
    confident_only: bool,
    sort_mode: str,
) -> list[Record]:
    filtered = list(records)
    if disagreements_only:
        filtered = [record for record in filtered if _is_disagreement(record)]
    if confident_only:
        filtered = [record for record in filtered if _is_disagreement(record) and _margin(record) > 0.08]
    return _sort_records(filtered, sort_mode)


def _try_load_evaluator(
    path: str | None,
    *,
    device: str,
) -> NeuralValueEvaluator | None:
    if not path:
        return None
    target = Path(path)
    if not target.is_file():
        print(f"inspector: model not found, disabled: {target}", file=sys.stderr)
        return None
    return NeuralValueEvaluator.from_checkpoint(target, device=device, precision="float32")


def _format_score(scores: Sequence[float], index: int) -> str:
    if index < 0 or index >= len(scores):
        return "-"
    return f"{float(scores[index]):+.3f}"


def _draw_text(pygame, screen, font, text: str, x: int, y: int, color, *, max_width: int | None = None) -> None:
    if max_width is None:
        screen.blit(font.render(text, True, color), (x, y))
        return
    rendered = font.render(text, True, color)
    if rendered.get_width() <= max_width:
        screen.blit(rendered, (x, y))
        return
    clipped = text
    while clipped and font.size(clipped + "...")[0] > max_width:
        clipped = clipped[:-1]
    screen.blit(font.render(clipped + "...", True, color), (x, y))


def _draw_board(
    pygame,
    screen,
    rows: Sequence[int],
    *,
    x: int,
    y: int,
    cell: int,
    title: str,
    subtitle: str,
    font,
    small,
    border_color,
    selected: bool,
) -> None:
    board_width = 10 * cell
    board_height = 24 * cell
    panel = (28, 31, 38)
    grid = (58, 64, 75)
    occupied = (126, 143, 160)
    hidden = (21, 23, 29)
    text = (230, 234, 240)
    muted = (155, 165, 178)

    _draw_text(pygame, screen, font, title, x, y - 43, text, max_width=board_width)
    _draw_text(pygame, screen, small, subtitle, x, y - 20, muted, max_width=board_width)
    pygame.draw.rect(screen, panel, (x - 5, y - 5, board_width + 10, board_height + 10), border_radius=5)
    pygame.draw.rect(
        screen,
        border_color,
        (x - 6, y - 6, board_width + 12, board_height + 12),
        3 if selected else 1,
        border_radius=6,
    )
    for row_index in range(24):
        mask = int(rows[row_index]) if row_index < len(rows) else 0
        for column in range(10):
            rect = pygame.Rect(x + column * cell, y + row_index * cell, cell, cell)
            if row_index < 4:
                pygame.draw.rect(screen, hidden, rect)
            pygame.draw.rect(screen, grid, rect, 1)
            if mask & (1 << column):
                pygame.draw.rect(screen, occupied, rect.inflate(-3, -3), border_radius=2)


def _draw_candidate_footer(
    pygame,
    screen,
    candidate: Mapping[str, object] | None,
    *,
    x: int,
    y: int,
    width: int,
    small,
    text,
    muted,
) -> None:
    if candidate is None:
        return
    snapshot = _candidate_snapshot(candidate)
    line1 = (
        f"move {snapshot.get('move', '?')}   h={snapshot.get('max_height', '?')} "
        f"holes={snapshot.get('holes', '?')} agg={snapshot.get('aggregate_height', '?')} "
        f"bump={snapshot.get('bumpiness', '?')} tspinSlots={snapshot.get('t_spin_slots', '?')}"
    )
    line2 = (
        f"clear={snapshot.get('lines', 0)} atk={snapshot.get('attack', 0)} "
        f"spin={int(bool(snapshot.get('spin', False)))} PC={int(bool(snapshot.get('perfect_clear', False)))} "
        f"combo={snapshot.get('combo', '?')} b2b={int(bool(snapshot.get('b2b', False)))} "
        f"chain={snapshot.get('b2b_chain', '?')} surge={snapshot.get('surge', '?')}"
    )
    _draw_text(pygame, screen, small, line1, x, y, text, max_width=width)
    _draw_text(pygame, screen, small, line2, x, y + 20, muted, max_width=width)


def _run_pygame(args: argparse.Namespace) -> int:
    try:
        import pygame
    except ImportError as error:
        raise SystemExit("Pygame is required for dagger-inspect. Install pygame-ce or the review extra.") from error

    dataset_path = Path(args.dataset)
    records = _load_jsonl(dataset_path)
    evaluator_a = _try_load_evaluator(args.model_a, device=args.device)
    evaluator_b = _try_load_evaluator(args.model_b, device=args.device)

    pygame.init()
    screen = pygame.display.set_mode((1560, 900), pygame.RESIZABLE)
    pygame.display.set_caption("MinoFlux DAgger Sample Inspector")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 23)
    small = pygame.font.SysFont("Consolas", 17)
    tiny = pygame.font.SysFont("Consolas", 15)

    bg = (15, 17, 21)
    panel = (25, 28, 34)
    text = (231, 234, 239)
    muted = (151, 160, 174)
    accent = (104, 180, 255)
    oracle_color = (255, 196, 92)
    model_a_color = (111, 208, 160)
    model_b_color = (201, 132, 255)
    selected_color = (255, 119, 141)
    bad = (255, 114, 114)
    good = (123, 218, 160)

    disagreements_only = not args.all
    confident_only = False
    sort_modes = ("margin", "piece", "seed")
    sort_index = sort_modes.index(args.sort)
    visible = _visible_records(
        records,
        disagreements_only=disagreements_only,
        confident_only=confident_only,
        sort_mode=sort_modes[sort_index],
    )
    record_index = 0
    selected_candidate = 0
    score_cache_a: dict[tuple[int, int], tuple[float, ...]] = {}
    score_cache_b: dict[tuple[int, int], tuple[float, ...]] = {}

    def refresh_visible(*, keep_key: tuple[int, int] | None = None) -> None:
        nonlocal visible, record_index, selected_candidate
        visible = _visible_records(
            records,
            disagreements_only=disagreements_only,
            confident_only=confident_only,
            sort_mode=sort_modes[sort_index],
        )
        if not visible:
            record_index = 0
            selected_candidate = 0
            return
        if keep_key is not None:
            found = next((i for i, record in enumerate(visible) if _record_key(record) == keep_key), None)
            record_index = found if found is not None else min(record_index, len(visible) - 1)
        else:
            record_index = min(record_index, len(visible) - 1)
        selected_candidate = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                if not visible:
                    continue
                current = visible[record_index]
                current_key = _record_key(current)
                current_candidates = _candidates(current)
                candidate_count = len(current_candidates)
                if event.key == pygame.K_RIGHT:
                    record_index = min(len(visible) - 1, record_index + 1)
                    selected_candidate = 0
                elif event.key == pygame.K_LEFT:
                    record_index = max(0, record_index - 1)
                    selected_candidate = 0
                elif event.key == pygame.K_PAGEDOWN:
                    record_index = min(len(visible) - 1, record_index + 10)
                    selected_candidate = 0
                elif event.key == pygame.K_PAGEUP:
                    record_index = max(0, record_index - 10)
                    selected_candidate = 0
                elif event.key == pygame.K_HOME:
                    record_index = 0
                    selected_candidate = 0
                elif event.key == pygame.K_END:
                    record_index = len(visible) - 1
                    selected_candidate = 0
                elif event.key == pygame.K_DOWN and candidate_count:
                    selected_candidate = (selected_candidate + 1) % candidate_count
                elif event.key == pygame.K_UP and candidate_count:
                    selected_candidate = (selected_candidate - 1) % candidate_count
                elif event.key == pygame.K_d:
                    disagreements_only = not disagreements_only
                    refresh_visible(keep_key=current_key)
                elif event.key == pygame.K_c:
                    confident_only = not confident_only
                    if confident_only:
                        disagreements_only = True
                    refresh_visible(keep_key=current_key)
                elif event.key == pygame.K_s:
                    sort_index = (sort_index + 1) % len(sort_modes)
                    refresh_visible(keep_key=current_key)
                elif event.key == pygame.K_o:
                    expert = _expert_index(current, candidate_count)
                    if expert >= 0:
                        selected_candidate = expert
                elif event.key in (pygame.K_a, pygame.K_b):
                    cache = score_cache_a if event.key == pygame.K_a else score_cache_b
                    evaluator = evaluator_a if event.key == pygame.K_a else evaluator_b
                    if current_key not in cache:
                        cache[current_key] = _score_candidates(current, evaluator)
                    best = _best_index(cache[current_key], candidate_count)
                    if best >= 0:
                        selected_candidate = best

        screen.fill(bg)
        if not visible:
            _draw_text(pygame, screen, font, "No samples match the active filters.", 30, 30, text)
            _draw_text(pygame, screen, small, "D = disagreement filter, C = confident filter, Esc = quit", 30, 65, muted)
            pygame.display.flip()
            clock.tick(30)
            continue

        record = visible[record_index]
        key = _record_key(record)
        candidates = _candidates(record)
        candidate_count = len(candidates)
        if candidate_count == 0:
            _draw_text(pygame, screen, font, "Current record has no candidates.", 30, 30, bad)
            pygame.display.flip()
            clock.tick(30)
            continue
        selected_candidate %= candidate_count

        if key not in score_cache_a:
            score_cache_a[key] = _score_candidates(record, evaluator_a)
        if key not in score_cache_b:
            score_cache_b[key] = _score_candidates(record, evaluator_b)
        scores_a = score_cache_a[key]
        scores_b = score_cache_b[key]
        expert = _expert_index(record, candidate_count)
        best_a = _best_index(scores_a, candidate_count)
        best_b = _best_index(scores_b, candidate_count)

        margin = _margin(record)
        margin_text = "-" if not math.isfinite(margin) else f"{margin:.4f}"
        disagree = _is_disagreement(record)
        header = (
            f"DAgger Inspector  {record_index + 1}/{len(visible)}  "
            f"seed={key[0]} piece={key[1]}  margin={margin_text}  "
            f"{'DISAGREE' if disagree else 'MATCH'}"
        )
        _draw_text(pygame, screen, font, header, 24, 14, bad if disagree else good, max_width=1500)
        reasons = ", ".join(_reasons(record)) or "none"
        filters = (
            f"reasons: {reasons}   filters: disagree={'ON' if disagreements_only else 'OFF'} "
            f"confident>0.08={'ON' if confident_only else 'OFF'} sort={sort_modes[sort_index]}"
        )
        _draw_text(pygame, screen, small, filters, 24, 43, muted, max_width=1500)

        board_y = 105
        cell = 18
        board_specs = (
            (expert, "Oracle expert", oracle_color, expert == selected_candidate, "E"),
            (best_a, args.label_a, model_a_color, best_a == selected_candidate, "A"),
            (best_b, args.label_b, model_b_color, best_b == selected_candidate, "B"),
            (selected_candidate, f"Selected #{selected_candidate}", selected_color, True, "S"),
        )
        board_xs = (30, 235, 440, 645)
        for x, (candidate_index, title, color, is_selected, role) in zip(board_xs, board_specs, strict=True):
            if candidate_index < 0 or candidate_index >= candidate_count:
                rows = ()
                subtitle = "model disabled"
            else:
                candidate = candidates[candidate_index]
                rows = _candidate_rows(candidate)
                subtitle = f"#{candidate_index} {_move_text(candidate)}"
                if role == "A":
                    subtitle += f"  v={_format_score(scores_a, candidate_index)}"
                elif role == "B":
                    subtitle += f"  v={_format_score(scores_b, candidate_index)}"
                elif role == "E":
                    subtitle += f"  A={_format_score(scores_a, candidate_index)} B={_format_score(scores_b, candidate_index)}"
                elif role == "S":
                    subtitle += f"  A={_format_score(scores_a, candidate_index)} B={_format_score(scores_b, candidate_index)}"
            _draw_board(
                pygame,
                screen,
                rows,
                x=x,
                y=board_y,
                cell=cell,
                title=title,
                subtitle=subtitle,
                font=font,
                small=tiny,
                border_color=color,
                selected=is_selected,
            )
            candidate = candidates[candidate_index] if 0 <= candidate_index < candidate_count else None
            _draw_candidate_footer(
                pygame,
                screen,
                candidate,
                x=x,
                y=board_y + 24 * cell + 14,
                width=190,
                small=tiny,
                text=text,
                muted=muted,
            )

        table_x = 850
        table_y = 86
        table_width = max(670, screen.get_width() - table_x - 20)
        pygame.draw.rect(screen, panel, (table_x, table_y, table_width, 625), border_radius=7)
        _draw_text(
            pygame,
            screen,
            font,
            "Candidates: E=Oracle A/B=model best *=selected",
            table_x + 12,
            table_y + 10,
            text,
            max_width=table_width - 24,
        )
        _draw_text(
            pygame,
            screen,
            tiny,
            "mark idx   A-value   B-value   h holes atk  move",
            table_x + 12,
            table_y + 42,
            muted,
            max_width=table_width - 24,
        )

        row_y = table_y + 65
        for index, candidate in enumerate(candidates[:28]):
            marks = ""
            marks += "E" if index == expert else "."
            marks += "A" if index == best_a else "."
            marks += "B" if index == best_b else "."
            marks += "*" if index == selected_candidate else "."
            snapshot = _candidate_snapshot(candidate)
            line = (
                f"{marks:4s} {index:2d}   {_format_score(scores_a, index):>7s}   {_format_score(scores_b, index):>7s}   "
                f"{str(snapshot.get('max_height', '?')):>2s} {str(snapshot.get('holes', '?')):>5s} "
                f"{str(snapshot.get('attack', '?')):>3s}  {_move_text(candidate)}"
            )
            line_color = selected_color if index == selected_candidate else (oracle_color if index == expert else text)
            _draw_text(pygame, screen, tiny, line, table_x + 12, row_y, line_color, max_width=table_width - 24)
            row_y += 20

        info_y = 735
        collector_match = record.get("learnerMatchedOracle")
        _draw_text(
            pygame,
            screen,
            small,
            f"collector learnerMatchedOracle={collector_match!r} | candidate count={candidate_count} | dataset={dataset_path}",
            24,
            info_y,
            text,
            max_width=1500,
        )
        _draw_text(
            pygame,
            screen,
            small,
            "Left/Right sample  PgUp/PgDn +/-10  Up/Down candidate  O/A/B jump  D disagree  C confident  S sort  Esc quit",
            24,
            info_y + 28,
            accent,
            max_width=1500,
        )
        _draw_text(
            pygame,
            screen,
            tiny,
            "A/B show value-model best among the recorded candidate set. The collector's exact learner move was not stored in these R4 records.",
            24,
            info_y + 55,
            muted,
            max_width=1500,
        )
        _draw_text(
            pygame,
            screen,
            tiny,
            f"A: {args.model_a or 'disabled'}    B: {args.model_b or 'disabled'}",
            24,
            info_y + 77,
            muted,
            max_width=1500,
        )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive Pygame inspector for Oracle DAgger ranking samples")
    parser.add_argument("--dataset", default="data/neural/native-oracle-dagger-r4.jsonl")
    parser.add_argument("--model-a", default="data/models/native-oracle-value-r3w.pt")
    parser.add_argument("--model-b", default="data/models/native-oracle-value-r4w.pt")
    parser.add_argument("--label-a", default="R3w candidate best")
    parser.add_argument("--label-b", default="R4w candidate best")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all", action="store_true", help="Start with matched samples visible too")
    parser.add_argument("--sort", choices=("margin", "piece", "seed"), default="margin")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not Path(args.dataset).is_file():
        raise FileNotFoundError(args.dataset)
    return _run_pygame(args)


if __name__ == "__main__":
    raise SystemExit(main())
