from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Callable, Iterable

from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE, VersusMatch
from minoflux_ai.features import BoardFeatures, extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
import minoflux_ai.search as search_mod
from minoflux_ai.search import SearchAction, SearchConfig, apply_search_action, clone_game, choose_search_action
import minoflux_ai.versus_search as versus_mod
from minoflux_ai.versus_search import DEFAULT_VERSUS_SEARCH_CONFIG, DEFAULT_VERSUS_WEIGHTS, VersusChoice

BASE_RANK = search_mod.rank_placements
BASELINE = "baseline"
CANDIDATES = (
    "spike_damage_delta",
    "spike_damage_variance",
    "spike_clean_floor",
    "spike_slot_quality_floor",
    "spike_t_urgency_floor",
    "spike_convex_danger",
    "alternating_hole_resilience",
    "split_hole_resilience",
    "spike_height_tail",
    "spike_hole_depth_tail",
    "hold_spike_reserve",
    "b2b_spike_reserve",
    "attack_spike_followthrough",
    "slot_survival_delta",
)
ALL = (BASELINE,) + CANDIDATES
SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
)


def _stress(board: list[list[str | None]], width: int, holes: Iterable[int]) -> list[list[str | None]]:
    stressed = [row.copy() for row in board]
    for raw_hole in holes:
        hole = min(width - 1, max(0, int(raw_hole)))
        stressed.pop(0)
        row: list[str | None] = ["G"] * width
        row[hole] = None
        stressed.append(row)
    return stressed


def _danger(f: BoardFeatures) -> float:
    return 2.15 * f.holes + 0.255 * f.hole_depth + 0.70 * f.max_height + 0.075 * f.bumpiness


def _clean_quality(f: BoardFeatures) -> float:
    return 1.0 / (1.0 + f.holes + 0.18 * f.hole_depth + f.max_height / 6.0)


def _slot_quality(f: BoardFeatures) -> float:
    return f.t_spin_slots / (1.0 + f.holes + 0.16 * f.hole_depth + f.max_height / 6.0)


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for i, piece in enumerate(game.queue):
        if piece == "T":
            return i + 1
        if i >= 5:
            break
    return 7


def _post_board(game: Game, evaluation: PlacementEvaluation) -> tuple[Game, list[list[str | None]], BoardFeatures]:
    child = clone_game(game)
    child.place(evaluation.placement)
    board = [row.copy() for row in child.board]
    return child, board, extract_board_features(board)


def _adjustment(name: str, game: Game, evaluation: PlacementEvaluation) -> float:
    if name == BASELINE:
        return 0.0
    child, board, current = _post_board(game, evaluation)
    width = child.width
    center_holes = tuple(sorted({max(0, min(width - 1, h)) for h in (3, 4, 5, 6)}))
    stressed = [extract_board_features(_stress(board, width, (h, h, h, h))) for h in center_holes]
    dangers = [_danger(f) for f in stressed]
    clean = [_clean_quality(f) for f in stressed]
    slots = [_slot_quality(f) for f in stressed]
    base_danger = _danger(current)

    if name == "spike_damage_delta":
        # Prefer boards whose structure degrades little, rather than merely boards
        # that happen to have a good absolute stressed score.
        return -0.17 * max(0.0, max(dangers) - base_danger)
    if name == "spike_damage_variance":
        # Robustness to garbage-hole RNG: avoid a placement that is excellent for
        # one center hole but catastrophically bad for another.
        return -0.28 * pstdev(dangers) if len(dangers) > 1 else 0.0
    if name == "spike_clean_floor":
        return 2.6 * min(clean, default=0.0)
    if name == "spike_slot_quality_floor":
        return 1.8 * min(slots, default=0.0)
    if name == "spike_t_urgency_floor":
        urgency = max(0.0, (6.0 - _next_t_distance(child)) / 6.0)
        return 2.1 * urgency * min(slots, default=0.0)
    if name == "spike_convex_danger":
        worst = max(stressed, key=_danger)
        excess_height = max(0, worst.max_height - 12)
        return -0.020 * worst.holes * worst.holes - 0.0030 * worst.hole_depth * worst.hole_depth - 0.045 * excess_height * excess_height
    if name == "alternating_hole_resilience":
        variants = []
        for a, b in ((3, 4), (4, 5), (5, 6), (3, 6)):
            variants.append(extract_board_features(_stress(board, width, (a, b, a, b))))
        return -0.13 * max((_danger(f) for f in variants), default=0.0)
    if name == "split_hole_resilience":
        variants = []
        for a, b in ((3, 4), (4, 5), (5, 6), (3, 6)):
            variants.append(extract_board_features(_stress(board, width, (a, a, b, b))))
        return -0.13 * max((_danger(f) for f in variants), default=0.0)
    if name == "spike_height_tail":
        heights = sorted((f.max_height for f in stressed), reverse=True)
        tail = mean(heights[:2]) if heights else 0.0
        return -0.11 * tail
    if name == "spike_hole_depth_tail":
        depths = sorted((f.hole_depth for f in stressed), reverse=True)
        tail = mean(depths[:2]) if depths else 0.0
        return -0.026 * tail
    if name == "hold_spike_reserve":
        # rank_search_actions evaluates Hold branches using a game with hold_used=True.
        if not game.hold_used:
            return 0.0
        return 1.7 * min(clean, default=0.0) + 0.55 * min(slots, default=0.0)
    if name == "b2b_spike_reserve":
        if not child.back_to_back:
            return 0.0
        return 1.35 * min(clean, default=0.0) + 0.18 * child.b2b_chain
    if name == "attack_spike_followthrough":
        if evaluation.features.attack <= 0:
            return 0.0
        return 0.24 * evaluation.features.attack * (1.0 + 2.0 * min(clean, default=0.0))
    if name == "slot_survival_delta":
        worst_slots = min((f.t_spin_slots for f in stressed), default=0)
        loss = max(0, current.t_spin_slots - worst_slots)
        keep = min(current.t_spin_slots, worst_slots)
        return 0.48 * keep - 0.62 * loss
    raise ValueError(name)


@contextmanager
def _candidate_ranker(name: str):
    if name == BASELINE:
        yield
        return

    def ranked(game: Game, weights=DEFAULT_WEIGHTS, *, placements=None, limit=None):
        # Ask the established evaluator for the whole branch before applying the
        # experimental judgment, so the baseline heuristic cannot pre-prune it.
        values = BASE_RANK(game, weights, placements=placements, limit=None)
        rescored = tuple(
            replace(item, score=item.score + _adjustment(name, game, item))
            for item in values
        )
        rescored = tuple(sorted(
            rescored,
            key=lambda item: (
                item.score,
                item.features.attack,
                item.features.spin_lines,
                item.features.lines,
                -item.features.board.holes,
                -item.features.board.max_height,
                -item.placement.rotation,
                -item.placement.x,
            ),
            reverse=True,
        ))
        if limit is None:
            return rescored
        return rescored[:max(0, int(limit))]

    old = search_mod.rank_placements
    search_mod.rank_placements = ranked
    try:
        yield
    finally:
        search_mod.rank_placements = old


def _solo_game(name: str, seed: int, max_pieces: int) -> dict[str, float | int | bool]:
    game = Game(seed)
    spins = spin_lines = tsd = tst = 0
    max_b2b = 0
    with _candidate_ranker(name):
        while not game.game_over and game.pieces_placed < max_pieces:
            choice = choose_search_action(game, DEFAULT_WEIGHTS, SEARCH)
            if choice is None:
                break
            result = apply_search_action(game, choice.action)
            if result.spin is not None:
                spins += 1
                spin_lines += result.lines
            if result.spin == T_SPIN_DOUBLE:
                tsd += 1
            elif result.spin == T_SPIN_TRIPLE:
                tst += 1
            max_b2b = max(max_b2b, game.b2b_chain)
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "app": game.attack / max(1, game.pieces_placed),
        "topout": int(game.game_over),
        "completed": int(not game.game_over and game.pieces_placed >= max_pieces),
        "spins": spins,
        "spinLines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "maxB2B": max_b2b,
    }


def _solo_candidate(args: tuple[str, tuple[int, ...], int]) -> tuple[str, dict[str, object]]:
    name, seeds, max_pieces = args
    games = [_solo_game(name, seed, max_pieces) for seed in seeds]
    pieces = sum(int(g["pieces"]) for g in games)
    attack = sum(int(g["attack"]) for g in games)
    return name, {
        "games": len(games),
        "maxPieces": max_pieces,
        "pieces": pieces,
        "attack": attack,
        "app": attack / max(1, pieces),
        "topouts": sum(int(g["topout"]) for g in games),
        "completed": sum(int(g["completed"]) for g in games),
        "spins": sum(int(g["spins"]) for g in games),
        "spinLines": sum(int(g["spinLines"]) for g in games),
        "tsd": sum(int(g["tsd"]) for g in games),
        "tst": sum(int(g["tst"]) for g in games),
        "maxB2B": max(int(g["maxB2B"]) for g in games),
        "perGame": games,
    }


def _quality(item: tuple[str, dict[str, object]], baseline: dict[str, object]) -> tuple[float, ...]:
    _, r = item
    # Safety first. Then completion, APP and meaningful attack structures.
    return (
        -float(r["topouts"]),
        float(r["completed"]),
        float(r["app"]),
        float(r["tsd"]) + 1.5 * float(r["tst"]),
        float(r["spinLines"]),
        float(r["maxB2B"]),
    )


def _run_solo(names: tuple[str, ...], seeds: tuple[int, ...], max_pieces: int) -> dict[str, dict[str, object]]:
    tasks = [(name, seeds, max_pieces) for name in names]
    with ProcessPoolExecutor(max_workers=min(4, len(tasks))) as executor:
        pairs = list(executor.map(_solo_candidate, tasks))
    return dict(pairs)


def _rank_top(results: dict[str, dict[str, object]], count: int) -> list[str]:
    baseline = results[BASELINE]
    candidates = [(name, result) for name, result in results.items() if name != BASELINE]
    # Reject candidates that are clearly less safe than baseline before sorting.
    safe = [
        item for item in candidates
        if int(item[1]["topouts"]) <= int(baseline["topouts"])
        and int(item[1]["completed"]) >= int(baseline["completed"])
    ]
    pool = safe or candidates
    pool.sort(key=lambda item: _quality(item, baseline), reverse=True)
    return [name for name, _ in pool[:count]]


def _rank_actions_for(name: str, game: Game, limit: int):
    with _candidate_ranker(name):
        return search_mod.rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=limit)


def _choose_versus(match: VersusMatch, side_name: str, name: str, opponent_name_model: str) -> VersusChoice | None:
    cfg = DEFAULT_VERSUS_SEARCH_CONFIG.normalized()
    own = versus_mod._side(match, side_name)
    ranked = _rank_actions_for(name, own.game, cfg.candidate_width)
    if not ranked:
        return None
    opponent_side = versus_mod._opponent_name(side_name)
    best = None
    for action, evaluation in ranked:
        after, resolution = versus_mod._simulate_action(match, side_name, action)
        score = versus_mod.score_versus_state(
            after,
            side_name,
            weights=DEFAULT_VERSUS_WEIGHTS,
            resolution=resolution,
            solo_score=evaluation.score,
            path_length=len(action.placement.path),
            action_side=side_name,
        )
        reply_action = None
        if cfg.opponent_reply_width > 0 and after.winner is None and not versus_mod._side(after, opponent_side).game.game_over:
            replies = _rank_actions_for(opponent_name_model, versus_mod._side(after, opponent_side).game, cfg.opponent_reply_width)
            worst_score = None
            for reply, reply_eval in replies:
                replied, reply_resolution = versus_mod._simulate_action(after, opponent_side, reply)
                reply_score = versus_mod.score_versus_state(
                    replied,
                    side_name,
                    weights=DEFAULT_VERSUS_WEIGHTS,
                    resolution=reply_resolution,
                    solo_score=-reply_eval.score,
                    path_length=len(reply.placement.path),
                    action_side=opponent_side,
                )
                if worst_score is None or reply_score < worst_score:
                    worst_score = reply_score
                    reply_action = reply
            if worst_score is not None:
                score = worst_score
        choice = VersusChoice(action, score, evaluation, resolution, reply_action)
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


def _versus_pair(candidate: str, seed: int, swapped: bool, max_turns: int) -> dict[str, object]:
    match = VersusMatch(seed, garbage_cap=8)
    # 'candidate' is the logical challenger regardless of which physical side it uses.
    candidate_side = "ai" if swapped else "player"
    baseline_side = "player" if swapped else "ai"
    turn = "player"
    turns = 0
    cand_max_b2b = base_max_b2b = 0
    while match.winner is None and turns < max_turns:
        if turn == candidate_side:
            choice = _choose_versus(match, turn, candidate, BASELINE)
        else:
            choice = _choose_versus(match, turn, BASELINE, candidate)
        if choice is None:
            versus_mod._side(match, turn).game.game_over = True
            match._update_winner()
            break
        side = versus_mod._side(match, turn)
        result = apply_search_action(side.game, choice.action)
        match.resolve_lock(turn, result)
        turns += 1
        cand_max_b2b = max(cand_max_b2b, versus_mod._side(match, candidate_side).game.b2b_chain)
        base_max_b2b = max(base_max_b2b, versus_mod._side(match, baseline_side).game.b2b_chain)
        turn = "ai" if turn == "player" else "player"
    winner = match.winner or "draw"
    logical_winner = "draw" if winner == "draw" else ("candidate" if winner == candidate_side else "baseline")
    cand = versus_mod._side(match, candidate_side)
    base = versus_mod._side(match, baseline_side)
    return {
        "seed": seed,
        "swapped": swapped,
        "winner": logical_winner,
        "turns": turns,
        "candidateAttack": cand.game.attack,
        "baselineAttack": base.game.attack,
        "candidateSent": cand.sent,
        "baselineSent": base.sent,
        "candidateCanceled": cand.canceled,
        "baselineCanceled": base.canceled,
        "candidateReceived": cand.received,
        "baselineReceived": base.received,
        "candidateTopout": int(cand.game.game_over),
        "baselineTopout": int(base.game.game_over),
        "candidateMaxB2B": cand_max_b2b,
        "baselineMaxB2B": base_max_b2b,
    }


def _versus(candidate: str, games: int = 8, max_turns: int = 120, seed_base: int = 9_100_001, seed_step: int = 101) -> dict[str, object]:
    rows = []
    for i in range(games):
        rows.append(_versus_pair(candidate, seed_base + (i // 2) * seed_step, bool(i % 2), max_turns))
    return {
        "games": games,
        "candidateWins": sum(r["winner"] == "candidate" for r in rows),
        "baselineWins": sum(r["winner"] == "baseline" for r in rows),
        "draws": sum(r["winner"] == "draw" for r in rows),
        "candidateMeanAttack": mean(float(r["candidateAttack"]) for r in rows),
        "baselineMeanAttack": mean(float(r["baselineAttack"]) for r in rows),
        "candidateMeanSent": mean(float(r["candidateSent"]) for r in rows),
        "baselineMeanSent": mean(float(r["baselineSent"]) for r in rows),
        "candidateMeanCanceled": mean(float(r["candidateCanceled"]) for r in rows),
        "baselineMeanCanceled": mean(float(r["baselineCanceled"]) for r in rows),
        "candidateMeanReceived": mean(float(r["candidateReceived"]) for r in rows),
        "baselineMeanReceived": mean(float(r["baselineReceived"]) for r in rows),
        "candidateTopouts": sum(int(r["candidateTopout"]) for r in rows),
        "baselineTopouts": sum(int(r["baselineTopout"]) for r in rows),
        "candidateMeanMaxB2B": mean(float(r["candidateMaxB2B"]) for r in rows),
        "baselineMeanMaxB2B": mean(float(r["baselineMaxB2B"]) for r in rows),
        "perGame": rows,
    }


def main() -> None:
    short_seeds = (6_100_001, 6_100_038)
    fresh_seeds = (7_300_001, 7_300_098, 7_300_195)
    short = _run_solo(ALL, short_seeds, 110)
    top3 = _rank_top(short, 3)
    fresh_names = (BASELINE,) + tuple(top3)
    fresh = _run_solo(fresh_names, fresh_seeds, 260)

    baseline_fresh = fresh[BASELINE]
    fresh_pass = []
    for name in top3:
        r = fresh[name]
        # Require safety at least baseline-like and either APP improvement or a
        # meaningful completion gain. This prevents short-seed winners from
        # advancing on denominator effects alone.
        safe = int(r["topouts"]) <= int(baseline_fresh["topouts"]) and int(r["completed"]) >= int(baseline_fresh["completed"])
        useful = float(r["app"]) >= float(baseline_fresh["app"]) * 1.01 or int(r["completed"]) > int(baseline_fresh["completed"])
        if safe and useful:
            fresh_pass.append(name)
    fresh_pass.sort(key=lambda n: _quality((n, fresh[n]), baseline_fresh), reverse=True)
    finalists = fresh_pass[:2]
    versus = {name: _versus(name) for name in finalists}

    winner = None
    for name in finalists:
        v = versus[name]
        # Safety-first adoption gate: must not lose the match, must not top out
        # more, and must improve at least one pressure metric without materially
        # regressing the other.
        wins_ok = int(v["candidateWins"]) > int(v["baselineWins"])
        safety_ok = int(v["candidateTopouts"]) <= int(v["baselineTopouts"])
        sent = float(v["candidateMeanSent"])
        base_sent = float(v["baselineMeanSent"])
        recv = float(v["candidateMeanReceived"])
        base_recv = float(v["baselineMeanReceived"])
        pressure_ok = sent >= base_sent and recv <= base_recv
        if wins_ok and safety_ok and pressure_ok:
            winner = name
            break

    payload = {
        "candidateCount": len(CANDIDATES),
        "candidates": list(CANDIDATES),
        "short": short,
        "shortTop3": top3,
        "fresh": fresh,
        "freshPass": fresh_pass,
        "finalists": finalists,
        "versus": versus,
        "winner": winner,
    }
    target = Path("data/self-improve/latest-tournament.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
