from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from statistics import mean

from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE, VersusMatch
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
import minoflux_ai.search as search_mod
from minoflux_ai.search import SearchConfig, apply_search_action, choose_search_action
import minoflux_ai.versus_search as versus_mod
from minoflux_ai.versus_search import DEFAULT_VERSUS_SEARCH_CONFIG, DEFAULT_VERSUS_WEIGHTS, VersusChoice

BASE_RANK = search_mod.rank_placements
BASELINE = "baseline"
CANDIDATES = (
    "worst_case_urgency_guard",
    "recovery_urgency_bonus",
    "recovery_attack_followthrough",
    "recovery_spin_conversion",
    "floor_clean_quality",
    "floor_height_margin",
    "worst_case_convex_guard",
    "resilience_worst_gap",
    "recovery_new_hole_guard",
    "hold_recovery_reserve",
    "b2b_recovery_reserve",
    "attack_worst_balance",
    "slot_delta_recovery",
    "danger_slot_survival",
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


def _next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 5:
            break
    return 7


def _t_supply(game: Game) -> int:
    count = int(game.current == "T") + int(game.hold_piece == "T")
    for index, piece in enumerate(game.queue):
        if index >= 5:
            break
        count += int(piece == "T")
    return count


def _adjustment(name: str, game: Game, evaluation: PlacementEvaluation) -> float:
    if name == BASELINE:
        return 0.0
    f = evaluation.features
    board = f.board
    # These are already computed by the production evaluator, so tournament
    # candidates can test richer interactions without rescanning hypothetical boards.
    worst_score = f.center_garbage_worst_case  # negative: higher is safer
    representative = f.center_garbage_resilience  # negative: higher is safer
    recovery = f.garbage_tspin_recovery
    floor = f.garbage_t_spin_slot_floor
    urgency = max(0.0, (6.0 - _next_t_distance(game)) / 6.0)
    supply = _t_supply(game)
    clean = 1.0 / (1.0 + board.holes + 0.16 * board.hole_depth + board.max_height / 6.0)
    danger = max(0.0, -worst_score)

    if name == "worst_case_urgency_guard":
        # When T is near, avoid placements whose worst likely garbage hole destroys
        # the board before the prepared conversion can happen.
        return 0.42 * urgency * worst_score
    if name == "recovery_urgency_bonus":
        return 0.72 * urgency * recovery
    if name == "recovery_attack_followthrough":
        return 0.13 * f.attack * recovery if f.attack > 0 else 0.0
    if name == "recovery_spin_conversion":
        return 0.38 * f.spin_lines * recovery if f.spin_lines > 0 else 0.0
    if name == "floor_clean_quality":
        return 0.82 * floor * clean
    if name == "floor_height_margin":
        height_margin = max(0.0, 18.0 - board.max_height) / 18.0
        return 0.62 * floor * height_margin
    if name == "worst_case_convex_guard":
        # Existing worst-case scoring is linear. This candidate tests whether the
        # very worst tails should become rapidly more expensive.
        return -0.075 * danger * danger
    if name == "resilience_worst_gap":
        # Penalize placements whose representative center-hole case looks fine but
        # whose worst nearby hole is much worse.
        normalized_rep = 5.5 * representative
        return -0.16 * max(0.0, normalized_rep - worst_score)
    if name == "recovery_new_hole_guard":
        return 0.70 * recovery / (1.0 + 1.8 * f.new_holes)
    if name == "hold_recovery_reserve":
        return 0.75 * recovery + 0.22 * floor if game.hold_used else 0.0
    if name == "b2b_recovery_reserve":
        if not game.back_to_back:
            return 0.0
        return 0.58 * recovery + 0.10 * max(0, game.b2b_chain) + 0.18 * floor
    if name == "attack_worst_balance":
        # Prefer attack that leaves a survivable post-spike board instead of raw
        # attack that immediately creates a garbage-vulnerable stack.
        safety = 1.0 / (1.0 + danger)
        return 0.30 * f.attack * safety
    if name == "slot_delta_recovery":
        return 0.36 * f.t_spin_slot_delta * max(0.0, recovery)
    if name == "danger_slot_survival":
        unmatched = max(0, floor - supply)
        matched = min(floor, supply)
        return 0.34 * matched - 0.28 * unmatched - 0.05 * danger * max(0, board.t_spin_slots - floor)
    raise ValueError(name)


@contextmanager
def _candidate_ranker(name: str):
    if name == BASELINE:
        yield
        return

    def ranked(game: Game, weights=DEFAULT_WEIGHTS, *, placements=None, limit=None):
        source = BASE_RANK(game, weights, placements=placements, limit=None)
        rescored = tuple(replace(item, score=item.score + _adjustment(name, game, item)) for item in source)
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


def _solo_game(name: str, seed: int, max_pieces: int) -> dict[str, float | int]:
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


def _quality(result: dict[str, object]) -> tuple[float, ...]:
    return (
        -float(result["topouts"]),
        float(result["completed"]),
        float(result["app"]),
        float(result["tsd"]) + 1.5 * float(result["tst"]),
        float(result["spinLines"]),
        float(result["maxB2B"]),
    )


def _run_solo(names: tuple[str, ...], seeds: tuple[int, ...], max_pieces: int) -> dict[str, dict[str, object]]:
    tasks = [(name, seeds, max_pieces) for name in names]
    with ProcessPoolExecutor(max_workers=min(4, len(tasks))) as executor:
        return dict(executor.map(_solo_candidate, tasks))


def _top(results: dict[str, dict[str, object]], count: int) -> list[str]:
    baseline = results[BASELINE]
    candidates = [(name, result) for name, result in results.items() if name != BASELINE]
    safe = [
        pair for pair in candidates
        if int(pair[1]["topouts"]) <= int(baseline["topouts"])
        and int(pair[1]["completed"]) >= int(baseline["completed"])
    ]
    pool = safe or candidates
    pool.sort(key=lambda pair: _quality(pair[1]), reverse=True)
    return [name for name, _ in pool[:count]]


def _rank_actions(name: str, game: Game, limit: int):
    with _candidate_ranker(name):
        return search_mod.rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=limit)


def _choose_versus(match: VersusMatch, side_name: str, name: str, opponent_model: str) -> VersusChoice | None:
    cfg = DEFAULT_VERSUS_SEARCH_CONFIG.normalized()
    own = versus_mod._side(match, side_name)
    ranked = _rank_actions(name, own.game, cfg.candidate_width)
    if not ranked:
        return None
    opponent_side = versus_mod._opponent_name(side_name)
    best = None
    for action, evaluation in ranked:
        after, resolution = versus_mod._simulate_action(match, side_name, action)
        score = versus_mod.score_versus_state(
            after, side_name, weights=DEFAULT_VERSUS_WEIGHTS, resolution=resolution,
            solo_score=evaluation.score, path_length=len(action.placement.path), action_side=side_name,
        )
        reply_action = None
        if cfg.opponent_reply_width > 0 and after.winner is None and not versus_mod._side(after, opponent_side).game.game_over:
            replies = _rank_actions(opponent_model, versus_mod._side(after, opponent_side).game, cfg.opponent_reply_width)
            worst_score = None
            for reply, reply_eval in replies:
                replied, reply_resolution = versus_mod._simulate_action(after, opponent_side, reply)
                reply_score = versus_mod.score_versus_state(
                    replied, side_name, weights=DEFAULT_VERSUS_WEIGHTS, resolution=reply_resolution,
                    solo_score=-reply_eval.score, path_length=len(reply.placement.path), action_side=opponent_side,
                )
                if worst_score is None or reply_score < worst_score:
                    worst_score = reply_score
                    reply_action = reply
            if worst_score is not None:
                score = worst_score
        choice = VersusChoice(action, score, evaluation, resolution, reply_action)
        key = (choice.score, choice.resolution.sent_lines, choice.resolution.canceled_lines, -len(choice.action.placement.path), -int(choice.action.use_hold))
        if best is None:
            best = choice
        else:
            best_key = (best.score, best.resolution.sent_lines, best.resolution.canceled_lines, -len(best.action.placement.path), -int(best.action.use_hold))
            if key > best_key:
                best = choice
    return best


def _versus_game(candidate: str, seed: int, swapped: bool, max_turns: int) -> dict[str, object]:
    match = VersusMatch(seed, garbage_cap=8)
    candidate_side = "ai" if swapped else "player"
    baseline_side = "player" if swapped else "ai"
    turn = "player"
    turns = 0
    candidate_max_b2b = baseline_max_b2b = 0
    while match.winner is None and turns < max_turns:
        choice = _choose_versus(match, turn, candidate if turn == candidate_side else BASELINE, BASELINE if turn == candidate_side else candidate)
        if choice is None:
            versus_mod._side(match, turn).game.game_over = True
            match._update_winner()
            break
        side = versus_mod._side(match, turn)
        result = apply_search_action(side.game, choice.action)
        match.resolve_lock(turn, result)
        turns += 1
        candidate_max_b2b = max(candidate_max_b2b, versus_mod._side(match, candidate_side).game.b2b_chain)
        baseline_max_b2b = max(baseline_max_b2b, versus_mod._side(match, baseline_side).game.b2b_chain)
        turn = "ai" if turn == "player" else "player"
    winner = match.winner or "draw"
    logical_winner = "draw" if winner == "draw" else ("candidate" if winner == candidate_side else "baseline")
    cand = versus_mod._side(match, candidate_side)
    base = versus_mod._side(match, baseline_side)
    return {
        "seed": seed, "swapped": swapped, "winner": logical_winner, "turns": turns,
        "candidateAttack": cand.game.attack, "baselineAttack": base.game.attack,
        "candidateSent": cand.sent, "baselineSent": base.sent,
        "candidateCanceled": cand.canceled, "baselineCanceled": base.canceled,
        "candidateReceived": cand.received, "baselineReceived": base.received,
        "candidateTopout": int(cand.game.game_over), "baselineTopout": int(base.game.game_over),
        "candidateMaxB2B": candidate_max_b2b, "baselineMaxB2B": baseline_max_b2b,
    }


def _versus(candidate: str, games: int = 8, max_turns: int = 110) -> dict[str, object]:
    rows = [_versus_game(candidate, 9_100_001 + (i // 2) * 101, bool(i % 2), max_turns) for i in range(games)]
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
    short = _run_solo(ALL, (6_100_001, 6_100_038), 100)
    top3 = _top(short, 3)
    fresh = _run_solo((BASELINE,) + tuple(top3), (7_300_001, 7_300_098, 7_300_195), 240)
    baseline = fresh[BASELINE]
    fresh_pass = []
    for name in top3:
        r = fresh[name]
        safe = int(r["topouts"]) <= int(baseline["topouts"]) and int(r["completed"]) >= int(baseline["completed"])
        useful = float(r["app"]) >= float(baseline["app"]) * 1.01 or int(r["completed"]) > int(baseline["completed"])
        if safe and useful:
            fresh_pass.append(name)
    fresh_pass.sort(key=lambda name: _quality(fresh[name]), reverse=True)
    finalists = fresh_pass[:2]
    versus = {name: _versus(name) for name in finalists}
    winner = None
    for name in finalists:
        v = versus[name]
        if (
            int(v["candidateWins"]) > int(v["baselineWins"])
            and int(v["candidateTopouts"]) <= int(v["baselineTopouts"])
            and float(v["candidateMeanSent"]) >= float(v["baselineMeanSent"])
            and float(v["candidateMeanReceived"]) <= float(v["baselineMeanReceived"])
        ):
            winner = name
            break
    payload = {
        "candidateCount": len(CANDIDATES), "candidates": list(CANDIDATES),
        "short": short, "shortTop3": top3, "fresh": fresh,
        "freshPass": fresh_pass, "finalists": finalists, "versus": versus, "winner": winner,
    }
    target = Path("data/self-improve/latest-tournament.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
