from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
from minoflux_ai.search import (
    SearchConfig,
    apply_search_action,
    choose_search_action,
    clone_game,
    rank_search_actions,
)
from minoflux_ai.versus_search import (
    VersusSearchConfig,
    choose_versus_action,
    clone_versus_match,
    score_versus_state,
)
from minoflux_engine import Game, VersusMatch

CANDIDATES = (
    "next_clean_frontier",
    "next_downstack_frontier",
    "next_clean_attack",
    "next_recovery_robustness",
    "next_low_variance",
    "next_option_depth",
    "next_hold_flexibility",
    "next_t_conversion",
    "next_slot_downstack",
    "next_b2b_routes",
    "next_attack_diversity",
    "next_safe_spin_routes",
    "next_height_escape",
    "next_queue_cleanliness",
)

SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=6,
    discount=0.90,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
)
FUTURE_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=6,
    discount=0.90,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=6_000,
)
VERSUS_CFG = VersusSearchConfig(
    placement_search=SEARCH,
    candidate_width=6,
    opponent_reply_width=1,
)


def _t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 4:
            break
    return 6


def _state_key(game: Game) -> tuple[object, ...]:
    rows = tuple(tuple(cell is not None for cell in row) for row in game.board)
    return rows, game.current, game.hold_piece, tuple(list(game.queue)[:5]), game.combo, game.b2b_chain


class FrontierScorer:
    def __init__(self, name: str) -> None:
        self.name = name

    def score_many(self, game: Game, evaluations: list[PlacementEvaluation] | tuple[PlacementEvaluation, ...]):
        result: list[float] = []
        for evaluation in evaluations:
            if evaluation.features.game_over:
                result.append(evaluation.score)
                continue
            child = clone_game(game)
            child.place(evaluation.placement)
            after = evaluation.features.board
            future_pairs = rank_search_actions(
                child,
                DEFAULT_WEIGHTS,
                FUTURE_SEARCH,
                limit=6,
            )
            futures = [item[1] for item in future_pairs]
            actions = [item[0] for item in future_pairs]
            if not futures:
                result.append(evaluation.score - 50.0)
                continue

            clean = [f for f in futures if not f.features.game_over and f.features.board.holes <= after.holes]
            down = [
                f for f in futures
                if not f.features.game_over
                and (f.features.board.holes < after.holes or f.features.board.hole_depth < after.hole_depth)
            ]
            safe = [f for f in futures if not f.features.game_over and f.features.board.max_height <= max(12, after.max_height)]
            spins = [f for f in futures if f.features.spin_lines > 0]
            difficult = [f for f in futures if f.features.spin_lines > 0 or f.features.lines == 4]
            clean_attack = [f for f in clean if f.features.attack > 0]
            hole_deltas = [f.features.board.holes - after.holes for f in futures]
            height_deltas = [f.features.board.max_height - after.max_height for f in futures]
            best_attack = max((f.features.attack for f in futures), default=0)
            best_clean_attack = max((f.features.attack for f in clean), default=0)
            best_hole_relief = max((after.holes - f.features.board.holes for f in futures), default=0)
            best_depth_relief = max((after.hole_depth - f.features.board.hole_depth for f in futures), default=0)
            best_height_relief = max((after.max_height - f.features.board.max_height for f in futures), default=0)
            slot_preserving_down = sum(
                f.features.board.t_spin_slots >= max(1, after.t_spin_slots)
                for f in down
            )
            hold_count = sum(action.use_hold for action in actions)
            direct_count = len(actions) - hold_count
            near_best = sum(f.score >= max(x.score for x in futures) - 1.0 for f in futures)
            attack_piece_kinds = len({f.placement.piece for f in futures if f.features.attack > 0})
            t_dist = _t_distance(child)
            bonus = 0.0

            if self.name == "next_clean_frontier":
                bonus = 0.16 * len(clean) + 0.10 * best_clean_attack
            elif self.name == "next_downstack_frontier":
                bonus = 0.20 * len(down) + 0.08 * best_hole_relief + 0.015 * best_depth_relief
            elif self.name == "next_clean_attack":
                bonus = 0.34 * best_clean_attack + 0.10 * len(clean_attack)
            elif self.name == "next_recovery_robustness":
                danger = max(0.0, (after.max_height - 9) / 6.0)
                bonus = danger * (0.20 * len(down) + 0.05 * best_depth_relief + 0.10 * best_height_relief)
            elif self.name == "next_low_variance":
                worst_holes = max(hole_deltas, default=0)
                worst_height = max(height_deltas, default=0)
                bonus = -0.16 * max(0, worst_holes) - 0.08 * max(0, worst_height) + 0.08 * len(safe)
            elif self.name == "next_option_depth":
                bonus = 0.15 * near_best + 0.05 * len(safe)
            elif self.name == "next_hold_flexibility":
                balance = min(hold_count, direct_count)
                bonus = 0.18 * balance + 0.08 * near_best
            elif self.name == "next_t_conversion":
                urgency = max(0.0, (4.0 - t_dist) / 4.0)
                bonus = urgency * (0.42 * len(spins) + 0.18 * max((f.features.spin_lines for f in spins), default=0))
            elif self.name == "next_slot_downstack":
                bonus = 0.22 * slot_preserving_down + 0.05 * best_depth_relief
            elif self.name == "next_b2b_routes":
                chain = 1.0 if child.back_to_back or child.b2b_chain > 0 else 0.35
                bonus = chain * (0.28 * len(difficult) + 0.12 * best_attack)
            elif self.name == "next_attack_diversity":
                bonus = 0.22 * attack_piece_kinds + 0.14 * best_attack
            elif self.name == "next_safe_spin_routes":
                safe_spins = sum(f.features.spin_lines > 0 and f.features.board.holes <= after.holes for f in futures)
                bonus = 0.34 * safe_spins + 0.08 * best_clean_attack
            elif self.name == "next_height_escape":
                danger = max(0.0, (after.max_height - 10) / 5.0)
                bonus = danger * (0.24 * best_height_relief + 0.08 * len(safe))
            elif self.name == "next_queue_cleanliness":
                # The actual next piece/hold branch is represented in future_pairs; reward states
                # where several concrete next-piece routes stay clean rather than one lucky route.
                clean_ratio = len(clean) / max(1, len(futures))
                bonus = 0.55 * clean_ratio + 0.08 * best_clean_attack
            result.append(evaluation.score + bonus)
        return tuple(result)


def _play(name: str, seed: int, max_pieces: int) -> dict[str, object]:
    game = Game(seed)
    scorer = None if name == "baseline" else FrontierScorer(name)
    spins = spin_lines = tsd = tst = perfect_clears = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < max_pieces:
        choice = choose_search_action(game, DEFAULT_WEIGHTS, SEARCH, scorer=scorer)
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
            if result.lines == 2:
                tsd += 1
            elif result.lines == 3:
                tst += 1
        perfect_clears += int(result.perfect_clear)
        max_b2b = max(max_b2b, game.b2b_chain)
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "app": game.attack / max(1, game.pieces_placed),
        "topout": game.game_over,
        "completed": (not game.game_over and game.pieces_placed >= max_pieces),
        "spins": spins,
        "spin_lines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "pc": perfect_clears,
        "max_b2b": max_b2b,
    }


def _benchmark(name: str, seeds: list[int], max_pieces: int) -> dict[str, object]:
    games = [_play(name, seed, max_pieces) for seed in seeds]
    pieces = sum(int(g["pieces"]) for g in games)
    attack = sum(int(g["attack"]) for g in games)
    return {
        "name": name,
        "games": len(games),
        "maxPieces": max_pieces,
        "pieces": pieces,
        "attack": attack,
        "app": attack / max(1, pieces),
        "topouts": sum(bool(g["topout"]) for g in games),
        "completed": sum(bool(g["completed"]) for g in games),
        "spins": sum(int(g["spins"]) for g in games),
        "spinLines": sum(int(g["spin_lines"]) for g in games),
        "tsd": sum(int(g["tsd"]) for g in games),
        "tst": sum(int(g["tst"]) for g in games),
        "perfectClears": sum(int(g["pc"]) for g in games),
        "maxB2B": max((int(g["max_b2b"]) for g in games), default=0),
        "perGame": games,
    }


def _run_bench_task(args: tuple[str, list[int], int]) -> dict[str, object]:
    return _benchmark(*args)


def _rank_key(result: dict[str, object]) -> tuple[float, ...]:
    return (
        float(result["completed"]),
        -float(result["topouts"]),
        float(result["app"]),
        float(result["spinLines"]),
        float(result["tsd"]),
        float(result["maxB2B"]),
    )


def _choose_candidate_action(match: VersusMatch, side_name: str, name: str):
    own = match.side(side_name)
    scorer = FrontierScorer(name)
    ranked = rank_search_actions(
        own.game,
        DEFAULT_WEIGHTS,
        SEARCH,
        limit=6,
        scorer=scorer,
    )
    if not ranked:
        return None
    opponent_name = "ai" if side_name == "player" else "player"
    best = None
    for action, evaluation in ranked:
        after = clone_versus_match(match)
        side = after.side(side_name)
        lock = apply_search_action(side.game, action)
        resolution = after.resolve_lock(side_name, lock)
        score = score_versus_state(
            after,
            side_name,
            resolution=resolution,
            solo_score=evaluation.score,
            path_length=len(action.placement.path),
            action_side=side_name,
        )
        if after.winner is None:
            replies = rank_search_actions(
                after.side(opponent_name).game,
                DEFAULT_WEIGHTS,
                SEARCH,
                limit=1,
            )
            if replies:
                reply, reply_eval = replies[0]
                replied = clone_versus_match(after)
                reply_lock = apply_search_action(replied.side(opponent_name).game, reply)
                reply_resolution = replied.resolve_lock(opponent_name, reply_lock)
                score = score_versus_state(
                    replied,
                    side_name,
                    resolution=reply_resolution,
                    solo_score=-reply_eval.score,
                    path_length=len(reply.placement.path),
                    action_side=opponent_name,
                )
        key = (score, resolution.sent_lines, resolution.canceled_lines, -len(action.placement.path))
        if best is None or key > best[0]:
            best = (key, action)
    return None if best is None else best[1]


def _versus_game(name: str, seed: int, candidate_side: str, max_turns: int = 120) -> dict[str, object]:
    match = VersusMatch(seed, garbage_cap=8)
    turn = "player"
    turns = 0
    max_b2b = {"player": 0, "ai": 0}
    while match.winner is None and turns < max_turns:
        if turn == candidate_side:
            action = _choose_candidate_action(match, turn, name)
        else:
            choice = choose_versus_action(match, turn, DEFAULT_WEIGHTS, VERSUS_CFG)
            action = None if choice is None else choice.action
        if action is None:
            match.side(turn).game.game_over = True
            match._update_winner()
            break
        lock = apply_search_action(match.side(turn).game, action)
        match.resolve_lock(turn, lock)
        max_b2b[turn] = max(max_b2b[turn], match.side(turn).game.b2b_chain)
        turns += 1
        turn = "ai" if turn == "player" else "player"
    candidate = match.side(candidate_side)
    baseline_side = match.side("ai" if candidate_side == "player" else "player")
    winner = match.winner or "draw"
    if winner == candidate_side:
        mapped = "candidate"
    elif winner == "draw":
        mapped = "draw"
    else:
        mapped = "baseline"
    return {
        "seed": seed,
        "candidateSide": candidate_side,
        "winner": mapped,
        "turns": turns,
        "candidateAttack": candidate.game.attack,
        "baselineAttack": baseline_side.game.attack,
        "candidateSent": candidate.sent,
        "baselineSent": baseline_side.sent,
        "candidateCanceled": candidate.canceled,
        "baselineCanceled": baseline_side.canceled,
        "candidateReceived": candidate.received,
        "baselineReceived": baseline_side.received,
        "candidateMaxB2B": max_b2b[candidate_side],
        "baselineMaxB2B": max_b2b["ai" if candidate_side == "player" else "player"],
        "candidateTopout": candidate.game.game_over,
        "baselineTopout": baseline_side.game.game_over,
    }


def _versus(name: str, games: int = 6) -> dict[str, object]:
    rows = []
    for index in range(games):
        pair = index // 2
        seed = 930_001 + pair * 97
        side = "player" if index % 2 == 0 else "ai"
        rows.append(_versus_game(name, seed, side))
    return {
        "name": name,
        "games": games,
        "wins": sum(r["winner"] == "candidate" for r in rows),
        "losses": sum(r["winner"] == "baseline" for r in rows),
        "draws": sum(r["winner"] == "draw" for r in rows),
        "candidateMeanAttack": mean(r["candidateAttack"] for r in rows),
        "baselineMeanAttack": mean(r["baselineAttack"] for r in rows),
        "candidateMeanSent": mean(r["candidateSent"] for r in rows),
        "baselineMeanSent": mean(r["baselineSent"] for r in rows),
        "candidateMeanCanceled": mean(r["candidateCanceled"] for r in rows),
        "baselineMeanCanceled": mean(r["baselineCanceled"] for r in rows),
        "candidateMeanReceived": mean(r["candidateReceived"] for r in rows),
        "baselineMeanReceived": mean(r["baselineReceived"] for r in rows),
        "candidateMaxB2B": max(r["candidateMaxB2B"] for r in rows),
        "baselineMaxB2B": max(r["baselineMaxB2B"] for r in rows),
        "candidateTopouts": sum(bool(r["candidateTopout"]) for r in rows),
        "baselineTopouts": sum(bool(r["baselineTopout"]) for r in rows),
        "perGame": rows,
    }


def main() -> None:
    short_seeds = [810_011, 810_108]
    fresh_seeds = [910_037, 910_134, 910_231]
    names = ["baseline", *CANDIDATES]
    short: list[dict[str, object]] = []
    # Candidate-level processes keep a slow lookahead candidate from blocking every other candidate.
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_run_bench_task, (name, short_seeds, 100)): name
            for name in names
        }
        for future in as_completed(futures):
            short.append(future.result())
    short.sort(key=lambda r: names.index(str(r["name"])))
    baseline_short = next(r for r in short if r["name"] == "baseline")
    candidate_short = [r for r in short if r["name"] != "baseline"]
    finalists = sorted(candidate_short, key=_rank_key, reverse=True)[:3]

    fresh_names = ["baseline", *(str(r["name"]) for r in finalists)]
    fresh: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_run_bench_task, (name, fresh_seeds, 180)): name
            for name in fresh_names
        }
        for future in as_completed(futures):
            fresh.append(future.result())
    fresh.sort(key=lambda r: fresh_names.index(str(r["name"])))
    baseline_fresh = next(r for r in fresh if r["name"] == "baseline")
    eligible = [
        r for r in fresh
        if r["name"] != "baseline"
        and int(r["topouts"]) <= int(baseline_fresh["topouts"])
        and int(r["completed"]) >= int(baseline_fresh["completed"])
        and float(r["app"]) >= float(baseline_fresh["app"]) * 1.01
    ]
    versus_names = [str(r["name"]) for r in sorted(eligible, key=_rank_key, reverse=True)[:2]]
    versus = [_versus(name, games=6) for name in versus_names]

    report = {
        "candidateCount": len(CANDIDATES),
        "short": short,
        "shortFinalists": [str(r["name"]) for r in finalists],
        "fresh": fresh,
        "versus": versus,
        "baselineShort": baseline_short,
        "baselineFresh": baseline_fresh,
    }
    Path("tournament-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "candidateCount": report["candidateCount"],
        "shortFinalists": report["shortFinalists"],
        "versusNames": versus_names,
    }))


if __name__ == "__main__":
    main()
