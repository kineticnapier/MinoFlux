from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from minoflux_ai import DEFAULT_WEIGHTS, SearchConfig
from minoflux_ai.search import SearchAction, apply_search_action, clone_game, rank_search_actions
from minoflux_ai.versus_search import VersusSearchConfig, clone_versus_match, score_versus_state
from minoflux_engine import Game, T_SPIN_DOUBLE, T_SPIN_TRIPLE, VersusMatch

SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    discount=0.9,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8_000,
)
VERSUS = VersusSearchConfig(
    placement_search=SEARCH,
    candidate_width=6,
    opponent_reply_width=1,
)
ROOT_WIDTH = 4
ROLLOUT_DEPTH = 2
CANDIDATES = (
    "trajectory_attack_sum",
    "trajectory_attack_floor",
    "trajectory_spin_conversion",
    "trajectory_slot_retention",
    "trajectory_slot_conversion",
    "trajectory_b2b_chain",
    "trajectory_hole_relief",
    "trajectory_height_relief",
    "trajectory_clean_attack",
    "trajectory_t_window",
    "trajectory_hold_release",
    "trajectory_survival",
    "trajectory_pressure_safe",
    "trajectory_balanced",
)


def next_t_distance(game: Game) -> int:
    if game.current == "T":
        return 0
    for index, piece in enumerate(game.queue):
        if piece == "T":
            return index + 1
        if index >= 4:
            break
    return 6


def trajectory_stats(game: Game, action: SearchAction) -> dict[str, float]:
    child = clone_game(game)
    initial_t_distance = next_t_distance(game)
    before_slots = 0.0
    before_holes = 0.0
    before_depth = 0.0
    before_height = 0.0
    attack_values: list[float] = []
    spin_lines = 0.0
    difficult_clears = 0.0
    t_conversions = 0.0
    slot_min = 99.0
    slot_last = 0.0
    hold_t_steps = 0.0
    survival_steps = 0.0

    root_ranked = rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=ROOT_WIDTH)
    root_eval = next((evaluation for candidate, evaluation in root_ranked if candidate == action), None)
    if root_eval is not None:
        before_slots = float(root_eval.features.board.t_spin_slots)
        before_holes = float(root_eval.features.board.holes)
        before_depth = float(root_eval.features.board.hole_depth)
        before_height = float(root_eval.features.board.max_height)

    root_result = apply_search_action(child, action)
    attack_values.append(float(root_result.attack))
    spin_lines += float(root_result.lines if root_result.spin is not None else 0)
    difficult_clears += float(root_result.spin is not None or root_result.lines == 4)
    t_conversions += float(action.placement.piece == "T" and root_result.spin is not None)
    hold_t_steps += float(child.hold_piece == "T")
    survival_steps += float(not child.game_over)

    last_eval = root_eval
    for _ in range(ROLLOUT_DEPTH):
        if child.game_over:
            break
        ranked = rank_search_actions(child, DEFAULT_WEIGHTS, SEARCH, limit=1)
        if not ranked:
            break
        future_action, future_eval = ranked[0]
        result = apply_search_action(child, future_action)
        attack_values.append(float(result.attack))
        spin_lines += float(result.lines if result.spin is not None else 0)
        difficult_clears += float(result.spin is not None or result.lines == 4)
        t_conversions += float(future_action.placement.piece == "T" and result.spin is not None)
        hold_t_steps += float(child.hold_piece == "T")
        survival_steps += float(not child.game_over)
        last_eval = future_eval
        slot_min = min(slot_min, float(future_eval.features.board.t_spin_slots))
        slot_last = float(future_eval.features.board.t_spin_slots)

    if last_eval is None:
        final_holes = before_holes
        final_depth = before_depth
        final_height = before_height
    else:
        final_holes = float(last_eval.features.board.holes)
        final_depth = float(last_eval.features.board.hole_depth)
        final_height = float(last_eval.features.board.max_height)
        if slot_min == 99.0:
            slot_min = float(last_eval.features.board.t_spin_slots)
            slot_last = slot_min

    if slot_min == 99.0:
        slot_min = before_slots
        slot_last = before_slots

    return {
        "attack_sum": sum(attack_values),
        "attack_floor": min(attack_values) if attack_values else 0.0,
        "spin_lines": spin_lines,
        "difficult": difficult_clears,
        "t_conversions": t_conversions,
        "slot_min": slot_min,
        "slot_last": slot_last,
        "slot_retained": min(before_slots, slot_min),
        "slot_consumed": max(0.0, before_slots - slot_last),
        "hole_relief": before_holes - final_holes,
        "depth_relief": (before_depth - final_depth) / 4.0,
        "height_relief": before_height - final_height,
        "hold_t_steps": hold_t_steps,
        "survival": survival_steps / (ROLLOUT_DEPTH + 1),
        "t_urgency": max(0.0, (5.0 - initial_t_distance) / 5.0),
        "final_holes": final_holes,
        "final_height": final_height,
    }


def bonus(mode: str, stats: dict[str, float], b2b_active: bool) -> float:
    safe_attack = stats["attack_sum"] / (1.0 + 0.40 * stats["final_holes"] + 0.08 * stats["final_height"])
    values = {
        "trajectory_attack_sum": 0.34 * stats["attack_sum"],
        "trajectory_attack_floor": 0.65 * stats["attack_floor"],
        "trajectory_spin_conversion": 0.65 * stats["spin_lines"] + 0.55 * stats["t_conversions"],
        "trajectory_slot_retention": 0.55 * stats["slot_retained"],
        "trajectory_slot_conversion": 0.65 * stats["t_urgency"] * stats["slot_consumed"] + 0.45 * stats["t_conversions"],
        "trajectory_b2b_chain": (0.55 if b2b_active else 0.28) * stats["difficult"],
        "trajectory_hole_relief": 0.55 * stats["hole_relief"] + 0.18 * stats["depth_relief"],
        "trajectory_height_relief": 0.30 * stats["height_relief"] + 0.10 * stats["depth_relief"],
        "trajectory_clean_attack": 0.48 * safe_attack,
        "trajectory_t_window": stats["t_urgency"] * (0.45 * stats["slot_retained"] + 0.60 * stats["t_conversions"] + 0.20 * stats["attack_sum"]),
        "trajectory_hold_release": 0.35 * stats["t_conversions"] - 0.12 * stats["hold_t_steps"] * stats["t_urgency"],
        "trajectory_survival": 1.80 * stats["survival"],
        "trajectory_pressure_safe": 0.36 * safe_attack + 0.22 * stats["hole_relief"] + 0.15 * stats["height_relief"],
        "trajectory_balanced": 0.22 * stats["attack_sum"] + 0.28 * stats["spin_lines"] + 0.22 * stats["slot_retained"] + 0.18 * stats["hole_relief"] + 0.65 * stats["survival"],
    }
    return values[mode]


def choose(game: Game, mode: str | None) -> SearchAction | None:
    ranked = rank_search_actions(game, DEFAULT_WEIGHTS, SEARCH, limit=1 if mode is None else ROOT_WIDTH)
    if not ranked:
        return None
    if mode is None:
        return ranked[0][0]
    best_action: SearchAction | None = None
    best_key: tuple[float, ...] | None = None
    for action, evaluation in ranked:
        stats = trajectory_stats(game, action)
        key = (
            evaluation.score + bonus(mode, stats, game.back_to_back),
            evaluation.features.attack,
            evaluation.features.spin_lines,
            evaluation.features.lines,
            -evaluation.features.board.holes,
            -evaluation.features.board.max_height,
            -int(action.use_hold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_action = action
    return best_action


def solo_game(seed: int, pieces: int, mode: str | None) -> dict[str, object]:
    game = Game(seed)
    spins = spin_lines = tsd = tst = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < pieces:
        action = choose(game, mode)
        if action is None:
            break
        result = apply_search_action(game, action)
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
        tsd += int(result.spin == T_SPIN_DOUBLE)
        tst += int(result.spin == T_SPIN_TRIPLE)
        max_b2b = max(max_b2b, game.b2b_chain)
    return {
        "pieces": game.pieces_placed,
        "attack": game.attack,
        "topout": game.game_over,
        "completed": not game.game_over and game.pieces_placed >= pieces,
        "spins": spins,
        "spin_lines": spin_lines,
        "tsd": tsd,
        "tst": tst,
        "max_b2b": max_b2b,
    }


def solo(mode: str | None, seeds: tuple[int, ...], pieces: int) -> dict[str, object]:
    games = [solo_game(seed, pieces, mode) for seed in seeds]
    total_pieces = sum(int(game["pieces"]) for game in games)
    total_attack = sum(int(game["attack"]) for game in games)
    return {
        "mode": mode or "baseline",
        "pieces": total_pieces,
        "attack": total_attack,
        "app": total_attack / max(1, total_pieces),
        "topouts": sum(bool(game["topout"]) for game in games),
        "completed": sum(bool(game["completed"]) for game in games),
        "spins": sum(int(game["spins"]) for game in games),
        "spin_lines": sum(int(game["spin_lines"]) for game in games),
        "tsd": sum(int(game["tsd"]) for game in games),
        "tst": sum(int(game["tst"]) for game in games),
        "max_b2b": max(int(game["max_b2b"]) for game in games),
    }


def solo_task(args: tuple[str | None, tuple[int, ...], int]) -> dict[str, object]:
    return solo(*args)


def quality_key(result: dict[str, object]) -> tuple[float, ...]:
    return (
        -float(result["topouts"]),
        float(result["completed"]),
        float(result["app"]),
        float(result["attack"]),
        float(result["tsd"]) + 1.5 * float(result["tst"]),
        float(result["spin_lines"]),
        float(result["max_b2b"]),
    )


def other(side: str) -> str:
    return "ai" if side == "player" else "player"


def baseline_versus_action(match: VersusMatch, side: str) -> SearchAction | None:
    from minoflux_ai.versus_search import choose_versus_action
    choice = choose_versus_action(match, side, DEFAULT_WEIGHTS, VERSUS)
    return choice.action if choice else None


def candidate_versus_action(match: VersusMatch, side: str, mode: str) -> SearchAction | None:
    own = match.side(side)
    opponent = other(side)
    ranked = rank_search_actions(own.game, DEFAULT_WEIGHTS, SEARCH, limit=VERSUS.candidate_width)
    best_action: SearchAction | None = None
    best_key: tuple[float, ...] | None = None
    for action, evaluation in ranked:
        stats = trajectory_stats(own.game, action)
        after = clone_versus_match(match)
        result = apply_search_action(after.side(side).game, action)
        resolution = after.resolve_lock(side, result)
        score = score_versus_state(
            after,
            side,
            resolution=resolution,
            solo_score=evaluation.score + bonus(mode, stats, own.game.back_to_back),
            path_length=len(action.placement.path),
            action_side=side,
        )
        if after.winner is None:
            replies = rank_search_actions(after.side(opponent).game, DEFAULT_WEIGHTS, SEARCH, limit=1)
            if replies:
                reply_action, reply_eval = replies[0]
                replied = clone_versus_match(after)
                reply_result = apply_search_action(replied.side(opponent).game, reply_action)
                reply_resolution = replied.resolve_lock(opponent, reply_result)
                score = score_versus_state(
                    replied,
                    side,
                    resolution=reply_resolution,
                    solo_score=-reply_eval.score,
                    path_length=len(reply_action.placement.path),
                    action_side=opponent,
                )
        key = (
            score,
            resolution.sent_lines,
            resolution.canceled_lines,
            -len(action.placement.path),
            -int(action.use_hold),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_action = action
    return best_action


def versus_game(seed: int, mode: str, swap: bool, max_turns: int = 90) -> dict[str, object]:
    match = VersusMatch(seed, garbage_cap=8)
    candidate_side = "ai" if swap else "player"
    turn = "player"
    turns = 0
    max_b2b = {"player": 0, "ai": 0}
    while match.winner is None and turns < max_turns:
        action = candidate_versus_action(match, turn, mode) if turn == candidate_side else baseline_versus_action(match, turn)
        if action is None:
            match.side(turn).game.game_over = True
            match._update_winner()
            break
        result = apply_search_action(match.side(turn).game, action)
        match.resolve_lock(turn, result)
        max_b2b[turn] = max(max_b2b[turn], match.side(turn).game.b2b_chain)
        turns += 1
        turn = other(turn)
    baseline_side = other(candidate_side)
    winner = match.winner or "draw"
    winner = "candidate" if winner == candidate_side else "baseline" if winner == baseline_side else "draw"
    candidate = match.side(candidate_side)
    baseline = match.side(baseline_side)
    return {
        "winner": winner,
        "candidate_attack": candidate.game.attack,
        "baseline_attack": baseline.game.attack,
        "candidate_sent": candidate.sent,
        "baseline_sent": baseline.sent,
        "candidate_canceled": candidate.canceled,
        "baseline_canceled": baseline.canceled,
        "candidate_received": candidate.received,
        "baseline_received": baseline.received,
        "candidate_b2b": max_b2b[candidate_side],
        "baseline_b2b": max_b2b[baseline_side],
        "candidate_topout": candidate.game.game_over,
        "baseline_topout": baseline.game.game_over,
    }


def versus(mode: str) -> dict[str, object]:
    games = [versus_game(920_001 + (index // 2) * 211, mode, bool(index % 2)) for index in range(6)]
    average = lambda key: sum(float(game[key]) for game in games) / len(games)
    return {
        "mode": mode,
        "candidate_wins": sum(game["winner"] == "candidate" for game in games),
        "baseline_wins": sum(game["winner"] == "baseline" for game in games),
        "draws": sum(game["winner"] == "draw" for game in games),
        "candidate_attack": average("candidate_attack"),
        "baseline_attack": average("baseline_attack"),
        "candidate_sent": average("candidate_sent"),
        "baseline_sent": average("baseline_sent"),
        "candidate_canceled": average("candidate_canceled"),
        "baseline_canceled": average("baseline_canceled"),
        "candidate_received": average("candidate_received"),
        "baseline_received": average("baseline_received"),
        "candidate_b2b": average("candidate_b2b"),
        "baseline_b2b": average("baseline_b2b"),
        "candidate_topouts": sum(bool(game["candidate_topout"]) for game in games),
        "baseline_topouts": sum(bool(game["baseline_topout"]) for game in games),
    }


def main() -> None:
    short_seeds = (52_001, 52_098)
    fresh_seeds = (252_001, 252_098, 252_195)
    baseline_short = solo(None, short_seeds, 55)
    with ProcessPoolExecutor(max_workers=4) as executor:
        short = list(executor.map(solo_task, [(mode, short_seeds, 55) for mode in CANDIDATES]))
    eligible = [
        result for result in short
        if int(result["topouts"]) <= int(baseline_short["topouts"])
        and int(result["completed"]) >= int(baseline_short["completed"])
    ]
    eligible.sort(key=quality_key, reverse=True)
    finalists = [str(result["mode"]) for result in eligible[:3]]

    baseline_fresh = solo(None, fresh_seeds, 120)
    with ProcessPoolExecutor(max_workers=3) as executor:
        fresh = list(executor.map(solo_task, [(mode, fresh_seeds, 120) for mode in finalists]))
    viable = [
        result for result in fresh
        if int(result["topouts"]) <= int(baseline_fresh["topouts"])
        and int(result["completed"]) >= int(baseline_fresh["completed"])
        and float(result["app"]) >= float(baseline_fresh["app"]) * 1.01
    ]
    viable.sort(key=quality_key, reverse=True)
    versus_modes = [str(result["mode"]) for result in viable[:2]]
    versus_results = [versus(mode) for mode in versus_modes]

    output = {
        "candidate_count": len(CANDIDATES),
        "candidates": list(CANDIDATES),
        "short_baseline": baseline_short,
        "short": short,
        "short_finalists": finalists,
        "fresh_baseline": baseline_fresh,
        "fresh": fresh,
        "versus_modes": versus_modes,
        "versus": versus_results,
    }
    path = Path("tournament-results.json")
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
