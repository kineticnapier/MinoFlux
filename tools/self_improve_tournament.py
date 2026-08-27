from __future__ import annotations

import argparse
import json
from dataclasses import replace

from minoflux_engine import Game
from minoflux_engine.pieces import SHAPES
from minoflux_ai import search
from minoflux_ai import versus_search
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
from minoflux_ai.search import DEFAULT_SEARCH_CONFIG, apply_search_action, clone_game

ORIGINAL_RANK = search.rank_search_actions

CANDIDATES = {
    "baseline": ("baseline", 0.0),
    "tsd_slot_delta": ("tsd_slot_delta", 0.8),
    "tst_slot_delta": ("tst_slot_delta", 1.2),
    "spin_quality_delta": ("spin_quality_delta", 0.7),
    "covered_holes_delta": ("covered_holes_delta", -0.35),
    "deep_holes_delta": ("deep_holes_delta", -0.22),
    "danger_hole_product": ("danger_hole_product", -0.025),
    "top_band": ("top_band", -0.20),
    "clean_well": ("clean_well", 0.16),
    "combo_continue": ("combo_continue", 0.35),
    "tsd_event": ("tsd_event", 0.65),
    "tst_event": ("tst_event", 1.0),
    "hold_t_for_slot": ("hold_t_for_slot", 0.35),
    "use_t_on_slot": ("use_t_on_slot", 0.45),
    "low_stack_slot": ("low_stack_slot", 0.25),
}


def heights(board):
    if not board:
        return ()
    h = len(board)
    w = len(board[0])
    out = []
    for x in range(w):
        y = 0
        while y < h and board[y][x] is None:
            y += 1
        out.append(h - y)
    return tuple(out)


def hole_stats(board):
    h = len(board)
    w = len(board[0]) if board else 0
    covered = deep = 0
    for x in range(w):
        blocks = 0
        for y in range(h):
            if board[y][x] is not None:
                blocks += 1
            elif blocks:
                covered += blocks
                deep += blocks * blocks
    return covered, deep


def spin_slot_lines(board):
    """Count geometric T slots that would clear exactly 2 or at least 3 rows."""
    h = len(board)
    w = len(board[0]) if board else 0
    if not board:
        return (0, 0)
    shapes = SHAPES["T"]
    tsd = tst = 0
    for shape in shapes:
        minx = min(dx for dx, dy in shape)
        maxx = max(dx for dx, dy in shape)
        miny = min(dy for dx, dy in shape)
        maxy = max(dy for dx, dy in shape)
        for x in range(-minx, w - maxx):
            for y in range(-miny, h - maxy):
                cells = [(x + dx, y + dy) for dx, dy in shape]
                if any(board[cy][cx] is not None for cx, cy in cells):
                    continue
                px = x + 1
                py = y + 1
                corners = 0
                for cx, cy in ((px - 1, py - 1), (px + 1, py - 1), (px - 1, py + 1), (px + 1, py + 1)):
                    if cx < 0 or cx >= w or cy < 0 or cy >= h or board[cy][cx] is not None:
                        corners += 1
                if corners < 3:
                    continue
                touched = {cy for cx, cy in cells}
                lines = 0
                for ry in touched:
                    if all(board[ry][cx] is not None or (cx, ry) in cells for cx in range(w)):
                        lines += 1
                if lines == 2:
                    tsd += 1
                elif lines >= 3:
                    tst += 1
    return tsd, tst


def clean_well_score(board):
    hs = heights(board)
    if not hs:
        return 0.0
    best = 0
    peak = max(hs)
    for x, value in enumerate(hs):
        left = hs[x - 1] if x > 0 else peak
        right = hs[x + 1] if x + 1 < len(hs) else peak
        best = max(best, min(left, right) - value)
    return float(max(0, best))


def feature_bonus(parent: Game, child: Game, action, evaluation: PlacementEvaluation, name: str, weight: float) -> float:
    if name == "baseline":
        return 0.0
    before = parent.board
    after = child.board
    if name in ("tsd_slot_delta", "tst_slot_delta", "spin_quality_delta"):
        b2, b3 = spin_slot_lines(before)
        a2, a3 = spin_slot_lines(after)
        if name == "tsd_slot_delta":
            value = a2 - b2
        elif name == "tst_slot_delta":
            value = a3 - b3
        else:
            value = (a2 - b2) + 1.7 * (a3 - b3)
        return weight * value
    if name in ("covered_holes_delta", "deep_holes_delta"):
        bc, bd = hole_stats(before)
        ac, ad = hole_stats(after)
        return weight * ((ac - bc) if name == "covered_holes_delta" else (ad - bd))
    hs = heights(after)
    maxh = max(hs, default=0)
    if name == "danger_hole_product":
        return weight * evaluation.features.board.holes * max(0, maxh - 8)
    if name == "top_band":
        cutoff = max(0, len(after) - 14)
        occ = sum(cell is not None for row in after[:cutoff] for cell in row)
        return weight * occ
    if name == "clean_well":
        return weight * clean_well_score(after)
    if name == "combo_continue":
        return weight * int(parent.combo >= 0 and evaluation.features.lines > 0)
    if name == "tsd_event":
        return weight * int(evaluation.features.spin == "t_spin_double")
    if name == "tst_event":
        return weight * int(evaluation.features.spin == "t_spin_triple")
    if name == "hold_t_for_slot":
        slots = sum(spin_slot_lines(before))
        return weight * int(slots > 0 and action.use_hold and child.hold_piece == "T")
    if name == "use_t_on_slot":
        slots = sum(spin_slot_lines(before))
        return weight * int(slots > 0 and action.placement.piece == "T")
    if name == "low_stack_slot":
        a2, a3 = spin_slot_lines(after)
        return weight * (a2 + a3) * (1.0 if maxh <= 10 else 0.0)
    return 0.0


def install_candidate(candidate: str):
    name, weight = CANDIDATES[candidate]
    if name == "baseline":
        search.rank_search_actions = ORIGINAL_RANK
        versus_search.rank_search_actions = ORIGINAL_RANK
        return

    def wrapped(game, weights=DEFAULT_WEIGHTS, config=DEFAULT_SEARCH_CONFIG, *, limit=None):
        raw = ORIGINAL_RANK(game, weights, config, limit=None)
        adjusted = []
        for action, evaluation in raw:
            child = clone_game(game)
            try:
                apply_search_action(child, action)
            except Exception:
                continue
            bonus = feature_bonus(game, child, action, evaluation, name, weight)
            adjusted.append((action, replace(evaluation, score=evaluation.score + bonus)))
        adjusted.sort(
            key=lambda item: (
                item[1].score,
                item[1].features.attack,
                item[1].features.spin_lines,
                -int(item[0].use_hold),
            ),
            reverse=True,
        )
        return tuple(adjusted if limit is None else adjusted[: max(0, int(limit))])

    search.rank_search_actions = wrapped
    versus_search.rank_search_actions = wrapped


def solo(candidate, games, pieces, seed_base, seed_step):
    install_candidate(candidate)
    total = {
        "pieces": 0,
        "attack": 0,
        "spins": 0,
        "spin_lines": 0,
        "tsd": 0,
        "tst": 0,
        "topouts": 0,
        "completed": 0,
        "max_b2b": 0,
    }
    for index in range(games):
        game = Game(seed_base + index * seed_step)
        spins = spin_lines = tsd = tst = max_b2b = 0
        while not game.game_over and game.pieces_placed < pieces:
            choice = search.choose_search_action(game, DEFAULT_WEIGHTS, DEFAULT_SEARCH_CONFIG)
            if choice is None:
                break
            result = apply_search_action(game, choice.action)
            if result.spin:
                spins += 1
                spin_lines += result.lines
                if result.spin == "t_spin_double":
                    tsd += 1
                elif result.spin == "t_spin_triple":
                    tst += 1
            max_b2b = max(max_b2b, game.b2b_chain)
        total["pieces"] += game.pieces_placed
        total["attack"] += game.attack
        total["spins"] += spins
        total["spin_lines"] += spin_lines
        total["tsd"] += tsd
        total["tst"] += tst
        total["topouts"] += int(game.game_over)
        total["completed"] += int(not game.game_over and game.pieces_placed >= pieces)
        total["max_b2b"] += max_b2b
    total["attack_per_piece"] = total["attack"] / max(1, total["pieces"])
    total["candidate"] = candidate
    total["games"] = games
    return total


def versus(candidate, games, turns, seed_base, seed_step):
    from minoflux_engine import VersusMatch
    from minoflux_ai.versus_search import choose_versus_action

    results = []
    for index in range(games):
        swapped = index % 2 == 1
        seed = seed_base + (index // 2) * seed_step
        match = VersusMatch(seed)
        turn = "player"
        count = 0
        candidate_max_b2b = baseline_max_b2b = 0
        while match.winner is None and count < turns:
            physical_is_candidate = (turn == "player") != swapped
            install_candidate(candidate if physical_is_candidate else "baseline")
            choice = choose_versus_action(match, turn)
            if choice is None:
                match.side(turn).game.game_over = True
                match._update_winner()
                break
            side = match.side(turn)
            result = apply_search_action(side.game, choice.action)
            match.resolve_lock(turn, result)
            if physical_is_candidate:
                candidate_max_b2b = max(candidate_max_b2b, side.game.b2b_chain)
            else:
                baseline_max_b2b = max(baseline_max_b2b, side.game.b2b_chain)
            count += 1
            turn = "ai" if turn == "player" else "player"
        candidate_side = match.ai if swapped else match.player
        baseline_side = match.player if swapped else match.ai
        winner = match.winner or "draw"
        if swapped:
            winner = {"player": "baseline", "ai": "candidate", "draw": "draw"}[winner]
        else:
            winner = {"player": "candidate", "ai": "baseline", "draw": "draw"}[winner]
        results.append({
            "winner": winner,
            "candidate_attack": candidate_side.game.attack,
            "baseline_attack": baseline_side.game.attack,
            "candidate_sent": candidate_side.sent,
            "baseline_sent": baseline_side.sent,
            "candidate_canceled": candidate_side.canceled,
            "baseline_canceled": baseline_side.canceled,
            "candidate_received": candidate_side.received,
            "baseline_received": baseline_side.received,
            "candidate_max_b2b": candidate_max_b2b,
            "baseline_max_b2b": baseline_max_b2b,
            "candidate_topout": int(candidate_side.game.game_over),
            "baseline_topout": int(baseline_side.game.game_over),
        })
    out = {
        "candidate": candidate,
        "games": games,
        "wins": sum(row["winner"] == "candidate" for row in results),
        "losses": sum(row["winner"] == "baseline" for row in results),
        "draws": sum(row["winner"] == "draw" for row in results),
    }
    for key in (
        "candidate_attack", "baseline_attack", "candidate_sent", "baseline_sent",
        "candidate_canceled", "baseline_canceled", "candidate_received", "baseline_received",
        "candidate_max_b2b", "baseline_max_b2b", "candidate_topout", "baseline_topout",
    ):
        out[key] = sum(row[key] for row in results) / games
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--mode", choices=("solo", "versus"), default="solo")
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--pieces", type=int, default=80)
    parser.add_argument("--turns", type=int, default=80)
    parser.add_argument("--seed-base", type=int, default=1701)
    parser.add_argument("--seed-step", type=int, default=97)
    args = parser.parse_args()
    output = (
        solo(args.candidate, args.games, args.pieces, args.seed_base, args.seed_step)
        if args.mode == "solo"
        else versus(args.candidate, args.games, args.turns, args.seed_base, args.seed_step)
    )
    print("TOURNAMENT_RESULT=" + json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
