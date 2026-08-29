from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from statistics import mean, pvariance

from minoflux_ai.features import extract_board_features
from minoflux_ai.heuristic import DEFAULT_WEIGHTS, PlacementEvaluation
from minoflux_ai.search import SearchConfig, apply_search_action, choose_search_action, clone_game, rank_search_actions
from minoflux_ai.versus_search import VersusSearchConfig, choose_versus_action, clone_versus_match, score_versus_state
from minoflux_engine import Game, VersusMatch

CANDIDATES = (
    "pressure_survival_floor",
    "pressure_height_floor",
    "pressure_hole_floor",
    "pressure_depth_floor",
    "pressure_shape_consistency",
    "pressure_line_block",
    "pressure_attack_cancel",
    "pressure_clean_counter",
    "pressure_spin_counter",
    "pressure_b2b_counter",
    "pressure_slot_preserve",
    "pressure_t_supply_reserve",
    "pressure_center_resilience",
    "pressure_edge_resilience",
)

SEARCH = SearchConfig(allow_hold=True, lookahead_pieces=0, beam_width=6, discount=0.90,
                      srs_reachable=True, allow_180=False, reachability_node_limit=8_000)
VERSUS_CFG = VersusSearchConfig(placement_search=SEARCH, candidate_width=6, opponent_reply_width=1)
STRESS_LINES = 4
STRESS_HOLES = (0, 4, 9)


def _t_supply(game: Game) -> int:
    value = int(game.current == "T") + int(game.hold_piece == "T")
    for piece in list(game.queue)[:6]:
        value += int(piece == "T")
    return value


class PressureScorer:
    def __init__(self, name: str) -> None:
        self.name = name

    def score_many(self, game: Game, evaluations: list[PlacementEvaluation] | tuple[PlacementEvaluation, ...]):
        scores: list[float] = []
        for ev in evaluations:
            if ev.features.game_over:
                scores.append(ev.score)
                continue
            child = clone_game(game)
            child.place(ev.placement)
            stressed = []
            for hole in STRESS_HOLES:
                probe = clone_game(child)
                probe.add_garbage(STRESS_LINES, hole)
                stressed.append((probe.game_over, extract_board_features(probe.board)))
            alive = sum(not dead for dead, _ in stressed)
            heights = [f.max_height for _, f in stressed]
            holes = [f.holes for _, f in stressed]
            depths = [f.hole_depth for _, f in stressed]
            slots = [f.t_spin_slots for _, f in stressed]
            worst_height = max(heights)
            worst_holes = max(holes)
            worst_depth = max(depths)
            mean_height = mean(heights)
            mean_holes = mean(holes)
            mean_depth = mean(depths)
            bonus = 0.0
            if self.name == "pressure_survival_floor":
                bonus = 0.75 * alive - 0.10 * worst_height - 0.10 * sum(dead for dead, _ in stressed)
            elif self.name == "pressure_height_floor":
                danger = max(0.0, (worst_height - 12) / 5.0)
                bonus = -0.34 * danger - 0.035 * mean_height
            elif self.name == "pressure_hole_floor":
                bonus = -0.24 * worst_holes - 0.08 * mean_holes
            elif self.name == "pressure_depth_floor":
                bonus = -0.025 * worst_depth - 0.010 * mean_depth
            elif self.name == "pressure_shape_consistency":
                bonus = -0.055 * pvariance(heights) - 0.035 * pvariance(holes) + 0.10 * alive
            elif self.name == "pressure_line_block":
                # A line clear blocks queued garbage for this placement in VersusMatch.
                bonus = 0.30 * ev.features.lines + 0.10 * alive - 0.025 * mean_height
            elif self.name == "pressure_attack_cancel":
                bonus = 0.26 * min(STRESS_LINES, ev.features.attack) + 0.06 * alive - 0.020 * mean_height
            elif self.name == "pressure_clean_counter":
                clean = 1.0 / (1.0 + mean_holes + mean_depth / 8.0)
                bonus = 0.24 * ev.features.attack + 0.70 * clean + 0.06 * alive
            elif self.name == "pressure_spin_counter":
                bonus = 0.34 * ev.features.spin_lines + 0.12 * ev.features.attack + 0.05 * alive - 0.02 * mean_holes
            elif self.name == "pressure_b2b_counter":
                difficult = int(ev.features.spin_lines > 0 or ev.features.lines == 4)
                chain = 1.0 if child.back_to_back or child.b2b_chain > 0 else 0.35
                bonus = 0.45 * difficult * chain + 0.08 * ev.features.attack - 0.02 * mean_height
            elif self.name == "pressure_slot_preserve":
                bonus = 0.16 * min(slots) + 0.08 * mean(slots) - 0.025 * mean_holes
            elif self.name == "pressure_t_supply_reserve":
                supply = _t_supply(child)
                matched = min(min(slots), supply)
                excess = max(0, min(slots) - supply)
                bonus = 0.22 * matched - 0.18 * excess - 0.02 * mean_holes
            elif self.name == "pressure_center_resilience":
                center = stressed[1][1]
                bonus = -0.075 * center.holes - 0.018 * center.hole_depth - 0.025 * center.max_height
            elif self.name == "pressure_edge_resilience":
                edge_holes = mean((stressed[0][1].holes, stressed[2][1].holes))
                edge_depth = mean((stressed[0][1].hole_depth, stressed[2][1].hole_depth))
                edge_height = mean((stressed[0][1].max_height, stressed[2][1].max_height))
                bonus = -0.075 * edge_holes - 0.018 * edge_depth - 0.025 * edge_height
            scores.append(ev.score + bonus)
        return tuple(scores)


def _play(name: str, seed: int, max_pieces: int) -> dict[str, object]:
    game = Game(seed)
    scorer = None if name == "baseline" else PressureScorer(name)
    spins = spin_lines = tsd = tst = 0
    max_b2b = 0
    while not game.game_over and game.pieces_placed < max_pieces:
        choice = choose_search_action(game, DEFAULT_WEIGHTS, SEARCH, scorer=scorer)
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        if result.spin is not None:
            spins += 1
            spin_lines += result.lines
            tsd += int(result.lines == 2)
            tst += int(result.lines == 3)
        max_b2b = max(max_b2b, game.b2b_chain)
    return {"seed": seed, "pieces": game.pieces_placed, "attack": game.attack,
            "topout": game.game_over, "completed": not game.game_over and game.pieces_placed >= max_pieces,
            "spins": spins, "spinLines": spin_lines, "tsd": tsd, "tst": tst, "maxB2B": max_b2b}


def _benchmark(name: str, seeds: list[int], pieces: int) -> dict[str, object]:
    games = [_play(name, seed, pieces) for seed in seeds]
    total_pieces = sum(int(g["pieces"]) for g in games)
    attack = sum(int(g["attack"]) for g in games)
    return {"name": name, "pieces": total_pieces, "attack": attack, "app": attack / max(1, total_pieces),
            "topouts": sum(bool(g["topout"]) for g in games), "completed": sum(bool(g["completed"]) for g in games),
            "spins": sum(int(g["spins"]) for g in games), "spinLines": sum(int(g["spinLines"]) for g in games),
            "tsd": sum(int(g["tsd"]) for g in games), "tst": sum(int(g["tst"]) for g in games),
            "maxB2B": max((int(g["maxB2B"]) for g in games), default=0), "perGame": games}


def _rank(r: dict[str, object]):
    return (int(r["completed"]), -int(r["topouts"]), float(r["app"]), int(r["spinLines"]), int(r["tsd"]), int(r["maxB2B"]))


def _run_task(args):
    return _benchmark(*args)


def _candidate_action(match: VersusMatch, side_name: str, name: str):
    own = match.side(side_name)
    ranked = rank_search_actions(own.game, DEFAULT_WEIGHTS, SEARCH, limit=6, scorer=PressureScorer(name))
    if not ranked:
        return None
    opponent = "ai" if side_name == "player" else "player"
    best = None
    for action, ev in ranked:
        after = clone_versus_match(match)
        lock = apply_search_action(after.side(side_name).game, action)
        resolution = after.resolve_lock(side_name, lock)
        score = score_versus_state(after, side_name, resolution=resolution, solo_score=ev.score,
                                   path_length=len(action.placement.path), action_side=side_name)
        if after.winner is None:
            replies = rank_search_actions(after.side(opponent).game, DEFAULT_WEIGHTS, SEARCH, limit=1)
            if replies:
                reply, rev = replies[0]
                replied = clone_versus_match(after)
                rr = apply_search_action(replied.side(opponent).game, reply)
                rres = replied.resolve_lock(opponent, rr)
                score = score_versus_state(replied, side_name, resolution=rres, solo_score=-rev.score,
                                           path_length=len(reply.placement.path), action_side=opponent)
        key = (score, resolution.sent_lines, resolution.canceled_lines, -len(action.placement.path))
        if best is None or key > best[0]:
            best = (key, action)
    return None if best is None else best[1]


def _versus_game(name: str, seed: int, candidate_side: str, turns_limit: int = 120):
    match = VersusMatch(seed, garbage_cap=8)
    turn = "player"
    max_b2b = {"player": 0, "ai": 0}
    turns = 0
    while match.winner is None and turns < turns_limit:
        if turn == candidate_side:
            action = _candidate_action(match, turn, name)
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
    other = "ai" if candidate_side == "player" else "player"
    cand, base = match.side(candidate_side), match.side(other)
    winner = "draw" if match.winner is None else ("candidate" if match.winner == candidate_side else "baseline")
    return {"winner": winner, "candidateAttack": cand.game.attack, "baselineAttack": base.game.attack,
            "candidateSent": cand.sent, "baselineSent": base.sent, "candidateCanceled": cand.canceled,
            "baselineCanceled": base.canceled, "candidateReceived": cand.received, "baselineReceived": base.received,
            "candidateMaxB2B": max_b2b[candidate_side], "baselineMaxB2B": max_b2b[other],
            "candidateTopout": cand.game.game_over, "baselineTopout": base.game.game_over}


def _versus(name: str, games: int = 6):
    rows = [_versus_game(name, 1_030_003 + (i // 2) * 97, "player" if i % 2 == 0 else "ai") for i in range(games)]
    return {"name": name, "games": games, "wins": sum(r["winner"] == "candidate" for r in rows),
            "losses": sum(r["winner"] == "baseline" for r in rows), "draws": sum(r["winner"] == "draw" for r in rows),
            "candidateMeanAttack": mean(r["candidateAttack"] for r in rows), "baselineMeanAttack": mean(r["baselineAttack"] for r in rows),
            "candidateMeanSent": mean(r["candidateSent"] for r in rows), "baselineMeanSent": mean(r["baselineSent"] for r in rows),
            "candidateMeanCanceled": mean(r["candidateCanceled"] for r in rows), "baselineMeanCanceled": mean(r["baselineCanceled"] for r in rows),
            "candidateMeanReceived": mean(r["candidateReceived"] for r in rows), "baselineMeanReceived": mean(r["baselineReceived"] for r in rows),
            "candidateMaxB2B": max(r["candidateMaxB2B"] for r in rows), "baselineMaxB2B": max(r["baselineMaxB2B"] for r in rows),
            "candidateTopouts": sum(bool(r["candidateTopout"]) for r in rows), "baselineTopouts": sum(bool(r["baselineTopout"]) for r in rows),
            "perGame": rows}


def _parallel(names: list[str], seeds: list[int], pieces: int):
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_task, (name, seeds, pieces)): name for name in names}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda r: names.index(str(r["name"])))
    return rows


def main() -> None:
    names = ["baseline", *CANDIDATES]
    short = _parallel(names, [1_010_011, 1_010_108], 70)
    finalists = sorted(short[1:], key=_rank, reverse=True)[:3]
    fresh_names = ["baseline", *(str(r["name"]) for r in finalists)]
    fresh = _parallel(fresh_names, [1_020_037, 1_020_134, 1_020_231], 160)
    baseline = fresh[0]
    eligible = [r for r in fresh[1:] if int(r["topouts"]) <= int(baseline["topouts"])
                and int(r["completed"]) >= int(baseline["completed"])
                and float(r["app"]) >= float(baseline["app"]) * 1.01]
    versus_names = [str(r["name"]) for r in sorted(eligible, key=_rank, reverse=True)[:2]]
    versus = [_versus(name, 6) for name in versus_names]
    report = {"candidateCount": len(CANDIDATES), "short": short,
              "shortFinalists": [str(r["name"]) for r in finalists], "fresh": fresh, "versus": versus}
    Path("pressure-tournament-result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"shortFinalists": report["shortFinalists"], "versusNames": versus_names}))


if __name__ == "__main__":
    main()
