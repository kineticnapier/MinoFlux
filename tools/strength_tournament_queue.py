from __future__ import annotations

import argparse
import json
from dataclasses import replace

from minoflux_engine import VersusMatch
import minoflux_ai.heuristic as heuristic
import minoflux_ai.search as search
from minoflux_ai.benchmark import run_heuristic_benchmark
from minoflux_ai.search import apply_search_action
from minoflux_ai.versus_search import choose_versus_action

CANDIDATES = {
    "baseline": (0.0, "baseline"),
    "supply_balance": (0.45, "supply_balance"),
    "supply_low_clean_balance": (0.65, "supply_low_clean_balance"),
    "supply_density_balance": (0.55, "supply_density_balance"),
    "supply_height_balance": (0.55, "supply_height_balance"),
    "supply_surplus_tax": (-0.55, "supply_surplus_tax"),
    "supply_deficit_build": (0.65, "supply_deficit_build"),
    "t_ready_quality": (0.70, "t_ready_quality"),
    "t_slot_conversion": (0.80, "t_slot_conversion"),
    "t_slot_waste_tax": (-0.75, "t_slot_waste_tax"),
    "non_t_slot_damage": (-0.55, "non_t_slot_damage"),
    "b2b_spin_safety": (0.60, "b2b_spin_safety"),
    "b2b_attack_efficiency": (0.45, "b2b_attack_efficiency"),
    "surge_attack_timing": (0.35, "surge_attack_timing"),
    "hold_t_release": (0.55, "hold_t_release"),
}

_ORIGINAL_RANK = heuristic.rank_placements


def _t_supply(game) -> int:
    pieces = [game.current, *list(game.queue)[:7]]
    return sum(piece == "T" for piece in pieces)


def _low_clean(board) -> float:
    return board.t_spin_slots / (1.0 + board.holes + board.max_height / 6.0)


def _height_quality(board) -> float:
    return board.t_spin_slots / (1.0 + board.max_height / 6.0)


def _bonus(name: str, weight: float, game, ev) -> float:
    f, b = ev.features, ev.features.board
    slots = b.t_spin_slots
    supply = _t_supply(game)
    if name == "baseline":
        return 0.0
    if name == "supply_balance":
        return -weight * abs(slots - supply)
    if name == "supply_low_clean_balance":
        return -weight * abs(_low_clean(b) - supply)
    if name == "supply_density_balance":
        return -weight * abs(b.t_spin_slot_density - supply)
    if name == "supply_height_balance":
        return -weight * abs(_height_quality(b) - supply)
    if name == "supply_surplus_tax":
        return weight * max(0, slots - supply)
    if name == "supply_deficit_build":
        return weight * max(0, supply - slots) * max(0, f.t_spin_slot_delta)
    if name == "t_ready_quality":
        return weight * _low_clean(b) if game.current == "T" else 0.0
    if name == "t_slot_conversion":
        consumed = max(0, -f.t_spin_slot_delta)
        return weight * consumed * (1 + f.attack) if game.current == "T" and f.spin_lines else 0.0
    if name == "t_slot_waste_tax":
        consumed = max(0, -f.t_spin_slot_delta)
        return weight * consumed if game.current == "T" and not f.spin_lines else 0.0
    if name == "non_t_slot_damage":
        return weight * max(0, -f.t_spin_slot_delta) * (1.0 + _low_clean(b)) if game.current != "T" else 0.0
    if name == "b2b_spin_safety":
        return weight * f.spin_lines / (1.0 + b.holes + b.max_height / 8.0) if game.back_to_back else 0.0
    if name == "b2b_attack_efficiency":
        chain = min(6, max(0, game.b2b_chain))
        return weight * f.attack * (1.0 + chain / 6.0) / (1.0 + b.holes + b.max_height / 10.0)
    if name == "surge_attack_timing":
        charge = min(20, max(0, game.surge_charge)) / 20.0
        return weight * charge * f.attack / (1.0 + b.holes + b.max_height / 10.0)
    if name == "hold_t_release":
        return weight * (1.0 + f.attack) * _low_clean(b) if game.current == "T" and game.hold_piece is not None else 0.0
    raise KeyError(name)


def install_candidate(candidate: str) -> None:
    heuristic.rank_placements = _ORIGINAL_RANK
    search.rank_placements = _ORIGINAL_RANK
    weight, name = CANDIDATES[candidate]
    if candidate == "baseline":
        return

    def rank(game, weights=heuristic.DEFAULT_WEIGHTS, *, placements=None, limit=None):
        ranked = list(_ORIGINAL_RANK(game, weights, placements=placements, limit=None))
        adjusted = [replace(ev, score=ev.score + _bonus(name, weight, game, ev)) for ev in ranked]
        adjusted.sort(key=heuristic._placement_key, reverse=True)
        return tuple(adjusted if limit is None else adjusted[: max(0, int(limit))])

    heuristic.rank_placements = rank
    search.rank_placements = rank


def summarize(result):
    return {
        "attack": result.attack,
        "pieces": result.pieces,
        "attack_per_piece": result.attack / result.pieces if result.pieces else 0.0,
        "topouts": result.topouts,
        "completed": result.completed,
        "spins": result.spins,
        "spin_lines": result.spin_lines,
        "tsd": result.t_spin_doubles,
        "tst": result.t_spin_triples,
    }


def run_mirrored(candidate: str):
    rows = []
    for leg in range(6):
        swapped = leg % 2 == 1
        seed = 140003 + (leg // 2) * 211
        match = VersusMatch(seed, garbage_cap=8)
        turn = "player"
        max_b2b = {"player": 0, "ai": 0}
        turns = 0
        candidate_side = "ai" if swapped else "player"
        while match.winner is None and turns < 160:
            install_candidate(candidate if turn == candidate_side else "baseline")
            choice = choose_versus_action(match, turn)
            if choice is None:
                match.side(turn).game.game_over = True
                match._update_winner()
                break
            result = apply_search_action(match.side(turn).game, choice.action)
            match.resolve_lock(turn, result)
            turns += 1
            max_b2b["player"] = max(max_b2b["player"], match.player.game.b2b_chain)
            max_b2b["ai"] = max(max_b2b["ai"], match.ai.game.b2b_chain)
            turn = "ai" if turn == "player" else "player"
        other = "ai" if candidate_side == "player" else "player"
        c = match.side(candidate_side)
        b = match.side(other)
        winner = "candidate" if match.winner == candidate_side else ("baseline" if match.winner in ("player", "ai") else "draw")
        rows.append({
            "winner": winner,
            "candidate_attack": c.game.attack,
            "baseline_attack": b.game.attack,
            "candidate_sent": c.sent,
            "baseline_sent": b.sent,
            "candidate_canceled": c.canceled,
            "baseline_canceled": b.canceled,
            "candidate_received": c.received,
            "baseline_received": b.received,
            "candidate_b2b": max_b2b[candidate_side],
            "baseline_b2b": max_b2b[other],
            "candidate_topout": int(c.game.game_over),
            "baseline_topout": int(b.game.game_over),
        })
    n = len(rows)
    keys = (
        "candidate_attack", "baseline_attack", "candidate_sent", "baseline_sent",
        "candidate_canceled", "baseline_canceled", "candidate_received", "baseline_received",
        "candidate_b2b", "baseline_b2b",
    )
    out = {
        "candidate": candidate,
        "phase": "versus",
        "wins": sum(r["winner"] == "candidate" for r in rows),
        "losses": sum(r["winner"] == "baseline" for r in rows),
        "draws": sum(r["winner"] == "draw" for r in rows),
        "candidate_topouts": sum(r["candidate_topout"] for r in rows),
        "baseline_topouts": sum(r["baseline_topout"] for r in rows),
    }
    out.update({k: sum(r[k] for r in rows) / n for k in keys})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--phase", choices=("screen", "fresh", "versus"), default="screen")
    args = parser.parse_args()
    if args.phase == "versus":
        print("TOURNAMENT_RESULT=" + json.dumps(run_mirrored(args.candidate), sort_keys=True))
        return
    install_candidate(args.candidate)
    if args.phase == "screen":
        result = run_heuristic_benchmark(games=2, max_pieces=160, seed_base=53003, seed_step=101, workers=1)
    else:
        result = run_heuristic_benchmark(games=3, max_pieces=250, seed_base=970301, seed_step=137, workers=1)
    print("TOURNAMENT_RESULT=" + json.dumps({"candidate": args.candidate, "phase": args.phase, **summarize(result)}, sort_keys=True))


if __name__ == "__main__":
    main()
