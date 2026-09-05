from __future__ import annotations

import hashlib
import json
import time

from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.neural import NeuralValueConfig, NeuralValueEvaluator, build_neural_value_model
from minoflux_ai.neural_fast import _ORIGINAL_SCORE_PLACEMENT_GROUPS
from minoflux_ai.search import SearchConfig, apply_search_action, choose_search_actions_batch
from minoflux_engine import Game

CFG = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=4,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=8000,
).normalized()
NCFG = NeuralValueConfig().normalized()


class OldScorer:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def score_placement_groups(self, groups):
        return _ORIGINAL_SCORE_PLACEMENT_GROUPS(self.evaluator, groups)

    def score_placements(self, game, placements):
        return self.score_placement_groups(((game, placements),))[0]


def action_payload(choice):
    if choice is None:
        return None
    p = choice.action.placement
    return [
        choice.action.use_hold,
        p.piece,
        p.x,
        p.y,
        p.rotation,
        p.last_move_was_rotation,
        p.rotation_kick_index,
        p.rotation_from,
        p.rotation_to,
        choice.score,
    ]


def run(scorer, *, games_count=20, max_pieces=300):
    games = [Game(8100001 + i * 97) for i in range(games_count)]
    digest = hashlib.sha256()
    started = time.perf_counter()
    for step in range(max_pieces):
        active_indices = [
            i for i, game in enumerate(games)
            if not game.game_over and game.pieces_placed < max_pieces
        ]
        if not active_indices:
            break
        active_games = tuple(games[i] for i in active_indices)
        choices = choose_search_actions_batch(
            active_games,
            DEFAULT_WEIGHTS,
            CFG,
            scorer=scorer,
        )
        for index, choice in zip(active_indices, choices):
            payload = action_payload(choice)
            digest.update(json.dumps([step, index, payload], separators=(",", ":")).encode())
            digest.update(b"\n")
            if choice is not None:
                apply_search_action(games[index], choice.action)
    elapsed = time.perf_counter() - started
    return {
        "signature": digest.hexdigest(),
        "pieces": sum(g.pieces_placed for g in games),
        "attack": sum(g.attack for g in games),
        "topouts": sum(int(g.game_over) for g in games),
        "completed": sum(int(not g.game_over and g.pieces_placed >= max_pieces) for g in games),
        "elapsed": elapsed,
    }


def main():
    import torch

    torch.manual_seed(20260905)
    model = build_neural_value_model(NCFG)
    evaluator = NeuralValueEvaluator(model, NCFG, device="cpu", precision="float32")
    old = run(OldScorer(evaluator))
    new = run(evaluator)
    print("old", old)
    print("new", new)
    for key in ("signature", "pieces", "attack", "topouts", "completed"):
        assert old[key] == new[key], key
    print("GAME/ACTION STREAM: IDENTICAL")
    print("speedup", old["elapsed"] / new["elapsed"])


if __name__ == "__main__":
    main()
