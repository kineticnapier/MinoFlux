from __future__ import annotations

import json
import time

from minoflux_ai import DEFAULT_WEIGHTS, FESTIVAL_SAFE_SCORER, SearchConfig, apply_search_action, choose_search_action, extract_board_features
from minoflux_engine import Game


def run(seed: int, max_pieces: int, *, lookahead: int, beam: int, scorer) -> dict[str, object]:
    game = Game(seed)
    config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=lookahead,
        beam_width=beam,
        discount=0.9,
        srs_reachable=True,
        allow_180=False,
        reachability_node_limit=8000,
    ).normalized()
    quads = weak = peak_holes = peak_height = 0
    started = time.perf_counter()
    while not game.game_over and game.pieces_placed < max_pieces:
        choice = choose_search_action(game, DEFAULT_WEIGHTS, config, scorer=scorer)
        if choice is None:
            break
        result = apply_search_action(game, choice.action)
        quads += int(result.lines == 4)
        weak += int(0 < result.lines < 4)
        features = extract_board_features(game.board)
        peak_holes = max(peak_holes, features.holes)
        peak_height = max(peak_height, features.max_height)
    elapsed = time.perf_counter() - started
    return {
        "seed": seed,
        "pieces": game.pieces_placed,
        "topout": game.game_over,
        "attack": game.attack,
        "quads": quads,
        "weak": weak,
        "peakHoles": peak_holes,
        "peakHeight": peak_height,
        "elapsed": elapsed,
        "pps": game.pieces_placed / max(elapsed, 1e-9),
    }


def main() -> None:
    variants = (
        ("festival-l0-b1", 0, 1, FESTIVAL_SAFE_SCORER),
        ("festival-l1-b2", 1, 2, FESTIVAL_SAFE_SCORER),
        ("festival-l1-b4", 1, 4, FESTIVAL_SAFE_SCORER),
        ("default-l0", 0, 1, None),
    )
    for name, lookahead, beam, scorer in variants:
        result = run(9_100_001, 1000, lookahead=lookahead, beam=beam, scorer=scorer)
        print(name + "=" + json.dumps(result, sort_keys=True), flush=True)

    # A small seed spread for the best-looking lookahead setting.
    games = [
        run(9_400_001 + 97 * i, 500, lookahead=1, beam=4, scorer=FESTIVAL_SAFE_SCORER)
        for i in range(6)
    ]
    print("festival-l1-b4-spread=" + json.dumps({
        "pieces": sum(int(g["pieces"]) for g in games),
        "topouts": sum(bool(g["topout"]) for g in games),
        "quads": sum(int(g["quads"]) for g in games),
        "peakHoles": max(int(g["peakHoles"]) for g in games),
        "peakHeight": max(int(g["peakHeight"]) for g in games),
        "games": games,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
