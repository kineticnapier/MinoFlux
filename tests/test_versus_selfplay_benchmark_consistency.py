from __future__ import annotations

from unittest.mock import patch

from minoflux_ai.features import extract_board_features
from minoflux_ai.search import SearchConfig
from minoflux_ai.versus_benchmark import run_versus_game
from minoflux_ai.versus_neural import (
    VersusSelfPlayConfig,
    _generate_versus_selfplay_dataset_impl,
)
from minoflux_ai.versus_search import VersusSearchConfig
from minoflux_engine import VersusMatch


class _PlacementBatchScorer:
    def score_placements(self, game, placements):
        return tuple(
            float(placement.x)
            + 0.01 * float(placement.rotation)
            + 0.0001 * float(placement.y)
            for placement in placements
        )

    def score_placement_groups(self, groups):
        return tuple(
            self.score_placements(game, placements)
            for game, placements in groups
        )

    def score_many(self, game, evaluations):
        return self.score_placements(
            game,
            tuple(evaluation.placement for evaluation in evaluations),
        )


class _TrackingVersusMatch(VersusMatch):
    instances: list["_TrackingVersusMatch"] = []

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hole_sequence: list[int] = []
        self.player_max_b2b_seen = 0
        self.ai_max_b2b_seen = 0
        self.player_max_surge_seen = 0
        self.ai_max_surge_seen = 0
        type(self).instances.append(self)

    def _next_hole(self, previous=None) -> int:
        hole = super()._next_hole(previous)
        self.hole_sequence.append(hole)
        return hole

    def resolve_lock(self, name, result):
        resolution = super().resolve_lock(name, result)
        self.player_max_b2b_seen = max(self.player_max_b2b_seen, self.player.game.b2b_chain)
        self.ai_max_b2b_seen = max(self.ai_max_b2b_seen, self.ai.game.b2b_chain)
        self.player_max_surge_seen = max(
            self.player_max_surge_seen, self.player.game.surge_charge
        )
        self.ai_max_surge_seen = max(self.ai_max_surge_seen, self.ai.game.surge_charge)
        return resolution


def _search_config() -> VersusSearchConfig:
    return VersusSearchConfig(
        placement_search=SearchConfig(
            allow_hold=True,
            lookahead_pieces=0,
            beam_width=2,
            discount=0.9,
            srs_reachable=True,
            allow_180=False,
            reachability_node_limit=8000,
        ),
        candidate_width=3,
        opponent_reply_width=1,
    ).normalized()


def _final_state(match: _TrackingVersusMatch) -> dict[str, object]:
    return {
        "winner": match.winner or "draw",
        "player_pieces": match.player.game.pieces_placed,
        "ai_pieces": match.ai.game.pieces_placed,
        "player_attack": match.player.game.attack,
        "ai_attack": match.ai.game.attack,
        "player_sent": match.player.sent,
        "ai_sent": match.ai.sent,
        "player_canceled": match.player.canceled,
        "ai_canceled": match.ai.canceled,
        "player_received": match.player.received,
        "ai_received": match.ai.received,
        "player_garbage_applied": match.player.garbage_applied,
        "ai_garbage_applied": match.ai.garbage_applied,
        "player_pending": match.player.pending.pending_lines,
        "ai_pending": match.ai.pending.pending_lines,
        "player_final_height": extract_board_features(match.player.game.board).max_height,
        "ai_final_height": extract_board_features(match.ai.game.board).max_height,
        "player_final_holes": extract_board_features(match.player.game.board).holes,
        "ai_final_holes": extract_board_features(match.ai.game.board).holes,
        "player_max_b2b": match.player_max_b2b_seen,
        "ai_max_b2b": match.ai_max_b2b_seen,
        "player_max_surge": match.player_max_surge_seen,
        "ai_max_surge": match.ai_max_surge_seen,
    }


def _engine_snapshot(match: _TrackingVersusMatch) -> dict[str, object]:
    def game_snapshot(game):
        return {
            "board": tuple(tuple(row) for row in game.board),
            "current": game.current,
            "hold": game.hold_piece,
            "queue": tuple(game.queue),
            "bagQueue": tuple(game._bag._queue),
            "bagRng": game._bag._rng.getstate(),
        }

    return {
        "player": game_snapshot(match.player.game),
        "ai": game_snapshot(match.ai.game),
        "playerPending": tuple(
            (packet.lines, packet.hole) for packet in match.player.pending.packets
        ),
        "aiPending": tuple((packet.lines, packet.hole) for packet in match.ai.pending.packets),
        "garbageHoles": tuple(match.hole_sequence),
        "garbageRng": match._garbage_rng.getstate(),
    }


def test_unswapped_same_seed_selfplay_matches_benchmark_exactly(tmp_path) -> None:
    seed = 424242
    max_turns = 18
    config = _search_config()
    scorer = _PlacementBatchScorer()

    _TrackingVersusMatch.instances.clear()
    with patch("minoflux_ai.versus_benchmark.VersusMatch", _TrackingVersusMatch):
        benchmark = run_versus_game(
            seed,
            max_turns=max_turns,
            player_config=config,
            ai_config=config,
            garbage_cap=8,
            player_starts=True,
            player_scorer=scorer,
            ai_scorer=scorer,
        )
    benchmark_match = _TrackingVersusMatch.instances[-1]

    _TrackingVersusMatch.instances.clear()
    with patch("minoflux_ai.versus_neural.VersusMatch", _TrackingVersusMatch):
        selfplay_summary = _generate_versus_selfplay_dataset_impl(
            tmp_path / "selfplay.jsonl",
            scorer,
            VersusSelfPlayConfig(
                games=1,
                max_turns=max_turns,
                seed_base=seed,
                seed_step=31,
                garbage_cap=8,
                search_config=config,
                game_batch=1,
                max_records_per_game=0,
            ),
            ai_scorer=scorer,
            progress=False,
        )
    selfplay_match = _TrackingVersusMatch.instances[-1]

    expected = {
        "winner": benchmark.winner,
        "player_pieces": benchmark.player_pieces,
        "ai_pieces": benchmark.ai_pieces,
        "player_attack": benchmark.player_attack,
        "ai_attack": benchmark.ai_attack,
        "player_sent": benchmark.player_sent,
        "ai_sent": benchmark.ai_sent,
        "player_canceled": benchmark.player_canceled,
        "ai_canceled": benchmark.ai_canceled,
        "player_received": benchmark.player_received,
        "ai_received": benchmark.ai_received,
        "player_garbage_applied": benchmark.player_garbage_applied,
        "ai_garbage_applied": benchmark.ai_garbage_applied,
        "player_pending": benchmark.player_pending,
        "ai_pending": benchmark.ai_pending,
        "player_final_height": benchmark.player_final_height,
        "ai_final_height": benchmark.ai_final_height,
        "player_final_holes": benchmark.player_final_holes,
        "ai_final_holes": benchmark.ai_final_holes,
        "player_max_b2b": benchmark.player_max_b2b,
        "ai_max_b2b": benchmark.ai_max_b2b,
        "player_max_surge": benchmark.player_max_surge,
        "ai_max_surge": benchmark.ai_max_surge,
    }
    assert _final_state(selfplay_match) == expected
    assert int(selfplay_summary["meanTurns"]) == benchmark.turns
    assert selfplay_summary[f"{benchmark.winner}Wins"] == 1

    # The actual engine states also match, including future bag RNG and garbage RNG.
    assert _engine_snapshot(selfplay_match) == _engine_snapshot(benchmark_match)


def test_selfplay_seed_metadata_is_one_seed_per_game(tmp_path) -> None:
    scorer = _PlacementBatchScorer()
    summary = _generate_versus_selfplay_dataset_impl(
        tmp_path / "seeds.jsonl",
        scorer,
        VersusSelfPlayConfig(
            games=3,
            max_turns=1,
            seed_base=5000,
            seed_step=17,
            garbage_cap=8,
            search_config=VersusSearchConfig(
                placement_search=SearchConfig(
                    allow_hold=False,
                    lookahead_pieces=0,
                    beam_width=1,
                    srs_reachable=False,
                ),
                candidate_width=1,
                opponent_reply_width=0,
            ),
            game_batch=1,
        ),
        ai_scorer=scorer,
        progress=False,
    )

    assert summary["games"] == 3
    assert summary["seedCount"] == 3
    assert summary["seedBase"] == 5000
    assert summary["seedStep"] == 17
    assert summary["seedStepUnit"] == "game"
    assert summary["mirroredSides"] is False
    assert summary["mirroredGameCount"] == 0
    assert summary["startingSideAlternates"] is True
