from __future__ import annotations

from collections import deque
from dataclasses import fields, replace
import json
import random
from unittest.mock import patch

import pytest

import _placement_teacher_reference as reference
from minoflux_ai import placement_teacher as teacher
from minoflux_ai.bitboard import board_row_masks
from minoflux_ai.features import extract_board_features
from minoflux_ai.search import SearchAction, apply_search_action, clone_game
from minoflux_engine import Game, Placement


def _signature(game):
    return (
        game.snapshot(queue_size=len(game.queue)),
        tuple(game.queue), tuple(game._bag._queue), game._bag._rng.getstate(),
        game.last_action, game.last_lock, game.last_move_was_rotation,
        game.last_rotation_kick_index, game.last_rotation_from, game.last_rotation_to,
        game.lock_elapsed_ms, game.lock_resets,
    )


def _varied_game(seed):
    game = Game(seed)
    rng = random.Random(seed)
    for y in range(game.height - 6, game.height):
        for x in range(game.width):
            game.board[y][x] = "G" if rng.random() < 0.30 else None
    game.hold_piece = (None, "T", "I")[seed % 3]
    game.combo = seed % 6 - 1
    game.back_to_back = bool(seed % 2)
    game.b2b_chain = seed % 12 if game.back_to_back else 0
    return game


@pytest.mark.parametrize("seed", range(810, 818))
def test_depth_two_all_scores_order_and_source_equal_original(seed):
    game = _varied_game(seed)
    config = teacher.PlacementTeacherConfig(
        depth=2, beam_width=24, allow_hold=bool(seed % 3), allow_180=bool(seed % 2),
    )
    before = _signature(game)
    expected = reference.rank_placement_teacher_actions(game, config=config)
    actual = teacher.rank_placement_teacher_actions(game, config=config)
    # Dataclass equality includes every path, rotation metadata, immediate,
    # future, total, breakdown, and the order of every legal root action.
    assert actual == expected
    assert actual
    assert _signature(game) == before
    assert teacher.rank_placement_teacher_actions(game, config=config) == actual


def test_depth_two_beam_is_irrelevant_and_public_config_is_normalized():
    game = Game(821)
    cfg = teacher.PlacementTeacherConfig(depth=2, beam_width=1)
    expected = reference.rank_placement_teacher_actions(game, config=cfg)
    for width in (0, 1, 24, 128, 999):
        assert teacher.rank_placement_teacher_actions(
            game, config=replace(cfg, beam_width=width)
        ) == expected


@pytest.mark.parametrize("beam", [1, 3])
def test_depth_three_preserves_original_beam_and_path_ties(beam):
    game = Game(823)
    game.current = "O"
    cfg = teacher.PlacementTeacherConfig(depth=3, beam_width=beam, allow_hold=False)
    # Attack-only creates many immediate ties; original path order must still
    # select the same intermediate beam before looking at the last ply.
    weights = teacher.PlacementTeacherWeights(**{
        field.name: (1.0 if field.name == "attack" else 0.0)
        for field in fields(teacher.PlacementTeacherWeights)
    })
    assert teacher.rank_placement_teacher_actions(game, weights, cfg) == (
        reference.rank_placement_teacher_actions(game, weights, cfg)
    )


def _assert_leaf(game, action, config=None, weights=None):
    cfg = (config or teacher.PlacementTeacherConfig()).normalized()
    weights = weights or teacher.PlacementTeacherWeights()
    before = _signature(game)
    child = clone_game(game)
    result = apply_search_action(child, action)
    expected = reference.placement_teacher_transition_score(game, child, result, weights, cfg)
    actual = teacher._leaf_action_score(
        game, action, weights, cfg,
        source_rows=board_row_masks(game.board), before_features=extract_board_features(game.board),
    )
    assert actual == expected.total
    assert _signature(game) == before
    return expected


@pytest.mark.parametrize("seed", range(830, 850))
def test_leaf_all_legal_actions_on_seeded_boards_match_engine(seed):
    game = _varied_game(seed)
    cfg = teacher.PlacementTeacherConfig(allow_180=bool(seed % 2))
    actions = reference._legal_actions(game, cfg)
    assert actions
    for action in actions:
        _assert_leaf(game, action, cfg)


@pytest.mark.parametrize("mini,kick,spin", [
    (False, 0, "T_SPIN_SINGLE"),
    (True, 0, "T_SPIN_MINI_SINGLE"),
    (True, 4, "T_SPIN_SINGLE"),
])
def test_leaf_t_spin_and_fifth_kick_metadata(mini, kick, spin):
    game = Game(851)
    game.current = "T"
    game.board[22] = ["G"] * 10
    for x in (3, 4, 5):
        game.board[22][x] = None
    for x, y in ([(3, 21), (3, 23), (5, 23)] if mini else [(3, 21), (5, 21), (3, 23)]):
        game.board[y][x] = "G"
    game.back_to_back, game.b2b_chain, game.combo = True, 5, 3
    placement = Placement(
        "T", 3, 21, 0, game.cells("T", 3, 21, 0),
        last_move_was_rotation=True, rotation_kick_index=kick,
        rotation_from=3, rotation_to=0,
    )
    child = clone_game(game)
    assert child.place(placement).spin == spin
    breakdown = _assert_leaf(game, SearchAction(False, placement))
    assert breakdown.spin_lines == breakdown.lines == 1
    assert breakdown.b2b_chain_growth == 1
    assert breakdown.combo == 4


@pytest.mark.parametrize("lines,perfect", [(1, False), (4, False), (4, True)])
def test_leaf_line_clear_b2b_break_growth_surge_combo_and_perfect_clear(lines, perfect):
    game = Game(853)
    game.current = "I"
    for y in range(24 - lines, 24):
        game.board[y] = ["G"] * 9 + [None]
    if not perfect:
        game.board[18][0] = "G"
    game.back_to_back, game.b2b_chain, game.combo = True, 10, 4
    p = Placement("I", 7, 20, 1, game.cells("I", 7, 20, 1))
    breakdown = _assert_leaf(game, SearchAction(False, p))
    assert breakdown.lines == lines
    assert breakdown.perfect_clear is perfect
    assert breakdown.b2b_broken is (lines == 1)
    assert breakdown.combo == 5
    if lines == 4:
        assert breakdown.b2b_chain_growth >= 1


@pytest.mark.parametrize("queue_length,hold", [(0, None), (1, None), (0, "I"), (1, "T")])
def test_sparse_queue_hold_enumeration_and_leaf_preserve_bag_rng(queue_length, hold):
    game = Game(857)
    game.queue = deque(tuple(game.queue)[:queue_length])
    game.hold_piece = hold
    before = _signature(game)
    cfg = teacher.PlacementTeacherConfig()
    expected_actions = reference._legal_actions(game, cfg)
    assert teacher._legal_actions(game, cfg) == expected_actions
    assert any(a.use_hold for a in expected_actions)
    for action in expected_actions:
        _assert_leaf(game, action)
    assert _signature(game) == before


def test_leaf_topout_hidden_occupancy_and_next_spawn_collision():
    game = Game(859)
    game.current = "I"
    game.board[1][4] = "G"
    game.queue[0] = "O"
    p = Placement("I", 0, 22, 0, game.cells("I", 0, 22, 0))
    assert _assert_leaf(game, SearchAction(False, p)).game_over
    # With no hidden rows, only the next O spawn detects the same obstruction.
    game.hidden_rows = 0
    assert _assert_leaf(game, SearchAction(False, p)).game_over


def test_no_actions_terminal_and_paused_match_reference():
    for attr in ("game_over", "paused"):
        game = Game(863)
        setattr(game, attr, True)
        assert teacher.rank_placement_teacher_actions(game) == reference.rank_placement_teacher_actions(game) == ()


def test_pathless_future_retains_exact_rotation_metadata_and_hold_used():
    game = _varied_game(867)
    game.current = "T"
    cfg = teacher.PlacementTeacherConfig(allow_180=True)
    for hold_used in (False, True):
        game.hold_used = hold_used
        expected = reference._legal_actions(game, cfg)
        actual = teacher._legal_actions(game, cfg, include_paths=False)
        assert actual == tuple(
            SearchAction(action.use_hold, replace(action.placement, path=()))
            for action in expected
        )
        assert expected and any(a.placement.last_move_was_rotation for a in expected)
        if hold_used:
            assert not any(a.use_hold for a in actual)


def test_transition_public_api_normalizes_high_stack_threshold():
    before = _varied_game(869)
    action = reference._legal_actions(before, teacher.PlacementTeacherConfig())[0]
    after = clone_game(before)
    result = apply_search_action(after, action)
    for height in (-20, 0, 100):
        cfg = teacher.PlacementTeacherConfig(high_stack_height=height)
        assert teacher.placement_teacher_transition_score(before, after, result, config=cfg) == (
            reference.placement_teacher_transition_score(before, after, result, config=cfg)
        )


def test_each_leaf_objective_term_independently_matches_engine():
    cases = []
    spin = Game(877)
    spin.current = "T"
    spin.board[22] = ["G"] * 10
    for x in (3, 4, 5):
        spin.board[22][x] = None
    for x, y in ((3, 21), (3, 23), (5, 23)):
        spin.board[y][x] = "G"
    spin.back_to_back, spin.b2b_chain, spin.combo = True, 5, 3
    cases.append((spin, SearchAction(False, Placement(
        "T", 3, 21, 0, spin.cells("T", 3, 21, 0),
        last_move_was_rotation=True, rotation_kick_index=4,
        rotation_from=3, rotation_to=0,
    ))))

    for lines, perfect in ((1, False), (4, True)):
        game = Game(879)
        game.current = "I"
        for y in range(24 - lines, 24):
            game.board[y] = ["G"] * 9 + [None]
        if not perfect:
            game.board[18][0] = "G"
        game.back_to_back, game.b2b_chain, game.combo = True, 10, 4
        cases.append((game, SearchAction(False, Placement(
            "I", 7, 20, 1, game.cells("I", 7, 20, 1),
        ))))

    holes = _varied_game(881)
    cases.append((holes, reference._legal_actions(holes, teacher.PlacementTeacherConfig())[0]))
    topout = Game(883)
    topout.current = "I"
    topout.hidden_rows = 0
    topout.board[1][4] = "G"
    topout.queue[0] = "O"
    cases.append((topout, SearchAction(False, Placement(
        "I", 0, 22, 0, topout.cells("I", 0, 22, 0),
    ))))

    names = tuple(field.name for field in fields(teacher.PlacementTeacherWeights))
    for selected in names:
        weights = teacher.PlacementTeacherWeights(**{
            name: float(name == selected) for name in names
        })
        for game, action in cases:
            # A unit basis isolates each positive/negative term, preventing
            # compensating mistakes from passing a total-score comparison.
            _assert_leaf(game, action, weights=weights)


def test_dataset_bytes_equal_original_and_multiprocess_order(tmp_path):
    cfg = teacher.PlacementV2DatasetConfig(
        games=2, max_pieces=2, seed_base=871, seed_step=31, max_candidates=24,
        teacher=teacher.PlacementTeacherConfig(depth=2, beam_width=24),
    )
    old_path, new_path, parallel_path = [tmp_path / name for name in ("old.jsonl", "new.jsonl", "parallel.jsonl")]
    with patch.object(teacher, "rank_placement_teacher_actions", reference.rank_placement_teacher_actions):
        teacher.write_placement_v2_dataset(old_path, cfg, workers=1)
    teacher.write_placement_v2_dataset(new_path, cfg, workers=1)
    teacher.write_placement_v2_dataset(parallel_path, cfg, workers=2)
    assert old_path.read_bytes() == new_path.read_bytes() == parallel_path.read_bytes()
    records = [json.loads(line) for line in new_path.read_text().splitlines()]
    assert len(records) == 4
    assert [(row["seed"], row["pieceIndex"]) for row in records] == [
        (871, 0), (871, 1), (902, 0), (902, 1),
    ]
