from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path

import pytest

from minoflux_ai.fusion_oracle import (
    FusionOracleConfig,
    FusionOracleMatchError,
    game_to_fusion_request,
    match_fusion_action,
    parse_fusion_label,
    run_fusion_oracle_requests,
    write_fusion_oracle_dataset,
)
from minoflux_ai.heuristic import DEFAULT_WEIGHTS
from minoflux_ai.neural_dataset import NEURAL_DATASET_FORMAT
from minoflux_ai.search import SearchAction, SearchConfig, rank_search_actions
from minoflux_engine import Game, Placement


_FUSION_PIECE_RAW = {"I": 0, "O": 1, "T": 2, "L": 3, "J": 4, "S": 5, "Z": 6}


def _placement(piece: str, cells: tuple[tuple[int, int], ...]) -> Placement:
    return Placement(piece=piece, x=0, y=0, rotation=0, cells=cells)


def _raw_move(*, piece_raw: int, rotation: int, x: int, y: int) -> int:
    return (
        (y & 0x3F)
        | ((x & 0xF) << 6)
        | ((piece_raw & 0x7) << 10)
        | ((rotation & 0x3) << 13)
    )


def _label_for_action(game: Game, action: SearchAction) -> dict[str, object]:
    cells = sorted((int(x), game.height - 1 - int(y)) for x, y in action.placement.cells)
    return {
        "best_move_raw": _FUSION_PIECE_RAW[action.placement.piece] << 10,
        "best_value": 5.0,
        "bestHoldUsed": bool(action.use_hold),
        "bestCells": [list(cell) for cell in cells],
        "position_complexity": 0.0,
        "root_scores": [],
        "policy_probs": [],
    }


def _write_fake_oracle(tmp_path: Path) -> Path:
    script = tmp_path / "fake_oracle.py"
    script.write_text(
        """from __future__ import annotations
import json
import os
from pathlib import Path
import sys

if os.environ.get("FAKE_ORACLE_FAIL") == "1":
    raise SystemExit(7)
if os.environ.get("RAYON_NUM_THREADS") != "1":
    raise SystemExit(8)
if "--time-budget-ms" in sys.argv:
    raise SystemExit(9)
args = sys.argv[1:]
if len(args) != 3 or args[0] != "--skip-failures":
    raise SystemExit(10)
input_path = Path(args[1])
output_path = Path(args[2])
log_path = os.environ.get("FAKE_ORACLE_LOG")
fixed_label = os.environ.get("FAKE_ORACLE_LABEL")
if log_path:
    with Path(log_path).open("a", encoding="utf-8") as log:
        log.write(json.dumps({"argv": args, "rayon": os.environ.get("RAYON_NUM_THREADS")}) + "\\n")
with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
    for line in source:
        if not line.strip():
            continue
        request = json.loads(line)
        if fixed_label:
            label = json.loads(fixed_label)
        else:
            label = {
                "best_move_raw": 0,
                "best_value": float(request["frame_id"]),
                "position_complexity": 0.0,
                "root_scores": [[0, float(request["frame_id"])]],
                "policy_probs": [1.0],
            }
        target.write(json.dumps(label) + "\\n")
""",
        encoding="utf-8",
    )
    if os.name == "nt":
        wrapper = tmp_path / "fake_oracle.cmd"
        wrapper.write_text(
            f'@"{os.fspath(Path(os.sys.executable))}" "{os.fspath(script)}" %*\r\n',
            encoding="utf-8",
        )
        return wrapper
    wrapper = tmp_path / "fake_oracle"
    wrapper.write_text(
        f"#!{os.sys.executable}\nimport runpy, sys\nsys.argv[0] = {str(script)!r}\nrunpy.run_path({str(script)!r}, run_name='__main__')\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    return wrapper


def test_game_to_fusion_request_converts_board_and_chain_state() -> None:
    game = Game(123)
    game.board = [[None] * 10 for _ in range(game.height)]
    game.board[23][0] = "G"
    game.board[22][9] = "T"
    game.current = "T"
    game.hold_piece = "I"
    game.queue = deque(["O", "S", "Z", "J", "L", "T", "I"] * 4)
    game.combo = 1
    game.back_to_back = True
    game.b2b_chain = 4
    game.lines = 12
    game.pieces_placed = 7

    request = game_to_fusion_request(game, request_id="seed123:7", queue_length=18)

    assert request["schema_version"] == "phase1-v1"
    assert request["replay_id"] == "seed123:7"
    assert request["frame_id"] == 7
    assert request["player_board_rows"] == [1, 1 << 9]
    assert request["opponent_board_rows"] == []
    assert request["current_piece"] == "t"
    assert request["hold_piece"] == "i"
    assert request["queue"] == [piece.lower() for piece in list(game.queue)[:18]]
    assert request["combo"] == 2
    assert request["b2b"] == 5
    assert request["lines"] == 12
    assert request["pending_garbage"] == 0
    assert request["bag_number"] == 0


@pytest.mark.parametrize(("combo", "expected"), [(-1, 0), (0, 1), (1, 2)])
def test_game_to_fusion_request_maps_combo_offset(combo: int, expected: int) -> None:
    game = Game(11)
    game.queue = deque(["I"] * 18)
    game.combo = combo
    request = game_to_fusion_request(game, request_id="combo", queue_length=18)
    assert request["combo"] == expected


@pytest.mark.parametrize(
    ("active", "chain", "expected"),
    [(False, 8, 0), (True, 0, 1), (True, 1, 2), (True, 4, 5)],
)
def test_game_to_fusion_request_maps_b2b_offset(active: bool, chain: int, expected: int) -> None:
    game = Game(12)
    game.queue = deque(["I"] * 18)
    game.back_to_back = active
    game.b2b_chain = chain
    request = game_to_fusion_request(game, request_id="b2b", queue_length=18)
    assert request["b2b"] == expected


def test_game_to_fusion_request_rejects_short_queue() -> None:
    game = Game(13)
    game.queue = deque(["I", "O"])
    with pytest.raises(ValueError, match="insufficient-queue"):
        game_to_fusion_request(game, request_id="short", queue_length=18)


def test_extended_label_matches_hold_action_exactly() -> None:
    game = Game(21)
    hold_cells = ((3, 23), (4, 23), (3, 22), (4, 22))
    direct_cells = ((0, 23), (1, 23), (0, 22), (1, 22))
    actions = (
        SearchAction(False, _placement("O", direct_cells)),
        SearchAction(True, _placement("O", hold_cells)),
    )
    raw = _raw_move(piece_raw=1, rotation=0, x=3, y=0)
    label = parse_fusion_label(
        {
            "best_move_raw": raw,
            "best_value": 3.5,
            "bestHoldUsed": True,
            "bestCells": [[3, 0], [4, 0], [3, 1], [4, 1]],
        }
    )
    assert match_fusion_action(game, actions, label) == actions[1]


def test_legacy_raw_label_matches_unique_action_by_piece_and_cells() -> None:
    game = Game(22)
    wanted = SearchAction(False, _placement("O", ((4, 22), (5, 22), (4, 21), (5, 21))))
    other = SearchAction(False, _placement("O", ((0, 22), (1, 22), (0, 21), (1, 21))))
    raw = _raw_move(piece_raw=1, rotation=0, x=4, y=1)
    label = parse_fusion_label({"best_move_raw": raw, "best_value": 2.0})
    assert match_fusion_action(game, (other, wanted), label) == wanted


def test_legacy_raw_label_rejects_hold_ambiguity() -> None:
    game = Game(23)
    cells = ((4, 22), (5, 22), (4, 21), (5, 21))
    actions = (SearchAction(False, _placement("O", cells)), SearchAction(True, _placement("O", cells)))
    raw = _raw_move(piece_raw=1, rotation=0, x=4, y=1)
    label = parse_fusion_label({"best_move_raw": raw, "best_value": 2.0})
    with pytest.raises(FusionOracleMatchError) as error:
        match_fusion_action(game, actions, label)
    assert error.value.reason == "oracle-action-ambiguous"


def test_legacy_raw_label_rejects_unmatched_action() -> None:
    game = Game(24)
    action = SearchAction(False, _placement("O", ((0, 22), (1, 22), (0, 21), (1, 21))))
    raw = _raw_move(piece_raw=1, rotation=0, x=4, y=1)
    label = parse_fusion_label({"best_move_raw": raw, "best_value": 2.0})
    with pytest.raises(FusionOracleMatchError) as error:
        match_fusion_action(game, (action,), label)
    assert error.value.reason == "oracle-action-unmatched"


def test_run_fusion_oracle_requests_batches_and_preserves_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _write_fake_oracle(tmp_path)
    log_path = tmp_path / "oracle.log"
    monkeypatch.setenv("FAKE_ORACLE_LOG", str(log_path))
    requests = tuple(
        {
            "schema_version": "phase1-v1",
            "replay_id": f"r{index}",
            "round_id": 0,
            "player_id": 0,
            "frame_id": index,
            "group_id": f"r{index}",
            "player_board_rows": [],
            "opponent_board_rows": [],
            "current_piece": "t",
            "hold_piece": None,
            "queue": ["i"] * 18,
            "combo": 0,
            "b2b": 0,
            "lines": 0,
            "pending_garbage": 0,
            "bag_number": 0,
        }
        for index in range(5)
    )
    labels = run_fusion_oracle_requests(requests, oracle, workers=2)
    assert [label.best_value for label in labels] == [0.0, 1.0, 2.0, 3.0, 4.0]
    calls = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(calls) == 2
    assert all(call["rayon"] == "1" for call in calls)
    assert all("--time-budget-ms" not in call["argv"] for call in calls)


def test_run_fusion_oracle_requests_fails_on_child_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle = _write_fake_oracle(tmp_path)
    monkeypatch.setenv("FAKE_ORACLE_FAIL", "1")
    request = {
        "schema_version": "phase1-v1",
        "replay_id": "fail",
        "round_id": 0,
        "player_id": 0,
        "frame_id": 0,
        "group_id": "fail",
        "player_board_rows": [],
        "opponent_board_rows": [],
        "current_piece": "t",
        "hold_piece": None,
        "queue": ["i"] * 18,
        "combo": 0,
        "b2b": 0,
        "lines": 0,
        "pending_garbage": 0,
        "bag_number": 0,
    }
    with pytest.raises(RuntimeError, match="oracle-process-failed"):
        run_fusion_oracle_requests((request,), oracle, workers=1)


def test_write_fusion_oracle_dataset_writes_existing_ranking_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed = 6_000_001
    game = Game(seed)
    search_config = SearchConfig(
        allow_hold=True,
        lookahead_pieces=0,
        beam_width=4,
        discount=0.90,
        srs_reachable=True,
        allow_180=True,
        reachability_node_limit=8_000,
    )
    ranked = rank_search_actions(game, DEFAULT_WEIGHTS, search_config, limit=None)
    assert ranked
    expected_action = ranked[0][0]
    monkeypatch.setenv("FAKE_ORACLE_LABEL", json.dumps(_label_for_action(game, expected_action)))
    oracle = _write_fake_oracle(tmp_path)
    output = tmp_path / "fusion-dataset.jsonl"

    summary = write_fusion_oracle_dataset(
        output,
        oracle,
        FusionOracleConfig(
            queue_length=18,
            max_candidates=4,
            workers=2,
            allow_180=True,
            reachability_node_limit=8_000,
        ),
        games=1,
        max_pieces=1,
        seed_base=seed,
        seed_step=97,
        trajectory="heuristic",
    )

    assert summary["samples"] == 1
    assert summary["skipped"] == {}
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert record["format"] == NEURAL_DATASET_FORMAT
    assert record["teacher"] == "fusion-offline-oracle"
    assert record["expertIndices"] == [record["expertIndex"]]
    expected_move = [
        int(expected_action.use_hold),
        expected_action.placement.piece,
        expected_action.placement.x,
        expected_action.placement.y,
        expected_action.placement.rotation,
    ]
    assert record["candidates"][record["expertIndex"]]["move"] == expected_move
    assert record["fusionOracle"] == {"beamWidth": 2000, "depth": 18, "timeBudgetMs": None}
    metadata = json.loads(output.with_suffix(".jsonl.meta.json").read_text(encoding="utf-8"))
    assert metadata["trajectory"] == "heuristic"
