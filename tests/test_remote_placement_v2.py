from __future__ import annotations

from minoflux.placement_v2_remote import DATASET, MODEL, build_parser
from tools.remote_agent import COMMANDS, COMMAND_PREFIX, _issue_command


def test_remote_whitelist_contains_placement_v2_jobs() -> None:
    assert COMMANDS["placement-v2-full-50"][:3] == (
        "-m",
        "minoflux.placement_v2_remote",
        "full",
    )
    assert COMMANDS["placement-v2-generate-50"][:3] == (
        "-m",
        "minoflux.placement_v2_remote",
        "generate",
    )
    assert COMMANDS["placement-v2-train"][:3] == (
        "-m",
        "minoflux.placement_v2_remote",
        "train",
    )
    assert COMMANDS["placement-v2-evaluate"][:3] == (
        "-m",
        "minoflux.placement_v2_remote",
        "evaluate",
    )


def test_remote_command_still_requires_owner_and_exact_whitelist() -> None:
    valid = {
        "title": COMMAND_PREFIX + "placement-v2-full-50",
        "user": {"login": "kineticnapier"},
    }
    assert _issue_command(valid) == "placement-v2-full-50"
    assert _issue_command({**valid, "user": {"login": "someone-else"}}) is None
    assert _issue_command({**valid, "title": COMMAND_PREFIX + "placement-v2-full-50 --games 999"}) is None


def test_remote_pipeline_defaults_match_fixed_training_job() -> None:
    args = build_parser().parse_args(["full"])
    assert args.games == 50
    assert args.max_pieces == 300
    assert args.teacher_depth == 2
    assert args.teacher_beam == 24
    assert args.workers == 6
    assert args.epochs == 8
    assert DATASET.as_posix() == "data/neural/placement-v2-ranking.jsonl"
    assert MODEL.as_posix() == "data/models/placement-v2.pt"
