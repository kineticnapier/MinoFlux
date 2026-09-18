from __future__ import annotations

from minoflux.neural_cli import build_parser


def test_fusion_dataset_cli_accepts_oracle_generation_options() -> None:
    args = build_parser().parse_args(
        [
            "fusion-dataset",
            "--oracle-bin",
            "/tmp/generate_policy_value_labels",
            "--output",
            "data/neural/fusion.jsonl",
            "--games",
            "2",
            "--max-pieces",
            "50",
            "--seed-base",
            "6000001",
            "--seed-step",
            "97",
            "--workers",
            "4",
            "--max-candidates",
            "24",
            "--allow-180",
        ]
    )

    assert args.command == "fusion-dataset"
    assert args.oracle_bin == "/tmp/generate_policy_value_labels"
    assert args.output == "data/neural/fusion.jsonl"
    assert args.games == 2
    assert args.max_pieces == 50
    assert args.workers == 4
    assert args.max_candidates == 24
    assert args.allow_180 is True
    assert args.trajectory == "heuristic"


def test_fusion_dataset_cli_defaults_do_not_expose_time_budget() -> None:
    args = build_parser().parse_args(
        ["fusion-dataset", "--oracle-bin", "/tmp/generate_policy_value_labels"]
    )

    assert args.workers == 1
    assert args.trajectory == "heuristic"
    assert args.max_candidates == 24
    assert not hasattr(args, "time_budget_ms")
