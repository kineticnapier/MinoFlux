from __future__ import annotations

from minoflux import versus_neural_cli

from minoflux.versus_neural_cli import _benchmark_summary


def test_benchmark_summary_names_policy_and_physical_wins_separately() -> None:
    text = _benchmark_summary(
        {
            "playerWins": 53,
            "aiWins": 47,
            "draws": 0,
            "playerPolicyWins": 53,
            "aiPolicyWins": 47,
            "physicalPlayerWins": 65,
            "physicalAiWins": 35,
            "seedCount": 50,
            "mirroredGameCount": 50,
        }
    )
    assert "Benchmark policies: player 53 - ai 47 - draws 0" in text
    assert "physical sides: player 65 - ai 35" in text
    assert "seeds 50, mirrored legs 50" in text


def test_selfplay_cli_does_not_leak_player_value_scorer_to_ai(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    player_solo = object()
    ai_solo = object()
    player_value = object()
    captured: dict[str, object] = {}

    def fake_load_solo(path, args, cache):
        if path == "player-solo.pt":
            return player_solo
        if path == "ai-solo.pt":
            return ai_solo
        raise AssertionError(f"unexpected solo model: {path!r}")

    def fake_load_versus_value(path, args, cache):
        if path == "player-value.pt":
            return player_value
        if path is None:
            return None
        raise AssertionError(f"unexpected versus model: {path!r}")

    def fake_generate(path, solo_scorer, config, **kwargs):
        assert solo_scorer is player_solo
        captured.update(kwargs)
        return {"format": "test"}

    monkeypatch.setattr(
        versus_neural_cli,
        "_load_solo",
        fake_load_solo,
    )
    monkeypatch.setattr(
        versus_neural_cli,
        "_load_versus_value",
        fake_load_versus_value,
    )
    monkeypatch.setattr(
        versus_neural_cli,
        "generate_versus_selfplay_dataset_progress",
        fake_generate,
    )

    args = versus_neural_cli.build_parser().parse_args(
        [
            "selfplay",
            "--output",
            str(tmp_path / "selfplay.jsonl"),
            "--player-solo-model",
            "player-solo.pt",
            "--ai-solo-model",
            "ai-solo.pt",
            "--player-versus-value-model",
            "player-value.pt",
        ]
    )

    assert versus_neural_cli._selfplay(args) == 0

    assert captured["value_scorer"] is player_value

    ai_value_scorer = captured["ai_value_scorer"]
    assert ai_value_scorer is not None
    assert ai_value_scorer is not player_value
    assert ai_value_scorer.score_match(None, "ai") == 0.0
    assert ai_value_scorer.score_matches((object(), object())) == (0.0, 0.0)

    capsys.readouterr()
