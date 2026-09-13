from __future__ import annotations

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
