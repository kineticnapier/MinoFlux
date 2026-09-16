from minoflux.versus_ai import FirstToScore, build_parser


def test_first_to_score_tracks_wins_and_ignores_draws() -> None:
    score = FirstToScore(target=3)

    score.record("player")
    score.record(None)
    score.record("ai")
    score.record("player")

    assert score.player_wins == 2
    assert score.ai_wins == 1
    assert score.winner is None

    score.record("player")
    assert score.winner == "player"


def test_first_to_score_reset_starts_new_series() -> None:
    score = FirstToScore(target=2, player_wins=2, ai_wins=1)

    score.reset()

    assert score.player_wins == 0
    assert score.ai_wins == 0
    assert score.winner is None


def test_first_to_zero_disables_series_winner() -> None:
    score = FirstToScore(target=0)
    score.record("ai")

    assert score.ai_wins == 1
    assert score.winner is None


def test_first_to_cli_option() -> None:
    args = build_parser().parse_args(["--first-to", "10"])

    assert args.first_to == 10
