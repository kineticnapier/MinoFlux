from minoflux.duel_cli import format_duel_summary, resolve_model


def test_duel_model_aliases():
    assert resolve_model("base") == "data/models/native-oracle-value.pt"
    assert resolve_model("r2w") == "data/models/native-oracle-value-r2w.pt"
    assert resolve_model("data/models/custom.pt") == "data/models/custom.pt"


def test_duel_summary_is_compact():
    result = {
        "playerPolicyWins": 111,
        "aiPolicyWins": 145,
        "draws": 0,
        "playerPolicyWinRate": 0.43359375,
        "aiPolicyWinRate": 0.56640625,
        "playerAttackPerPiece": 0.2775827482447342,
        "aiAttackPerPiece": 0.28465641669281455,
        "playerSentPerPiece": 0.23928034102306922,
        "aiSentPerPiece": 0.2418575462817697,
    }
    assert format_duel_summary("r2w", "r3w", result) == (
        "r2w 111 - 145 r3w (draw 0)\n"
        "win 43.4% - 56.6% | APP 0.2776 - 0.2847 | sent/piece 0.2393 - 0.2419"
    )
