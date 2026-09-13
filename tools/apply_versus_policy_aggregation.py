from __future__ import annotations

from pathlib import Path


BENCHMARK = Path("src/minoflux_ai/versus_benchmark.py")
SELFPLAY = Path("src/minoflux_ai/versus_neural.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_benchmark() -> None:
    text = BENCHMARK.read_text(encoding="utf-8")
    start = text.index("    def to_dict(self) -> dict[str, object]:\n")
    end = text.index("\n\n\ndef run_versus_game(\n", start)
    new_method = '''    def to_dict(self) -> dict[str, object]:
        # Per-game results are stored in logical policy orientation. Swapped
        # physical sides are remapped by _remap_swapped_result() before they
        # reach this aggregate, so the legacy playerWins/aiWins fields already
        # mean player-policy/ai-policy wins rather than physical-side wins.
        player_pieces = max(1.0, sum(item.player_pieces for item in self.per_game))
        ai_pieces = max(1.0, sum(item.ai_pieces for item in self.per_game))
        unswapped = tuple(item for item in self.per_game if not item.models_swapped)
        swapped = tuple(item for item in self.per_game if item.models_swapped)

        unswapped_player_policy_wins = sum(item.winner == "player" for item in unswapped)
        unswapped_ai_policy_wins = sum(item.winner == "ai" for item in unswapped)
        swapped_player_policy_wins = sum(item.winner == "player" for item in swapped)
        swapped_ai_policy_wins = sum(item.winner == "ai" for item in swapped)

        def physical_winner(item: VersusGameResult) -> str:
            if item.winner == "draw" or not item.models_swapped:
                return item.winner
            return "ai" if item.winner == "player" else "player"

        physical_player_wins = sum(physical_winner(item) == "player" for item in self.per_game)
        physical_ai_wins = sum(physical_winner(item) == "ai" for item in self.per_game)
        seed_count = len({item.seed for item in self.per_game})
        mirrored_game_count = len(swapped)
        unswapped_game_count = len(unswapped)

        return {
            "games": self.games,
            "maxTurns": self.max_turns,
            "seedBase": self.seed_base,
            "seedStep": self.seed_step,
            "seedCount": seed_count,
            "seedStepUnit": "mirroredPair",
            "mirroredSides": True,
            "mirroredGameCount": mirrored_game_count,
            "unswappedGameCount": unswapped_game_count,
            "perGameOrientation": "policy",
            "playerWins": self.player_wins,
            "aiWins": self.ai_wins,
            "draws": self.draws,
            "playerWinRate": self.player_wins / self.games,
            "aiWinRate": self.ai_wins / self.games,
            "playerPolicyWins": self.player_wins,
            "aiPolicyWins": self.ai_wins,
            "playerPolicyWinRate": self.player_wins / self.games,
            "aiPolicyWinRate": self.ai_wins / self.games,
            "unswappedPlayerPolicyWins": unswapped_player_policy_wins,
            "unswappedAiPolicyWins": unswapped_ai_policy_wins,
            "swappedPlayerPolicyWins": swapped_player_policy_wins,
            "swappedAiPolicyWins": swapped_ai_policy_wins,
            "physicalPlayerWins": physical_player_wins,
            "physicalAiWins": physical_ai_wins,
            "physicalPlayerWinRate": physical_player_wins / self.games,
            "physicalAiWinRate": physical_ai_wins / self.games,
            "meanTurns": self.mean_turns,
            "playerMeanPieces": self.player_mean_pieces,
            "aiMeanPieces": self.ai_mean_pieces,
            "playerMeanAttack": self.player_mean_attack,
            "aiMeanAttack": self.ai_mean_attack,
            "playerMeanSent": self.player_mean_sent,
            "aiMeanSent": self.ai_mean_sent,
            "playerMeanCanceled": self.player_mean_canceled,
            "aiMeanCanceled": self.ai_mean_canceled,
            "playerMeanReceived": self.player_mean_received,
            "aiMeanReceived": self.ai_mean_received,
            "playerSentPerPiece": sum(item.player_sent for item in self.per_game) / player_pieces,
            "aiSentPerPiece": sum(item.ai_sent for item in self.per_game) / ai_pieces,
            "playerAttackPerPiece": sum(item.player_attack for item in self.per_game) / player_pieces,
            "aiAttackPerPiece": sum(item.ai_attack for item in self.per_game) / ai_pieces,
            "perGame": [asdict(item) for item in self.per_game],
        }
'''
    text = text[:start] + new_method + text[end:]

    text = replace_once(
        text,
        '''                    game_bar.set_postfix(
                        P=player_wins,
                        A=ai_wins,
                        D=draws,
                        turns=logical.turns,
                        active=len(active),
                    )''',
        '''                    game_bar.set_postfix(
                        playerPolicy=player_wins,
                        aiPolicy=ai_wins,
                        D=draws,
                        turns=logical.turns,
                        active=len(active),
                    )''',
        label="batched benchmark progress",
    )
    text = replace_once(
        text,
        "            game_bar.set_postfix(P=player_wins, A=ai_wins, D=draws, turns=logical.turns)",
        "            game_bar.set_postfix(playerPolicy=player_wins, aiPolicy=ai_wins, D=draws, turns=logical.turns)",
        label="serial benchmark progress",
    )
    text = replace_once(
        text,
        "def _remap_swapped_result(result: VersusGameResult) -> VersusGameResult:\n",
        '''def _remap_swapped_result(result: VersusGameResult) -> VersusGameResult:
    # Normalize a physically swapped leg back to logical policy orientation.
    # After this point, winner="player" and player_* always refer to the
    # configured player policy, not the physical player side.
''',
        label="swapped-result semantics comment",
    )
    BENCHMARK.write_text(text, encoding="utf-8")


def patch_selfplay() -> None:
    text = SELFPLAY.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''        "games": cfg.games,
        "records": total_records,''',
        '''        "games": cfg.games,
        "seedCount": cfg.games,
        "seedBase": cfg.seed_base,
        "seedStep": cfg.seed_step,
        "seedStepUnit": "game",
        "mirroredSides": False,
        "mirroredGameCount": 0,
        "startingSideAlternates": True,
        "records": total_records,''',
        label="selfplay seed metadata",
    )
    SELFPLAY.write_text(text, encoding="utf-8")


def main() -> None:
    patch_benchmark()
    patch_selfplay()


if __name__ == "__main__":
    main()
