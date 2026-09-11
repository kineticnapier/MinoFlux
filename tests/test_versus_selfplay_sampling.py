from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minoflux.versus_neural_cli import _selfplay, build_parser
from minoflux_ai.versus_neural import (
    VERSUS_SELFPLAY_FORMAT,
    VersusSelfPlayConfig,
    _sample_selfplay_records,
    generate_versus_selfplay_dataset,
)


class VersusSelfPlaySamplingTests(unittest.TestCase):
    def test_legacy_symmetric_cli_remains_symmetric(self) -> None:
        args = build_parser().parse_args(
            [
                "selfplay",
                "--solo-model", "legacy-solo.pt",
                "--versus-value-model", "legacy-value.pt",
                "--games", "2",
            ]
        )
        solo = object()
        value = object()
        captured: dict[str, object] = {}

        def fake_generate(path, player_scorer, config, **kwargs):
            captured.update(
                path=path,
                player_scorer=player_scorer,
                config=config,
                kwargs=kwargs,
            )
            return {"format": VERSUS_SELFPLAY_FORMAT}

        with (
            patch("minoflux.versus_neural_cli._load_solo", return_value=solo),
            patch("minoflux.versus_neural_cli._load_versus_value", return_value=value),
            patch("minoflux.versus_neural_cli.generate_versus_selfplay_dataset_progress", side_effect=fake_generate),
            patch("minoflux.versus_neural_cli._print"),
        ):
            self.assertEqual(_selfplay(args), 0)

        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        config = captured["config"]
        assert isinstance(config, VersusSelfPlayConfig)
        self.assertIs(captured["player_scorer"], solo)
        self.assertIs(kwargs["ai_scorer"], solo)
        self.assertIs(kwargs["value_scorer"], value)
        self.assertIs(kwargs["ai_value_scorer"], value)
        self.assertEqual(config.max_records_per_game, 0)

    def test_asymmetric_cli_passes_separate_policies(self) -> None:
        args = build_parser().parse_args(
            [
                "selfplay",
                "--player-solo-model", "player-solo.pt",
                "--ai-solo-model", "ai-solo.pt",
                "--player-versus-value-model", "player-value.pt",
                "--ai-versus-value-model", "ai-value.pt",
                "--max-records-per-game", "48",
            ]
        )
        models = {
            "player-solo.pt": object(),
            "ai-solo.pt": object(),
            "player-value.pt": object(),
            "ai-value.pt": object(),
        }
        captured: dict[str, object] = {}

        def load_solo(path, _args, _cache):
            return models[path]

        def load_value(path, _args, _cache):
            return None if path is None else models[path]

        def fake_generate(path, player_scorer, config, **kwargs):
            captured.update(
                path=path,
                player_scorer=player_scorer,
                config=config,
                kwargs=kwargs,
            )
            return {"format": VERSUS_SELFPLAY_FORMAT}

        with (
            patch("minoflux.versus_neural_cli._load_solo", side_effect=load_solo),
            patch("minoflux.versus_neural_cli._load_versus_value", side_effect=load_value),
            patch("minoflux.versus_neural_cli.generate_versus_selfplay_dataset_progress", side_effect=fake_generate),
            patch("minoflux.versus_neural_cli._print"),
        ):
            self.assertEqual(_selfplay(args), 0)

        kwargs = captured["kwargs"]
        assert isinstance(kwargs, dict)
        config = captured["config"]
        assert isinstance(config, VersusSelfPlayConfig)
        self.assertIs(captured["player_scorer"], models["player-solo.pt"])
        self.assertIs(kwargs["ai_scorer"], models["ai-solo.pt"])
        self.assertIs(kwargs["value_scorer"], models["player-value.pt"])
        self.assertIs(kwargs["ai_value_scorer"], models["ai-value.pt"])
        self.assertEqual(config.max_records_per_game, 48)

    def test_asymmetric_generator_routes_policies_by_acting_side(self) -> None:
        player_solo = object()
        ai_solo = object()
        player_value = object()
        ai_value = object()
        captured = []

        def finish_immediately(requests):
            captured.extend(requests)
            return tuple(None for _ in requests)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selfplay.jsonl"
            with patch(
                "minoflux_ai.versus_neural.choose_versus_actions_batch",
                side_effect=finish_immediately,
            ):
                generate_versus_selfplay_dataset(
                    output,
                    player_solo,  # type: ignore[arg-type]
                    VersusSelfPlayConfig(games=2, game_batch=2, max_turns=2),
                    value_scorer=player_value,  # type: ignore[arg-type]
                    ai_scorer=ai_solo,  # type: ignore[arg-type]
                    ai_value_scorer=ai_value,  # type: ignore[arg-type]
                )

        self.assertEqual(len(captured), 2)
        player_request, ai_request = captured
        self.assertEqual(player_request.side_name, "player")
        self.assertIs(player_request.scorer, player_solo)
        self.assertIs(player_request.opponent_scorer, ai_solo)
        self.assertIs(player_request.state_scorer, player_value)
        self.assertEqual(ai_request.side_name, "ai")
        self.assertIs(ai_request.scorer, ai_solo)
        self.assertIs(ai_request.opponent_scorer, player_solo)
        self.assertIs(ai_request.state_scorer, ai_value)

    def test_max_records_per_game_limits_written_records(self) -> None:
        def finish_immediately(requests):
            return tuple(None for _ in requests)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "selfplay.jsonl"
            with patch(
                "minoflux_ai.versus_neural.choose_versus_actions_batch",
                side_effect=finish_immediately,
            ):
                result = generate_versus_selfplay_dataset(
                    output,
                    object(),  # type: ignore[arg-type]
                    VersusSelfPlayConfig(
                        games=1,
                        max_turns=2,
                        max_records_per_game=2,
                    ),
                )
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result["records"], 2)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(record["format"] == VERSUS_SELFPLAY_FORMAT for record in records))
        self.assertTrue(all("outcome" in record for record in records))
        self.assertTrue(any(not record["terminal"] for record in records))
        self.assertTrue(any(record["terminal"] for record in records))

    def test_sampling_is_deterministic_and_spans_the_game(self) -> None:
        records = [({"ply": index}, "player") for index in range(20)]
        first = _sample_selfplay_records(records, 5, seed=123456)
        second = _sample_selfplay_records(records, 5, seed=123456)
        first_plys = [int(record["ply"]) for record, _side in first]
        second_plys = [int(record["ply"]) for record, _side in second]
        self.assertEqual(first_plys, second_plys)
        self.assertEqual(len(first_plys), 5)
        self.assertLess(first_plys[0], 4)
        self.assertGreaterEqual(first_plys[-1], 16)
        self.assertNotEqual(first_plys, list(range(5)))

    def test_short_game_keeps_every_record(self) -> None:
        records = [
            ({"ply": 0, "side": "player"}, "player"),
            ({"ply": 0, "side": "ai"}, "ai"),
            ({"ply": 1, "side": "player"}, "player"),
            ({"ply": 1, "side": "ai"}, "ai"),
        ]
        sampled = _sample_selfplay_records(records, 48, seed=7)
        self.assertEqual(sampled, records)
        self.assertEqual(_sample_selfplay_records(records, 0, seed=7), records)


if __name__ == "__main__":
    unittest.main()
