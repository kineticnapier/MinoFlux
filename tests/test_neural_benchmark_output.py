from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch

from minoflux.versus_neural_cli import _benchmark, _print_report, build_parser


class FakeBenchmarkResult:
    def to_dict(self) -> dict[str, object]:
        return {"games": 2, "playerWins": 1, "aiWins": 1, "draws": 0}


class NeuralBenchmarkOutputTests(unittest.TestCase):
    def capture(self, result: dict[str, object], **kwargs: bool) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            _print_report(result, **kwargs)
        return buffer.getvalue()

    def test_benchmark_output_flags_are_mutually_exclusive(self) -> None:
        parser = build_parser()
        default = parser.parse_args(["benchmark"])
        self.assertFalse(default.print_json)
        self.assertFalse(default.pretty_json)

        compact = parser.parse_args(["benchmark", "--print-json"])
        self.assertTrue(compact.print_json)
        self.assertFalse(compact.pretty_json)

        pretty = parser.parse_args(["benchmark", "--pretty-json"])
        self.assertFalse(pretty.print_json)
        self.assertTrue(pretty.pretty_json)

        with self.assertRaises(SystemExit):
            parser.parse_args(["benchmark", "--print-json", "--pretty-json"])

    def test_default_output_preserves_pretty_json(self) -> None:
        result = {"games": 2, "playerWins": 1, "nested": {"draws": 1}}
        self.assertEqual(
            self.capture(result),
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        )

    def test_print_json_is_compact_single_line(self) -> None:
        result = {"games": 2, "playerWins": 1, "model": "攻撃型"}
        text = self.capture(result, print_json=True)
        self.assertEqual(
            text,
            json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n",
        )
        self.assertEqual(len(text.rstrip("\n").splitlines()), 1)

    def test_pretty_json_matches_default_output(self) -> None:
        result = {"games": 2, "playerWins": 1, "nested": {"draws": 1}}
        self.assertEqual(
            self.capture(result, pretty_json=True),
            self.capture(result),
        )

    def test_benchmark_print_json_keeps_progress_enabled(self) -> None:
        args = build_parser().parse_args(["benchmark", "--print-json"])
        buffer = io.StringIO()
        with (
            patch("minoflux.versus_neural_cli._load_solo", return_value=object()),
            patch("minoflux.versus_neural_cli._load_versus_value", return_value=None),
            patch(
                "minoflux.versus_neural_cli.run_versus_benchmark",
                return_value=FakeBenchmarkResult(),
            ) as run_benchmark,
            redirect_stdout(buffer),
        ):
            self.assertEqual(_benchmark(args), 0)

        output = buffer.getvalue()
        self.assertEqual(len(output.rstrip("\n").splitlines()), 1)
        self.assertEqual(json.loads(output)["games"], 2)
        self.assertTrue(run_benchmark.call_args.kwargs["progress"])


if __name__ == "__main__":
    unittest.main()
