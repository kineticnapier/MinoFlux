from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minoflux.lab import (
    MODEL_SOURCE_BUILTIN,
    MODEL_SOURCE_CUSTOM,
    MODEL_SOURCE_LATEST,
    _default_model_source,
    _load_selected_weights,
)
from minoflux_ai import DEFAULT_WEIGHTS, HeuristicWeights, save_weights


class LabModelSelectionTests(unittest.TestCase):
    def test_builtin_weights(self) -> None:
        weights, source = _load_selected_weights(MODEL_SOURCE_BUILTIN)
        self.assertEqual(weights, DEFAULT_WEIGHTS)
        self.assertEqual(source, MODEL_SOURCE_BUILTIN)

    def test_latest_model_is_default_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "latest-cem.json"
            expected = HeuristicWeights(lines=3.25)
            save_weights(model_path, expected)
            with patch("minoflux.lab.LATEST_CEM_MODEL", model_path):
                self.assertEqual(_default_model_source(), MODEL_SOURCE_LATEST)
                actual, source = _load_selected_weights(MODEL_SOURCE_LATEST)
            self.assertEqual(actual, expected)
            self.assertEqual(source, str(model_path.resolve()))

    def test_builtin_is_default_when_latest_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with patch("minoflux.lab.LATEST_CEM_MODEL", missing):
                self.assertEqual(_default_model_source(), MODEL_SOURCE_BUILTIN)

    def test_custom_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "custom.json"
            expected = HeuristicWeights(holes=-9.5)
            save_weights(model_path, expected)
            actual, source = _load_selected_weights(MODEL_SOURCE_CUSTOM, str(model_path))
            self.assertEqual(actual, expected)
            self.assertEqual(source, str(model_path.resolve()))

    def test_custom_model_requires_a_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            _load_selected_weights(MODEL_SOURCE_CUSTOM, "")

    def test_latest_model_must_exist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            with patch("minoflux.lab.LATEST_CEM_MODEL", missing):
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    _load_selected_weights(MODEL_SOURCE_LATEST)


if __name__ == "__main__":
    unittest.main()
