from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from minoflux.handling import HandlingController, RepeatTimer
from minoflux.settings import GameSettings, load_settings, save_settings


class RepeatTimerTests(unittest.TestCase):
    def test_repeat_waits_for_initial_delay(self) -> None:
        timer = RepeatTimer()
        timer.press(1.0, 100)
        self.assertEqual(timer.poll(1.099, 20).count, 0)
        self.assertEqual(timer.poll(1.100, 20).count, 1)
        self.assertEqual(timer.poll(1.160, 20).count, 3)

    def test_zero_interval_emits_one_instant_batch(self) -> None:
        timer = RepeatTimer()
        timer.press(2.0, 50)
        self.assertFalse(timer.poll(2.049, 0).instant)
        self.assertTrue(timer.poll(2.050, 0).instant)
        self.assertFalse(timer.poll(3.0, 0).instant)

    def test_last_pressed_horizontal_direction_wins(self) -> None:
        handling = HandlingController()
        handling.press_horizontal(-1, 0.0, 100)
        handling.press_horizontal(1, 0.02, 100)
        direction, batch = handling.poll_horizontal(0.12, 20)
        self.assertEqual(direction, 1)
        self.assertEqual(batch.count, 1)
        handling.release_horizontal(1, 0.13, 100)
        self.assertEqual(handling.active_horizontal, -1)


class SettingsTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            settings = GameSettings(das_ms=90, arr_ms=0, soft_drop_ms=12)
            settings.bindings["rotate_180"] = "v"
            save_settings(settings, path)
            loaded = load_settings(path)
            self.assertEqual(loaded.das_ms, 90)
            self.assertEqual(loaded.arr_ms, 0)
            self.assertEqual(loaded.soft_drop_ms, 12)
            self.assertEqual(loaded.bindings["rotate_180"], "v")

    def test_invalid_values_are_clamped(self) -> None:
        settings = GameSettings(das_ms=-4, arr_ms=9999, soft_drop_ms=-1).normalize()
        self.assertEqual(settings.das_ms, 0)
        self.assertEqual(settings.arr_ms, 500)
        self.assertEqual(settings.soft_drop_ms, 0)


if __name__ == "__main__":
    unittest.main()
