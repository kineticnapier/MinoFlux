from __future__ import annotations

from types import SimpleNamespace

from minoflux import neural_cli


def test_torch_compile_relaunches_under_utf8_on_windows(monkeypatch) -> None:
    calls: list[tuple[list[str], bool]] = []

    def fake_run(command, *, check):
        calls.append((list(command), check))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(neural_cli.sys, "platform", "win32")
    monkeypatch.setattr(neural_cli.sys, "flags", SimpleNamespace(utf8_mode=0))
    monkeypatch.setattr(neural_cli.sys, "executable", r"C:\Python\python.exe")
    monkeypatch.setattr(neural_cli.subprocess, "run", fake_run)

    result = neural_cli._run_in_utf8_mode_if_needed(
        ["evaluate", "--torch-compile", "--games", "2"]
    )

    assert result == 7
    assert calls == [
        (
            [
                r"C:\Python\python.exe",
                "-X",
                "utf8",
                "-m",
                "minoflux.neural_cli",
                "evaluate",
                "--torch-compile",
                "--games",
                "2",
            ],
            False,
        )
    ]


def test_torch_compile_does_not_relaunch_when_utf8_is_already_enabled(monkeypatch) -> None:
    monkeypatch.setattr(neural_cli.sys, "platform", "win32")
    monkeypatch.setattr(neural_cli.sys, "flags", SimpleNamespace(utf8_mode=1))

    assert neural_cli._run_in_utf8_mode_if_needed(["evaluate", "--torch-compile"]) is None


def test_non_compile_commands_do_not_relaunch(monkeypatch) -> None:
    monkeypatch.setattr(neural_cli.sys, "platform", "win32")
    monkeypatch.setattr(neural_cli.sys, "flags", SimpleNamespace(utf8_mode=0))

    assert neural_cli._run_in_utf8_mode_if_needed(["evaluate"]) is None
