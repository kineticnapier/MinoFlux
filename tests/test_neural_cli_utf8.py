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


def test_evaluate_parser_accepts_profile_and_dynamic_graph_skip() -> None:
    args = neural_cli.build_parser().parse_args(
        [
            "evaluate",
            "--torch-compile",
            "--torch-compile-skip-dynamic-graphs",
            "--profile",
        ]
    )

    assert args.torch_compile is True
    assert args.torch_compile_skip_dynamic_graphs is True
    assert args.profile is True


def test_profiled_scorer_preserves_grouped_values() -> None:
    class FakeEvaluator:
        def score_placement_groups(self, groups):
            return tuple(tuple(float(index) for index, _placement in enumerate(placements)) for _game, placements in groups)

    scorer = neural_cli._ProfiledNeuralScorer(FakeEvaluator())
    groups = ((object(), (object(), object())), (object(), (object(),)))

    assert scorer.score_placement_groups(groups) == ((0.0, 1.0), (0.0,))
    assert scorer.calls == 1
    assert scorer.groups == 2
    assert scorer.states == 3
    assert scorer.seconds >= 0.0
