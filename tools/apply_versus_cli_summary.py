from __future__ import annotations

from pathlib import Path


CLI = Path("src/minoflux/versus_neural_cli.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = CLI.read_text(encoding="utf-8")
    if "def _benchmark_summary(" not in text:
        text = replace_once(
            text,
            '''def _execution_note(payload: dict[str, object]) -> str:
    if payload.get("skipped"):
        return f"skipped: {payload.get('reason', 'skipped')}"
    if payload.get("reused"):
        return f"reused from {payload.get('reusedFrom', '?')}"
    return "executed"


''',
            '''def _execution_note(payload: dict[str, object]) -> str:
    if payload.get("skipped"):
        return f"skipped: {payload.get('reason', 'skipped')}"
    if payload.get("reused"):
        return f"reused from {payload.get('reusedFrom', '?')}"
    return "executed"


def _benchmark_summary(report: dict[str, object]) -> str:
    player_policy_wins = int(report.get("playerPolicyWins", report.get("playerWins", 0)))
    ai_policy_wins = int(report.get("aiPolicyWins", report.get("aiWins", 0)))
    draws = int(report.get("draws", 0))
    physical_player_wins = int(report.get("physicalPlayerWins", 0))
    physical_ai_wins = int(report.get("physicalAiWins", 0))
    seed_count = int(report.get("seedCount", 0))
    mirrored_games = int(report.get("mirroredGameCount", 0))
    return (
        "Benchmark policies: "
        f"player {player_policy_wins} - ai {ai_policy_wins} - draws {draws}; "
        "physical sides: "
        f"player {physical_player_wins} - ai {physical_ai_wins}; "
        f"seeds {seed_count}, mirrored legs {mirrored_games}"
    )


''',
            label="benchmark summary helper",
        )
    if "print(_benchmark_summary(result), file=sys.stderr)" not in text:
        text = replace_once(
            text,
            '''    if profile is not None:
        result["versusProfile"] = profile.to_dict()
        print(profile.format_table(), file=sys.stderr)
    _print_report(
        result,
        print_json=bool(args.print_json),
        pretty_json=bool(args.pretty_json),
    )
''',
            '''    if profile is not None:
        result["versusProfile"] = profile.to_dict()
        print(profile.format_table(), file=sys.stderr)
    if not args.print_json and not args.pretty_json:
        print(_benchmark_summary(result), file=sys.stderr)
    _print_report(
        result,
        print_json=bool(args.print_json),
        pretty_json=bool(args.pretty_json),
    )
''',
            label="benchmark final summary",
        )
    CLI.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
