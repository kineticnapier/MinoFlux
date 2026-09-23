from __future__ import annotations

from argparse import ArgumentParser
import json
import subprocess
import sys


def resolve_model(value: str) -> str:
    """Resolve a short native-oracle model alias to its checkpoint path."""

    text = str(value)
    if text.lower() == "base":
        return "data/models/native-oracle-value.pt"
    if text.lower().endswith(".pt") or "/" in text or "\\" in text:
        return text
    return f"data/models/native-oracle-value-{text}.pt"


def format_duel_summary(left: str, right: str, result: dict[str, object]) -> str:
    left_wins = int(result.get("playerPolicyWins", result.get("playerWins", 0)))
    right_wins = int(result.get("aiPolicyWins", result.get("aiWins", 0)))
    draws = int(result.get("draws", 0))
    left_rate = 100.0 * float(result.get("playerPolicyWinRate", result.get("playerWinRate", 0.0)))
    right_rate = 100.0 * float(result.get("aiPolicyWinRate", result.get("aiWinRate", 0.0)))
    left_app = float(result.get("playerAttackPerPiece", 0.0))
    right_app = float(result.get("aiAttackPerPiece", 0.0))
    left_sent = float(result.get("playerSentPerPiece", 0.0))
    right_sent = float(result.get("aiSentPerPiece", 0.0))
    return (
        f"{left} {left_wins} - {right_wins} {right} (draw {draws})\n"
        f"win {left_rate:.1f}% - {right_rate:.1f}% | "
        f"APP {left_app:.4f} - {right_app:.4f} | "
        f"sent/piece {left_sent:.4f} - {right_sent:.4f}"
    )


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="minoflux-duel",
        description="Run a compact mirrored duel between two solo neural checkpoints",
    )
    parser.add_argument("left", help="Short alias such as r2w, or a checkpoint path")
    parser.add_argument("right", help="Short alias such as r3w, or a checkpoint path")
    parser.add_argument("-n", "--games", type=int, default=256)
    parser.add_argument("-t", "--max-turns", type=int, default=1000)
    parser.add_argument("--seed-base", type=int, default=10_000_001)
    parser.add_argument("--seed-step", type=int, default=31)
    parser.add_argument("--garbage-cap", type=int, default=8)
    parser.add_argument("--game-batch", type=int, default=16)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-180", action="store_true", help="Disable 180-degree rotations")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        "minoflux.versus_neural_cli",
        "benchmark",
        "--games",
        str(max(1, int(args.games))),
        "--max-turns",
        str(max(1, int(args.max_turns))),
        "--seed-base",
        str(int(args.seed_base)),
        "--seed-step",
        str(int(args.seed_step)),
        "--garbage-cap",
        str(max(1, int(args.garbage_cap))),
        "--player-neural-model",
        resolve_model(args.left),
        "--ai-neural-model",
        resolve_model(args.right),
        "--game-batch",
        str(max(1, int(args.game_batch))),
        "--beam",
        str(max(1, int(args.beam))),
        "--lookahead",
        str(max(0, int(args.lookahead))),
        "--device",
        str(args.device),
        "--print-json",
    ]
    if not args.no_180:
        command.append("--allow-180")

    completed = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)

    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        print("duel benchmark produced no JSON output", file=sys.stderr)
        return 2
    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        print(f"failed to parse duel benchmark output: {exc}", file=sys.stderr)
        return 2

    print(format_duel_summary(args.left, args.right, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
