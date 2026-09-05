from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DATASET = Path("data/neural/placement-v2-ranking.jsonl")
MODEL = Path("data/models/placement-v2.pt")
RESULT_PATH = Path("data/remote/latest-placement-evaluation.json")


def _run(label: str, *args: str) -> None:
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join((str(sys.executable), *args)), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def _run_json(label: str, *args: str) -> dict[str, Any]:
    """Run a child command, preserve stderr progress, and parse its JSON stdout."""
    print(f"\n=== {label} ===", flush=True)
    print("$ " + " ".join((str(sys.executable), *args)), flush=True)
    completed = subprocess.run(
        [sys.executable, *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    raw = completed.stdout.strip()
    if raw:
        print(raw, flush=True)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} did not return JSON on stdout") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} returned non-object JSON")
    return value


def _generate(*, games: int, max_pieces: int, teacher_depth: int, teacher_beam: int, workers: int) -> None:
    _run(
        "placement-v2 teacher dataset",
        "-m",
        "minoflux.placement_v2_cli",
        "generate",
        "--output",
        str(DATASET),
        "--games",
        str(games),
        "--max-pieces",
        str(max_pieces),
        "--seed-base",
        "8000001",
        "--seed-step",
        "31",
        "--max-candidates",
        "24",
        "--teacher-depth",
        str(teacher_depth),
        "--teacher-beam",
        str(teacher_beam),
        "--workers",
        str(workers),
    )


def _train(*, epochs: int) -> None:
    if not DATASET.is_file():
        raise SystemExit(f"Dataset not found: {DATASET}")
    _run(
        "placement-v2 neural distillation",
        "-m",
        "minoflux.neural_cli",
        "train",
        "--dataset",
        str(DATASET),
        "--output",
        str(MODEL),
        "--epochs",
        str(epochs),
        "--batch-size",
        "64",
        "--learning-rate",
        "3e-4",
        "--teacher-weight",
        "0.5",
        "--rollout-weight",
        "0",
        "--device",
        "auto",
    )


def _evaluate(model: Path, label: str) -> dict[str, Any]:
    if not model.is_file():
        raise SystemExit(f"Model not found: {model}")
    return _run_json(
        label,
        "-m",
        "minoflux.neural_cli",
        "evaluate",
        "--model",
        str(model),
        "--games",
        "8",
        "--max-pieces",
        "300",
        "--seed-base",
        "8100001",
        "--seed-step",
        "97",
        "--game-batch-size",
        "8",
        "--device",
        "auto",
    )


def _summary_line(label: str, result: dict[str, Any]) -> str:
    return (
        f"{label}: pieces={result.get('pieces')} attack={result.get('attack')} "
        f"APP={result.get('attackPerPiece')} topouts={result.get('topouts')} "
        f"completed={result.get('completed')}"
    )


def _publish_cloudflare_result(payload: dict[str, Any]) -> None:
    """Publish structured results directly to the Worker when run by the CF agent."""
    base_url = os.environ.get("MINOFLUX_CF_REMOTE_URL", "").strip().rstrip("/")
    token = os.environ.get("MINOFLUX_CF_AGENT_TOKEN", "").strip()
    if not base_url or not token:
        return

    body = {
        "state": "done",
        "command": payload.get("command"),
        "message": "Placement-v2 evaluation completed",
        "logTail": [
            _summary_line("placement-v2", payload["placementV2"]),
            _summary_line("baseline", payload["baseline"]),
        ],
        "result": payload,
    }
    request = Request(
        base_url + "/api/agent/status",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "MinoFlux-Placement-v2-Remote",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            response.read()
    except (HTTPError, URLError, TimeoutError) as error:
        print(f"warning: could not publish result to Cloudflare: {error}", file=sys.stderr, flush=True)
        return
    print("Published structured result directly to Cloudflare", flush=True)


def _write_evaluation_result(
    *,
    command: str,
    placement_v2: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Save locally and, for remote runs, publish directly through Cloudflare."""
    payload = {
        "command": command,
        "updatedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "placementV2": placement_v2,
        "baseline": baseline,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT_PATH.with_suffix(RESULT_PATH.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(RESULT_PATH)
    print(_summary_line("placement-v2", placement_v2), flush=True)
    print(_summary_line("baseline", baseline), flush=True)
    print(f"Saved structured result: {RESULT_PATH}", flush=True)
    _publish_cloudflare_result(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed Placement-v2 jobs used by the MinoFlux remote agent"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    full = sub.add_parser("full", help="Generate, train, then compare Placement-v2")
    full.add_argument("--games", type=int, default=50)
    full.add_argument("--max-pieces", type=int, default=300)
    full.add_argument("--teacher-depth", type=int, default=2)
    full.add_argument("--teacher-beam", type=int, default=24)
    full.add_argument("--workers", type=int, default=6)
    full.add_argument("--epochs", type=int, default=8)

    generate = sub.add_parser("generate", help="Generate the fixed Placement-v2 dataset")
    generate.add_argument("--games", type=int, default=50)
    generate.add_argument("--max-pieces", type=int, default=300)
    generate.add_argument("--teacher-depth", type=int, default=2)
    generate.add_argument("--teacher-beam", type=int, default=24)
    generate.add_argument("--workers", type=int, default=6)

    train = sub.add_parser("train", help="Train Placement-v2 from the fixed dataset")
    train.add_argument("--epochs", type=int, default=8)

    sub.add_parser("evaluate", help="Evaluate Placement-v2 and the current human model")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"full", "generate"}:
        _generate(
            games=args.games,
            max_pieces=args.max_pieces,
            teacher_depth=args.teacher_depth,
            teacher_beam=args.teacher_beam,
            workers=args.workers,
        )
        if args.command == "generate":
            return 0

    if args.command in {"full", "train"}:
        _train(epochs=args.epochs)
        if args.command == "train":
            return 0

    if args.command in {"full", "evaluate"}:
        placement_v2 = _evaluate(MODEL, "placement-v2 evaluation")
        baseline = _evaluate(Path("data/models/neural-value-human.pt"), "current human-model baseline")
        _write_evaluation_result(
            command=(
                "placement-v2-full-50"
                if args.command == "full"
                else "placement-v2-evaluate"
            ),
            placement_v2=placement_v2,
            baseline=baseline,
        )
        return 0

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
