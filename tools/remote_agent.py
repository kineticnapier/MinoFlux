from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

REPOSITORY = "kineticnapier/MinoFlux"
OWNER = "kineticnapier"
CONTROL_BRANCH = "self-improve"
STATUS_BRANCH = "remote-status"
COMMAND_PREFIX = "[MinoFlux Remote] "

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
STATE_PATH = DATA_DIR / "remote-agent-state.json"
LOG_DIR = DATA_DIR / "remote-logs"
STATUS_WORKTREE = DATA_DIR / "remote-status-worktree"
STATUS_RELATIVE_PATH = Path("docs/remote/status.json")

COMMANDS: dict[str, tuple[str, ...]] = {
    "train-v1": (
        "-m", "minoflux.versus_neural_cli", "train",
        "data/neural/versus-selfplay-v1.jsonl",
        "--output", "data/models/versus-value-v1.pt",
        "--epochs", "6",
        "--batch-size", "256",
        "--teacher-weight", "0.25",
        "--device", "auto",
    ),
    "benchmark-v1-20": (
        "-m", "minoflux.versus_neural_cli", "benchmark",
        "--games", "20",
        "--max-turns", "500",
        "--seed-base", "7100001",
        "--seed-step", "31",
        "--player-neural-model", "data/models/neural-value-human.pt",
        "--player-versus-value-model", "data/models/versus-value-v1.pt",
        "--ai-neural-model", "data/models/neural-value-human.pt",
        "--candidate-width", "16",
        "--reply-width", "4",
        "--torch-compile",
        "--device", "auto",
    ),
    "selfplay-v2-50": (
        "-m", "minoflux.versus_neural_cli", "selfplay",
        "--output", "data/neural/versus-selfplay-v2.jsonl",
        "--games", "50",
        "--max-turns", "500",
        "--seed-base", "6200001",
        "--seed-step", "31",
        "--solo-model", "data/models/neural-value-human.pt",
        "--versus-value-model", "data/models/versus-value-v1.pt",
        "--candidate-width", "16",
        "--reply-width", "4",
        "--torch-compile",
        "--device", "auto",
    ),
    "train-v2": (
        "-m", "minoflux.versus_neural_cli", "train",
        "data/neural/versus-selfplay-v2.jsonl",
        "--output", "data/models/versus-value-v2.pt",
        "--resume", "data/models/versus-value-v1.pt",
        "--epochs", "6",
        "--batch-size", "256",
        "--teacher-weight", "0.25",
        "--device", "auto",
    ),
    "benchmark-v2-20": (
        "-m", "minoflux.versus_neural_cli", "benchmark",
        "--games", "20",
        "--max-turns", "500",
        "--seed-base", "7200001",
        "--seed-step", "31",
        "--player-neural-model", "data/models/neural-value-human.pt",
        "--player-versus-value-model", "data/models/versus-value-v2.pt",
        "--ai-neural-model", "data/models/neural-value-human.pt",
        "--ai-versus-value-model", "data/models/versus-value-v1.pt",
        "--candidate-width", "16",
        "--reply-width", "4",
        "--torch-compile",
        "--device", "auto",
    ),
}
STOP_COMMAND = "stop"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _now().isoformat().replace("+00:00", "Z")


def _run_git(*args: str, cwd: Path = PROJECT_ROOT, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _load_state() -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.exists():
        return {"seenIssues": [], "createdAt": _iso_now()}
    try:
        value = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"seenIssues": [], "createdAt": _iso_now()}
    if not isinstance(value, dict):
        return {"seenIssues": [], "createdAt": _iso_now()}
    value.setdefault("seenIssues", [])
    return value


def _save_state(state: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "MinoFlux-Remote-Agent",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("MINOFLUX_REMOTE_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _fetch_open_issues() -> list[dict[str, Any]]:
    url = f"https://api.github.com/repos/{REPOSITORY}/issues?state=open&per_page=100&sort=created&direction=asc"
    request = Request(url, headers=_headers())
    try:
        with urlopen(request, timeout=20) as response:
            payload = json.load(response)
    except HTTPError as error:
        remaining = error.headers.get("X-RateLimit-Remaining")
        raise RuntimeError(f"GitHub API HTTP {error.code}; rate-limit remaining={remaining}") from error
    except (URLError, TimeoutError) as error:
        raise RuntimeError(f"GitHub API request failed: {error}") from error
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected GitHub Issues response")
    return [item for item in payload if isinstance(item, dict)]


def _issue_command(issue: dict[str, Any]) -> str | None:
    if "pull_request" in issue:
        return None
    user = issue.get("user")
    if not isinstance(user, dict) or user.get("login") != OWNER:
        return None
    title = issue.get("title")
    if not isinstance(title, str) or not title.startswith(COMMAND_PREFIX):
        return None
    command = title[len(COMMAND_PREFIX):].strip()
    if command not in COMMANDS and command != STOP_COMMAND:
        return None
    return command


def _created_at(issue: dict[str, Any]) -> datetime | None:
    raw = issue.get("created_at")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _pending_commands(state: dict[str, Any], *, include_stop: bool = True) -> list[tuple[int, str, dict[str, Any]]]:
    seen = {int(value) for value in state.get("seenIssues", []) if isinstance(value, int)}
    cutoff = _now() - timedelta(hours=24)
    pending: list[tuple[int, str, dict[str, Any]]] = []
    for issue in _fetch_open_issues():
        number = issue.get("number")
        if not isinstance(number, int) or number in seen:
            continue
        command = _issue_command(issue)
        if command is None or (command == STOP_COMMAND and not include_stop):
            continue
        created = _created_at(issue)
        if created is not None and created < cutoff:
            seen.add(number)
            continue
        pending.append((number, command, issue))
    state["seenIssues"] = sorted(seen)
    return pending


def _mark_seen(state: dict[str, Any], issue_number: int) -> None:
    seen = {int(value) for value in state.get("seenIssues", []) if isinstance(value, int)}
    seen.add(int(issue_number))
    state["seenIssues"] = sorted(seen)
    _save_state(state)


def _ensure_status_worktree() -> None:
    status_file = STATUS_WORKTREE / STATUS_RELATIVE_PATH
    if (STATUS_WORKTREE / ".git").exists() or (STATUS_WORKTREE / ".git").is_file():
        return

    STATUS_WORKTREE.parent.mkdir(parents=True, exist_ok=True)
    _run_git("fetch", "origin", STATUS_BRANCH)
    result = _run_git(
        "worktree",
        "add",
        "-B",
        STATUS_BRANCH,
        str(STATUS_WORKTREE),
        f"origin/{STATUS_BRANCH}",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not prepare status worktree:\n{result.stdout}")
    status_file.parent.mkdir(parents=True, exist_ok=True)


def _publish_status(payload: dict[str, Any]) -> None:
    try:
        _ensure_status_worktree()
        _run_git("pull", "--ff-only", "origin", STATUS_BRANCH, cwd=STATUS_WORKTREE)
        target = STATUS_WORKTREE / STATUS_RELATIVE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _run_git("add", str(STATUS_RELATIVE_PATH).replace("\\", "/"), cwd=STATUS_WORKTREE)
        diff = _run_git("diff", "--cached", "--quiet", cwd=STATUS_WORKTREE, check=False)
        if diff.returncode == 0:
            return
        commit = _run_git(
            "-c", "user.name=MinoFlux Remote",
            "-c", "user.email=minoflux-remote@users.noreply.github.com",
            "commit", "-m", "Update remote status",
            cwd=STATUS_WORKTREE,
            check=False,
        )
        if commit.returncode != 0:
            raise RuntimeError(commit.stdout)
        pushed = _run_git("push", "origin", STATUS_BRANCH, cwd=STATUS_WORKTREE, check=False)
        if pushed.returncode != 0:
            raise RuntimeError(pushed.stdout)
    except Exception as error:
        print(f"[remote] status publish failed: {error}", file=sys.stderr)


def _tail_log(path: Path, limit: int = 18) -> list[str]:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 32768), os.SEEK_SET)
            raw = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    parts = [part.strip() for part in re.split(r"[\r\n]+", raw) if part.strip()]
    return parts[-limit:]


def _sync_code() -> None:
    fetched = _run_git("fetch", "origin", CONTROL_BRANCH, check=False)
    if fetched.returncode != 0:
        raise RuntimeError(f"git fetch failed:\n{fetched.stdout}")
    merged = _run_git("merge", "--ff-only", f"origin/{CONTROL_BRANCH}", check=False)
    if merged.returncode != 0:
        raise RuntimeError(
            "Could not fast-forward the local self-improve branch. "
            "Resolve local Git changes before using remote control.\n"
            + merged.stdout
        )


def _python_executable() -> Path:
    candidate = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not candidate.exists():
        raise RuntimeError(f"Virtualenv Python not found: {candidate}")
    return candidate


def _terminate_process_tree(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def _check_stop(state: dict[str, Any]) -> int | None:
    try:
        pending = _pending_commands(state, include_stop=True)
    except RuntimeError:
        return None
    for number, command, _issue in pending:
        if command == STOP_COMMAND:
            _mark_seen(state, number)
            return number
    return None


def _execute(issue_number: int, command: str, state: dict[str, Any], poll_seconds: float) -> None:
    started = _iso_now()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{stamp}-{command}.log"

    base = {
        "agentOnline": True,
        "command": command,
        "issue": issue_number,
        "startedAt": started,
        "updatedAt": started,
        "logTail": [],
    }
    _publish_status({**base, "state": "preparing", "message": "Syncing self-improve"})

    try:
        _sync_code()
        python = _python_executable()
        argv = [str(python), *COMMANDS[command]]
    except Exception as error:
        _publish_status({
            **base,
            "state": "error",
            "updatedAt": _iso_now(),
            "finishedAt": _iso_now(),
            "message": str(error),
        })
        return

    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            argv,
            cwd=PROJECT_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        _publish_status({
            **base,
            "state": "running",
            "pid": process.pid,
            "updatedAt": _iso_now(),
            "message": "Running",
        })

        last_publish = 0.0
        last_stop_check = 0.0
        stopped_by: int | None = None
        status_interval = 20.0
        stop_interval = max(10.0, poll_seconds)

        while process.poll() is None:
            now = time.monotonic()
            if now - last_publish >= status_interval:
                _publish_status({
                    **base,
                    "state": "running",
                    "pid": process.pid,
                    "updatedAt": _iso_now(),
                    "message": "Running",
                    "logTail": _tail_log(log_path),
                })
                last_publish = now

            if now - last_stop_check >= stop_interval:
                stopped_by = _check_stop(state)
                last_stop_check = now
                if stopped_by is not None:
                    _terminate_process_tree(process)
                    break
            time.sleep(2.0)

        try:
            exit_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            exit_code = process.wait(timeout=10)

    finished = _iso_now()
    log_tail = _tail_log(log_path)
    if stopped_by is not None:
        status = "stopped"
        message = f"Stopped by issue #{stopped_by}"
    elif exit_code == 0:
        status = "done"
        message = "Completed successfully"
    else:
        status = "error"
        message = f"Process exited with code {exit_code}"

    _publish_status({
        **base,
        "state": status,
        "pid": None,
        "updatedAt": finished,
        "finishedAt": finished,
        "exitCode": exit_code,
        "message": message,
        "logPath": str(log_path.relative_to(PROJECT_ROOT)),
        "logTail": log_tail,
    })


def _idle_status(message: str = "Waiting for a command") -> dict[str, Any]:
    return {
        "agentOnline": True,
        "state": "idle",
        "command": None,
        "issue": None,
        "pid": None,
        "updatedAt": _iso_now(),
        "message": message,
        "logTail": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Poll GitHub Issues for fixed MinoFlux remote commands")
    parser.add_argument("--once", action="store_true", help="Poll once and exit when there is no command")
    parser.add_argument("--poll-seconds", type=float, default=None)
    args = parser.parse_args()

    token = bool(os.environ.get("MINOFLUX_REMOTE_TOKEN", "").strip())
    poll_seconds = args.poll_seconds if args.poll_seconds is not None else (10.0 if token else 60.0)
    poll_seconds = max(5.0, float(poll_seconds))

    state = _load_state()
    _publish_status(_idle_status())
    print(f"[remote] watching {REPOSITORY}; poll={poll_seconds:.0f}s; token={'yes' if token else 'no'}")

    while True:
        try:
            pending = _pending_commands(state, include_stop=False)
            _save_state(state)
        except Exception as error:
            print(f"[remote] poll failed: {error}", file=sys.stderr)
            _publish_status(_idle_status(f"Poll error: {error}"))
            if args.once:
                return 1
            time.sleep(poll_seconds)
            continue

        if pending:
            number, command, _issue = pending[0]
            _mark_seen(state, number)
            print(f"[remote] issue #{number}: {command}")
            _execute(number, command, state, poll_seconds)
            _publish_status(_idle_status(f"Last command finished: {command}"))
            if args.once:
                return 0
            continue

        if args.once:
            print("[remote] no pending command")
            return 0

        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
