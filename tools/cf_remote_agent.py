from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from remote_agent import (
    COMMANDS,
    LOG_DIR,
    PROJECT_ROOT,
    _python_executable,
    _sync_code,
    _tail_log,
    _terminate_process_tree,
)

RESULT_PATH = PROJECT_ROOT / "data" / "remote" / "latest-placement-evaluation.json"
RESULT_COMMANDS = {"placement-v2-evaluate", "placement-v2-full-50"}


@dataclass
class ActiveTask:
    command: str
    process: subprocess.Popen[Any]
    log_path: Path
    log_handle: Any


def _load_latest_result() -> dict[str, Any] | None:
    try:
        value = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class CloudflareRemoteAgent:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.active: ActiveTask | None = None
        self.last_status_publish = 0.0

    def _request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout: float = 35.0,
    ) -> dict[str, Any]:
        data = None
        headers = {
            "authorization": f"Bearer {self.token}",
            "accept": "application/json",
            "user-agent": "MinoFlux-CF-Remote-Agent",
        }
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"
        request = Request(
            self.base_url + path,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as error:
            try:
                detail = error.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(f"HTTP {error.code}: {detail or error.reason}") from error
        except (URLError, TimeoutError) as error:
            raise RuntimeError(f"request failed: {error}") from error
        if not raw:
            return {}
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("unexpected non-object response")
        return value

    def publish_status(
        self,
        state: str,
        *,
        command: str | None = None,
        pid: int | None = None,
        message: str = "",
        log_tail: list[str] | None = None,
        result: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "state": state,
            "command": command,
            "pid": pid,
            "message": message,
            "logTail": log_tail or [],
        }
        if result is not None:
            payload["result"] = result
        self._request_json(
            "/api/agent/status",
            method="POST",
            payload=payload,
            timeout=15.0,
        )
        self.last_status_publish = time.monotonic()

    def poll(self, wait_seconds: float) -> dict[str, Any] | None:
        query = urlencode({"wait": max(0.0, min(25.0, wait_seconds))})
        result = self._request_json(
            f"/api/agent/poll?{query}",
            timeout=max(10.0, wait_seconds + 10.0),
        )
        command = result.get("command")
        return command if isinstance(command, dict) else None

    def ack(self, command_id: str) -> None:
        self._request_json(
            "/api/agent/ack",
            method="POST",
            payload={"id": command_id},
            timeout=15.0,
        )

    def start_task(self, command: str) -> None:
        if command not in COMMANDS:
            raise RuntimeError(f"command is not locally whitelisted: {command}")

        self.publish_status("preparing", command=command, message="Syncing self-improve")
        _sync_code()
        python = _python_executable()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if command in RESULT_COMMANDS:
            try:
                RESULT_PATH.unlink()
            except FileNotFoundError:
                pass
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = LOG_DIR / f"{stamp}-cf-{command}.log"
        log_handle = log_path.open("wb")
        try:
            process = subprocess.Popen(
                [str(python), *COMMANDS[command]],
                cwd=PROJECT_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
        except Exception:
            log_handle.close()
            raise

        self.active = ActiveTask(command, process, log_path, log_handle)
        self.publish_status(
            "running",
            command=command,
            pid=process.pid,
            message="Running",
        )
        print(f"[cf-remote] started {command}; pid={process.pid}; log={log_path}")

    def stop_task(self) -> None:
        task = self.active
        if task is None:
            self.publish_status("idle", message="Stop requested, but no task is running")
            return
        _terminate_process_tree(task.process)
        try:
            exit_code = task.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(task.process)
            exit_code = task.process.wait(timeout=10)
        task.log_handle.close()
        log_tail = _tail_log(task.log_path)
        self.publish_status(
            "stopped",
            command=task.command,
            message=f"Stopped; exit={exit_code}",
            log_tail=log_tail,
        )
        print(f"[cf-remote] stopped {task.command}; exit={exit_code}")
        self.active = None

    def finish_if_done(self) -> None:
        task = self.active
        if task is None:
            return
        exit_code = task.process.poll()
        if exit_code is None:
            return
        task.log_handle.close()
        log_tail = _tail_log(task.log_path)
        state = "done" if exit_code == 0 else "error"
        message = "Completed successfully" if exit_code == 0 else f"Process exited with code {exit_code}"
        result = (
            _load_latest_result()
            if exit_code == 0 and task.command in RESULT_COMMANDS
            else None
        )
        self.publish_status(
            state,
            command=task.command,
            message=message,
            log_tail=log_tail,
            result=result,
        )
        print(f"[cf-remote] {task.command} finished; exit={exit_code}")
        self.active = None

    def publish_running_status(self, *, force: bool = False) -> None:
        task = self.active
        if task is None:
            return
        if not force and time.monotonic() - self.last_status_publish < 2.0:
            return
        self.publish_status(
            "running",
            command=task.command,
            pid=task.process.pid,
            message="Running",
            log_tail=_tail_log(task.log_path),
        )

    def handle_remote_command(self, envelope: dict[str, Any]) -> None:
        command_id = envelope.get("id")
        command = envelope.get("command")
        if not isinstance(command_id, str) or not isinstance(command, str):
            print("[cf-remote] malformed command envelope", file=sys.stderr)
            return

        # Ack only after validating the envelope. This prevents a malformed response
        # from silently deleting a real queued command.
        if command != "stop" and command not in COMMANDS:
            print(f"[cf-remote] refusing unknown command: {command}", file=sys.stderr)
            return
        self.ack(command_id)

        if command == "stop":
            self.stop_task()
            return
        if self.active is not None:
            task = self.active
            self.publish_status(
                "running",
                command=task.command,
                pid=task.process.pid,
                message=f"Ignored {command}: agent is busy",
                log_tail=_tail_log(task.log_path),
            )
            return

        try:
            self.start_task(command)
        except Exception as error:
            self.publish_status("error", command=command, message=str(error))
            print(f"[cf-remote] could not start {command}: {error}", file=sys.stderr)

    def run(self) -> int:
        reconnect_delay = 1.0
        while True:
            try:
                self.finish_if_done()
                if self.active is None:
                    if time.monotonic() - self.last_status_publish > 20.0:
                        self.publish_status("idle", message="Waiting for a command")
                    envelope = self.poll(25.0)
                else:
                    self.publish_running_status()
                    envelope = self.poll(2.0)
                    self.finish_if_done()
                if envelope is not None:
                    self.handle_remote_command(envelope)
                reconnect_delay = 1.0
            except KeyboardInterrupt:
                print("\n[cf-remote] exiting")
                return 0
            except Exception as error:
                print(f"[cf-remote] relay error: {error}; retrying in {reconnect_delay:.0f}s", file=sys.stderr)
                time.sleep(reconnect_delay)
                reconnect_delay = min(15.0, reconnect_delay * 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="MinoFlux Cloudflare remote command agent")
    parser.add_argument("--url", default=os.environ.get("MINOFLUX_CF_REMOTE_URL", ""))
    parser.add_argument("--token", default=os.environ.get("MINOFLUX_CF_AGENT_TOKEN", ""))
    args = parser.parse_args()

    url = args.url.strip()
    token = args.token.strip()
    if not url:
        raise SystemExit("MINOFLUX_CF_REMOTE_URL (or --url) is required")
    if not token:
        raise SystemExit("MINOFLUX_CF_AGENT_TOKEN (or --token) is required")
    if not url.startswith(("https://", "http://")):
        raise SystemExit("remote URL must start with https:// or http://")

    print(f"[cf-remote] connecting to {url.rstrip('/')}")
    return CloudflareRemoteAgent(url, token).run()


if __name__ == "__main__":
    raise SystemExit(main())
