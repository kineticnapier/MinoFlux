# MinoFlux Remote

A small outbound-only remote control path for the MinoFlux development machine.

Flow:

1. Open the GitHub Pages dashboard on an iPad.
2. Tap a fixed command button.
3. GitHub opens a pre-filled Issue. Submit it without changing the title.
4. `tools/remote_agent.py` polls public Issues and accepts only exact whitelisted titles created by `kineticnapier`.
5. The home PC runs the corresponding fixed MinoFlux command.
6. The agent publishes progress to the `remote-status` branch; the dashboard refreshes it every 10 seconds.

No inbound port is opened on the home PC, and the dashboard never stores a shell command or GitHub secret.

## One-time home PC setup

After pulling `self-improve`, start:

```powershell
start-remote-agent.bat
```

The agent uses the existing `.venv` directly. Without a GitHub token it polls once per minute, which stays within the public API rate limit.

Optional: set `MINOFLUX_REMOTE_TOKEN` to a fine-grained token limited to this repository with read-only Issues access. When present, polling defaults to every 10 seconds.

The agent uses existing Git credentials only to push `docs/remote/status.json` to the `remote-status` branch. Its status worktree lives under ignored `data/`.

## Accepted commands

- `train-v1`
- `benchmark-v1-20`
- `selfplay-v2-50`
- `train-v2`
- `benchmark-v2-20`
- `stop`

The Issue body is ignored. Arbitrary shell commands are never executed.
