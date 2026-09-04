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

After pulling `self-improve`, install the ML environment and start the agent:

```powershell
uv sync --extra game --extra ml
start-remote-agent.bat
```

Restart the agent after changing `tools/remote_agent.py`; its command whitelist is loaded when the process starts.

The agent uses the existing `.venv` directly. Without a GitHub token it polls once per minute, which stays within the public API rate limit.

Optional: set `MINOFLUX_REMOTE_TOKEN` to a fine-grained token limited to this repository with read-only Issues access. When present, polling defaults to every 10 seconds.

The agent uses existing Git credentials only to push `docs/remote/status.json` to the `remote-status` branch. Its status worktree lives under ignored `data/`.

## Placement v2 commands

- `placement-v2-full-50`
  - Generate 50 × up-to-300-piece Placement Teacher v2 games with depth 2 / beam 24 / 6 workers.
  - Train `data/models/placement-v2.pt` for 8 epochs.
  - Evaluate it and `data/models/neural-value-human.pt` on the same 8 deterministic seeds.
- `placement-v2-generate-50`
  - Only generate `data/neural/placement-v2-ranking.jsonl` with the same teacher settings.
- `placement-v2-train`
  - Train from the existing fixed Placement v2 dataset.
- `placement-v2-evaluate`
  - Evaluate the new model and the current human-model baseline on the same seeds.
- `stop`
  - Terminate the active remote process tree. This also stops a child dataset-generation/training process started by the full pipeline.

The full pipeline deliberately uses fixed arguments. Issue bodies are ignored and arbitrary shell commands are never executed.

## Older accepted commands

- `train-v1`
- `benchmark-v1-20`
- `selfplay-v2-50`
- `train-v2`
- `benchmark-v2-20`

These are retained for reproducibility of the older Versus Value experiments.
