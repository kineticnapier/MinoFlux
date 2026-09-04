# MinoFlux Cloudflare Remote

Cloudflare is the preferred low-latency remote-control path. The old GitHub-Issue polling agent remains available as a fallback.

## Architecture

- The dashboard is served directly by the Worker.
- The dashboard sends only fixed whitelisted command names to the Worker.
- A singleton Durable Object stores status and at most one queued command.
- The home PC keeps an outbound long-poll connection to Cloudflare. No inbound port is opened.
- Idle command latency is normally sub-second because the pending long-poll request is completed as soon as a command is queued.
- While a task is running, the agent polls every two seconds so `stop` remains responsive and publishes the latest log tail.
- Arbitrary shell commands are never accepted.

## GitHub Actions deploy

Repository secrets:

- `CLOUDFLARE_API_TOKEN`
- `CLOUDFLARE_ACCOUNT_ID`

The API token only needs permission to deploy Workers for the selected Cloudflare account. The workflow automatically skips deployment instead of failing when either secret is absent.

Worker runtime secrets are intentionally not stored in the repository. Configure these once in Cloudflare after the first deploy:

- `COMMAND_TOKEN`: entered in the teacher-iPad dashboard; authorizes `/api/status` and `/api/command`.
- `AGENT_TOKEN`: stored only on the home PC; authorizes `/api/agent/*`.

For example, from `worker/` with Wrangler authenticated:

```text
npx wrangler secret put COMMAND_TOKEN
npx wrangler secret put AGENT_TOKEN
```

The same values can be created in the Cloudflare dashboard instead.

## Home PC setup

After the Worker is deployed and its runtime secrets exist, set:

```powershell
setx MINOFLUX_CF_REMOTE_URL "https://minoflux-remote.YOUR_SUBDOMAIN.workers.dev"
setx MINOFLUX_CF_AGENT_TOKEN "THE_SAME_VALUE_AS_WORKER_AGENT_TOKEN"
```

Open a new PowerShell window, pull `self-improve`, then run:

```powershell
git pull
uv sync --extra game --extra ml
.\start-cf-remote-agent.bat
```

The agent uses only Python's standard library for the relay connection, so no extra Python dependency is required.

## Dashboard

Open the Worker URL on the teacher iPad. Enter `COMMAND_TOKEN` once and press `保存して接続`. The token is saved only to that browser's localStorage; it is not embedded in the HTML.

The dashboard refreshes status every two seconds. Commands are delivered through the home agent's held Cloudflare request, so command start does not wait for the dashboard refresh interval.

## Fixed commands

- `placement-v2-full-50`
- `placement-v2-generate-50`
- `placement-v2-train`
- `placement-v2-evaluate`
- `train-v1`
- `benchmark-v1-20`
- `selfplay-v2-50`
- `train-v2`
- `benchmark-v2-20`
- `stop`

Before starting a task the home agent fast-forwards the local `self-improve` branch, matching the previous remote-agent behavior.
