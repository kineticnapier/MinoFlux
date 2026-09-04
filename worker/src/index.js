const ALLOWED_COMMANDS = new Set([
  "train-v1",
  "benchmark-v1-20",
  "selfplay-v2-50",
  "train-v2",
  "benchmark-v2-20",
  "placement-v2-full-50",
  "placement-v2-generate-50",
  "placement-v2-train",
  "placement-v2-evaluate",
  "stop",
]);

function json(data, status = 200, extraHeaders = {}) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "content-type": "application/json; charset=utf-8",
      "cache-control": "no-store",
      ...extraHeaders,
    },
  });
}

function bearerToken(request) {
  const header = request.headers.get("authorization") || "";
  return header.startsWith("Bearer ") ? header.slice(7) : "";
}

function sameToken(left, right) {
  if (typeof left !== "string" || typeof right !== "string") return false;
  if (left.length !== right.length || left.length === 0) return false;
  let diff = 0;
  for (let i = 0; i < left.length; i += 1) {
    diff |= left.charCodeAt(i) ^ right.charCodeAt(i);
  }
  return diff === 0;
}

function authorized(request, expected) {
  return sameToken(bearerToken(request), expected || "");
}

function remoteStub(env) {
  const id = env.REMOTE.idFromName("singleton");
  return env.REMOTE.get(id);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/health") {
      return json({ ok: true, service: "minoflux-remote" });
    }

    if (url.pathname.startsWith("/api/agent/")) {
      if (!env.AGENT_TOKEN) return json({ error: "AGENT_TOKEN is not configured" }, 503);
      if (!authorized(request, env.AGENT_TOKEN)) return json({ error: "unauthorized" }, 401);
      return remoteStub(env).fetch(request);
    }

    if (url.pathname.startsWith("/api/")) {
      return remoteStub(env).fetch(request);
    }

    return env.ASSETS.fetch(request);
  },
};

export class RemoteHub {
  constructor(state) {
    this.state = state;
    this.waiters = new Set();
  }

  async fetch(request) {
    const url = new URL(request.url);

    if (url.pathname === "/api/status" && request.method === "GET") {
      return this.getStatus();
    }
    if (url.pathname === "/api/command" && request.method === "POST") {
      return this.enqueueCommand(request);
    }
    if (url.pathname === "/api/agent/poll" && request.method === "GET") {
      return this.poll(request);
    }
    if (url.pathname === "/api/agent/ack" && request.method === "POST") {
      return this.ack(request);
    }
    if (url.pathname === "/api/agent/status" && request.method === "POST") {
      return this.updateStatus(request);
    }

    return json({ error: "not found" }, 404);
  }

  async getStatus() {
    const [status, lastAgentSeen, pending] = await Promise.all([
      this.state.storage.get("status"),
      this.state.storage.get("lastAgentSeen"),
      this.state.storage.get("pending"),
    ]);

    const seenMs = lastAgentSeen ? Date.parse(lastAgentSeen) : Number.NaN;
    const agentOnline = Number.isFinite(seenMs) && Date.now() - seenMs < 40_000;

    return json({
      agentOnline,
      state: status?.state || "idle",
      command: status?.command || null,
      pid: status?.pid ?? null,
      message: status?.message || (agentOnline ? "Waiting for a command" : "Agent offline"),
      logTail: Array.isArray(status?.logTail) ? status.logTail : [],
      updatedAt: status?.updatedAt || lastAgentSeen || null,
      lastAgentSeen: lastAgentSeen || null,
      queuedCommand: pending?.command || null,
      queuedCommandId: pending?.id || null,
    });
  }

  async enqueueCommand(request) {
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid json" }, 400);
    }

    const command = typeof body?.command === "string" ? body.command.trim() : "";
    if (!ALLOWED_COMMANDS.has(command)) {
      return json({ error: "command is not whitelisted" }, 400);
    }

    const [pending, status] = await Promise.all([
      this.state.storage.get("pending"),
      this.state.storage.get("status"),
    ]);

    if (pending && command !== "stop") {
      return json({ error: "a command is already queued", pending }, 409);
    }
    if (command !== "stop" && ["preparing", "running"].includes(status?.state)) {
      return json({ error: "agent is busy", state: status.state, command: status.command }, 409);
    }

    const queued = {
      id: crypto.randomUUID(),
      command,
      createdAt: new Date().toISOString(),
    };
    await this.state.storage.put("pending", queued);
    this.wakeWaiters();
    return json({ ok: true, queued, replaced: Boolean(pending) }, 202);
  }

  async poll(request) {
    const url = new URL(request.url);
    const requestedWait = Number(url.searchParams.get("wait") || "25");
    const waitSeconds = Math.max(0, Math.min(25, Number.isFinite(requestedWait) ? requestedWait : 25));
    const now = new Date().toISOString();
    await this.state.storage.put("lastAgentSeen", now);

    const pending = await this.state.storage.get("pending");
    if (pending || waitSeconds === 0) {
      return json({ command: pending || null, serverTime: now });
    }

    return new Promise((resolve) => {
      let finished = false;
      const waiter = async () => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        this.waiters.delete(waiter);
        const current = await this.state.storage.get("pending");
        resolve(json({ command: current || null, serverTime: new Date().toISOString() }));
      };
      const timer = setTimeout(() => {
        if (finished) return;
        finished = true;
        this.waiters.delete(waiter);
        resolve(json({ command: null, serverTime: new Date().toISOString() }));
      }, waitSeconds * 1000);
      this.waiters.add(waiter);
    });
  }

  async ack(request) {
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid json" }, 400);
    }

    const id = typeof body?.id === "string" ? body.id : "";
    const pending = await this.state.storage.get("pending");
    if (!pending) return json({ ok: true, cleared: false });
    if (!id || pending.id !== id) return json({ error: "command id mismatch" }, 409);

    await this.state.storage.delete("pending");
    return json({ ok: true, cleared: true });
  }

  async updateStatus(request) {
    let body;
    try {
      body = await request.json();
    } catch {
      return json({ error: "invalid json" }, 400);
    }

    const now = new Date().toISOString();
    const status = {
      state: typeof body?.state === "string" ? body.state : "idle",
      command: typeof body?.command === "string" ? body.command : null,
      pid: Number.isInteger(body?.pid) ? body.pid : null,
      message: typeof body?.message === "string" ? body.message : "",
      logTail: Array.isArray(body?.logTail)
        ? body.logTail.filter((line) => typeof line === "string").slice(-24)
        : [],
      updatedAt: now,
    };

    await Promise.all([
      this.state.storage.put("status", status),
      this.state.storage.put("lastAgentSeen", now),
    ]);
    return json({ ok: true, updatedAt: now });
  }

  wakeWaiters() {
    for (const waiter of [...this.waiters]) waiter();
  }
}
