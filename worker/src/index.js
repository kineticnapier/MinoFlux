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
  "placement-baseline-100",
  "stop",
]);

const FULL_LOG_CHUNK_CHARS = 24_000;

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
    if (url.pathname === "/api/result" && request.method === "GET") {
      return this.getResult();
    }
    if (url.pathname === "/api/log" && request.method === "GET") {
      return this.getFullLog();
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
    const [status, lastAgentSeen, pending, fullLogMeta] = await Promise.all([
      this.state.storage.get("status"),
      this.state.storage.get("lastAgentSeen"),
      this.state.storage.get("pending"),
      this.state.storage.get("fullLogMeta"),
    ]);

    const seenMs = lastAgentSeen ? Date.parse(lastAgentSeen) : Number.NaN;
    const agentOnline = Number.isFinite(seenMs) && Date.now() - seenMs < 40_000;
    const currentState = status?.state || "idle";
    const currentLogTail = Array.isArray(status?.logTail) ? status.logTail : [];
    const lastLogTail = Array.isArray(status?.lastLogTail) ? status.lastLogTail : [];
    const logTail = currentState === "idle" && currentLogTail.length === 0
      ? lastLogTail
      : currentLogTail;
    const defaultMessage = agentOnline ? "Waiting for a command" : "Agent offline";
    let message = status?.message || defaultMessage;
    if (currentState === "idle" && status?.lastState) {
      const lastSummary = status.lastMessage || status.lastCommand || "completed";
      message = `${status?.message || defaultMessage} · last ${status.lastState}: ${lastSummary}`;
    }

    return json({
      agentOnline,
      state: currentState,
      command: status?.command || null,
      pid: status?.pid ?? null,
      message,
      logTail,
      updatedAt: status?.updatedAt || lastAgentSeen || null,
      lastAgentSeen: lastAgentSeen || null,
      queuedCommand: pending?.command || null,
      queuedCommandId: pending?.id || null,
      lastCommand: status?.lastCommand || null,
      lastState: status?.lastState || null,
      lastMessage: status?.lastMessage || null,
      lastFinishedAt: status?.lastFinishedAt || null,
      latestResult: status?.lastResult || null,
      fullLogAvailable: Boolean(fullLogMeta?.chunks),
      fullLogCommand: fullLogMeta?.command || null,
      fullLogUpdatedAt: fullLogMeta?.updatedAt || null,
    });
  }

  async getResult() {
    const status = await this.state.storage.get("status");
    return json({
      result: status?.lastResult || null,
      lastCommand: status?.lastCommand || null,
      lastState: status?.lastState || null,
      lastFinishedAt: status?.lastFinishedAt || null,
    });
  }

  async getFullLog() {
    const meta = await this.state.storage.get("fullLogMeta");
    const chunkCount = Number.isInteger(meta?.chunks) ? meta.chunks : 0;
    if (chunkCount <= 0) {
      return json({ log: null, command: null, updatedAt: null, length: 0 });
    }
    const chunks = await Promise.all(
      Array.from({ length: chunkCount }, (_, index) => this.state.storage.get(`fullLog:${index}`)),
    );
    const log = chunks.map((chunk) => typeof chunk === "string" ? chunk : "").join("");
    return json({
      log,
      command: meta.command || null,
      updatedAt: meta.updatedAt || null,
      length: log.length,
    });
  }

  async storeFullLog(log, command, updatedAt) {
    const previous = await this.state.storage.get("fullLogMeta");
    const previousCount = Number.isInteger(previous?.chunks) ? previous.chunks : 0;
    const chunks = [];
    for (let offset = 0; offset < log.length; offset += FULL_LOG_CHUNK_CHARS) {
      chunks.push(log.slice(offset, offset + FULL_LOG_CHUNK_CHARS));
    }
    if (chunks.length === 0) chunks.push("");

    await Promise.all(
      chunks.map((chunk, index) => this.state.storage.put(`fullLog:${index}`, chunk)),
    );
    if (previousCount > chunks.length) {
      await Promise.all(
        Array.from(
          { length: previousCount - chunks.length },
          (_, index) => this.state.storage.delete(`fullLog:${chunks.length + index}`),
        ),
      );
    }
    await this.state.storage.put("fullLogMeta", {
      command,
      updatedAt,
      chunks: chunks.length,
      length: log.length,
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
    const previous = await this.state.storage.get("status");
    const state = typeof body?.state === "string" ? body.state : "idle";
    const command = typeof body?.command === "string" ? body.command : null;
    const message = typeof body?.message === "string" ? body.message : "";
    const logTail = Array.isArray(body?.logTail)
      ? body.logTail.filter((line) => typeof line === "string").slice(-24)
      : [];
    const result = body?.result && typeof body.result === "object" && !Array.isArray(body.result)
      ? body.result
      : null;
    const terminal = ["done", "error", "stopped"].includes(state);
    const fullLog = terminal && typeof body?.fullLog === "string" ? body.fullLog : null;

    if (fullLog !== null) {
      await this.storeFullLog(fullLog, command, now);
    }

    const status = {
      state,
      command,
      pid: Number.isInteger(body?.pid) ? body.pid : null,
      message,
      logTail,
      updatedAt: now,
      lastCommand: terminal ? command : (previous?.lastCommand || null),
      lastState: terminal ? state : (previous?.lastState || null),
      lastMessage: terminal ? message : (previous?.lastMessage || null),
      lastLogTail: terminal ? logTail : (Array.isArray(previous?.lastLogTail) ? previous.lastLogTail : []),
      lastFinishedAt: terminal ? now : (previous?.lastFinishedAt || null),
      lastResult: result || previous?.lastResult || null,
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
