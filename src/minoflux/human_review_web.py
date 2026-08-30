from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

from minoflux_ai.human_review import append_human_label, load_review_queue, review_key

_REVIEW_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MinoFlux Human Review</title>
<style>
:root { color-scheme: dark; font-family: ui-monospace, Consolas, monospace; }
body { margin: 0; background: #111318; color: #e8eaf0; }
main { max-width: 1500px; margin: 0 auto; padding: 20px; }
h1 { margin: 0 0 8px; font-size: 24px; }
#status { color: #aeb6c8; margin-bottom: 14px; }
.panel { border: 1px solid #343a49; border-radius: 10px; padding: 12px; background: #181b22; }
.source-wrap { display: flex; gap: 18px; align-items: flex-start; flex-wrap: wrap; }
.board { white-space: pre; line-height: 1.0; letter-spacing: .08em; font-size: 17px; margin: 8px 0 0; }
.meta { line-height: 1.7; }
#candidates { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-top: 14px; }
.candidate { text-align: left; color: inherit; cursor: pointer; border: 1px solid #3a4152; border-radius: 10px; background: #1b1f28; padding: 10px; }
.candidate:hover { border-color: #9ca8c8; transform: translateY(-1px); }
.candidate .title { font-weight: 700; margin-bottom: 5px; }
.candidate .score { color: #9ba7bc; font-size: 12px; }
.controls { display: flex; gap: 8px; margin: 14px 0; }
.controls button { cursor: pointer; border: 1px solid #3a4152; border-radius: 8px; background: #202531; color: inherit; padding: 8px 14px; }
#message { min-height: 1.5em; color: #b8d6a8; }
kbd { border: 1px solid #4a5265; border-radius: 4px; padding: 1px 5px; background: #222733; }
</style>
</head>
<body><main>
<h1>MinoFlux Human Review</h1>
<div id="status">Loading…</div><div id="message"></div>
<div class="panel source-wrap">
  <div><strong>Current position</strong><pre class="board" id="source-board"></pre></div>
  <div class="meta" id="source-meta"></div>
</div>
<div class="controls"><button id="prev">← Previous</button><button id="skip">Skip →</button></div>
<div id="candidates"></div>
<p style="color:#8e98ad">Click the move you would play. Keyboard: <kbd>1</kbd>…<kbd>9</kbd> choose, <kbd>N</kbd> skip, <kbd>←</kbd> previous.</p>
<script>
let index = 0, current = null;
function board(rows) { return rows.slice(-20).map(mask => Array.from({length:10}, (_,x) => (mask & (1<<x)) ? '█' : '·').join('')).join('\n'); }
function moveText(move) { return `${move.hold ? 'HOLD → ' : ''}${move.piece}  x=${move.x} y=${move.y} r=${move.rotation}`; }
async function load() {
  const response = await fetch(`/api/sample?index=${index}`), data = await response.json();
  current = data; const candidates = document.getElementById('candidates'); candidates.innerHTML = '';
  if (data.done) {
    document.getElementById('status').textContent = `All pending positions reviewed (${data.total} total).`;
    document.getElementById('source-board').textContent = ''; document.getElementById('source-meta').textContent = ''; return;
  }
  index = data.index;
  document.getElementById('status').textContent = `${data.index + 1} / ${data.total} — seed ${data.seed}, piece ${data.pieceIndex} — ${data.reasons.join(', ')}`;
  document.getElementById('source-board').textContent = board(data.source.rows);
  document.getElementById('source-meta').innerHTML = `Current: <b>${data.source.current}</b><br>Hold: <b>${data.source.hold || '-'}</b><br>Next: <b>${data.source.next.join(' ')}</b><br>Combo: ${data.source.combo}<br>B2B: ${data.source.b2b ? 'yes' : 'no'} (${data.source.b2bChain})`;
  data.candidates.forEach((candidate, candidateIndex) => {
    const button = document.createElement('button'); button.className = 'candidate';
    button.innerHTML = `<div class="title">${candidateIndex + 1}. ${moveText(candidate.move)}</div><div class="score">NN value ${Number(candidate.nnValue).toFixed(4)}</div><pre class="board">${board(candidate.rows)}</pre>`;
    button.onclick = () => label(candidateIndex); candidates.appendChild(button);
  });
}
async function label(candidate) {
  if (!current || current.done) return;
  const response = await fetch('/api/label', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({index, candidate})});
  const data = await response.json(); document.getElementById('message').textContent = data.message;
  if (data.ok) { index += 1; await load(); }
}
document.getElementById('skip').onclick = async () => { index += 1; document.getElementById('message').textContent = 'Skipped.'; await load(); };
document.getElementById('prev').onclick = async () => { index = Math.max(0, index - 1); document.getElementById('message').textContent = ''; await load(); };
document.addEventListener('keydown', event => {
  if (event.key >= '1' && event.key <= '9' && current && !current.done) { const candidate = Number(event.key) - 1; if (candidate < current.candidates.length) label(candidate); }
  else if (event.key.toLowerCase() === 'n') document.getElementById('skip').click();
  else if (event.key === 'ArrowLeft') document.getElementById('prev').click();
});
load();
</script>
</main></body></html>"""


def launch_human_review_app(queue_path: str | Path, output_path: str | Path, *, port: int = 7861) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import parse_qs, urlparse
    import webbrowser

    records = load_review_queue(queue_path)
    output = Path(output_path)
    reviewed: set[tuple[int, int]] = set()
    if output.is_file():
        with output.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, Mapping):
                        reviewed.add(review_key(value))
    pending = [record for record in records if review_key(record) not in reviewed]

    class Handler(BaseHTTPRequestHandler):
        def _json(self, value: object, status: int = 200) -> None:
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                body = _REVIEW_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers(); self.wfile.write(body); return
            if parsed.path == "/api/sample":
                try: index = int(parse_qs(parsed.query).get("index", ["0"])[0])
                except ValueError: index = 0
                if not pending: self._json({"done": True, "total": 0}); return
                if index >= len(pending): self._json({"done": True, "total": len(pending)}); return
                index = max(0, index)
                record = dict(pending[index]); record.pop("championMove", None)
                record.update({"index": index, "total": len(pending), "done": False})
                self._json(record); return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/label": self.send_error(404); return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size).decode("utf-8"))
                index, candidate = int(payload["index"]), int(payload["candidate"])
                if index < 0 or index >= len(pending): raise ValueError("Position index is out of range")
                candidates = pending[index].get("candidates")
                if not isinstance(candidates, Sequence) or candidate < 0 or candidate >= len(candidates):
                    raise ValueError("Candidate index is out of range")
                added = append_human_label(output, pending[index], candidate)
                self._json({"ok": True, "message": "Saved human label." if added else "Already labeled; kept the existing label."})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._json({"ok": False, "message": str(error)}, status=400)

        def log_message(self, format: str, *args: object) -> None: return

    server = ThreadingHTTPServer(("127.0.0.1", int(port)), Handler)
    url = f"http://127.0.0.1:{int(port)}/"
    print(f"Human review UI: {url}"); print(f"Labels: {output}"); webbrowser.open(url)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
