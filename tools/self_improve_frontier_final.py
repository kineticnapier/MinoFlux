from __future__ import annotations

import json
from pathlib import Path

import self_improve_frontier_tournament as t
from minoflux_ai.search import SearchConfig

# Keep the finalist comparison affordable while preserving SRS reachability at the root.
t.FUTURE_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=3,
    srs_reachable=False,
)

FINALISTS = ("next_clean_frontier", "next_height_escape")


def main() -> None:
    versus = [t._versus(name, games=6) for name in FINALISTS]
    report = {"finalists": list(FINALISTS), "versus": versus}
    Path("tournament-final.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
