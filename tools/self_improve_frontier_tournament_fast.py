from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import self_improve_frontier_tournament as t
from minoflux_ai.search import SearchConfig

# Same 14 judgments, but only the top three heuristic continuations are inspected
# during screening. Fresh/versus still use unseen seeds and mirrored sides.
t.FUTURE_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=3,
    srs_reachable=True,
    allow_180=False,
    reachability_node_limit=3_500,
)


def run_parallel(names: list[str], seeds: list[int], pieces: int):
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(t._run_bench_task, (name, seeds, pieces)): name for name in names}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: names.index(str(row["name"])))
    return rows


def main() -> None:
    names = ["baseline", *t.CANDIDATES]
    short = run_parallel(names, [820_003, 820_100], 60)
    candidates = [row for row in short if row["name"] != "baseline"]
    finalists = sorted(candidates, key=t._rank_key, reverse=True)[:3]

    fresh_names = ["baseline", *(str(row["name"]) for row in finalists)]
    fresh = run_parallel(fresh_names, [920_029, 920_126, 920_223], 100)
    baseline = fresh[0]
    eligible = [
        row for row in fresh[1:]
        if int(row["topouts"]) <= int(baseline["topouts"])
        and int(row["completed"]) >= int(baseline["completed"])
        and float(row["app"]) >= float(baseline["app"]) * 1.01
    ]
    versus_names = [str(row["name"]) for row in sorted(eligible, key=t._rank_key, reverse=True)[:2]]
    versus = [t._versus(name, games=4) for name in versus_names]

    result = {
        "candidateCount": len(t.CANDIDATES),
        "short": short,
        "shortFinalists": [str(row["name"]) for row in finalists],
        "fresh": fresh,
        "versus": versus,
    }
    Path("tournament-result-fast.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"shortFinalists": result["shortFinalists"], "versusNames": versus_names}))


if __name__ == "__main__":
    main()
