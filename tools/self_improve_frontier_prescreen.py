from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import self_improve_frontier_tournament as t
from minoflux_ai.search import SearchConfig

# Root moves remain SRS-reachable. Only the post-root option probe is made cheap.
t.FUTURE_SEARCH = SearchConfig(
    allow_hold=True,
    lookahead_pieces=0,
    beam_width=3,
    srs_reachable=False,
)

def parallel(names, seeds, pieces):
    out=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        fs={pool.submit(t._run_bench_task,(name,seeds,pieces)):name for name in names}
        for f in as_completed(fs): out.append(f.result())
    out.sort(key=lambda r:names.index(str(r['name'])))
    return out

def main():
    names=['baseline',*t.CANDIDATES]
    short=parallel(names,[830_009,830_106],40)
    finalists=sorted([r for r in short if r['name']!='baseline'],key=t._rank_key,reverse=True)[:3]
    fn=['baseline',*(str(r['name']) for r in finalists)]
    fresh=parallel(fn,[940_031,940_128,940_225],80)
    base=fresh[0]
    eligible=[r for r in fresh[1:] if int(r['topouts'])<=int(base['topouts']) and int(r['completed'])>=int(base['completed']) and float(r['app'])>=float(base['app'])*1.01]
    report={'candidateCount':len(t.CANDIDATES),'short':short,'shortFinalists':[str(r['name']) for r in finalists],'fresh':fresh,'eligible':[str(r['name']) for r in sorted(eligible,key=t._rank_key,reverse=True)[:2]]}
    Path('tournament-prescreen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'shortFinalists':report['shortFinalists'],'eligible':report['eligible']}))
if __name__=='__main__': main()
