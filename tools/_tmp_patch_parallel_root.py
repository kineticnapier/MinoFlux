from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)


# Release the GIL only around the pure C++ exact-SRS core. Python argument
# decoding and result construction remain under the GIL.
p = Path('src/minoflux_ai/_reachability_native.cpp')
text = p.read_text()
text = replace_once(
    text,
    '''    return profile
        ? run_native<true>(*table, rows, start_x, start_y, start_rotation, max_nodes)
        : run_native<false>(*table, rows, start_x, start_y, start_rotation, max_nodes);
''',
    '''    RunResult result;
    {
        py::gil_scoped_release release;
        result = profile
            ? run_native<true>(*table, rows, start_x, start_y, start_rotation, max_nodes)
            : run_native<false>(*table, rows, start_x, start_y, start_rotation, max_nodes);
    }
    return result;
''',
    'GIL release',
)
p.write_text(text)


# Run root reachability in two deterministic waves: all direct branches, then
# all Hold branches. Results are consumed strictly in original game order.
p = Path('src/minoflux_ai/neural_search_fast.py')
text = p.read_text()
text = replace_once(
    text,
    '''from heapq import nlargest
import time
from typing import Sequence
''',
    '''from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from functools import cache
from heapq import nlargest
import os
import time
from typing import Sequence
''',
    'imports',
)
insert = '''\n\n_NATIVE_THREADS_ENV = "MINOFLUX_NATIVE_REACHABILITY_THREADS"\n\n\ndef _native_reachability_workers() -> int:\n    raw = os.environ.get(_NATIVE_THREADS_ENV, "1").strip()\n    try:\n        workers = int(raw)\n    except ValueError:\n        return 1\n    return max(1, min(32, workers))\n\n\n@cache\ndef _native_reachability_executor(workers: int) -> ThreadPoolExecutor:\n    return ThreadPoolExecutor(\n        max_workers=workers,\n        thread_name_prefix="minoflux-srs",\n    )\n\n\ndef _submit_record_reachability(\n    executor: ThreadPoolExecutor,\n    game: Game,\n    *,\n    allow_180: bool,\n    max_nodes: int,\n    rows=None,\n):\n    context = copy_context()\n    return executor.submit(\n        context.run,\n        reachable_placement_records_native,\n        game,\n        allow_180=allow_180,\n        max_nodes=max_nodes,\n        rows=rows,\n    )\n'''
anchor = '_ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH = _search.choose_search_actions_batch\n'
text = replace_once(text, anchor, anchor + insert, 'parallel helpers')
old_loop = '''    for index, game in enumerate(games):
        if game.game_over:
            prepared.append((NativePlacementRecords.empty(game.current), None, None))
            continue

        # Short queues are rare near synthetic/test boundaries. Preserve the
        # existing generic fallback instead of changing its semantics.
        if len(game.queue) < queue_length + 1:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        held = _search._held_search_game(game) if cfg.allow_hold else None
        if held is not None and len(held.queue) < queue_length + 1:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        direct = reachable_placement_records_native(
            game,
            allow_180=cfg.allow_180,
            max_nodes=cfg.reachability_node_limit,
        )
        if direct is None:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )
        held_records = (
            reachable_placement_records_native(
                held,
                allow_180=cfg.allow_180,
                max_nodes=cfg.reachability_node_limit,
                rows=direct.rows,
            )
            if held is not None
            else None
        )
        if held is not None and held_records is None:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        prepared.append((direct, held, held_records))
        if direct:
            groups.append((game, direct))
            group_keys.append((index, False))
        if held is not None and held_records:
            groups.append((held, held_records))
            group_keys.append((index, True))
'''
new_loop = '''    branch_games = []
    for game in games:
        if game.game_over:
            branch_games.append((game, None))
            continue

        # Short queues are rare near synthetic/test boundaries. Preserve the
        # existing generic fallback instead of changing its semantics.
        if len(game.queue) < queue_length + 1:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

        held = _search._held_search_game(game) if cfg.allow_hold else None
        if held is not None and len(held.queue) < queue_length + 1:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )
        branch_games.append((game, held))

    workers = _native_reachability_workers()
    executor = _native_reachability_executor(workers) if workers > 1 else None

    if executor is None:
        direct_results = [
            (
                NativePlacementRecords.empty(game.current)
                if game.game_over
                else reachable_placement_records_native(
                    game,
                    allow_180=cfg.allow_180,
                    max_nodes=cfg.reachability_node_limit,
                )
            )
            for game, _held in branch_games
        ]
    else:
        direct_futures = [
            (
                None
                if game.game_over
                else _submit_record_reachability(
                    executor,
                    game,
                    allow_180=cfg.allow_180,
                    max_nodes=cfg.reachability_node_limit,
                )
            )
            for game, _held in branch_games
        ]
        direct_results = [
            (
                NativePlacementRecords.empty(game.current)
                if future is None
                else future.result()
            )
            for (game, _held), future in zip(branch_games, direct_futures)
        ]

    if any(result is None for result in direct_results):
        return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
            games,
            weights,
            cfg,
            scorer=scorer,
        )

    if executor is None:
        held_results = [
            (
                reachable_placement_records_native(
                    held,
                    allow_180=cfg.allow_180,
                    max_nodes=cfg.reachability_node_limit,
                    rows=direct.rows,
                )
                if held is not None
                else None
            )
            for (_game, held), direct in zip(branch_games, direct_results)
        ]
    else:
        held_futures = [
            (
                _submit_record_reachability(
                    executor,
                    held,
                    allow_180=cfg.allow_180,
                    max_nodes=cfg.reachability_node_limit,
                    rows=direct.rows,
                )
                if held is not None
                else None
            )
            for (_game, held), direct in zip(branch_games, direct_results)
        ]
        held_results = [
            future.result() if future is not None else None
            for future in held_futures
        ]

    for held, held_records in zip(
        (held for _game, held in branch_games),
        held_results,
    ):
        if held is not None and held_records is None:
            return _ORIGINAL_CHOOSE_SEARCH_ACTIONS_BATCH(
                games,
                weights,
                cfg,
                scorer=scorer,
            )

    for index, ((game, held), direct, held_records) in enumerate(
        zip(branch_games, direct_results, held_results)
    ):
        assert direct is not None
        prepared.append((direct, held, held_records))
        if direct:
            groups.append((game, direct))
            group_keys.append((index, False))
        if held is not None and held_records:
            groups.append((held, held_records))
            group_keys.append((index, True))
'''
text = replace_once(text, old_loop, new_loop, 'batch reachability loop')
p.write_text(text)
print('patched GIL release + deterministic root reachability parallelism')
