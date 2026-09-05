from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)


# ---------------------------------------------------------------------------
# reachability native: keep legacy run(), add packed record output for neural.
# ---------------------------------------------------------------------------
p = Path('src/minoflux_ai/_reachability_native.cpp')
text = p.read_text()
text = replace_once(
    text,
    'constexpr size_t kMaskBytes = 32;\n',
    'constexpr size_t kMaskBytes = 32;\nconstexpr size_t kPlacementRecordInts = 7;\nconstexpr size_t kPlacementRecordBytes = kPlacementRecordInts * sizeof(int32_t);\n',
    'reachability constants',
)
start = text.index('py::dict run(\n')
end = text.index('\n}  // namespace\n', start)
old = text[start:end]
new = r'''RunResult execute_run(
    int table_handle,
    const py::sequence& row_values,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes,
    bool profile
) {
    if (table_handle < 0 || static_cast<size_t>(table_handle) >= g_tables.size()) {
        throw std::runtime_error("invalid native reachability table handle");
    }
    const auto table = g_tables[static_cast<size_t>(table_handle)];
    if (static_cast<int>(py::len(row_values)) != table->height) {
        throw std::runtime_error("board row count mismatch");
    }
    std::vector<uint64_t> rows;
    rows.reserve(static_cast<size_t>(table->height));
    const uint64_t width_mask = table->width == 64 ? ~uint64_t{0} : ((uint64_t{1} << table->width) - 1);
    for (py::handle value : row_values) {
        rows.push_back(py::cast<uint64_t>(value) & width_mask);
    }

    return profile
        ? run_native<true>(*table, rows, start_x, start_y, start_rotation, max_nodes)
        : run_native<false>(*table, rows, start_x, start_y, start_rotation, max_nodes);
}

void add_run_metadata(py::dict& output, const RunResult& native_result) {
    py::dict counters;
    counters["bfsNodes"] = native_result.counters.bfs_nodes;
    counters["collisionChecks"] = native_result.counters.collision_checks;
    counters["collisionEvaluations"] = native_result.counters.collision_evaluations;
    counters["collisionCacheHits"] = native_result.counters.collision_cache_hits;
    counters["kickChecks"] = native_result.counters.kick_checks;
    counters["landingQueries"] = native_result.counters.landing_queries;
    counters["landingCacheHits"] = native_result.counters.landing_cache_hits;
    counters["representativeNodes"] = native_result.counters.representative_nodes;
    counters["representativeDuplicateSkips"] = native_result.counters.representative_duplicate_skips;

    py::dict timings;
    timings["setupSeconds"] = native_result.timings.setup_seconds;
    timings["bfsSeconds"] = native_result.timings.bfs_seconds;
    timings["rotationSeconds"] = native_result.timings.rotation_seconds;
    timings["landingSeconds"] = native_result.timings.landing_seconds;
    timings["representativeSeconds"] = native_result.timings.representative_seconds;
    timings["placementSeconds"] = native_result.timings.placement_seconds;

    output["counters"] = counters;
    output["timings"] = timings;
}

void write_i32_le(char* dest, int32_t value) noexcept {
    const uint32_t bits = static_cast<uint32_t>(value);
    dest[0] = static_cast<char>(bits & 0xffU);
    dest[1] = static_cast<char>((bits >> 8) & 0xffU);
    dest[2] = static_cast<char>((bits >> 16) & 0xffU);
    dest[3] = static_cast<char>((bits >> 24) & 0xffU);
}

py::dict run(
    int table_handle,
    const py::sequence& row_values,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes,
    bool profile
) {
    RunResult native_result = execute_run(
        table_handle,
        row_values,
        start_x,
        start_y,
        start_rotation,
        max_nodes,
        profile
    );

    py::list placements;
    for (const PlacementRecord& item : native_result.placements) {
        placements.append(py::make_tuple(
            item.x,
            item.y,
            item.rotation,
            item.last_rotation,
            item.kick_index,
            item.rotation_from,
            item.rotation_to
        ));
    }

    py::dict output;
    output["placements"] = placements;
    add_run_metadata(output, native_result);
    return output;
}

py::dict run_packed(
    int table_handle,
    const py::sequence& row_values,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes,
    bool profile
) {
    RunResult native_result = execute_run(
        table_handle,
        row_values,
        start_x,
        start_y,
        start_rotation,
        max_nodes,
        profile
    );

    std::string packed(native_result.placements.size() * kPlacementRecordBytes, '\0');
    char* dest = packed.data();
    for (const PlacementRecord& item : native_result.placements) {
        const std::array<int32_t, kPlacementRecordInts> fields = {
            item.x,
            item.y,
            item.rotation,
            item.last_rotation ? int32_t{1} : int32_t{0},
            item.kick_index,
            item.rotation_from,
            item.rotation_to,
        };
        for (int32_t field : fields) {
            write_i32_le(dest, field);
            dest += sizeof(int32_t);
        }
    }

    py::dict output;
    output["placementsPacked"] = py::bytes(packed);
    output["placementCount"] = native_result.placements.size();
    add_run_metadata(output, native_result);
    return output;
}
'''
text = text[:start] + new + text[end:]
text = replace_once(
    text,
    '''    module.def(\n        "run",\n        &run,\n        py::arg("table_handle"),\n        py::arg("rows"),\n        py::arg("start_x"),\n        py::arg("start_y"),\n        py::arg("start_rotation"),\n        py::arg("max_nodes"),\n        py::arg("profile") = false\n    );\n''',
    '''    module.def(\n        "run",\n        &run,\n        py::arg("table_handle"),\n        py::arg("rows"),\n        py::arg("start_x"),\n        py::arg("start_y"),\n        py::arg("start_rotation"),\n        py::arg("max_nodes"),\n        py::arg("profile") = false\n    );\n    module.def(\n        "run_packed",\n        &run_packed,\n        py::arg("table_handle"),\n        py::arg("rows"),\n        py::arg("start_x"),\n        py::arg("start_y"),\n        py::arg("start_rotation"),\n        py::arg("max_nodes"),\n        py::arg("profile") = false\n    );\n''',
    'reachability binding',
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Python record wrapper: packed bytes + lazy compatibility decoding + row reuse.
# ---------------------------------------------------------------------------
p = Path('src/minoflux_ai/reachability_native.py')
text = p.read_text()
text = replace_once(text, 'import os\nimport time\n', 'import os\nimport struct\nimport time\n', 'struct import')
text = replace_once(
    text,
    '_NATIVE_RECORD_CACHE_MAXSIZE = 8_192\n',
    '_NATIVE_RECORD_CACHE_MAXSIZE = 8_192\n_NATIVE_RECORD_STRUCT = struct.Struct("<7i")\n',
    'record struct',
)
old_class = '''@dataclass(frozen=True, slots=True)\nclass NativePlacementRecords:\n    """One ordered native reachability result without Python Placement materialization."""\n\n    piece: str\n    records: Sequence[Sequence[object]]\n\n    def __len__(self) -> int:\n        return len(self.records)\n\n    def materialize(self, index: int) -> Placement:\n        return _materialize_record(self.piece, self.records[index])\n'''
new_class = '''@dataclass(frozen=True, slots=True)\nclass NativePlacementRecords:\n    """Ordered native reachability records kept packed until a winner is needed."""\n\n    piece: str\n    packed: bytes\n    count: int\n    rows: tuple[int, ...] = ()\n\n    @classmethod\n    def empty(cls, piece: str, rows: Sequence[int] = ()) -> "NativePlacementRecords":\n        return cls(piece, b"", 0, tuple(rows))\n\n    def __len__(self) -> int:\n        return self.count\n\n    def record(self, index: int) -> tuple[int, int, int, int, int, int, int]:\n        if index < 0:\n            index += self.count\n        if index < 0 or index >= self.count:\n            raise IndexError(index)\n        return _NATIVE_RECORD_STRUCT.unpack_from(\n            self.packed,\n            index * _NATIVE_RECORD_STRUCT.size,\n        )\n\n    @property\n    def records(self) -> tuple[tuple[int, int, int, int, int, int, int], ...]:\n        """Compatibility view; neural fast paths intentionally do not call this."""\n\n        return tuple(self.record(index) for index in range(self.count))\n\n    def materialize(self, index: int) -> Placement:\n        return _materialize_record(self.piece, self.record(index))\n'''
text = replace_once(text, old_class, new_class, 'NativePlacementRecords')
text = replace_once(
    text,
    '''def reachable_placement_records_native(\n    game: Game,\n    *,\n    allow_180: bool = False,\n    max_nodes: int = 8_000,\n) -> NativePlacementRecords | None:\n''',
    '''def reachable_placement_records_native(\n    game: Game,\n    *,\n    allow_180: bool = False,\n    max_nodes: int = 8_000,\n    rows: Sequence[int] | None = None,\n) -> NativePlacementRecords | None:\n''',
    'record function signature',
)
text = replace_once(
    text,
    '''    if game.game_over or game.paused:\n        result = NativePlacementRecords(game.current, ())\n        if profile is not None:\n            profile.total_seconds += time.perf_counter() - profile_started\n        return result\n\n    board_mask_started = time.perf_counter() if profiling else 0.0\n    rows = board_row_masks(game.board)\n    if profile is not None:\n        profile.board_mask_seconds += time.perf_counter() - board_mask_started\n''',
    '''    if game.game_over or game.paused:\n        result = NativePlacementRecords.empty(game.current)\n        if profile is not None:\n            profile.total_seconds += time.perf_counter() - profile_started\n        return result\n\n    if rows is None:\n        board_mask_started = time.perf_counter() if profiling else 0.0\n        resolved_rows = board_row_masks(game.board)\n        if profile is not None:\n            profile.board_mask_seconds += time.perf_counter() - board_mask_started\n    else:\n        resolved_rows = tuple(int(row) for row in rows)\n        if len(resolved_rows) != game.height:\n            raise ValueError("precomputed board row count does not match game height")\n    rows = resolved_rows\n''',
    'row reuse',
)
text = replace_once(
    text,
    '''    native_result = _native.run(\n        table_handle,\n        rows,\n        game.x,\n        game.y,\n        game.rotation & 3,\n        normalized_max_nodes,\n        profiling,\n    )\n    result = NativePlacementRecords(game.current, native_result["placements"])\n''',
    '''    native_result = _native.run_packed(\n        table_handle,\n        rows,\n        game.x,\n        game.y,\n        game.rotation & 3,\n        normalized_max_nodes,\n        profiling,\n    )\n    packed = native_result["placementsPacked"]\n    count = int(native_result["placementCount"])\n    if len(packed) != count * _NATIVE_RECORD_STRUCT.size:\n        raise RuntimeError("native packed placement record length mismatch")\n    result = NativePlacementRecords(game.current, packed, count, rows)\n''',
    'packed run call',
)
p.write_text(text)


# ---------------------------------------------------------------------------
# neural native: direct decode of packed 7xi32 records, no py::sequence per item.
# ---------------------------------------------------------------------------
p = Path('src/minoflux_ai/_neural_native.cpp')
text = p.read_text()
text = replace_once(
    text,
    'namespace {\n\nstruct Cell {\n',
    'namespace {\n\nconstexpr size_t kPackedRecordInts = 7;\nconstexpr size_t kPackedRecordBytes = kPackedRecordInts * sizeof(int32_t);\n\nint32_t read_i32_le(const char* ptr) noexcept {\n    const uint32_t bits =\n        static_cast<uint32_t>(static_cast<unsigned char>(ptr[0])) |\n        (static_cast<uint32_t>(static_cast<unsigned char>(ptr[1])) << 8) |\n        (static_cast<uint32_t>(static_cast<unsigned char>(ptr[2])) << 16) |\n        (static_cast<uint32_t>(static_cast<unsigned char>(ptr[3])) << 24);\n    return static_cast<int32_t>(bits);\n}\n\nstruct Cell {\n',
    'neural packed helpers',
)
text = replace_once(
    text,
    '''py::tuple encode_group(\n    const py::sequence& source_rows_object,\n    const py::sequence& placements,\n''',
    '''py::tuple encode_group(\n    const py::sequence& source_rows_object,\n    const py::object& placements_object,\n''',
    'encode_group signature',
)
text = replace_once(
    text,
    '''    const py::sequence& normal_prefix_object,\n    const py::sequence& locked_prefix_object,\n    bool raw_records\n) {\n''',
    '''    const py::sequence& normal_prefix_object,\n    const py::sequence& locked_prefix_object,\n    int record_mode\n) {\n''',
    'record mode signature',
)
text = replace_once(
    text,
    '''    const size_t prefix_size = normal_prefix.size();\n    const size_t count = static_cast<size_t>(py::len(placements));\n    const size_t context_size = prefix_size + 9;\n''',
    '''    const size_t prefix_size = normal_prefix.size();\n    py::sequence placements;\n    const char* packed_records = nullptr;\n    size_t count = 0;\n    if (record_mode == 2) {\n        char* packed_ptr = nullptr;\n        Py_ssize_t packed_size = 0;\n        if (PyBytes_AsStringAndSize(placements_object.ptr(), &packed_ptr, &packed_size) != 0) {\n            throw py::error_already_set();\n        }\n        if (packed_size < 0 || static_cast<size_t>(packed_size) % kPackedRecordBytes != 0) {\n            throw std::runtime_error("packed placement record byte length mismatch");\n        }\n        packed_records = packed_ptr;\n        count = static_cast<size_t>(packed_size) / kPackedRecordBytes;\n    } else {\n        placements = py::reinterpret_borrow<py::sequence>(placements_object);\n        count = static_cast<size_t>(py::len(placements));\n    }\n    const size_t context_size = prefix_size + 9;\n''',
    'packed count setup',
)
old_loop = '''        py::handle placement = placements[static_cast<py::ssize_t>(placement_index)];\n        int x = 0;\n        int y = 0;\n        int rotation = 0;\n        bool last_rotation = false;\n        int kick_index = -1;\n        if (raw_records) {\n            py::sequence record = py::reinterpret_borrow<py::sequence>(placement);\n            if (py::len(record) < 5) {\n                throw std::runtime_error("native placement record must contain at least five fields");\n            }\n            x = py::cast<int>(record[0]);\n            y = py::cast<int>(record[1]);\n            rotation = py::cast<int>(record[2]) & 3;\n            last_rotation = py::cast<bool>(record[3]);\n            kick_index = py::cast<int>(record[4]);\n        } else {\n            const std::string placement_piece = py::cast<std::string>(placement.attr("piece"));\n            if (placement_piece != piece_name) {\n                throw std::runtime_error("placement piece does not match group piece");\n            }\n            x = py::cast<int>(placement.attr("x"));\n            y = py::cast<int>(placement.attr("y"));\n            rotation = py::cast<int>(placement.attr("rotation")) & 3;\n            last_rotation = py::cast<bool>(placement.attr("last_move_was_rotation"));\n            py::object kick_object = py::reinterpret_borrow<py::object>(placement.attr("rotation_kick_index"));\n            kick_index = kick_object.is_none() ? -1 : py::cast<int>(kick_object);\n        }\n'''
new_loop = '''        int x = 0;\n        int y = 0;\n        int rotation = 0;\n        bool last_rotation = false;\n        int kick_index = -1;\n        if (record_mode == 2) {\n            const char* record = packed_records + placement_index * kPackedRecordBytes;\n            x = read_i32_le(record);\n            y = read_i32_le(record + 4);\n            rotation = read_i32_le(record + 8) & 3;\n            last_rotation = read_i32_le(record + 12) != 0;\n            kick_index = read_i32_le(record + 16);\n        } else {\n            py::handle placement = placements[static_cast<py::ssize_t>(placement_index)];\n            if (record_mode == 1) {\n                py::sequence record = py::reinterpret_borrow<py::sequence>(placement);\n                if (py::len(record) < 5) {\n                    throw std::runtime_error("native placement record must contain at least five fields");\n                }\n                x = py::cast<int>(record[0]);\n                y = py::cast<int>(record[1]);\n                rotation = py::cast<int>(record[2]) & 3;\n                last_rotation = py::cast<bool>(record[3]);\n                kick_index = py::cast<int>(record[4]);\n            } else {\n                const std::string placement_piece = py::cast<std::string>(placement.attr("piece"));\n                if (placement_piece != piece_name) {\n                    throw std::runtime_error("placement piece does not match group piece");\n                }\n                x = py::cast<int>(placement.attr("x"));\n                y = py::cast<int>(placement.attr("y"));\n                rotation = py::cast<int>(placement.attr("rotation")) & 3;\n                last_rotation = py::cast<bool>(placement.attr("last_move_was_rotation"));\n                py::object kick_object = py::reinterpret_borrow<py::object>(placement.attr("rotation_kick_index"));\n                kick_index = kick_object.is_none() ? -1 : py::cast<int>(kick_object);\n            }\n        }\n'''
text = replace_once(text, old_loop, new_loop, 'candidate decode loop')
text = text.replace('        false\n    );\n}\n\npy::tuple encode_record_group(', '        0\n    );\n}\n\npy::tuple encode_record_group(', 1)
text = text.replace('        true\n    );\n}\n\n}  // namespace', '        1\n    );\n}\n\npy::tuple encode_packed_record_group(\n    const py::sequence& source_rows_object,\n    const py::bytes& records,\n    const std::string& piece_name,\n    const std::string& next_piece_name,\n    int width,\n    int height,\n    int hidden_rows,\n    int combo_before,\n    bool back_to_back_before,\n    int b2b_chain_before,\n    const py::sequence& normal_prefix_object,\n    const py::sequence& locked_prefix_object\n) {\n    return encode_group(\n        source_rows_object,\n        records,\n        piece_name,\n        next_piece_name,\n        width,\n        height,\n        hidden_rows,\n        combo_before,\n        back_to_back_before,\n        b2b_chain_before,\n        normal_prefix_object,\n        locked_prefix_object,\n        2\n    );\n}\n\n}  // namespace', 1)
text = replace_once(
    text,
    '''    module.def(\n        "encode_record_group",\n        &encode_record_group,\n        py::arg("source_rows"),\n        py::arg("records"),\n        py::arg("piece"),\n        py::arg("next_piece"),\n        py::arg("width"),\n        py::arg("height"),\n        py::arg("hidden_rows"),\n        py::arg("combo"),\n        py::arg("back_to_back"),\n        py::arg("b2b_chain"),\n        py::arg("normal_prefix"),\n        py::arg("locked_prefix")\n    );\n''',
    '''    module.def(\n        "encode_record_group",\n        &encode_record_group,\n        py::arg("source_rows"),\n        py::arg("records"),\n        py::arg("piece"),\n        py::arg("next_piece"),\n        py::arg("width"),\n        py::arg("height"),\n        py::arg("hidden_rows"),\n        py::arg("combo"),\n        py::arg("back_to_back"),\n        py::arg("b2b_chain"),\n        py::arg("normal_prefix"),\n        py::arg("locked_prefix")\n    );\n    module.def(\n        "encode_packed_record_group",\n        &encode_packed_record_group,\n        py::arg("source_rows"),\n        py::arg("records"),\n        py::arg("piece"),\n        py::arg("next_piece"),\n        py::arg("width"),\n        py::arg("height"),\n        py::arg("hidden_rows"),\n        py::arg("combo"),\n        py::arg("back_to_back"),\n        py::arg("b2b_chain"),\n        py::arg("normal_prefix"),\n        py::arg("locked_prefix")\n    );\n''',
    'packed neural binding',
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Python scorer/search: use packed bytes and reuse direct board rows for Hold.
# ---------------------------------------------------------------------------
p = Path('src/minoflux_ai/neural_fast.py')
text = p.read_text()
text = replace_once(
    text,
    '''        source_rows = board_row_masks(game.board)\n        board_chunk, context_chunk = _native.encode_record_group(\n            source_rows,\n            batch.records,\n''',
    '''        source_rows = batch.rows or board_row_masks(game.board)\n        board_chunk, context_chunk = _native.encode_packed_record_group(\n            source_rows,\n            batch.packed,\n''',
    'packed neural call',
)
p.write_text(text)

p = Path('src/minoflux_ai/neural_search_fast.py')
text = p.read_text()
text = replace_once(
    text,
    '            prepared.append((NativePlacementRecords(game.current, ()), None, None))\n',
    '            prepared.append((NativePlacementRecords.empty(game.current), None, None))\n',
    'empty records',
)
text = replace_once(
    text,
    '''            reachable_placement_records_native(\n                held,\n                allow_180=cfg.allow_180,\n                max_nodes=cfg.reachability_node_limit,\n            )\n''',
    '''            reachable_placement_records_native(\n                held,\n                allow_180=cfg.allow_180,\n                max_nodes=cfg.reachability_node_limit,\n                rows=direct.rows,\n            )\n''',
    'hold row reuse',
)
p.write_text(text)


# ---------------------------------------------------------------------------
# Differential tests: packed ABI and encoder must match legacy tuple records.
# ---------------------------------------------------------------------------
p = Path('tests/test_neural_native_record_path.py')
text = p.read_text()
insert_after = '''def _record_buffers(game: Game, batch):\n    queue, normal, locked = _prefixes(game)\n    return _neural_native.encode_record_group(\n        board_row_masks(game.board),\n        batch.records,\n        game.current,\n        queue[0],\n        game.width,\n        game.height,\n        game.hidden_rows,\n        game.combo,\n        game.back_to_back,\n        game.b2b_chain,\n        normal,\n        locked,\n    )\n\n\n'''
packed_helper = '''def _packed_record_buffers(game: Game, batch):\n    queue, normal, locked = _prefixes(game)\n    return _neural_native.encode_packed_record_group(\n        batch.rows or board_row_masks(game.board),\n        batch.packed,\n        game.current,\n        queue[0],\n        game.width,\n        game.height,\n        game.hidden_rows,\n        game.combo,\n        game.back_to_back,\n        game.b2b_chain,\n        normal,\n        locked,\n    )\n\n\n'''
text = replace_once(text, insert_after, insert_after + packed_helper, 'packed test helper')
text = replace_once(
    text,
    '''            self.assertEqual(\n                _record_buffers(game, raw_direct),\n                _placement_buffers(game, direct),\n            )\n''',
    '''            self.assertEqual(\n                _record_buffers(game, raw_direct),\n                _placement_buffers(game, direct),\n            )\n            self.assertEqual(\n                _packed_record_buffers(game, raw_direct),\n                _record_buffers(game, raw_direct),\n            )\n''',
    'direct packed buffer assertion',
)
text = replace_once(
    text,
    '''                self.assertEqual(\n                    _record_buffers(held, raw_hold),\n                    _placement_buffers(held, hold),\n                )\n''',
    '''                self.assertEqual(\n                    _record_buffers(held, raw_hold),\n                    _placement_buffers(held, hold),\n                )\n                self.assertEqual(\n                    _packed_record_buffers(held, raw_hold),\n                    _record_buffers(held, raw_hold),\n                )\n''',
    'hold packed buffer assertion',
)
p.write_text(text)

print('patched packed native record path + board row reuse')
