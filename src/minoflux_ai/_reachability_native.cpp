#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "native/reachability_pybind.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;
namespace reach = minoflux::reachability;
namespace support = minoflux::reachability::pybind_support;

namespace {
std::vector<std::shared_ptr<reach::Table>> g_tables;

int register_table(
    const std::string& piece,
    int width,
    int height,
    int x_min,
    int x_max,
    int x_count,
    int y_min,
    const py::sequence& state_x,
    const py::sequence& state_y,
    const py::sequence& left_state,
    const py::sequence& right_state,
    const py::sequence& down_state,
    const py::bytes& collision_invalid,
    const py::bytes& collision_masks,
    const py::bytes& geometry_invalid,
    const py::bytes& geometry_masks,
    const py::sequence& rotation_transitions
) {
    g_tables.push_back(support::table_from_python(
        piece,
        width,
        height,
        x_min,
        x_max,
        x_count,
        y_min,
        state_x,
        state_y,
        left_state,
        right_state,
        down_state,
        collision_invalid,
        collision_masks,
        geometry_invalid,
        geometry_masks,
        rotation_transitions
    ));
    return static_cast<int>(g_tables.size() - 1);
}

reach::RunResult execute_run(
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
    const auto& table = *g_tables[static_cast<size_t>(table_handle)];
    const auto rows = support::rows_from_python(row_values, table.width, table.height);
    return reach::run(table, rows, start_x, start_y, start_rotation, max_nodes, profile);
}

void add_run_metadata(py::dict& output, const reach::RunResult& native_result) {
    const uint64_t logical_known_clear = native_result.counters.rotation_known_clear_skips;
    py::dict counters;
    counters["bfsNodes"] = native_result.counters.bfs_nodes;
    counters["collisionChecks"] = native_result.counters.collision_checks + logical_known_clear;
    counters["collisionEvaluations"] = native_result.counters.collision_evaluations;
    counters["collisionCacheHits"] = native_result.counters.collision_cache_hits + logical_known_clear;
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

std::string pack_records(const std::vector<reach::PlacementRecord>& placements) {
    std::string packed(placements.size() * reach::kPlacementRecordBytes, '\0');
    char* dest = packed.data();
    for (const auto& item : placements) {
        const std::array<int32_t, reach::kPlacementRecordInts> fields = {
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
    return packed;
}

py::dict run(
    int table_handle,
    const py::sequence& rows,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes,
    bool profile
) {
    const auto native_result = execute_run(
        table_handle,
        rows,
        start_x,
        start_y,
        start_rotation,
        max_nodes,
        profile
    );
    py::list placements;
    for (const auto& item : native_result.placements) {
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
    const py::sequence& rows,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes,
    bool profile
) {
    const auto native_result = execute_run(
        table_handle,
        rows,
        start_x,
        start_y,
        start_rotation,
        max_nodes,
        profile
    );
    py::dict output;
    output["placementsPacked"] = py::bytes(pack_records(native_result.placements));
    output["placementCount"] = native_result.placements.size();
    add_run_metadata(output, native_result);
    return output;
}

py::tuple run_packed_fast(
    int table_handle,
    const py::sequence& rows,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes
) {
    const auto native_result = execute_run(
        table_handle,
        rows,
        start_x,
        start_y,
        start_rotation,
        max_nodes,
        false
    );
    return py::make_tuple(
        py::bytes(pack_records(native_result.placements)),
        native_result.placements.size()
    );
}
}  // namespace

PYBIND11_MODULE(_reachability_native, module) {
    module.doc() = "Native exact-SRS pathless reachability core";
    module.def(
        "register_table",
        &register_table,
        py::arg("piece"),
        py::arg("width"),
        py::arg("height"),
        py::arg("x_min"),
        py::arg("x_max"),
        py::arg("x_count"),
        py::arg("y_min"),
        py::arg("state_x"),
        py::arg("state_y"),
        py::arg("left_state"),
        py::arg("right_state"),
        py::arg("down_state"),
        py::arg("collision_invalid"),
        py::arg("collision_masks"),
        py::arg("geometry_invalid"),
        py::arg("geometry_masks"),
        py::arg("rotation_transitions")
    );
    module.def(
        "run",
        &run,
        py::arg("table_handle"),
        py::arg("rows"),
        py::arg("start_x"),
        py::arg("start_y"),
        py::arg("start_rotation"),
        py::arg("max_nodes"),
        py::arg("profile") = false
    );
    module.def(
        "run_packed",
        &run_packed,
        py::arg("table_handle"),
        py::arg("rows"),
        py::arg("start_x"),
        py::arg("start_y"),
        py::arg("start_rotation"),
        py::arg("max_nodes"),
        py::arg("profile") = false
    );
    module.def(
        "run_packed_fast",
        &run_packed_fast,
        py::arg("table_handle"),
        py::arg("rows"),
        py::arg("start_x"),
        py::arg("start_y"),
        py::arg("start_rotation"),
        py::arg("max_nodes")
    );
}
