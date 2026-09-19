#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "native/oracle_core.hpp"
#include "native/reachability_pybind.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace oracle = minoflux::oracle;
namespace reach_support = minoflux::reachability::pybind_support;

namespace {

oracle::Piece parse_piece(const std::string& value, const char* name) {
    if (value.size() != 1) {
        throw py::value_error(std::string(name) + " must be one tetromino letter");
    }
    const oracle::Piece piece = oracle::piece_from_char(value[0]);
    if (piece == oracle::Piece::None) {
        throw py::value_error(std::string(name) + " contains an unknown tetromino");
    }
    return piece;
}

std::array<uint16_t, oracle::kHeight> parse_rows(const py::sequence& values) {
    if (py::len(values) != oracle::kHeight) {
        throw py::value_error("oracle board must contain exactly 24 rows");
    }
    std::array<uint16_t, oracle::kHeight> rows{};
    for (py::ssize_t index = 0; index < oracle::kHeight; ++index) {
        const uint64_t raw = py::cast<uint64_t>(values[index]);
        if ((raw & ~uint64_t{oracle::kFullRow}) != 0) {
            throw py::value_error("oracle board row exceeds 10 bits");
        }
        rows[static_cast<size_t>(index)] = static_cast<uint16_t>(raw);
    }
    return rows;
}

std::vector<oracle::Piece> parse_queue(const py::sequence& values) {
    std::vector<oracle::Piece> result;
    result.reserve(static_cast<size_t>(py::len(values)));
    for (py::handle item : values) {
        result.push_back(parse_piece(py::cast<std::string>(item), "queue item"));
    }
    return result;
}

oracle::State parse_state(
    const py::sequence& rows_value,
    const std::string& current_value,
    const py::object& hold_value,
    int combo,
    bool back_to_back,
    int b2b_chain,
    bool can_hold
) {
    oracle::State state;
    state.rows = parse_rows(rows_value);
    state.current = parse_piece(current_value, "current");
    state.combo = static_cast<int16_t>(combo);
    state.b2b_active = back_to_back;
    state.b2b_chain = static_cast<uint16_t>(std::max(0, b2b_chain));
    state.can_hold = can_hold;
    state.has_hold = !hold_value.is_none();
    if (state.has_hold) {
        state.hold = parse_piece(py::cast<std::string>(hold_value), "hold");
    }
    return state;
}

oracle::Move parse_move(const py::dict& value) {
    oracle::Move move;
    move.piece = parse_piece(py::cast<std::string>(value["piece"]), "move piece");
    move.x = static_cast<int8_t>(py::cast<int>(value["x"]));
    move.y = static_cast<int8_t>(py::cast<int>(value["y"]));
    move.rotation = static_cast<int8_t>(py::cast<int>(value["rotation"]));
    move.use_hold = py::cast<bool>(value["holdUsed"]);
    move.last_rotation = py::cast<bool>(value["lastMoveWasRotation"]);
    move.kick_index = static_cast<int8_t>(py::cast<int>(value["kickIndex"]));
    move.rotation_from = static_cast<int8_t>(py::cast<int>(value["rotationFrom"]));
    move.rotation_to = static_cast<int8_t>(py::cast<int>(value["rotationTo"]));
    return move;
}

py::dict move_dict(const oracle::Move& move) {
    py::dict output;
    output["piece"] = std::string(1, oracle::piece_to_char(move.piece));
    output["x"] = move.x;
    output["y"] = move.y;
    output["rotation"] = move.rotation;
    output["holdUsed"] = move.use_hold;
    output["lastMoveWasRotation"] = move.last_rotation;
    output["kickIndex"] = move.kick_index;
    output["rotationFrom"] = move.rotation_from;
    output["rotationTo"] = move.rotation_to;
    return output;
}

py::object spin_name(int event) {
    switch (event) {
        case 1: return py::str("T_SPIN_MINI");
        case 2: return py::str("T_SPIN_MINI_SINGLE");
        case 3: return py::str("T_SPIN");
        case 4: return py::str("T_SPIN_SINGLE");
        case 5: return py::str("T_SPIN_DOUBLE");
        case 6: return py::str("T_SPIN_TRIPLE");
        default: return py::none();
    }
}

void register_reachability_table_native(
    const std::string& piece_value,
    bool allow_180,
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
    const oracle::Piece piece = parse_piece(piece_value, "piece");
    auto table = reach_support::table_from_python(
        piece_value,
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
    );
    oracle::register_reachability_table(piece, allow_180, std::move(table));
}

py::list reachable_native(
    const py::sequence& rows_value,
    const std::string& piece_value,
    bool allow_180,
    int max_nodes
) {
    const auto rows = parse_rows(rows_value);
    const oracle::Piece piece = parse_piece(piece_value, "piece");
    std::vector<oracle::Move> moves;
    {
        py::gil_scoped_release release;
        moves = oracle::reachable_moves(rows, piece, allow_180, max_nodes);
    }
    py::list output;
    for (const oracle::Move& move : moves) {
        output.append(move_dict(move));
    }
    return output;
}

py::dict transition_native(
    const py::sequence& rows_value,
    const std::string& current_value,
    const py::object& hold_value,
    const py::sequence& queue_value,
    int combo,
    bool back_to_back,
    int b2b_chain,
    bool can_hold,
    const py::dict& move_value
) {
    oracle::State state = parse_state(
        rows_value,
        current_value,
        hold_value,
        combo,
        back_to_back,
        b2b_chain,
        can_hold
    );
    const std::vector<oracle::Piece> queue = parse_queue(queue_value);
    const oracle::Move move = parse_move(move_value);

    oracle::TransitionResult result;
    {
        py::gil_scoped_release release;
        result = oracle::transition(state, queue, move);
    }
    if (!result.valid) {
        throw py::value_error("invalid native oracle transition");
    }

    py::dict output;
    output["rows"] = result.state.rows;
    output["current"] = std::string(1, oracle::piece_to_char(result.state.current));
    output["hold"] = result.state.has_hold
        ? py::object(py::str(std::string(1, oracle::piece_to_char(result.state.hold))))
        : py::object(py::none());
    output["queueIndex"] = result.state.queue_index;
    output["combo"] = result.state.combo;
    output["backToBack"] = result.state.b2b_active;
    output["b2bChain"] = result.state.b2b_chain;
    output["canHold"] = result.state.can_hold;
    output["gameOver"] = result.state.game_over;
    output["lines"] = result.lines;
    output["attack"] = result.attack;
    output["spin"] = spin_name(result.spin_event);
    output["perfectClear"] = result.perfect_clear;
    output["surgeReleased"] = result.surge_released;
    output["surgeCharge"] = result.surge_charge;
    return output;
}

oracle::Result run_search(
    const oracle::State& state,
    const std::vector<oracle::Piece>& queue,
    int beam_width,
    int depth,
    bool allow_180,
    int max_nodes
) {
    oracle::Config config;
    config.beam_width = beam_width;
    config.depth = depth;
    config.allow_180 = allow_180;
    config.max_nodes = max_nodes;
    return oracle::search(state, queue, config);
}

py::object search_native(
    const py::sequence& rows_value,
    const std::string& current_value,
    const py::object& hold_value,
    const py::sequence& queue_value,
    int combo,
    bool back_to_back,
    int b2b_chain,
    bool can_hold,
    int beam_width,
    int depth,
    bool allow_180,
    int max_nodes
) {
    const oracle::State state = parse_state(
        rows_value,
        current_value,
        hold_value,
        combo,
        back_to_back,
        b2b_chain,
        can_hold
    );
    const std::vector<oracle::Piece> queue = parse_queue(queue_value);

    oracle::Result result;
    {
        py::gil_scoped_release release;
        result = run_search(state, queue, beam_width, depth, allow_180, max_nodes);
    }
    if (!result.found) {
        return py::none();
    }

    py::dict output = move_dict(result.move);
    output["score"] = result.score;
    return output;
}

py::dict search_profile_native(
    const py::sequence& rows_value,
    const std::string& current_value,
    const py::object& hold_value,
    const py::sequence& queue_value,
    int combo,
    bool back_to_back,
    int b2b_chain,
    bool can_hold,
    int beam_width,
    int depth,
    bool allow_180,
    int max_nodes
) {
    const oracle::State state = parse_state(
        rows_value,
        current_value,
        hold_value,
        combo,
        back_to_back,
        b2b_chain,
        can_hold
    );
    const std::vector<oracle::Piece> queue = parse_queue(queue_value);

    oracle::begin_reachability_profile();
    oracle::begin_search_profile();
    oracle::Result result;
    const auto started = std::chrono::steady_clock::now();
    try {
        py::gil_scoped_release release;
        result = run_search(state, queue, beam_width, depth, allow_180, max_nodes);
    } catch (...) {
        oracle::end_search_profile();
        oracle::end_reachability_profile();
        throw;
    }
    const auto stopped = std::chrono::steady_clock::now();
    const oracle::SearchProfile search_profile = oracle::end_search_profile();
    const oracle::ReachabilityProfile reachability_profile = oracle::end_reachability_profile();

    py::dict output;
    output["searchSeconds"] = std::chrono::duration<double>(stopped - started).count();
    output["reachabilitySeconds"] = reachability_profile.total_seconds;
    output["movegenCalls"] = reachability_profile.calls;
    output["generatedMoves"] = reachability_profile.generated_moves;
    output["featureCalls"] = search_profile.feature_calls;
    output["featureSeconds"] = search_profile.feature_seconds;
    output["dedupSeconds"] = search_profile.dedup_seconds;
    output["pruneSeconds"] = search_profile.prune_seconds;
    output["expandedChildren"] = search_profile.expanded_children;
    output["dedupHits"] = search_profile.dedup_hits;
    output["dedupReplacements"] = search_profile.dedup_replacements;

    py::dict reachability;
    reachability["setupSeconds"] = reachability_profile.setup_seconds;
    reachability["bfsSeconds"] = reachability_profile.bfs_seconds;
    reachability["rotationSeconds"] = reachability_profile.rotation_seconds;
    reachability["landingSeconds"] = reachability_profile.landing_seconds;
    reachability["representativeSeconds"] = reachability_profile.representative_seconds;
    reachability["placementSeconds"] = reachability_profile.placement_seconds;
    output["reachability"] = reachability;

    if (result.found) {
        py::dict choice = move_dict(result.move);
        choice["score"] = result.score;
        output["choice"] = choice;
    } else {
        output["choice"] = py::none();
    }
    return output;
}

}  // namespace

PYBIND11_MODULE(_oracle_native, module) {
    module.doc() = "Native offline exact-SRS beam-search oracle";
    module.def("api_version", []() { return 1; });
    module.def("reachability_backend", []() { return "shared-table-v1"; });
    module.def(
        "register_reachability_table",
        &register_reachability_table_native,
        py::arg("piece"),
        py::arg("allow_180"),
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
        "reachable",
        &reachable_native,
        py::arg("rows"),
        py::arg("piece"),
        py::arg("allow_180") = false,
        py::arg("max_nodes") = 8000
    );
    module.def(
        "transition",
        &transition_native,
        py::arg("rows"),
        py::arg("current"),
        py::arg("hold"),
        py::arg("queue"),
        py::arg("combo"),
        py::arg("back_to_back"),
        py::arg("b2b_chain"),
        py::arg("can_hold"),
        py::arg("move")
    );
    module.def(
        "search",
        &search_native,
        py::arg("rows"),
        py::arg("current"),
        py::arg("hold"),
        py::arg("queue"),
        py::arg("combo"),
        py::arg("back_to_back"),
        py::arg("b2b_chain"),
        py::arg("can_hold"),
        py::arg("beam_width") = 2000,
        py::arg("depth") = 18,
        py::arg("allow_180") = true,
        py::arg("max_nodes") = 8000
    );
    module.def(
        "search_profile",
        &search_profile_native,
        py::arg("rows"),
        py::arg("current"),
        py::arg("hold"),
        py::arg("queue"),
        py::arg("combo"),
        py::arg("back_to_back"),
        py::arg("b2b_chain"),
        py::arg("can_hold"),
        py::arg("beam_width") = 2000,
        py::arg("depth") = 18,
        py::arg("allow_180") = true,
        py::arg("max_nodes") = 8000
    );
}
