#include "oracle_search_core.cpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace minoflux::oracle {
namespace {
namespace reach = minoflux::reachability;

constexpr uint64_t kDetailedProfileSampleStride = 257;

std::array<std::shared_ptr<const reach::Table>, 14> g_reachability_tables{};
thread_local bool g_reachability_profile_enabled = false;
thread_local ReachabilityProfile g_reachability_profile{};
thread_local uint64_t g_detailed_profile_sample_calls = 0;
thread_local double g_sampled_rotation_seconds = 0.0;
thread_local double g_sampled_landing_seconds = 0.0;
thread_local double g_sampled_representative_seconds = 0.0;

bool sample_detailed_profile(uint64_t call_index) noexcept {
    return call_index % kDetailedProfileSampleStride == 0;
}

size_t table_index(Piece piece, bool allow_180) {
    const int value = static_cast<int>(piece);
    if (value < 0 || value >= 7) {
        throw std::runtime_error("invalid oracle piece for reachability table");
    }
    return static_cast<size_t>(value * 2 + (allow_180 ? 1 : 0));
}

const reach::Table& table_for(Piece piece, bool allow_180) {
    const auto& table = g_reachability_tables[table_index(piece, allow_180)];
    if (!table) {
        throw std::runtime_error("oracle reachability table is not registered");
    }
    return *table;
}
}  // namespace

void register_reachability_table(
    Piece piece,
    bool allow_180,
    std::shared_ptr<const minoflux::reachability::Table> table
) {
    if (!table) {
        throw std::runtime_error("oracle reachability table is null");
    }
    if (table->width != kWidth || table->height != kHeight) {
        throw std::runtime_error("oracle reachability table dimensions do not match");
    }
    g_reachability_tables[table_index(piece, allow_180)] = std::move(table);
}

void begin_reachability_profile() {
    g_reachability_profile = ReachabilityProfile{};
    g_detailed_profile_sample_calls = 0;
    g_sampled_rotation_seconds = 0.0;
    g_sampled_landing_seconds = 0.0;
    g_sampled_representative_seconds = 0.0;
    reach::set_profile_detailed_timings(false);
    g_reachability_profile_enabled = true;
}

ReachabilityProfile end_reachability_profile() {
    g_reachability_profile_enabled = false;
    reach::set_profile_detailed_timings(true);
    if (g_detailed_profile_sample_calls != 0) {
        const double scale =
            static_cast<double>(g_reachability_profile.calls) /
            static_cast<double>(g_detailed_profile_sample_calls);
        g_reachability_profile.rotation_seconds = g_sampled_rotation_seconds * scale;
        g_reachability_profile.landing_seconds = g_sampled_landing_seconds * scale;
        g_reachability_profile.representative_seconds =
            g_sampled_representative_seconds * scale;
    }
    return g_reachability_profile;
}

std::vector<Move> reachable_moves(
    const std::array<uint16_t, kHeight>& rows,
    Piece piece,
    bool allow_180,
    int max_nodes,
    bool use_hold
) {
    const reach::Table& table = table_for(piece, allow_180);
    std::vector<uint64_t> native_rows;
    native_rows.reserve(kHeight);
    for (uint16_t row : rows) {
        native_rows.push_back(row);
    }

    const bool profiling = g_reachability_profile_enabled;
    const bool detailed_sample =
        profiling && sample_detailed_profile(g_reachability_profile.calls);
    if (profiling) {
        reach::set_profile_detailed_timings(detailed_sample);
    }
    const auto started = profiling
        ? std::chrono::steady_clock::now()
        : std::chrono::steady_clock::time_point{};
    const reach::RunResult native_result = reach::run(
        table,
        native_rows,
        3,
        1,
        0,
        max_nodes,
        profiling
    );
    if (profiling) {
        const auto stopped = std::chrono::steady_clock::now();
        ++g_reachability_profile.calls;
        g_reachability_profile.generated_moves += native_result.placements.size();
        g_reachability_profile.bfs_nodes += native_result.counters.bfs_nodes;
        g_reachability_profile.collision_checks += native_result.counters.collision_checks;
        g_reachability_profile.collision_evaluations +=
            native_result.counters.collision_evaluations;
        g_reachability_profile.collision_cache_hits +=
            native_result.counters.collision_cache_hits;
        g_reachability_profile.kick_checks += native_result.counters.kick_checks;
        g_reachability_profile.movement_edges += native_result.counters.movement_edges;
        g_reachability_profile.movement_visited_skips +=
            native_result.counters.movement_visited_skips;
        g_reachability_profile.movement_collision_checks +=
            native_result.counters.movement_collision_checks;
        g_reachability_profile.movement_enqueues += native_result.counters.movement_enqueues;
        g_reachability_profile.rotation_groups += native_result.counters.rotation_groups;
        g_reachability_profile.rotation_collision_checks +=
            native_result.counters.rotation_collision_checks;
        g_reachability_profile.rotation_successes += native_result.counters.rotation_successes;
        g_reachability_profile.rotation_geometry_enqueues +=
            native_result.counters.rotation_geometry_enqueues;
        g_reachability_profile.landing_collision_checks +=
            native_result.counters.landing_collision_checks;
        g_reachability_profile.landing_queries += native_result.counters.landing_queries;
        g_reachability_profile.landing_cache_hits +=
            native_result.counters.landing_cache_hits;
        g_reachability_profile.representative_nodes +=
            native_result.counters.representative_nodes;
        g_reachability_profile.representative_duplicate_skips +=
            native_result.counters.representative_duplicate_skips;
        g_reachability_profile.total_seconds +=
            std::chrono::duration<double>(stopped - started).count();
        g_reachability_profile.setup_seconds += native_result.timings.setup_seconds;
        g_reachability_profile.bfs_seconds += native_result.timings.bfs_seconds;
        g_reachability_profile.placement_seconds +=
            native_result.timings.placement_seconds;
        if (detailed_sample) {
            ++g_detailed_profile_sample_calls;
            g_sampled_rotation_seconds += native_result.timings.rotation_seconds;
            g_sampled_landing_seconds += native_result.timings.landing_seconds;
            g_sampled_representative_seconds +=
                native_result.timings.representative_seconds;
        }
    }

    std::vector<Move> result;
    result.reserve(native_result.placements.size());
    for (const reach::PlacementRecord& placement : native_result.placements) {
        Move move;
        move.piece = piece;
        move.x = static_cast<int8_t>(placement.x);
        move.y = static_cast<int8_t>(placement.y);
        move.rotation = static_cast<int8_t>(placement.rotation);
        move.use_hold = use_hold;
        move.last_rotation = placement.last_rotation;
        move.kick_index = static_cast<int8_t>(placement.kick_index);
        move.rotation_from = static_cast<int8_t>(placement.rotation_from);
        move.rotation_to = static_cast<int8_t>(placement.rotation_to);
        result.push_back(move);
    }
    return result;
}

TransitionResult transition(
    const State& root,
    std::span<const Piece> queue,
    const Move& move
) {
    TransitionResult result;
    if (root.game_over || root.current == Piece::None) {
        return result;
    }

    State state = root;
    Piece placing = state.current;
    if (move.use_hold) {
        if (!state.can_hold) {
            return result;
        }
        const Piece outgoing = state.current;
        if (state.has_hold) {
            placing = state.hold;
        } else {
            if (state.queue_index >= queue.size()) {
                return result;
            }
            placing = queue[state.queue_index++];
            if (placing == Piece::None) {
                return result;
            }
        }
        state.hold = outgoing;
        state.has_hold = true;
        state.can_hold = false;
        state.current = placing;
        if (collides(state.rows, placing, 3, 1, 0)) {
            return result;
        }
    }

    if (
        move.piece != placing ||
        collides(state.rows, move.piece, move.x, move.y, move.rotation) ||
        !collides(state.rows, move.piece, move.x, move.y + 1, move.rotation)
    ) {
        return result;
    }

    const int spin_kind =
        (move.piece == Piece::T && move.last_rotation)
            ? classify_t_spin(state.rows, move.x, move.y, move.rotation, move.kick_index)
            : 0;
    const bool was_active = state.b2b_active;
    const int previous_chain = was_active ? state.b2b_chain : 0;
    const Features parent_features = features(state.rows);
    LockValue value;
    Features after_features;
    apply_placement(
        state,
        queue,
        move,
        parent_features.holes,
        value,
        after_features
    );

    const int event = spin_event(spin_kind, value.lines);
    const bool difficult = difficult_clear(value.lines, event);
    const int released =
        (value.lines > 0 && !value.pc && !difficult && was_active && previous_chain >= 4)
            ? previous_chain
            : 0;

    result.valid = true;
    result.state = state;
    result.lines = value.lines;
    result.attack = value.attack;
    result.spin_event = event;
    result.perfect_clear = value.pc;
    result.surge_released = released;
    result.surge_charge =
        state.b2b_active && state.b2b_chain >= 4 ? state.b2b_chain : 0;
    return result;
}

}  // namespace minoflux::oracle
