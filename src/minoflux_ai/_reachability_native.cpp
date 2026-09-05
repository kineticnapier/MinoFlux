#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using Clock = std::chrono::steady_clock;

namespace {

constexpr int32_t kNoState = -1;
constexpr int32_t kNoLanding = -1;
constexpr uint8_t kCollisionUnknown = 0;
constexpr uint8_t kCollisionClear = 1;
constexpr uint8_t kCollisionBlocked = 2;
constexpr int kKickIndexBits = 3;
constexpr int kKickIndexMask = (1 << kKickIndexBits) - 1;
constexpr size_t kMaskBytes = 32;
constexpr size_t kPlacementRecordInts = 7;
constexpr size_t kPlacementRecordBytes = kPlacementRecordInts * sizeof(int32_t);

struct Mask256 {
    std::array<uint64_t, 4> words{};

    bool operator==(const Mask256& other) const noexcept {
        return words == other.words;
    }
};

struct MaskHash {
    size_t operator()(const Mask256& value) const noexcept {
        size_t h = 0x9e3779b97f4a7c15ULL;
        for (uint64_t word : value.words) {
            h ^= static_cast<size_t>(word + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
        }
        return h;
    }
};

struct Table {
    int width = 0;
    int height = 0;
    int x_min = 0;
    int x_max = 0;
    int x_count = 0;
    int y_min = 0;
    int state_count = 0;
    int limb_count = 0;
    bool piece_is_t = false;

    std::vector<int32_t> state_x;
    std::vector<int32_t> state_y;
    std::vector<int32_t> left_state;
    std::vector<int32_t> right_state;
    std::vector<int32_t> down_state;
    std::vector<uint8_t> collision_invalid;
    std::vector<uint8_t> geometry_invalid;
    std::vector<Mask256> collision_masks;
    std::vector<Mask256> geometry_masks;
    std::vector<int32_t> geometry_ids;
    int32_t geometry_count = 0;

    std::vector<uint32_t> state_group_offsets;
    std::vector<uint32_t> group_kick_offsets;
    std::vector<int32_t> kick_targets;
    std::vector<int8_t> kick_indices;
};

struct Counters {
    uint64_t bfs_nodes = 0;
    uint64_t collision_checks = 0;
    uint64_t collision_evaluations = 0;
    uint64_t collision_cache_hits = 0;
    uint64_t kick_checks = 0;
    uint64_t landing_queries = 0;
    uint64_t landing_cache_hits = 0;
    uint64_t representative_nodes = 0;
    uint64_t representative_duplicate_skips = 0;
};

struct Timings {
    double setup_seconds = 0.0;
    double bfs_seconds = 0.0;
    double rotation_seconds = 0.0;
    double landing_seconds = 0.0;
    double representative_seconds = 0.0;
    double placement_seconds = 0.0;
};

struct PlacementRecord {
    int32_t x = 0;
    int32_t y = 0;
    int32_t rotation = 0;
    bool last_rotation = false;
    int32_t kick_index = -1;
    int32_t rotation_from = -1;
    int32_t rotation_to = -1;
};

struct RunResult {
    std::vector<PlacementRecord> placements;
    Counters counters;
    Timings timings;
};

struct BestRecord {
    int spin_any = 0;
    int spin_full = 0;
    int negative_depth = 0;
    int32_t order_rank = 0;
    PlacementRecord placement;
};

struct Scratch {
    std::vector<uint8_t> collision_cache;
    std::vector<int32_t> landing_state;
    std::vector<int32_t> state_depths;
    std::vector<int32_t> state_kick_infos;
    std::vector<int32_t> rotation_depths;
    std::vector<int32_t> rotation_kick_infos;
    std::vector<uint8_t> rotation_is_geometry;
    std::vector<int32_t> frontier;
    std::vector<int32_t> visited_state_ids;
    std::vector<int32_t> visited_rotation_ids;
    std::vector<int32_t> landing_trail;
    std::vector<BestRecord> best_records;
    std::vector<uint32_t> best_generations;
    std::vector<int32_t> touched_geometry_ids;
    uint32_t best_generation = 0;
};

std::vector<std::shared_ptr<Table>> g_tables;
thread_local Scratch g_scratch;

double seconds_between(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double>(end - start).count();
}

uint64_t read_u64_le(const char* ptr) {
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(static_cast<unsigned char>(ptr[i])) << (8 * i);
    }
    return value;
}

std::vector<int32_t> to_i32_vector(const py::sequence& values, size_t expected, const char* name) {
    if (static_cast<size_t>(py::len(values)) != expected) {
        throw std::runtime_error(std::string(name) + " length mismatch");
    }
    std::vector<int32_t> result;
    result.reserve(expected);
    for (py::handle value : values) {
        result.push_back(py::cast<int32_t>(value));
    }
    return result;
}

std::vector<uint8_t> to_u8_bytes(const py::bytes& values, size_t expected, const char* name) {
    std::string raw = values;
    if (raw.size() != expected) {
        throw std::runtime_error(std::string(name) + " length mismatch");
    }
    return std::vector<uint8_t>(raw.begin(), raw.end());
}

std::vector<Mask256> decode_masks(const py::bytes& values, size_t count, const char* name) {
    std::string raw = values;
    if (raw.size() != count * kMaskBytes) {
        throw std::runtime_error(std::string(name) + " length mismatch");
    }
    std::vector<Mask256> result(count);
    for (size_t i = 0; i < count; ++i) {
        const char* base = raw.data() + i * kMaskBytes;
        for (size_t limb = 0; limb < 4; ++limb) {
            result[i].words[limb] = read_u64_le(base + limb * 8);
        }
    }
    return result;
}

bool mask_intersects(const Mask256& left, const Mask256& right, int limb_count) noexcept {
    for (int i = 0; i < limb_count; ++i) {
        if ((left.words[static_cast<size_t>(i)] & right.words[static_cast<size_t>(i)]) != 0) {
            return true;
        }
    }
    return false;
}

Mask256 pack_board(const std::vector<uint64_t>& rows, int width, int height) {
    Mask256 result{};
    for (int y = 0; y < height; ++y) {
        uint64_t row = rows[static_cast<size_t>(y)];
        const int bit_index = y * width;
        const int limb = bit_index >> 6;
        const int shift = bit_index & 63;
        if (limb < 4) {
            result.words[static_cast<size_t>(limb)] |= row << shift;
        }
        if (shift != 0 && limb + 1 < 4) {
            result.words[static_cast<size_t>(limb + 1)] |= row >> (64 - shift);
        }
    }
    return result;
}

template <bool Profile>
inline bool checked_collision(
    const Table& table,
    const Mask256& board,
    int32_t state_id,
    std::vector<uint8_t>& collision_cache,
    Counters& counters
) {
    if constexpr (Profile) {
        ++counters.collision_checks;
    }
    const uint8_t cached = collision_cache[static_cast<size_t>(state_id)];
    if (cached != kCollisionUnknown) {
        if constexpr (Profile) {
            ++counters.collision_cache_hits;
        }
        return cached == kCollisionBlocked;
    }

    if constexpr (Profile) {
        ++counters.collision_evaluations;
    }
    const bool blocked = table.collision_invalid[static_cast<size_t>(state_id)] != 0 ||
        mask_intersects(board, table.collision_masks[static_cast<size_t>(state_id)], table.limb_count);
    collision_cache[static_cast<size_t>(state_id)] = blocked ? kCollisionBlocked : kCollisionClear;
    return blocked;
}

bool occupied_or_wall(
    const std::vector<uint64_t>& rows,
    int width,
    int height,
    int x,
    int y
) noexcept {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return true;
    }
    return (rows[static_cast<size_t>(y)] & (uint64_t{1} << x)) != 0;
}

int classify_t_spin(
    const std::vector<uint64_t>& rows,
    int width,
    int height,
    int x,
    int y,
    int rotation,
    int kick_index
) noexcept {
    const int pivot_x = x + 1;
    const int pivot_y = y + 1;
    const std::array<bool, 4> corners = {
        occupied_or_wall(rows, width, height, pivot_x - 1, pivot_y - 1),
        occupied_or_wall(rows, width, height, pivot_x + 1, pivot_y - 1),
        occupied_or_wall(rows, width, height, pivot_x - 1, pivot_y + 1),
        occupied_or_wall(rows, width, height, pivot_x + 1, pivot_y + 1),
    };
    const int count = static_cast<int>(corners[0]) + static_cast<int>(corners[1]) +
        static_cast<int>(corners[2]) + static_cast<int>(corners[3]);
    if (count < 3) {
        return 0;
    }
    static constexpr std::array<std::array<int, 2>, 4> front = {{
        {{0, 1}}, {{1, 3}}, {{2, 3}}, {{0, 2}},
    }};
    const auto pair = front[static_cast<size_t>(rotation & 3)];
    if ((corners[static_cast<size_t>(pair[0])] && corners[static_cast<size_t>(pair[1])]) || kick_index == 4) {
        return 2;
    }
    return 1;
}

bool better_non_t(const BestRecord& candidate, const BestRecord& current) noexcept {
    return candidate.negative_depth > current.negative_depth ||
        (candidate.negative_depth == current.negative_depth && candidate.order_rank < current.order_rank);
}

bool better_t(const BestRecord& candidate, const BestRecord& current) noexcept {
    if (candidate.spin_any != current.spin_any) {
        return candidate.spin_any > current.spin_any;
    }
    if (candidate.spin_full != current.spin_full) {
        return candidate.spin_full > current.spin_full;
    }
    if (candidate.negative_depth != current.negative_depth) {
        return candidate.negative_depth > current.negative_depth;
    }
    return candidate.order_rank < current.order_rank;
}

template <bool Profile>
RunResult run_native(
    const Table& table,
    const std::vector<uint64_t>& rows,
    int start_x,
    int start_y,
    int start_rotation,
    int max_nodes
) {
    RunResult result;
    auto& counters = result.counters;
    auto& timings = result.timings;
    const auto run_setup_started = Clock::now();

    if (start_x < table.x_min || start_x > table.x_max || start_y < table.y_min || start_y >= table.height) {
        if constexpr (Profile) {
            timings.setup_seconds = seconds_between(run_setup_started, Clock::now());
        }
        return result;
    }

    const int32_t start_state = static_cast<int32_t>(
        ((((start_y - table.y_min) * table.x_count) + (start_x - table.x_min)) << 2) |
        (start_rotation & 3)
    );
    const Mask256 board = pack_board(rows, table.width, table.height);
    Scratch& scratch = g_scratch;
    const size_t n = static_cast<size_t>(table.state_count);

    scratch.collision_cache.assign(n, kCollisionUnknown);
    scratch.landing_state.assign(n, kNoLanding);
    scratch.state_depths.assign(n, kNoState);
    scratch.state_kick_infos.assign(n, -1);
    scratch.frontier.clear();
    scratch.visited_state_ids.clear();
    scratch.visited_rotation_ids.clear();
    scratch.landing_trail.clear();
    scratch.frontier.reserve(n);
    scratch.visited_state_ids.reserve(n);
    scratch.visited_rotation_ids.reserve(n);
    scratch.landing_trail.reserve(n);

    if (table.piece_is_t) {
        scratch.rotation_depths.assign(n, kNoState);
        scratch.rotation_kick_infos.assign(n, -1);
        scratch.rotation_is_geometry.assign(n, uint8_t{0});
    } else {
        scratch.rotation_depths.clear();
        scratch.rotation_kick_infos.clear();
        scratch.rotation_is_geometry.clear();
    }

    if constexpr (Profile) {
        counters.collision_checks = 1;
        counters.collision_evaluations = 1;
    }
    const bool start_blocked = table.collision_invalid[static_cast<size_t>(start_state)] != 0 ||
        mask_intersects(board, table.collision_masks[static_cast<size_t>(start_state)], table.limb_count);
    scratch.collision_cache[static_cast<size_t>(start_state)] = start_blocked ? kCollisionBlocked : kCollisionClear;
    if (start_blocked) {
        if constexpr (Profile) {
            timings.setup_seconds = seconds_between(run_setup_started, Clock::now());
        }
        return result;
    }

    scratch.state_depths[static_cast<size_t>(start_state)] = 0;
    scratch.frontier.push_back(start_state);
    scratch.visited_state_ids.push_back(start_state);
    size_t frontier_index = 0;
    int reachable_count = 1;
    const int budget = std::max(1, max_nodes);

    if constexpr (Profile) {
        timings.setup_seconds = seconds_between(run_setup_started, Clock::now());
    }
    const auto bfs_started = Clock::now();

    while (frontier_index < scratch.frontier.size() && reachable_count <= budget) {
        const int32_t state_id = scratch.frontier[frontier_index++];
        if constexpr (Profile) {
            ++counters.bfs_nodes;
        }
        const int32_t new_depth = scratch.state_depths[static_cast<size_t>(state_id)] + 1;

        const std::array<int32_t, 3> movement_targets = {
            table.left_state[static_cast<size_t>(state_id)],
            table.right_state[static_cast<size_t>(state_id)],
            table.down_state[static_cast<size_t>(state_id)],
        };
        for (int32_t target_state : movement_targets) {
            if (target_state == kNoState || scratch.state_depths[static_cast<size_t>(target_state)] != kNoState) {
                continue;
            }
            if (!checked_collision<Profile>(table, board, target_state, scratch.collision_cache, counters)) {
                scratch.state_depths[static_cast<size_t>(target_state)] = new_depth;
                scratch.state_kick_infos[static_cast<size_t>(target_state)] = -1;
                scratch.visited_state_ids.push_back(target_state);
                ++reachable_count;
                scratch.frontier.push_back(target_state);
            }
        }

        Clock::time_point rotation_started{};
        if constexpr (Profile) {
            rotation_started = Clock::now();
        }
        const uint32_t group_begin = table.state_group_offsets[static_cast<size_t>(state_id)];
        const uint32_t group_end = table.state_group_offsets[static_cast<size_t>(state_id) + 1];
        for (uint32_t group_index = group_begin; group_index < group_end; ++group_index) {
            int32_t successful_state = kNoState;
            int32_t successful_kick = -1;
            const uint32_t kick_begin = table.group_kick_offsets[static_cast<size_t>(group_index)];
            const uint32_t kick_end = table.group_kick_offsets[static_cast<size_t>(group_index) + 1];
            for (uint32_t kick_index = kick_begin; kick_index < kick_end; ++kick_index) {
                if constexpr (Profile) {
                    ++counters.kick_checks;
                }
                const int32_t target_state = table.kick_targets[static_cast<size_t>(kick_index)];
                if (checked_collision<Profile>(table, board, target_state, scratch.collision_cache, counters)) {
                    continue;
                }
                successful_state = target_state;
                successful_kick = table.kick_indices[static_cast<size_t>(kick_index)];
                break;
            }
            if (successful_state == kNoState) {
                continue;
            }

            const bool adds_geometry = scratch.state_depths[static_cast<size_t>(successful_state)] == kNoState;
            int32_t previous_rotation_depth = kNoState;
            bool improves_rotation = false;
            if (table.piece_is_t) {
                previous_rotation_depth = scratch.rotation_depths[static_cast<size_t>(successful_state)];
                improves_rotation = previous_rotation_depth == kNoState || new_depth < previous_rotation_depth;
            }
            if (!improves_rotation && !adds_geometry) {
                continue;
            }

            const int32_t rotation_info = ((state_id & 3) << kKickIndexBits) | successful_kick;
            if (improves_rotation) {
                if (previous_rotation_depth == kNoState) {
                    scratch.visited_rotation_ids.push_back(successful_state);
                }
                scratch.rotation_depths[static_cast<size_t>(successful_state)] = new_depth;
                scratch.rotation_kick_infos[static_cast<size_t>(successful_state)] = rotation_info;
                scratch.rotation_is_geometry[static_cast<size_t>(successful_state)] = static_cast<uint8_t>(adds_geometry);
            }
            if (adds_geometry) {
                scratch.state_depths[static_cast<size_t>(successful_state)] = new_depth;
                scratch.state_kick_infos[static_cast<size_t>(successful_state)] = rotation_info;
                scratch.visited_state_ids.push_back(successful_state);
                ++reachable_count;
                scratch.frontier.push_back(successful_state);
            }
        }
        if constexpr (Profile) {
            timings.rotation_seconds += seconds_between(rotation_started, Clock::now());
        }

        if (reachable_count > budget) {
            break;
        }
    }

    if constexpr (Profile) {
        timings.bfs_seconds = seconds_between(bfs_started, Clock::now());
    }

    auto compute_landing = [&](int32_t state_id) -> int32_t {
        scratch.landing_trail.clear();
        scratch.landing_trail.push_back(state_id);
        int32_t current_id = state_id;
        int32_t landing = state_id;
        while (true) {
            const int32_t target_id = table.down_state[static_cast<size_t>(current_id)];
            if (target_id == kNoState) {
                landing = current_id;
                break;
            }
            if (checked_collision<Profile>(table, board, target_id, scratch.collision_cache, counters)) {
                landing = current_id;
                break;
            }
            current_id = target_id;
            const int32_t cached_landing = scratch.landing_state[static_cast<size_t>(current_id)];
            if (cached_landing != kNoLanding) {
                if constexpr (Profile) {
                    ++counters.landing_cache_hits;
                }
                landing = cached_landing;
                break;
            }
            scratch.landing_trail.push_back(current_id);
        }
        for (int32_t cached_id : scratch.landing_trail) {
            scratch.landing_state[static_cast<size_t>(cached_id)] = landing;
        }
        return landing;
    };

    const size_t geometry_count = static_cast<size_t>(table.geometry_count);
    if (scratch.best_records.size() < geometry_count) {
        scratch.best_records.resize(geometry_count);
    }
    if (scratch.best_generations.size() < geometry_count) {
        scratch.best_generations.resize(geometry_count, 0);
    }
    scratch.touched_geometry_ids.clear();
    scratch.touched_geometry_ids.reserve(geometry_count);
    ++scratch.best_generation;
    if (scratch.best_generation == 0) {
        std::fill(scratch.best_generations.begin(), scratch.best_generations.end(), uint32_t{0});
        scratch.best_generation = 1;
    }
    const uint32_t best_generation = scratch.best_generation;
    const auto representative_started = Clock::now();

    auto landing_for = [&](int32_t state_id) -> int32_t {
        Clock::time_point landing_started{};
        if constexpr (Profile) {
            ++counters.landing_queries;
            landing_started = Clock::now();
        }
        int32_t final_state = scratch.landing_state[static_cast<size_t>(state_id)];
        if (final_state != kNoLanding) {
            if constexpr (Profile) {
                ++counters.landing_cache_hits;
            }
        } else {
            final_state = compute_landing(state_id);
        }
        if constexpr (Profile) {
            timings.landing_seconds += seconds_between(landing_started, Clock::now());
        }
        return final_state;
    };

    if (!table.piece_is_t) {
        for (int32_t state_id : scratch.visited_state_ids) {
            const int32_t final_state = landing_for(state_id);
            if (table.geometry_invalid[static_cast<size_t>(final_state)] != 0) {
                if constexpr (Profile) {
                    ++counters.representative_nodes;
                }
                continue;
            }
            const int32_t rotation_info = scratch.state_kick_infos[static_cast<size_t>(state_id)];
            const bool last_rotation = rotation_info >= 0;
            BestRecord candidate;
            candidate.negative_depth = -scratch.state_depths[static_cast<size_t>(state_id)];
            candidate.order_rank = state_id;
            candidate.placement.x = table.state_x[static_cast<size_t>(state_id)];
            candidate.placement.y = table.state_y[static_cast<size_t>(final_state)];
            candidate.placement.rotation = state_id & 3;
            candidate.placement.last_rotation = last_rotation;
            candidate.placement.kick_index = last_rotation ? (rotation_info & kKickIndexMask) : -1;
            candidate.placement.rotation_from = last_rotation ? (rotation_info >> kKickIndexBits) : -1;
            candidate.placement.rotation_to = last_rotation ? (state_id & 3) : -1;

            const int32_t geometry_id = table.geometry_ids[static_cast<size_t>(final_state)];
            auto& generation = scratch.best_generations[static_cast<size_t>(geometry_id)];
            BestRecord& current = scratch.best_records[static_cast<size_t>(geometry_id)];
            if (generation != best_generation) {
                generation = best_generation;
                scratch.touched_geometry_ids.push_back(geometry_id);
                current = candidate;
            } else if (better_non_t(candidate, current)) {
                current = candidate;
            }
            if constexpr (Profile) {
                ++counters.representative_nodes;
            }
        }
    } else {
        auto emit = [&](int32_t state_id, int32_t depth, int32_t rotation_info, int32_t order_rank) {
            const int32_t final_state = landing_for(state_id);
            if (table.geometry_invalid[static_cast<size_t>(final_state)] == 0) {
                const bool last_rotation = rotation_info >= 0;
                const int32_t kick_index = last_rotation ? (rotation_info & kKickIndexMask) : -1;
                BestRecord candidate;
                if (last_rotation) {
                    const int spin = classify_t_spin(
                        rows,
                        table.width,
                        table.height,
                        table.state_x[static_cast<size_t>(state_id)],
                        table.state_y[static_cast<size_t>(final_state)],
                        state_id & 3,
                        kick_index
                    );
                    candidate.spin_any = spin != 0;
                    candidate.spin_full = spin == 2;
                }
                candidate.negative_depth = -depth;
                candidate.order_rank = order_rank;
                candidate.placement.x = table.state_x[static_cast<size_t>(state_id)];
                candidate.placement.y = table.state_y[static_cast<size_t>(final_state)];
                candidate.placement.rotation = state_id & 3;
                candidate.placement.last_rotation = last_rotation;
                candidate.placement.kick_index = kick_index;
                candidate.placement.rotation_from = last_rotation ? (rotation_info >> kKickIndexBits) : -1;
                candidate.placement.rotation_to = last_rotation ? (state_id & 3) : -1;

                const int32_t geometry_id = table.geometry_ids[static_cast<size_t>(final_state)];
                auto& generation = scratch.best_generations[static_cast<size_t>(geometry_id)];
                BestRecord& current = scratch.best_records[static_cast<size_t>(geometry_id)];
                if (generation != best_generation) {
                    generation = best_generation;
                    scratch.touched_geometry_ids.push_back(geometry_id);
                    current = candidate;
                } else if (better_t(candidate, current)) {
                    current = candidate;
                }
            }
            if constexpr (Profile) {
                ++counters.representative_nodes;
            }
        };

        for (int32_t state_id : scratch.visited_state_ids) {
            emit(
                state_id,
                scratch.state_depths[static_cast<size_t>(state_id)],
                scratch.state_kick_infos[static_cast<size_t>(state_id)],
                state_id
            );
        }
        const int32_t rotation_phase_base = table.state_count;
        for (int32_t state_id : scratch.visited_rotation_ids) {
            if (scratch.rotation_is_geometry[static_cast<size_t>(state_id)] != 0) {
                if constexpr (Profile) {
                    ++counters.representative_duplicate_skips;
                }
                continue;
            }
            emit(
                state_id,
                scratch.rotation_depths[static_cast<size_t>(state_id)],
                scratch.rotation_kick_infos[static_cast<size_t>(state_id)],
                rotation_phase_base + state_id
            );
        }
    }

    if constexpr (Profile) {
        const double representative_total = seconds_between(representative_started, Clock::now());
        timings.representative_seconds = std::max(0.0, representative_total - timings.landing_seconds);
    }

    const auto placement_started = Clock::now();
    result.placements.reserve(scratch.touched_geometry_ids.size());
    for (int32_t geometry_id : scratch.touched_geometry_ids) {
        result.placements.push_back(
            scratch.best_records[static_cast<size_t>(geometry_id)].placement
        );
    }
    std::sort(
        result.placements.begin(),
        result.placements.end(),
        [](const PlacementRecord& a, const PlacementRecord& b) {
            if (a.rotation != b.rotation) return a.rotation < b.rotation;
            if (a.x != b.x) return a.x < b.x;
            return a.y < b.y;
        }
    );
    if constexpr (Profile) {
        timings.placement_seconds = seconds_between(placement_started, Clock::now());
    }
    return result;
}

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
    if (width <= 0 || width > 64 || height <= 0 || width * height > 256) {
        throw std::runtime_error("native reachability table exceeds supported board mask size");
    }
    const size_t state_count = static_cast<size_t>(py::len(state_x));
    if (static_cast<size_t>(py::len(rotation_transitions)) != state_count) {
        throw std::runtime_error("rotation transition state count mismatch");
    }

    auto table = std::make_shared<Table>();
    table->width = width;
    table->height = height;
    table->x_min = x_min;
    table->x_max = x_max;
    table->x_count = x_count;
    table->y_min = y_min;
    table->state_count = static_cast<int>(state_count);
    table->limb_count = (width * height + 63) / 64;
    table->piece_is_t = piece == "T";
    table->state_x = to_i32_vector(state_x, state_count, "state_x");
    table->state_y = to_i32_vector(state_y, state_count, "state_y");
    table->left_state = to_i32_vector(left_state, state_count, "left_state");
    table->right_state = to_i32_vector(right_state, state_count, "right_state");
    table->down_state = to_i32_vector(down_state, state_count, "down_state");
    table->collision_invalid = to_u8_bytes(collision_invalid, state_count, "collision_invalid");
    table->geometry_invalid = to_u8_bytes(geometry_invalid, state_count, "geometry_invalid");
    table->collision_masks = decode_masks(collision_masks, state_count, "collision_masks");
    table->geometry_masks = decode_masks(geometry_masks, state_count, "geometry_masks");
    table->geometry_ids.assign(state_count, -1);
    std::unordered_map<Mask256, int32_t, MaskHash> geometry_lookup;
    geometry_lookup.reserve(state_count);
    for (size_t state_id = 0; state_id < state_count; ++state_id) {
        if (table->geometry_invalid[state_id] != 0) {
            continue;
        }
        const Mask256& mask = table->geometry_masks[state_id];
        const auto inserted = geometry_lookup.emplace(mask, table->geometry_count);
        if (inserted.second) {
            ++table->geometry_count;
        }
        table->geometry_ids[state_id] = inserted.first->second;
    }

    table->state_group_offsets.reserve(state_count + 1);
    table->state_group_offsets.push_back(0);
    table->group_kick_offsets.push_back(0);
    for (py::handle state_groups_handle : rotation_transitions) {
        py::sequence state_groups = py::reinterpret_borrow<py::sequence>(state_groups_handle);
        for (py::handle kicks_handle : state_groups) {
            py::sequence kicks = py::reinterpret_borrow<py::sequence>(kicks_handle);
            for (py::handle kick_handle : kicks) {
                py::sequence kick = py::reinterpret_borrow<py::sequence>(kick_handle);
                if (py::len(kick) != 2) {
                    throw std::runtime_error("kick tuple must have two entries");
                }
                table->kick_targets.push_back(py::cast<int32_t>(kick[0]));
                table->kick_indices.push_back(py::cast<int8_t>(kick[1]));
            }
            table->group_kick_offsets.push_back(static_cast<uint32_t>(table->kick_targets.size()));
        }
        table->state_group_offsets.push_back(static_cast<uint32_t>(table->group_kick_offsets.size() - 1));
    }

    g_tables.push_back(std::move(table));
    return static_cast<int>(g_tables.size() - 1);
}

RunResult execute_run(
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

    RunResult result;
    {
        py::gil_scoped_release release;
        result = profile
            ? run_native<true>(*table, rows, start_x, start_y, start_rotation, max_nodes)
            : run_native<false>(*table, rows, start_x, start_y, start_rotation, max_nodes);
    }
    return result;
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
}
