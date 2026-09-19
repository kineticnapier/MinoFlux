#include "oracle_core.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace minoflux::oracle {
namespace {

struct Cell {
    int8_t x;
    int8_t y;
};

constexpr Cell C(int x, int y) {
    return Cell{static_cast<int8_t>(x), static_cast<int8_t>(y)};
}

using Shape = std::array<Cell, 4>;
using Rotations = std::array<Shape, 4>;

constexpr std::array<Rotations, 7> kShapes = {
    Rotations{
        Shape{C(0, 1), C(1, 1), C(2, 1), C(3, 1)},
        Shape{C(2, 0), C(2, 1), C(2, 2), C(2, 3)},
        Shape{C(0, 2), C(1, 2), C(2, 2), C(3, 2)},
        Shape{C(1, 0), C(1, 1), C(1, 2), C(1, 3)},
    },
    Rotations{
        Shape{C(1, 0), C(2, 0), C(1, 1), C(2, 1)},
        Shape{C(1, 0), C(2, 0), C(1, 1), C(2, 1)},
        Shape{C(1, 0), C(2, 0), C(1, 1), C(2, 1)},
        Shape{C(1, 0), C(2, 0), C(1, 1), C(2, 1)},
    },
    Rotations{
        Shape{C(1, 0), C(0, 1), C(1, 1), C(2, 1)},
        Shape{C(1, 0), C(1, 1), C(2, 1), C(1, 2)},
        Shape{C(0, 1), C(1, 1), C(2, 1), C(1, 2)},
        Shape{C(1, 0), C(0, 1), C(1, 1), C(1, 2)},
    },
    Rotations{
        Shape{C(1, 0), C(2, 0), C(0, 1), C(1, 1)},
        Shape{C(1, 0), C(1, 1), C(2, 1), C(2, 2)},
        Shape{C(1, 1), C(2, 1), C(0, 2), C(1, 2)},
        Shape{C(0, 0), C(0, 1), C(1, 1), C(1, 2)},
    },
    Rotations{
        Shape{C(0, 0), C(1, 0), C(1, 1), C(2, 1)},
        Shape{C(2, 0), C(1, 1), C(2, 1), C(1, 2)},
        Shape{C(0, 1), C(1, 1), C(1, 2), C(2, 2)},
        Shape{C(1, 0), C(0, 1), C(1, 1), C(0, 2)},
    },
    Rotations{
        Shape{C(0, 0), C(0, 1), C(1, 1), C(2, 1)},
        Shape{C(1, 0), C(2, 0), C(1, 1), C(1, 2)},
        Shape{C(0, 1), C(1, 1), C(2, 1), C(2, 2)},
        Shape{C(1, 0), C(1, 1), C(0, 2), C(1, 2)},
    },
    Rotations{
        Shape{C(2, 0), C(0, 1), C(1, 1), C(2, 1)},
        Shape{C(1, 0), C(1, 1), C(1, 2), C(2, 2)},
        Shape{C(0, 1), C(1, 1), C(2, 1), C(0, 2)},
        Shape{C(0, 0), C(1, 0), C(1, 1), C(1, 2)},
    },
};

thread_local bool g_search_profile_enabled = false;
thread_local SearchProfile g_search_profile{};

bool collides(
    const std::array<uint16_t, kHeight>& rows,
    Piece piece,
    int x,
    int y,
    int rotation
) noexcept {
    const auto& cells = kShapes[static_cast<size_t>(piece)][static_cast<size_t>(rotation & 3)];
    for (const Cell cell : cells) {
        const int cx = x + cell.x;
        const int cy = y + cell.y;
        if (cx < 0 || cx >= kWidth || cy >= kHeight) {
            return true;
        }
        if (cy >= 0 && (rows[static_cast<size_t>(cy)] & (uint16_t{1} << cx))) {
            return true;
        }
    }
    return false;
}

bool occupied_or_wall(
    const std::array<uint16_t, kHeight>& rows,
    int x,
    int y
) noexcept {
    if (x < 0 || x >= kWidth || y < 0 || y >= kHeight) {
        return true;
    }
    return (rows[static_cast<size_t>(y)] & (uint16_t{1} << x)) != 0;
}

int classify_t_spin(
    const std::array<uint16_t, kHeight>& rows,
    int x,
    int y,
    int rotation,
    int kick_index
) noexcept {
    const int px = x + 1;
    const int py = y + 1;
    const std::array<bool, 4> corners = {
        occupied_or_wall(rows, px - 1, py - 1),
        occupied_or_wall(rows, px + 1, py - 1),
        occupied_or_wall(rows, px - 1, py + 1),
        occupied_or_wall(rows, px + 1, py + 1),
    };
    if (int(corners[0]) + int(corners[1]) + int(corners[2]) + int(corners[3]) < 3) {
        return 0;
    }
    constexpr std::array<std::array<int, 2>, 4> front = {{{{0, 1}}, {{1, 3}}, {{2, 3}}, {{0, 2}}}};
    const auto pair = front[static_cast<size_t>(rotation & 3)];
    return (corners[pair[0]] && corners[pair[1]]) || kick_index == 4 ? 2 : 1;
}

struct Features {
    int aggregate_height = 0;
    int max_height = 0;
    int holes = 0;
    int hole_depth = 0;
    int bumpiness = 0;
    int wells = 0;
    int t_spin_slots = 0;
};

bool cell_occupied(
    const std::array<uint16_t, kHeight>& rows,
    int x,
    int y
) noexcept {
    if (x < 0 || x >= kWidth || y < 0 || y >= kHeight) {
        return true;
    }
    return (rows[static_cast<size_t>(y)] & (uint16_t{1} << x)) != 0;
}

bool cell_empty(
    const std::array<uint16_t, kHeight>& rows,
    int x,
    int y
) noexcept {
    return x >= 0 && x < kWidth && y >= 0 && y < kHeight && !cell_occupied(rows, x, y);
}

Features features_impl(const std::array<uint16_t, kHeight>& rows) {
    Features result;
    std::array<int, kWidth> heights{};

    for (int x = 0; x < kWidth; ++x) {
        int top = -1;
        int occupied_above = 0;
        for (int y = 0; y < kHeight; ++y) {
            const bool occupied = (rows[static_cast<size_t>(y)] & (uint16_t{1} << x)) != 0;
            if (occupied) {
                if (top < 0) {
                    top = y;
                }
                ++occupied_above;
            } else if (top >= 0) {
                ++result.holes;
                result.hole_depth += occupied_above;
            }
        }
        heights[static_cast<size_t>(x)] = top < 0 ? 0 : kHeight - top;
    }

    for (int height : heights) {
        result.aggregate_height += height;
    }
    result.max_height = *std::max_element(heights.begin(), heights.end());
    for (int x = 0; x < kWidth - 1; ++x) {
        result.bumpiness += std::abs(
            heights[static_cast<size_t>(x)] - heights[static_cast<size_t>(x + 1)]
        );
    }

    for (int x = 0; x < kWidth; ++x) {
        int run = 0;
        for (int y = 0; y < kHeight; ++y) {
            const bool well = !cell_occupied(rows, x, y) &&
                (x == 0 || cell_occupied(rows, x - 1, y)) &&
                (x == kWidth - 1 || cell_occupied(rows, x + 1, y));
            if (well) {
                ++run;
            } else {
                result.wells += run * (run + 1) / 2;
                run = 0;
            }
        }
        result.wells += run * (run + 1) / 2;
    }

    for (int y = 0; y < kHeight; ++y) {
        for (int x = 0; x < kWidth; ++x) {
            if (cell_occupied(rows, x, y)) {
                continue;
            }
            const int corners =
                int(cell_occupied(rows, x - 1, y - 1)) +
                int(cell_occupied(rows, x + 1, y - 1)) +
                int(cell_occupied(rows, x - 1, y + 1)) +
                int(cell_occupied(rows, x + 1, y + 1));
            if (corners < 3) {
                continue;
            }
            const int cardinals =
                int(cell_empty(rows, x, y - 1)) +
                int(cell_empty(rows, x, y + 1)) +
                int(cell_empty(rows, x - 1, y)) +
                int(cell_empty(rows, x + 1, y));
            if (cardinals >= 3) {
                ++result.t_spin_slots;
            }
        }
    }
    return result;
}

Features features(const std::array<uint16_t, kHeight>& rows) {
    if (!g_search_profile_enabled) {
        return features_impl(rows);
    }
    const auto started = std::chrono::steady_clock::now();
    Features result = features_impl(rows);
    const auto stopped = std::chrono::steady_clock::now();
    ++g_search_profile.feature_calls;
    g_search_profile.feature_seconds +=
        std::chrono::duration<double>(stopped - started).count();
    return result;
}

double board_score(const Features& features_value, bool game_over) noexcept {
    const double density =
        double(features_value.t_spin_slots) / (1.0 + double(features_value.holes));
    double score =
        features_value.aggregate_height * -0.510066 +
        features_value.max_height * -0.08 +
        features_value.holes * -0.8 +
        features_value.hole_depth * -0.12 +
        features_value.bumpiness * -0.184483 +
        features_value.wells * -0.06 +
        features_value.t_spin_slots * 1.2 +
        density * 0.28;
    if (game_over) {
        score -= 1'000'000.0;
    }
    return score;
}

int spin_event(int kind, int lines) noexcept {
    if (!kind) {
        return 0;
    }
    if (kind == 1) {
        if (lines == 0) return 1;
        if (lines == 1) return 2;
        return lines == 2 ? 5 : 6;
    }
    if (lines == 0) return 3;
    if (lines == 1) return 4;
    if (lines == 2) return 5;
    return 6;
}

int base_attack(int lines, int event) noexcept {
    if (event == 2) return 0;
    if (event == 4) return 2;
    if (event == 5) return 4;
    if (event == 6) return 6;
    switch (lines) {
        case 2: return 1;
        case 3: return 2;
        case 4: return 4;
        default: return 0;
    }
}

bool difficult_clear(int lines, int event) noexcept {
    return lines == 4 || (event != 0 && lines > 0);
}

struct LockValue {
    int lines = 0;
    int attack = 0;
    int spin_lines = 0;
    bool pc = false;
    int new_holes = 0;
};

bool spawn_next(State& state, std::span<const Piece> queue) noexcept {
    if (state.queue_index >= queue.size()) {
        return false;
    }
    state.current = queue[state.queue_index++];
    state.can_hold = true;
    return state.current != Piece::None && !collides(state.rows, state.current, 3, 1, 0);
}

bool apply_placement(
    State& state,
    std::span<const Piece> queue,
    const Move& move,
    int parent_holes,
    LockValue& value,
    Features& after_features
) {
    const int spin_kind =
        (move.piece == Piece::T && move.last_rotation)
            ? classify_t_spin(state.rows, move.x, move.y, move.rotation, move.kick_index)
            : 0;
    bool topped = false;
    const auto& cells =
        kShapes[static_cast<size_t>(move.piece)][static_cast<size_t>(move.rotation & 3)];
    for (const Cell cell : cells) {
        const int x = move.x + cell.x;
        const int y = move.y + cell.y;
        if (y < 0) {
            topped = true;
        } else {
            state.rows[static_cast<size_t>(y)] |= uint16_t{1} << x;
        }
    }

    std::array<uint16_t, kHeight> cleared{};
    int write = kHeight - 1;
    int lines = 0;
    for (int y = kHeight - 1; y >= 0; --y) {
        if (state.rows[static_cast<size_t>(y)] == kFullRow) {
            ++lines;
        } else {
            cleared[static_cast<size_t>(write--)] = state.rows[static_cast<size_t>(y)];
        }
    }
    state.rows = cleared;

    const bool pc = std::all_of(
        state.rows.begin(), state.rows.end(), [](uint16_t row) { return row == 0; }
    );
    const int event = spin_event(spin_kind, lines);
    const bool difficult = difficult_clear(lines, event);
    const bool was_active = state.b2b_active;
    const int chain = was_active ? state.b2b_chain : 0;
    int bonus = 0;
    int released = 0;
    int next_chain = chain;
    bool next_active = was_active;

    if (lines == 0) {
    } else if (pc) {
        bonus = was_active ? 1 : 0;
        next_active = true;
        next_chain = was_active ? chain + 2 : 2;
    } else if (difficult) {
        bonus = was_active ? 1 : 0;
        next_active = true;
        next_chain = was_active ? chain + 1 : 0;
    } else {
        released = (was_active && chain >= 4) ? chain : 0;
        next_active = false;
        next_chain = 0;
    }

    int attack = base_attack(lines, event) + bonus + released;
    if (lines) {
        ++state.combo;
        if (state.combo > 0) {
            attack += std::min(4, state.combo / 2 + 1);
        }
    } else {
        state.combo = -1;
    }
    if (pc && lines) {
        attack += 10;
    }

    state.b2b_active = next_active;
    state.b2b_chain = static_cast<uint16_t>(std::max(0, next_chain));
    state.game_over = topped;
    for (int y = 0; y < kHiddenRows; ++y) {
        if (state.rows[static_cast<size_t>(y)] != 0) {
            state.game_over = true;
        }
    }

    after_features = features(state.rows);
    value.lines = lines;
    value.attack = attack;
    value.spin_lines = event ? lines : 0;
    value.pc = pc;
    value.new_holes = std::max(0, after_features.holes - parent_holes);

    if (!state.game_over && !spawn_next(state, queue)) {
        state.game_over = true;
    }
    return true;
}

double event_score(const State& state, const LockValue& value) noexcept {
    return value.attack * 0.85 +
        value.lines * 0.760666 +
        value.spin_lines * 1.25 +
        (value.pc ? 8.0 : 0.0) +
        value.new_holes * -1.2 +
        std::max<int>(0, state.combo) * 0.03 +
        state.b2b_chain * 0.05;
}

struct StateKey {
    std::array<uint16_t, kHeight> rows{};
    Piece current = Piece::None;
    Piece hold = Piece::None;
    uint16_t queue_index = 0;
    int16_t combo = -1;
    uint16_t b2b_chain = 0;
    uint8_t flags = 0;

    bool operator==(const StateKey& other) const noexcept {
        return rows == other.rows &&
            current == other.current &&
            hold == other.hold &&
            queue_index == other.queue_index &&
            combo == other.combo &&
            b2b_chain == other.b2b_chain &&
            flags == other.flags;
    }
};

struct StateHash {
    size_t operator()(const StateKey& state) const noexcept {
        uint64_t hash = 1469598103934665603ULL;
        auto mix = [&](uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ULL;
        };
        for (uint16_t row : state.rows) {
            mix(row);
        }
        mix(static_cast<uint8_t>(state.current));
        mix(static_cast<uint8_t>(state.hold));
        mix(state.queue_index);
        mix(static_cast<uint16_t>(state.combo));
        mix(state.b2b_chain);
        mix(state.flags);
        return static_cast<size_t>(hash ^ (hash >> 32));
    }
};

StateKey key_of(const State& state) noexcept {
    StateKey key;
    key.rows = state.rows;
    key.current = state.current;
    key.hold = state.hold;
    key.queue_index = state.queue_index;
    key.combo = state.combo;
    key.b2b_chain = state.b2b_chain;
    key.flags =
        uint8_t(state.has_hold) |
        (uint8_t(state.can_hold) << 1) |
        (uint8_t(state.b2b_active) << 2) |
        (uint8_t(state.game_over) << 3);
    return key;
}

struct Node {
    State state{};
    double path_score = 0.0;
    double rank_score = 0.0;
    Move root{};
    bool has_root = false;
};

bool move_less(const Move& left, const Move& right) noexcept {
    if (left.use_hold != right.use_hold) return left.use_hold < right.use_hold;
    if (left.piece != right.piece) return static_cast<int>(left.piece) < static_cast<int>(right.piece);
    if (left.rotation != right.rotation) return left.rotation < right.rotation;
    if (left.x != right.x) return left.x < right.x;
    if (left.y != right.y) return left.y < right.y;
    if (left.last_rotation != right.last_rotation) return left.last_rotation < right.last_rotation;
    return left.kick_index < right.kick_index;
}

bool better_node(const Node& left, const Node& right) noexcept {
    if (left.rank_score != right.rank_score) return left.rank_score > right.rank_score;
    if (left.path_score != right.path_score) return left.path_score > right.path_score;
    if (left.state.game_over != right.state.game_over) return !left.state.game_over;
    if (left.state.rows != right.state.rows) return left.state.rows < right.state.rows;
    return move_less(left.root, right.root);
}

void insert_dedup(
    std::vector<Node>& children,
    std::unordered_map<StateKey, size_t, StateHash>& indices,
    Node&& child
) {
    const bool profiling = g_search_profile_enabled;
    const auto started = profiling ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

    StateKey key = key_of(child.state);
    auto [iterator, inserted] = indices.emplace(std::move(key), children.size());
    if (inserted) {
        children.push_back(std::move(child));
    } else {
        if (profiling) {
            ++g_search_profile.dedup_hits;
        }
        if (better_node(child, children[iterator->second])) {
            if (profiling) {
                ++g_search_profile.dedup_replacements;
            }
            children[iterator->second] = std::move(child);
        }
    }

    if (profiling) {
        const auto stopped = std::chrono::steady_clock::now();
        g_search_profile.dedup_seconds +=
            std::chrono::duration<double>(stopped - started).count();
    }
}

void expand_branch(
    const Node& node,
    std::span<const Piece> queue,
    const Config& config,
    bool use_hold,
    int ply,
    const Features& parent_features,
    std::vector<Node>& children,
    std::unordered_map<StateKey, size_t, StateHash>& indices
) {
    State branch = node.state;
    Piece placing = branch.current;
    if (use_hold) {
        if (!branch.can_hold) {
            return;
        }
        const Piece outgoing = branch.current;
        if (branch.has_hold) {
            placing = branch.hold;
        } else {
            if (branch.queue_index >= queue.size()) {
                return;
            }
            placing = queue[branch.queue_index++];
            if (placing == Piece::None) {
                return;
            }
        }
        branch.hold = outgoing;
        branch.has_hold = true;
        branch.can_hold = false;
        branch.current = placing;
        if (collides(branch.rows, placing, 3, 1, 0)) {
            return;
        }
    }

    const auto moves = reachable_moves(
        branch.rows,
        placing,
        config.allow_180,
        config.max_nodes,
        use_hold
    );
    for (const Move& move : moves) {
        if (g_search_profile_enabled) {
            ++g_search_profile.expanded_children;
        }
        State child_state = branch;
        LockValue value;
        Features child_features;
        apply_placement(
            child_state,
            queue,
            move,
            parent_features.holes,
            value,
            child_features
        );

        Node child;
        child.state = child_state;
        child.path_score = node.path_score + event_score(child_state, value);
        child.rank_score =
            child.path_score + board_score(child_features, child_state.game_over);
        child.root = ply == 0 ? move : node.root;
        child.has_root = true;
        insert_dedup(children, indices, std::move(child));
    }
}

}  // namespace

void begin_search_profile() {
    g_search_profile = SearchProfile{};
    g_search_profile_enabled = true;
}

SearchProfile end_search_profile() {
    g_search_profile_enabled = false;
    return g_search_profile;
}

Piece piece_from_char(char value) {
    switch (value) {
        case 'I': case 'i': return Piece::I;
        case 'O': case 'o': return Piece::O;
        case 'T': case 't': return Piece::T;
        case 'S': case 's': return Piece::S;
        case 'Z': case 'z': return Piece::Z;
        case 'J': case 'j': return Piece::J;
        case 'L': case 'l': return Piece::L;
        default: return Piece::None;
    }
}

char piece_to_char(Piece piece) {
    constexpr std::array<char, 7> names = {'I', 'O', 'T', 'S', 'Z', 'J', 'L'};
    const int index = static_cast<int>(piece);
    return index >= 0 && index < 7 ? names[static_cast<size_t>(index)] : '?';
}

Result search(
    const State& root,
    std::span<const Piece> queue,
    const Config& raw_config
) {
    Config config = raw_config;
    config.beam_width = std::max(1, config.beam_width);
    config.depth = std::max(1, config.depth);
    config.max_nodes = std::max(1, config.max_nodes);
    if (root.game_over || root.current == Piece::None) {
        return {};
    }

    Node start;
    start.state = root;
    start.rank_score = board_score(features(root.rows), false);
    std::vector<Node> frontier{start};

    for (int ply = 0; ply < config.depth; ++ply) {
        std::vector<Node> children;
        children.reserve(static_cast<size_t>(config.beam_width) * 48);
        std::unordered_map<StateKey, size_t, StateHash> indices;
        indices.reserve(static_cast<size_t>(config.beam_width) * 64);

        for (const Node& node : frontier) {
            if (node.state.game_over) {
                if (node.has_root) {
                    insert_dedup(children, indices, Node(node));
                }
                continue;
            }

            const Features parent_features = features(node.state.rows);
            expand_branch(
                node,
                queue,
                config,
                false,
                ply,
                parent_features,
                children,
                indices
            );
            expand_branch(
                node,
                queue,
                config,
                true,
                ply,
                parent_features,
                children,
                indices
            );
        }

        if (children.empty()) {
            break;
        }

        const bool profiling = g_search_profile_enabled;
        const auto prune_started = profiling
            ? std::chrono::steady_clock::now()
            : std::chrono::steady_clock::time_point{};
        if (static_cast<int>(children.size()) > config.beam_width) {
            auto middle = children.begin() + config.beam_width;
            std::nth_element(
                children.begin(),
                middle,
                children.end(),
                [](const Node& left, const Node& right) { return better_node(left, right); }
            );
            children.resize(static_cast<size_t>(config.beam_width));
        }
        std::sort(
            children.begin(),
            children.end(),
            [](const Node& left, const Node& right) { return better_node(left, right); }
        );
        if (profiling) {
            const auto prune_stopped = std::chrono::steady_clock::now();
            g_search_profile.prune_seconds +=
                std::chrono::duration<double>(prune_stopped - prune_started).count();
        }
        frontier.swap(children);
    }

    if (frontier.empty() || !frontier.front().has_root) {
        return {};
    }

    const Node& best = frontier.front();
    Result result;
    result.found = true;
    result.move = best.root;
    result.score = best.rank_score;
    return result;
}

}  // namespace minoflux::oracle
