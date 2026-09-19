#pragma once

#include "reachability_core.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace minoflux::oracle {

constexpr int kWidth = 10;
constexpr int kHeight = 24;
constexpr int kHiddenRows = 4;
constexpr uint16_t kFullRow = (1u << kWidth) - 1u;

enum class Piece : int8_t {
    I = 0,
    O = 1,
    T = 2,
    S = 3,
    Z = 4,
    J = 5,
    L = 6,
    None = -1,
};

struct Move {
    Piece piece = Piece::None;
    int8_t x = 0;
    int8_t y = 0;
    int8_t rotation = 0;
    bool use_hold = false;
    bool last_rotation = false;
    int8_t kick_index = -1;
    int8_t rotation_from = -1;
    int8_t rotation_to = -1;
};

struct State {
    std::array<uint16_t, kHeight> rows{};
    Piece current = Piece::T;
    Piece hold = Piece::None;
    uint16_t queue_index = 0;
    int16_t combo = -1;
    uint16_t b2b_chain = 0;
    bool has_hold = false;
    bool can_hold = true;
    bool b2b_active = false;
    bool game_over = false;
};

struct Config {
    int beam_width = 2000;
    int depth = 18;
    bool allow_180 = true;
    int max_nodes = 8000;
};

struct Result {
    bool found = false;
    Move move{};
    double score = 0.0;
};

struct TransitionResult {
    bool valid = false;
    State state{};
    int lines = 0;
    int attack = 0;
    int spin_event = 0;
    bool perfect_clear = false;
    int surge_released = 0;
    int surge_charge = 0;
};

struct BoardFeatures {
    int aggregate_height = 0;
    int max_height = 0;
    int holes = 0;
    int hole_depth = 0;
    int bumpiness = 0;
    int wells = 0;
    int t_spin_slots = 0;
};

struct ReachabilityProfile {
    uint64_t calls = 0;
    uint64_t generated_moves = 0;
    double total_seconds = 0.0;
    double setup_seconds = 0.0;
    double bfs_seconds = 0.0;
    double rotation_seconds = 0.0;
    double landing_seconds = 0.0;
    double representative_seconds = 0.0;
    double placement_seconds = 0.0;
};

struct SearchProfile {
    uint64_t feature_calls = 0;
    uint64_t expanded_children = 0;
    uint64_t dedup_hits = 0;
    uint64_t dedup_replacements = 0;
    double feature_seconds = 0.0;
    double dedup_seconds = 0.0;
    double prune_seconds = 0.0;
};

Piece piece_from_char(char value);
char piece_to_char(Piece piece);
BoardFeatures board_features(const std::array<uint16_t, kHeight>& rows);
void register_reachability_table(
    Piece piece,
    bool allow_180,
    std::shared_ptr<const minoflux::reachability::Table> table
);
void begin_reachability_profile();
ReachabilityProfile end_reachability_profile();
void begin_search_profile();
SearchProfile end_search_profile();
std::vector<Move> reachable_moves(
    const std::array<uint16_t, kHeight>& rows,
    Piece piece,
    bool allow_180,
    int max_nodes,
    bool use_hold = false
);
TransitionResult transition(
    const State& root,
    std::span<const Piece> queue,
    const Move& move
);
Result search(
    const State& root,
    std::span<const Piece> queue,
    const Config& config
);

}  // namespace minoflux::oracle
